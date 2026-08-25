# 2026-08-23 — bigPP L1: sliced pipelined staging + shm 2-rank reduce (fp8 lane)

Context: `docs/bigPP.md` lever L1. On qwen38tp2fp8 the host-staged
collective path is 94-100% of prefill forward (E0 split in
`bench/reference/bigPP-fp8-report.md`). A fully async cross-layer
pipeline is not implementable inside XpuCommunicator (the reduced tensor is
consumed immediately after all_reduce returns, so H2D must be
stream-ordered before the consumer — a host-mediated reduce forces one
host wait per collective). What is implemented instead:

1. **Sliced pipelined staging** (`VLLM_XPU_STAGE_SLICE_KB`, default 0=off):
   each staged all_reduce is split into ~8 MiB slices; D2H slice copies are
   enqueued async with one event per slice; a dedicated worker thread runs
   the CPU reduce per slice as its D2H lands; the engine enqueues each H2D
   slice as its reduce completes. Wall per collective goes from
   D2H+reduce+H2D toward max(D2H+H2D, reduce). Bit-exact: all_reduce is
   elementwise, slices are independent, engine returns only after all H2D
   enqueued (consumer stream-order unchanged), gloo call order per rank
   unchanged (a sliced op fully completes before the engine issues anything
   else on the CPU group).
2. **shm 2-rank reduce** (`VLLM_XPU_STAGE_SHM`, default 0=off, sliced path
   only, world==2): replaces gloo's loopback TCP (~4.6 GB/s measured) with
   a /dev/shm exchange (random per-boot name rendezvoused over gloo
   broadcast; region-per-rank + rdy/ack spin barriers with 60 s deadlock
   timeout; torch CPU add on frombuffer views). Measured bit-identical to
   gloo's 2-rank sum. ~7x faster than gloo at 8 MiB slices on this host.
   Segments: owner unlinks at exit; crashed runs leak one (~16 MB);
   `_gc_stale` reaps >1h-old ones at the next boot.

## Files

- Fork: `src/vllm/vllm/distributed/device_communicators/xpu_communicator.py`
  (`_staged_all_reduce_sliced`, `_stage_worker`, `_ShmPair`; blocking path
  untouched and still the default when the envs are unset).
- `ut_stage_async.py` — 2-process (one B70 each) test driving the REAL
  XpuCommunicator: bit-exactness vs CPU reference sum across shapes/dtypes/
  configs under `torch.inference_mode()`, plus a GEMM+all_reduce throughput
  loop. PASS criteria: bit-exact everywhere; sliced >=1.2x blocking;
  sliced+shm > sliced+gloo.

## Regressions found by this pack during development

- Worker-thread exceptions previously killed the worker silently and hung
  the engine on the done-queue; worker now propagates exceptions.
- vLLM forwards run under `torch.inference_mode()`, so the pinned staging
  buffers are inference tensors; the shm worker's `torch.add(out=...)`
  raised `Inplace update to inference tensor outside InferenceMode`
  (boot hang at the memory-profile dummy run). Fix: re-enter
  inference_mode in the reduce. The test covers this by running under
  inference_mode.

## Results (unit, 84 MB payload, GEMM+all_reduce loop)

| config | ms/iter | vs blocking |
| --- | ---: | ---: |
| blocking | 63.1-64.0 | 1.00x |
| sliced 512 KiB | 49.9-51.6 | ~1.25x |
| sliced 2 MiB | 43.7-45.7 | ~1.45x |
| sliced 8 MiB | 43.8-45.0 | ~1.43x |
| sliced 2 MiB + shm | 27.3 | 2.31x |
| sliced 8 MiB + shm | 27.3-27.4 | 2.34x |

Server cells (fp8 lane, cold probes, strict suite, quality gate):
`bench/reference/bigPP-fp8-l1a-*` (sliced, gloo) and
`bench/reference/bigPP-fp8-l1b-*` (sliced + shm). Headline: PP 658->969 @240, 733->1379
@633, 825->1620 @8192, 792->1463 @32768; strict decode 66.78 (reference
66.32, in band); quality PASS (documented `logic` casing tolerance; one
non-reproducing cold-boot repeat outlier, attributed to first-request
cold-shape numerics — the repeat prefill DOES cross the 1 MiB staging
threshold, so it was investigated: 20/20 identical steady-state greedy
outputs, cold-boot quality-first rerun PASS).

Production: flipped on for qwen38tp2fp8 via the switch mode env block
(EXTRA_LAUNCH_ENV default `VLLM_XPU_STAGE_SLICE_KB=8192
VLLM_XPU_STAGE_SHM=1`). Revert: `EXTRA_LAUNCH_ENV="VLLM_XPU_STAGE_SLICE_KB=0"
bash ~/b70-switch.sh qwen38tp2fp8 --force`.
