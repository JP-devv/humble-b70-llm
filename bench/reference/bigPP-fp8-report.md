# bigPP on the FP8 lane (qwen38tp2fp8) — E0 + env-day report

Date: 2026-08-23. Host: reference box (2x Arc Pro B70, TP2, no GPU P2P).
Lane: `qwen38tp2fp8` — /data/models/Qwen3.8-27B-Uncensored-bf16-full,
`--quantization fp8` (W8A8 e4m3 dynamic act at load), TP2 + MTP3, INT8
lm_head, `--dtype float16`, 262144 ctx, FP8 KV, sharp chat template,
served id `Qwen3.8-27B-UNC-FP8`, `--max-num-seqs 1`.
**All numbers below are "FP8 lane, no TP1-fp8 baseline"** (a TP1 FP8 lane
cannot exist: 27 GB fp8 trunk > 32 GB card). Number classes are never
mixed: strict cold suite rows and PP probes are cold; no warm rows are
promoted.

Tenants at every cell start: ComfyUI instances on both cards (ports
18188/18189, idle), vram_sampler + portal/simple servers. VRAM used after
vLLM boot: GPU0 29477 MiB, GPU1 28991 MiB (of 32768). Nothing was stopped
except the vLLM server itself (via killvllm.sh through the switch).

## 1. E0 — instrumentation, baseline, split

### 1a. Instrumentation (fork, off by default)

`VLLM_XPU_STAGE_TIMING=1` added to
`vllm/distributed/device_communicators/xpu_communicator.py`:
perf_counter around each blocking phase (D2H / gloo-reduce / H2D) of every
staged collective (XpuCommunicator methods + the functional wrappers used
by the traced MTP proposer path), cumulative calls/bytes/ms, drained once
per engine step and printed as `[xpu-stage-timing]` inside the existing
`VLLM_XPU_STEP_TIMING` channel in `gpu_model_runner.py`. Host-side only —
no `.item()`, no stream sync; decode graph capture unaffected (verified:
decode in band with it on, and graph-capture boot succeeded). One `if`
per collective when unset; env unset = zero change (restore boot shows 0
stage-timing lines).

Switch: two generic hooks added to `switch/b70-switch.sh`:
`EXTRA_LAUNCH_ENV` (placed after the fixed env in start_tp2's tmux line so
per-cell overrides win) and `EXTRA_VLLM_ARGS` (appended in the
qwen38tp2fp8 mode block). Empty in production.

Diffs:
- full fork diff vs HEAD: `bench-results/bigPP-fp8-fork-full-diff-after.patch`
  (includes the pre-existing host-staging/GDN/step-timing patches)
- this session's incremental instrumentation only:
  `bench-results/bigPP-fp8-instrumentation-incremental.patch`
- switch diff: `EXTRA_LAUNCH_ENV` + `EXTRA_VLLM_ARGS` hooks (2 hunks).

### 1b. PP probes (cold; prompt_tokens exact, cached_tokens==0 per request)

| prompt_tokens | TTFT (ms) | PP tok/s | steps (num_sched) |
| ---: | ---: | ---: | --- |
| 240 | 364.9 | 658 | 240 |
| 633 | 863.9 | 733 | 633 |
| 8192 | 9924.2 | 825 | 6400 + 1792 |
| 32768 | 41386 | 792 | 8000 x3 + 6400 + 2368 |

(rep-1 / shape-warm-content-cold values; rep-0 carries lazy shape init,
e.g. 1724 ms at 240. Note the scheduler never emits 8192: observed chunks
cap at 8000, and an 8192 prompt splits 6400+1792 — per-chunk fixed costs
hit more often than `--max-num-batched-tokens 8192` suggests. Relevant to
L5 later.)

Strict-suite TTFT @240-token prompts (same boot): 359.7 ms median —
matches the probe (365 ms) and the pre-investigation reference (370.7 ms).

### 1c. Staged-ms split (the E0 verdict)

`[xpu-stage-timing]` per prefill step, ms (rank0 = x16 card / rank1 = x4
card); 133 calls per step = 128 trunk all-reduces + 5 MTP-proposer
collectives:

| num_sched | fwd ms | staged ms (% fwd) | rank1 d2h / gloo / h2d | rank0 d2h / gloo / h2d |
| ---: | ---: | ---: | --- | --- |
| 240 | 343.5 | 321.9 (94%) | 97.3 / 173.4 / 51.3 | 57.1 / 250.0 / 14.4 |
| 633 | 793.8 | 784.3 (99%) | 245.1 / 413.3 / 125.9 | 139.8 / 607.2 / 33.5 |
| 6400 | 7449.5 | 7555.7 (~100%) | 2535.1 / 3761.8 / 1258.7 | 1486.7 / 5802.3 / 253.9 |
| 8000 | 9285.1 | 9491.0 (~100%) | 3200.8 / 4720.1 / 1570.2 | 2048.4 / 7081.7 / 345.0 |
| 2368 | 3191.2 | 3258.9 (~100%) | 1295.2 / 1477.6 / 486.2 | 900.6 / 2228.7 / 119.7 |

(E0 and L3 splits identical within noise; L3 rows shown where cleaner.
staged_total can exceed fwd slightly — drain-boundary overlap; rank0 gloo
exceeds rank1 because it absorbs the peer-wait skew from the x4 card's
slower D2H. True gloo reduce time = rank1's bucket.)

**The staged path is 94-100% of the prefill forward at every size on this
lane.** The int4-lane TTFT delta mechanism (128 blocking staged
collectives) transfers — but where int4's 240-token delta was ~185 ms of a
360 ms TTFT (~51%), the fp8 lane's comm share is ~94% because its compute
is so much faster.

### 1d. Link state under load

sysfs reads (`current_link_speed/width` for 0000:03:00.0 and
0000:0b:00.0) showed **2.5 GT/s x1 on both cards for the entire 41 s of a
32768-token prefill** — the values never leave the ASPM low-power state on
this platform, even under load. They are not usable as runtime evidence
here. Bandwidth-derived truth instead (H2D phases): rank0/card-03 moved
8.68 GB in 254 ms = **~34 GB/s** (exceeds gen4 x16 raw — gen5 x16-class);
rank1/card-0b moved 8.68 GB in 1259 ms = **~6.9 GB/s** (gen4 x4
practical ceiling). The {x16, x4} topology holds at runtime; the x4 card
is the pacing item, exactly as bigPP §2.3 models it.

### 1e. Prefill-step breakdown (torch profiler, one 633 + one 8192 probe)

Per-rank window ≈ 11 s wall. rank1 (x4 card, critical path), duplicate
parent/child rows resolved:

| bucket | GPU time | share |
| --- | ---: | ---: |
| comm copies (H2D 1709 + D2H 1675 ms) | 3434 ms | 31% |
| GEMM (`fp8_gemm_w8a16` 1329 + `int8_gemm_w8a8` 43) | 1372 ms | 12% |
| attention + GDN (fa2 varlen 181 + gdn_attention 151 + gdn kernels ~150) | ~480 ms | 4% |
| fused quant/norm/elementwise (triton fp8_gemm_w8a16_* etc.) | ~320 ms | 3% |
| **GPU idle** (host blocked: `zeEventHostSynchronize` self CPU 5033 ms; gloo + queue drains) | ~5400 ms | **49%** |

Answers to the sharper-question candidates:
- **Per-token quant dispatch at prefill M**: not it. Dynamic fp8 act-quant
  is inductor-fused into the triton kernels around the GEMMs (~3% total).
- **GDN in_proj bf16**: not it. `_xpu_C::gdn_attention` + gdn kernels ~4%.
- **Chunked-attention path**: not it (fa2 varlen 181 ms per window).
- The fp8 GEMM is fast as expected (1.43 ms avg at M~6400-8000). **The
  expected prefill gain over int4 went entirely into the serialized
  host-staged comm: compute is only ~20% of the window; the GEMM speedup
  is invisible behind 31% copies + 49% idle-in-blocking-comm.**

Effective per-collective numbers at M=8000 (payload 81.9 MB): card2 H2D
6.9 GB/s (x4 ceiling); card1 H2D 34 GB/s; gloo loopback TCP reduce
~35.5 ms/call ≈ **4.6 GB/s** — slow for loopback (single-threaded TCP
path; this venv's gloo has no shm transport).

Decode path: exactly one staged collective per decode step (0.99 MB,
gloo phase 0.5 ms; its 31.7 ms "d2h" is the blocking copy absorbing the
forward-stream sync — unavoidable at any threshold since bookkeeping must
sync there anyway). Decode collectives otherwise stay native (10-40 KB <
1 MB). Strict decode with instrumentation on: **66.89 tok/s** (reference
66.32, +0.9%, in band).

## 2. Env-day cells

### L3 — GLOO_SOCKET_IFNAME=lo + NUMA

- NUMA: `numactl -H` shows **one node** (EPYC 9015, 24 CPUs). Nothing to
  pin; the socket-locality half of L3 is moot on this host.
- Gloo binding before the pin (E0, `ss -tnp` on the workers): gloo data
  connections were already on loopback (127.0.0.1 <-> 127.0.1.1, the
  /etc/hosts hostname alias). The only LAN-addressed sockets
  (HOST <-> HOST hairpin) are c10d TCPStore rendezvous —
  metadata only, negligible traffic. So the pin was expected to be ~flat.
- Measured (full cell: probes + strict + quality):
  probes 658 / 712 / 825 / 796 tok/s @ 240/633/8192/32768 (E0: 658* /
  733 / 825 / 792; *strict-TTFT-derived) — **flat within noise**;
  decode 65.95 tok/s (-0.6% vs 66.32, in band); quality PASS (only the
  documented `logic` "Yes" casing tolerance, code_execution canary 14).
- Verdict: **flat (~0 ms). Not retained** in production (no effect; the
  CPU-copy/interface bucket is ruled out of the split). Evidence:
  `bigPP-fp8-l3-{probe,steps}-*.json/txt`, `bigPP-fp8-l3-strict.json`,
  `bigPP-fp8-l3-quality.json`.

### L4 — everything native oneCCL (HOST_STAGED_COLLECTIVES=0)

- **Boot crash at engine init.** The memory-profile/dummy pass issues a
  >=1 MB native device all_reduce; oneCCL tries
  `zeMemOpenIpcHandle` cross-device -> `ZE_RESULT_ERROR_INVALID_ARGUMENT`
  -> both workers die (`Worker proc VllmWorker-0 died unexpectedly`),
  APIServer RuntimeError. This is the topology's known IPC wall: oneCCL's
  large-payload path requires cross-device IPC that separate root ports
  forbid, and `CCL_ATL_TRANSPORT=ofi` does not route around it.
- Verdict: **the L4 question is closed by failure — native oneCCL is not
  a slower alternative, it is not viable at prefill payload sizes on this
  board.** Host staging is the only working path; L1's design stays
  "make staging async", not "route prefill native". Evidence:
  `bigPP-fp8-l4-native-oneccl-CRASH.txt` (full traceback).

### 2c — fused quant+GEMM A/B

Correctly **skipped**: the precondition (prefill kernel-launch-bound) is
false. Prefill is 94-100% serialized-comm-bound; all kernel buckets
together are ~20% of the window and dispatch overhead is in the noise.

## 3. Expected gains on THIS lane

Per-chunk resource floors from the E0 measurements (compute ~0.25 ms per
token per chunk; card2 copy volume 2 x payload at 6.9 GB/s; gloo reduce at
4.6 GB/s effective):

| size | current TTFT | L1 floor (max of compute / card2 copies / gloo) | est. PP after L1 |
| ---: | ---: | --- | ---: |
| 240 | ~360 ms | max(0.06, 0.15, 0.17 s) ~= 0.17-0.20 s | ~1200-1400 tok/s (1.8-2x) |
| 633 | ~865 ms | max(0.16, 0.37, 0.40 s) ~= 0.40-0.45 s | ~1400-1600 tok/s (~2x) |
| 8192 (2 chunks) | ~9.9 s | max(2.0, 3.1, 4.7 s) ~= 4.7-5.6 s/chunk... per 8000-chunk | ~1450-1700 tok/s (~2x) |

- **L1 (async double-buffered staging)**: hides gloo under compute and
  overlaps copies with the next layer; floor = slowest of the three
  resources. Expected **~1.8-2x PP at every size**, clearing bigPP Tier A
  (PP@240 >= 900, PP@633 >= 950) with margin. Decode risk ~zero (decode
  stays native below the 1 MB threshold; the one 0.99 MB decode-path
  staged call is already the sync point). After L1, the residual binder is
  the **gloo loopback TCP bandwidth (4.6 GB/s)** — the next lever is a
  gloo with shm transport (this venv's gloo lacks it; documented follow-up),
  not L2.
- **L2 (second CPU x16 / board change)**: alone (staging still blocking)
  ~1.4x (card2 copy term shrinks 3.7x). Combined with L1: **no additional
  gain until gloo improves** — the floor stays gloo-bound (4.7 s per
  8000-chunk vs card2 copies 3.1 s). L2's value is real only after a
  gloo/shm fix, at which point the floor becomes compute (~2 s per
  8000-chunk, PP@8192 ~4000). A board purchase is not justified by PP on
  current data.
- **Go/no-go on L1: GO.** Comm is 94-100% of prefill forward, L4 proved
  there is no native fallback, and L1 is the only code lever that attacks
  the serialized 80%. Same design as bigPP §4.3; on this lane the relative
  win is larger than on int4 because compute is a smaller share.

## 3b. L1 IMPLEMENTED + VALIDATED (2026-08-23, same day)

Implementation reality: a fully async cross-layer pipeline is not possible
inside XpuCommunicator (the reduced tensor is consumed immediately after
all_reduce returns; H2D must precede the consumer on the stream, so a
host-mediated reduce forces one host wait per collective). What shipped:

- **Sliced pipelined staging** (`VLLM_XPU_STAGE_SLICE_KB=8192`): each staged
  all_reduce splits into 8 MiB slices; D2H slices enqueued async with one
  event per slice; a worker thread reduces each slice as its D2H lands; the
  engine enqueues each H2D as its reduce completes. Wall per collective:
  D2H+reduce+H2D -> ~max(D2H+H2D, reduce).
- **shm 2-rank reduce** (`VLLM_XPU_STAGE_SHM=1`): /dev/shm exchange +
  rdy/ack spin barriers + torch CPU add on frombuffer views, replacing
  gloo loopback TCP. Measured bit-identical to gloo's 2-rank sum (unit
  test). ~7x faster than gloo per 8 MiB slice (gloo 4.7 s -> shm 0.63 s
  per 8000-token chunk).

Unit regression (regressions/2026-08-23-bigPP-async-staging/, 2 procs on
the real cards, real XpuCommunicator, under inference_mode): bit-exact on
all configs incl. odd shapes and fp32; throughput at 84 MB payload:
blocking 63.1 -> sliced 43.8 (1.44x) -> sliced+shm 27.3 ms (2.34x).
Two bugs found and fixed by the pack: silent worker-thread exception hang
(now propagated), and inference-tensor inplace restriction in the shm
worker (boot hang; fixed by re-entering inference_mode).

Server cells (this lane, cold probes; strict + quality each cell):

| config | PP@240 | PP@633 | PP@8192 | PP@32768 | strict decode | quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E0 baseline (blocking) | 658 | 733 | 825 | 792 | 66.89 | PASS* |
| L1a sliced (gloo) | 635 | 792 | 1060 | 1020 | 66.74 | PASS* |
| **L1b sliced + shm** | **969** | **1379** | **1620** | **1463** | **66.78** | PASS* |

*PASS = all items green except the documented `logic` "Yes" casing
tolerance. L1b caveat investigated: one cold-boot repeat-case outlier
(run 0 of 8 differed in token spacing; the repeat prefill at 185 tokens
crosses the 1 MiB staging threshold so the shm path WAS involved).
Attribution evidence: unit bit-exactness, 20/20 identical steady-state
greedy repeats at 181 tokens, and a cold-boot quality-first rerun PASS
(`bigPP-fp8-l1b-quality-coldboot.json`). Conclusion: first-request
cold-shape numerics, not a shm defect.

Final warm measurement on the production boot: 240 tokens at 206-240 ms
(**~1000-1160 tok/s**), 633 at 497-512 ms (~1240-1274), 8192 at 5075 ms
(~1614). Tier A met: staged wall share slashed, PP@240 >= 900 and PP@633
>= 950, decode in band (66.78 vs 66.32 ref, +0.7%), quality PASS.

**Production state: L1 is flipped ON by default for qwen38tp2fp8** via the
switch mode env block (EXTRA_LAUNCH_ENV default
`VLLM_XPU_STAGE_SLICE_KB=8192 VLLM_XPU_STAGE_SHM=1`). Revert:
`EXTRA_LAUNCH_ENV="VLLM_XPU_STAGE_SLICE_KB=0" bash ~/b70-switch.sh
qwen38tp2fp8 --force`. Other lanes untouched (out of scope).

Post-L1 residual floor at 8192: ~5.1 s for two chunks = card2 x4 copies
(~3.1 s/chunk-pair share) now dominating; next levers in order: L2 (x16
board) or larger chunks (L5), then L7 only if a compute-bound residual
remains.

## 3c. In-session caching investigation + L5 (2026-08-23 evening)

**Prefix caching: structurally inert on this stack, and correctly so.**
Motivation: mid-session PP re-prefills the whole conversation every turn.
Measured on the live lane: `enable_prefix_caching=True` (mamba "align"
mode auto-selected), yet `Prefix cache hit rate: 0.0%` always,
`cached_tokens=0` on byte-identical back-to-back prompts (185 and 2001
tokens), zero TTFT reduction. Prometheus: queries counted, hits always 0.
Verdict from upstream research (verified against this fork's source):
**known-broken/degenerate upstream, not a fork regression** —
- vllm#45238: align mode registers only ~1-2 mamba state checkpoints per
  request, near prompt end; if that boundary is outside the shared prefix,
  the hybrid per-group intersection vetoes every hit → flat 0%
  (geometry-dependent). Matches this fork (`_cache_partial_tail_block`
  only registers at the last prompt boundary).
- vllm#43559 / #50630: when a hit DOES land under MTP, the MambaManager
  hit path ignores the eagle drop-block margin → recurrent state can't
  rewind → silent output corruption (~20% accuracy drop reported). This
  fork's MambaManager indeed never reads `drop_eagle_block`.
- The escape hatch (boot without MTP) is closed twice over: the no-spec
  path in this fork dies on the first request (scheduler KeyError in
  `update_from_output`) AND 3 of 4 hard host resets this evening
  correlated with no-MTP boots. Do not retry.
- Real cure is upstream RFC #52959 (internal state checkpoints for align
  mode) or a fork feature (checkpoint mamba state at every block boundary
  + honor drop_eagle_block). Multi-day, correctness-critical.
Decision: **do not force-enable anything**; "no compromises" forbids the
corruption-prone hit path.

**L5 (16K prefill chunks)**: flat. PP 950/1341/1607/1489 @240/633/8192/
32768 vs L1b's 969/1379/1620/1463; decode 65.91 in band; quality PASS.
The 8192→6400+1792 split cost was not material; copy volume dominates.
Reverted to 8192 (keeps record-lane comparability). KV cache at 16K
chunks: 600,441 tokens.

**Host instability note**: 4 hard resets this evening (20:09, 20:20,
20:28, 20:50), no MCE/panic in journals (logs just stop). 3 of 4
correlated with the unsupported no-MTP config; the 20:09 one had no
proximate vLLM load. If resets recur under the normal config, treat as a
hardware incident (PSU/board), independent of this work. The resets
corrupted part of the torch compile cache (EOFError artifacts) — cleared
and rebuilt.

## 4. Restore state

`b70-switch.sh qwen38tp2fp8 --force`, clean env: READY, served id
`Qwen3.8-27B-UNC-FP8`, arithmetic canary **14**, timing channels off.
ComfyUI tenants untouched throughout (idle, ~0.3 GB/card baseline).
Fork state: stage-timing + sliced staging + shm reduce present, all
env-gated; slice/shm default ON for this lane via the switch (§3b).
Stale shm segment from the one crashed boot cleaned; the live server's
segment is owner-unlinked at exit. Diffs: `bigPP-fp8-fork-diff.patch`
(communicator + runner vs HEAD), switch diff (EXTRA hooks + fp8 mode
default).

## 5. Evidence inventory (all files now in `bench/reference/`)

Promotion candidates into `humble-b70-llm/bench/reference/` (per bigPP
§4 "every run lands as bench/reference/bigPP-<exp>-<config>.json"):
- `bigPP-fp8-e0-strict.json` (decode 66.89, TTFT 359.7, gate passed) and
  `bigPP-fp8-e0-quality.json` — the same-boot decode/quality rows.
- `bigPP-fp8-e0-probe-{633,8192,32768}.json` +
  `bigPP-fp8-e0-steps-{633,8192,32768}.txt` — the E0 split.
- `bigPP-fp8-e0-profiler-rank{0,1}.txt` — the 1e breakdown.
- `bigPP-fp8-l4-native-oneccl-CRASH.txt` — the closed door.
- `bigPP-fp8-l3-*` — the flat cell (record, don't retain).
- Reference README row should carry the "FP8 lane, no TP1-fp8 baseline"
  label on every PP number.
Left in place (not promoted): raw profiler traces (/tmp/bigpp-prof),
switch logs, linkstate poll log.
