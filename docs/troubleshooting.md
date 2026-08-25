# Troubleshooting (failure signatures we actually hit)

## Build-time

### CMake FindPython "Cannot find source file ... .cpp"
The kernels source tree is mid-merge or has a stale CMake reference to a
file that was moved/removed upstream. Verify `csrc/moe/fused_moe_prologue.cpp`
exists; if the header exists but the .cpp was deleted, remove the stale
`list(APPEND)` entry. (This is why we ship `patches/` with pinned base
commits — the base commits are known-good.)

### icpx "Killed" during a heavy SYCL unit
The compiler was OOM-killed. The fattest units (`grouped_gemm_xe2.cpp`,
`chunk_prefill_kernel_template_*`) can each peak at 7-12 GB. Fixes:
- `MAX_JOBS=3..6` (the builder script default: 6).
- Free host RAM: stop inference servers while building.
- Use the reduced attention presets to shrink the variant set:
  `VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf`
  `VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf`
  (Qwen-class models are covered by the default presets).

### CMake 4.x FindPython error at configure time
The pip-resolved `cmake>=3.26` can land on 4.x which has stricter Python
target rules. Pin `cmake==3.31.x` or use the distro cmake (3.28 works).

## Serve-time

### Engine core init fails with "not enough GPU memory" right after weights load
The INT8 head load-time quantization used to allocate a full-size FP16 copy
(5 GB on a single card, 2.5 GB per rank at TP2). The shipped patch chunks
the row loop (`VLLM_XPU_LM_HEAD_INT8_CHUNK_ROWS`, default 4096) and the
transient disappears. If you still OOM: lower `GPU_MEMORY_UTILIZATION`
(0.88 on one card is the tested value).

### Device lost / engine reset during graph capture on a TP2 host without peer IPC
XPU command-graph capture on systems without L0 peer IPC must not record
host-staged collectives into a graph segment. The shipped patches split
collectives out of capture (piecewise capture with staged comms between
segments). If you hit this on a different stack, the signature is a kernel
timeout ~270 GPU submissions into the decode capture.

## Measurement-time

### Warmed numbers look wildly better than the published ones
That's the contract doing its job — see `docs/metrics-contract.md`. Run the
in-repo harness, don't hand-time.

### Two fresh rebuilds differ token-for-token but are internally deterministic
torch.compile cache identity. The compile cache is part of run identity;
compare within the tolerance band and record the cache state in the
evidence manifest. Do not gate on byte equality.

### Quality gate fails the logic case with a case-only difference (expected `yes`, got `Yes`)
The uncensored trunk exhibits stylistic casing differences on short
answers; this reproduces identically with the FP16 head and is not a
quantization regression. All other canaries must pass.

## Model fetch

### `hf download` stalls at 0 KB/s
The xet transport stalls on some hosts; set `HF_HUB_DISABLE_XET=1` or
`HF_XET_HIGH_PERFORMANCE=1`, or fetch with `curl -L -C -` at ~34 MB/s.
Resume with `-C -`.


## W8A8 campaign signatures (2026-08-21/22)

### Boot fails after a vLLM source edit or env-mode change with
`MergedColumnParallelLinear ... no attribute '_xpu_gdn_bf16_weight'`
The torch.compile cache keys do not include our env flags (VLLM_XPU_GDN_BF16 /
VLLM_XPU_W8A16 / VLLM_XPU_STEP_TIMING) or the fork's python source, so a boot
can load a stale compiled artifact that binds module attributes the new mode
does not create. Fix: `rm -rf ~/.cache/vllm/torch_compile_cache` then boot.
First boot after a clear costs 2-4 min compile (17 min for the W8A16 graph).

### A 20-40x decode collapse (e.g. 2.9 tok/s) after enabling a mixed-precision path
The oneDNN `bf16_s8` joint has no native XPU primitive and falls back to a
per-call reorder path. Both operands must be the same width (s8 x s8) for the
fast DPAS path.

### Kernels wheel rebuild (ABI and toolchain)
The wheel ABI must match the venv's torch (libsycl.so.9 -> oneAPI 2026.x;
2025.3 links .so.8 and fails dlopen). Pinned recipe: CMPLR_ROOT=.../2026.1,
CXXFLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15" (2026.1 cannot
resolve headers against gcc-16 otherwise), HX_SKIP_ZE_GATE=1 (vendored oneDNN
LevelZero version gate false-fails on fresh configures), venv lib FIRST in
LD_LIBRARY_PATH (urDeviceWaitExp), rm -rf build before toolchain changes.
Full command in regressions/2026-08-21-w8a8-gdn-bf16-thinking/README.md.

### Manual fp8/other boots failing with zeMemOpenIpcHandle INVALID_ARGUMENT
b70-switch injects VLLM_XPU_HOST_STAGED_COLLECTIVES=1 (and the served name);
manual tmux launches of the fp8 launcher must add it (plus SERVED_NAME).

### Decode "too fast" in hand timing
MTP emits multi-token SSE deltas; counting chunks overstates decode ~2.8x.
Measure with return_token_ids (tokens 1-100 after TTFT) only.
