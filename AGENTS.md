# Guidance for agents working in this repo

This file is loaded automatically by the pi coding agent. The repo reproduces
the lab's production Qwen3.8-27B inference stack on 2x Intel Arc Pro B70
(reference box — SSH alias in ~/.ssh/config; live-stack
truth lives in `~/mypi-coding-agent/AGENTS.md` on the reference box, which pi also loads
there). All claimed numbers must follow the metrics contract
(`docs/metrics-contract.md`).

## If you are new here: read these in order

1. `README.md` — headline numbers and what the repo guarantees.
2. `docs/metrics-contract.md` — the two number classes (cold one-shot vs
   LocalMaxxing warm) and why they differ. Never mix them.
3. `docs/w8a8-int8-campaign.md` — the full 2026-08-21/22 measured campaign:
   kernel integration, quality dispositions, the thinking-mode root cause,
   the INT8-vs-FP8 decode-gap verdict, operational rules.
4. `regressions/2026-08-21-w8a8-gdn-bf16-thinking/` — unit tests + evidence.

## Invariants and hard-won pitfalls (do not rediscover)

- **Measure decode only with `return_token_ids`** (tokens 1-100 after TTFT).
  MTP emits multi-token SSE deltas; counting chunks overstates ~2.8x. Scripts:
  `scripts/bench-strict.py` (true tokens), reference-box /tmp/lane-decode.py,
  ut_int8_w8a8.py in regressions.
- **Quality gates before speed claims** (repo rule): the lossy-weights
  discriminator is `sum(i*i for i in range(4))` must be `14` (a `30` answers a
  prompt-class/quantization sensitivity — see
  bench/reference/quality-w8a8-codeexec-disposition.json). `logic` "Yes"
  casing is a documented tolerance.
- **torch.compile cache vs env/source**: any vLLM source edit or env mode
  change (VLLM_XPU_GDN_BF16 / VLLM_XPU_W8A16 / VLLM_XPU_STEP_TIMING) requires
  `rm -rf ~/.cache/vllm/torch_compile_cache` before the next boot, or boots
  fail with stale-artifact bind errors (MergedColumnParallelLinear ...
  _xpu_gdn_bf16_weight). First boot after a clear = 2-4 min compile (17 min
  for the W8A16 graph) — normal, not a bug.
- **Memory walls (measured)**: bf16-dequant of the 27B int8 checkpoint is
  infeasible on 32 GB cards at TP2 (27 GB/card leaves ~0.3 GB KV; 64 GB host
  RAM is irrelevant to VRAM residency). oneDNN `bf16 x s8` mixed GEMM has no
  native XPU primitive (reorder fallback, ~20x slow). Both shapes are closed.
- **The config flip**: `quant_method: auto-round` resolves to the fork's
  ARK/inc path (slow), shadowing `--quantization gptq`; the AutoRound recipe
  ships flip-ready (`gptq` + `desc_act: false` + `pack_dtype: int32`).
- **Kernel wheel rebuild**: toolchain 2026.1 (ABI = libsycl.so.9),
  CXXFLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15" (2026.1 can't
  resolve headers vs gcc-16), HX_SKIP_ZE_GATE=1, venv lib FIRST in
  LD_LIBRARY_PATH (urDeviceWaitExp), rm -rf build on toolchain change. Full
  recipe in regressions/.../README.md.
- **Thinking policy**: effort high/xhigh is intrinsically non-convergent on
  this uncensored model family (measured at full precision). Low/medium work
  everywhere. The stack clamps effort at the server (template) and pi
  (thinkingLevelMap medium cap on all b70 providers).
- **Prefix caching is inert on this model family** (hybrid GDN + MTP,
  upstream-known: zero-hit geometry vllm#45238, corruption-on-hit
  #43559/#50630): 0.0% hit rate always; do NOT force-enable. In-session
  turns re-prefill full history (L1-accelerated). No-MTP boots are banned
  (broken fork path + correlated with hard host resets 2026-08-23). Cure =
  per-boundary GDN state checkpoints (bigPP.md §8.2).
- **Lane roles today**: qwen38tp2 (int4 + INT8 head) = fast lane (95.26 cold
  re-verified / 105.6 lmx APPROVED); qwen38tp2fp8 = second-fast (66.3 cold);
  since 2026-08-23 it runs **bigPP L1 staging** by default
  (`VLLM_XPU_STAGE_SLICE_KB=8192 VLLM_XPU_STAGE_SHM=1` in the switch mode
  block): PP 969/1379/1620 tok/s @240/633/8192 (~1.9x), decode unchanged —
  see docs/bigPP.md §8 + regressions/2026-08-23-bigPP-async-staging/;
  qwen38w8a8 = genuine
  pure-INT8 quality lane (63.6 cold, ~40-43 on prose — FP8 wins the decode
  gap: oneDNN primitive availability, see campaign doc); qwen38tp1 = 68.
- **Never**: `git add -A`/`reset --hard`/`clean`; kill servers except via
  the reference box's killvllm.sh; measure with SSE deltas; edit generated files.
  Manual fp8 boots need VLLM_XPU_HOST_STAGED_COLLECTIVES=1 + the served name
  (the switch normally injects both).

## Where things live

- Evidence JSONs: `bench/reference/` (19 rows, index in its README).
- Forward plan (dual x8 bifurcation + P2P probe): `docs/p2p.md`.
- Regression pack: `regressions/2026-08-21-w8a8-gdn-bf16-thinking/`.
- Live stack (reference box): $HOME/humble-b70-llm (fork, editable vllm +
  kernels tree kernels-int8-build + wheels in dist-w8a16), launchers in
  $HOME/bin/, switch in ~/mypi-coding-agent/switch/b70-switch.sh,
  prod venv $HOME/.venvs/vllm-xpu (wheel 0.1.14.dev4+g27214a3 with
  int8_gemm_w8a8/w8a16/fused ops), pi configs on both hosts
  (models.json/settings.json, providers b70-unc/tp1/fp8/w8a8).
