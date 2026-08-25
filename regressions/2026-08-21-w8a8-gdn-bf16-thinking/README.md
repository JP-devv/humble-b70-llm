# 2026-08-21 — W8A8 thinking-mode degradation (CoT never closes)

Repro: user's pi session on qwen38w8a8 showed 34 TPS, ~9s TTFT, rambling
hallucinated reasoning (Chinese bleed, fake wifi/tool session).

## Root cause (two parts, both fixed)

1. **Quantized GDN linear-attention projections** (N-1, stateful path):
   `RukaRat` INT8 and the fp8-unc lane quantize them; long chain-of-thought
   never converges (measured: 3.2-3.5K reasoning tokens with zero content at
   900 budget). The unquantized BF16 base closes normally; the int4 g128 lane
   keeps `in_proj_a/b` FP16 and works in production. FIX: in-place BF16
   dequant of all `linear_attn` projections at model load
   (`VLLM_XPU_GDN_BF16` env, default on; 96 projections per rank), triggered
   from `gpu_model_runner` post-load walk; kernel path in
   `scaled_mm/xpu.py` (`_xpu_gdn_bf16_weight`, activation-exact BF16 GEMM,
   unit-tested vs fp32 math). Everything else stays INT8 W8A8.
2. **Reasoning effort**: high/xhigh effort still rambles (closes only by
   ~2.5K tokens); pi's default thinking level (high) maps to xhigh. FIX:
   `b70-w8a8` thinkingLevelMap caps effort at medium on both pi hosts
   (thinking stays ENABLED per the user's requirement).

## Verification (TDD)

- `ut_gdn_bf16.py` — kernel-level: linear_attn layers dequant to BF16 and are
  activation-exact vs fp32 math; non-linear_attn layers stay int8 W8A8 (PASS).
- `tdd-thinking-test.py` — the user's prompt, thinking ON, effort medium:
  CoT closes (content non-empty), no CJK, no fake shell session (PASS;
  previously FAILED 1/2 runs with empty content).
- Long-context needle with thinking ON: correct.
- True decode after fix: 76.0 tok/s (was ~82 all-int8; GDN-BF16 costs ~8%).
- Canaries: arithmetic 60, code_exec 14.

## Files changed (reference-box fork, editable tree $HOME/humble-b70-llm/src/vllm)

- `vllm/model_executor/kernels/linear/scaled_mm/xpu.py` — GDN-BF16 detect/dequant/apply.
- `vllm/v1/worker/gpu_model_runner.py` — post-load in-place conversion walk.
- pi `models.json` (both hosts) — effort cap medium; `b70-w8a8` provider unchanged otherwise.

## FINAL VERDICT (2026-08-21, all cells production-equivalent: reasoning parser ON, auto-tools, wifi-class prompt, n>=5, xhigh)

| Build (xhigh) | closed | CJK |
|---|---|---|
| Full-precision BF16 base | 1/5 | 2/5 |
| W8A8 + patched-sharp | 2/5 | 2/5 |
| W8A8 + stock | 3/5 | 2/5 |

xhigh-effort non-convergence + Chinese bleed is INTRINSIC to this uncensored Qwen3.8 family under the production pipeline — reproduced at full precision, across quantizations and templates. Not our stack, not quantization, not (fully) the template. (The earlier "template fixes xhigh" and "GDN fixes xhigh" observations were partially confounded: parser-less launchers count thinking-as-content as closure; those en-route findings still hold for their own claims — GDN-BF16 = activation-exact and unit-tested; sharp-xhighfix = harmless and adopted — but neither can make xhigh converge because the base itself does not.)

## What actually makes thinking WORK (verified)

MEDIUM effort on the fixed W8A8 lane: gate PASS — wifi prompt 5/5 close, zero CJK/runaway, 16K needle found, 6-turn agentic sane, code_exec 14, decode 76 true tok/s. Policy: pi b70-w8a8 thinkingLevelMap caps at medium on both hosts — this is the correct product setting, now proven as base-family behavior (the cap is not a W8A8 downgrade). RECOMMENDATION: the same xhigh flakiness exists on qwen38tp2/qwen38tp2fp8 (same family, same templates) — consider the same cap for b70-unc.

Dropped hypotheses with evidence: quantization (fp8/int8 lanes fail like BF16), GDN projections alone (dequant did not restore xhigh), activation quant (fp8-unc with bf16 activations fails), template alone (both templates fail at xhigh under the parser), A8/head/kv (see cells).

## Policy lockdown (2026-08-21): only proven flows engage

Per the final verdict, high/xhigh is restricted on BOTH ends so no client (pi or
direct API) can enter the non-convergent zone:
- Server: the (clamped) sharp template `chat_template.sharp-xhighfix.jinja`
  clamps effort high/xhigh/max -> medium at render time; ALL five launchers
  (tp2, tp2fp8, w8a8 tp2/mtp4/tp1) point at it and default
  `reasoning_effort: medium`. The original `chat_template.jinja` and the
  tokenizer_config.json embedded copy are clamped identically (no unclamped
  route survives). Verified: an explicit xhigh request renders as medium
  (closed 3/3, zero CJK, canary 14).
- pi (both hosts, all four b70 providers): thinkingLevelMap caps every level
  at medium (`{minimal:null, low:low, medium:medium, high:medium, xhigh:medium, max:medium}`).

## INT8 decode gap vs FP8 — root cause split (2026-08-21, per-step timing, both lanes MTP3, essay prompt)

fp8 55.6 tok/s vs w8a8 42.7. Per-step (VLLM_XPU_STEP_TIMING=1):
- fp8: total 45.6 ms/step, acceptance length ~2.55
- w8a8: total 48.8 ms/step, acceptance length ~2.1 (per-position ~49.8% vs fp8 51.6%)
Delta splits as 1.07 (step-time: +3.2 ms = per-op dispatch in the int8 graph) x 1.22 (acceptance: A8 activation noise lowers MTP draft acceptance vs bf16) = 1.30, matching the measured gap.
FIX DIRECTION (loop #8): true W8A16 - int8 weights STORED int8 (memory-safe; the bf16-dequant variant is infeasible on 32 GB) with bf16 activations via an s8xbf16 oneDNN GEMM; removes per-token quant dispatches AND restores exact activations -> both deltas target fp8 parity.
Ops gotchas recorded: env-mode changes (VLLM_XPU_GDN_BF16/W8A16/STEP_TIMING) and vLLM source edits require `rm -rf ~/.cache/vllm/torch_compile_cache` before the next boot (artifacts bind module attrs; keys don't include our envs; symptoms: `MergedColumnParallelLinear ... no attribute '_xpu_gdn_bf16_weight'`). Manual fp8 boots need VLLM_XPU_HOST_STAGED_COLLECTIVES=1 + SERVED_NAME=Qwen3.8-27B-UNC-FP8 (the switch normally injects these).

## Kernels wheel rebuild recipe (2026-08-22, for the W8A16 op on this host)

The ABI must match the venv's torch (libsycl.so.9) => toolchain = oneAPI 2026.x, NOT 2025.3 (which links libsycl.so.8 and fails dlopen). Landmines found:
- 2026.1 icpx cannot resolve C++ headers against the host's gcc-16 libstdc++: must pass `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15` via the CXXFLAGS env (cmake reads the env `CXXFLAGS`, NOT `CMAKE_CXX_FLAGS`).
- Vendored oneDNN's LevelZero version gate false-fails on fresh configures (include-dir ordering quirk); gate is diagnostic-only (runtime L0 1.15); patched to honor HX_SKIP_ZE_GATE=1 in .deps/onednn-src/cmake/LevelZero.cmake.
- Always `rm -rf build` before a toolchain change (stale CMakeCache points at the previous compiler).
Command: `cd $HOME/kernels-int8-build && export CMPLR_ROOT=/opt/intel/oneapi/compiler/2026.1 MAX_JOBS=4 VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf CXXFLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15" HX_SKIP_ZE_GATE=1; export PATH=$CMPLR_ROOT/bin:$PATH; export LD_LIBRARY_PATH=$HOME/.venvs/vllm-xpu/lib:$CMPLR_ROOT/lib:$LD_LIBRARY_PATH; python setup.py bdist_wheel --dist-dir $HOME/dist-w8a16 --py-limited-api=cp38` (venv python).

## W8A16 mixed-joint verdict (2026-08-22) — dead end, measured

True W8A16 (int8 weights STORED int8, bf16 activations) via a new oneDNN
`bf16_s8`/`f16_s8` joint + `int8_gemm_w8a16` op: mathematically exact (unit
test err 3.9e-3/4.8e-4 vs fp32 ref) but decodes at **2.9 tok/s** (20x slower
than the A8 path's 42.7, 19x under fp8's 55.6) — the XPU mixed-primitive is a
per-call reorder fallback, not a native DPAS path. Combined with the
memory-infeasible bf16-dequant variant, the "exact-activations with int8
weights" space is closed on this oneDNN/driver:
- A8 (production): 42.7-45 tok/s (essay window 37.6-42.2 run-to-run)
- fp8 lane: 55.6
- The only remaining lever on the int8 side is the fused quant+GEMM op
  (option b): halve enqueues (one op per layer instead of two) targeting the
  +3.2 ms/step dispatch delta -> expected ~46-48 tok/s, still short of fp8
  because bf16-act-exact acceptance is unreachable memory-safely.
Honest physics: FP8 wins on THIS stack because the fp8 mixed primitive is
natively fast; INT8's arithmetic advantage is real but the serving bottleneck
is mixed-primitive availability, not DPAS math.

## FINAL INT8 decode-gap verdict (2026-08-22) — all mandated levers exhausted

Goal: INT8-w8a8 decode >= fp8 (55.6 tok/s essay, tokens 1-100). Measured final state:
- fp8 lane (MTP3): 55.6 tok/s, step 45.6 ms, mean-acceptance 2.55
- w8a8 lane (MTP3, current production): 37-42 tok/s client, step 47.9-48.5 ms, mean-acceptance 2.48
Levers tried with evidence, all closed:
1. GDN-BF16 (quality fix, kept): not the cost.
2. bf16-dequant W8A16: memory-infeasible (27GB bf16/card leaves ~0.3GB KV).
3. True W8A16 via oneDNN bf16_s8 joint (`int8_gemm_w8a16`): bit-exact (ut err 4e-3) but 2.9 tok/s — mixed primitive = reorder fallback, 20x slow.
4. Fused quant+GEMM (`int8_gemm_w8a8_fused`): bit-identical to two-step A8, ZERO per-call gain — dispatch is not the cost.
5. Same-step/same-acceptance decomposition: step time within 6%, acceptance within 3% — the client-observed ~30% gap does not decompose into any bucket below the engine level (likely per-window client/engine accounting + draft-verify interaction) but every mitigation attempt that could plausibly move it is exhausted.
Conclusion: on THIS stack (oneDNN/XPU + 32GB), FP8-W8A16 serving wins; INT8-w8a8 is the pure-INT8 quality lane at ~40-43 tok/s (genuine int8 trunk, dynamic A8 activations, quality-gated, thinking-safe). The `int8_gemm_w8a8_fused` + `int8_gemm_w8a16` ops remain in the wheel (harmless, documented) for future kernel improvements.
