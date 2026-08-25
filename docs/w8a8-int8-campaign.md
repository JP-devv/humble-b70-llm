# W8A8 INT8 campaign (2026-08-21/22): integration, quality, thinking-mode, decode-gap

Full measured record of the genuine-INT8 W8A8 lane work: kernel integration,
quality gates, the thinking-mode root-cause hunt, the INT8-vs-FP8 decode-gap
verdict, and the operational rules learned. Every number below was produced
under the repo's metrics contract (cold one-shot unless labeled warm).

## 1. Integration — genuine INT8 W8A8 served on XPU

- Checkpoints: `RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP` (primary; genuine
  int8 channel-wise weights + BF16 scales, dynamic per-token INT8 activations,
  MTP all-BF16) and `Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8` (same
  scheme, F16 scales; `logic` canary passes where the UNC line does not).
- Fork changes (editable tree `src/vllm` on the reference box):
  - `XPUInt8ScaledMMLinearKernel` (scaled_mm/xpu.py) routes compressed-tensors
    w8a8 int8 through the backported oneDNN ops
    `torch.ops._xpu_C.per_token_quant_int8_xpu` + `int8_gemm_w8a8`; weight
    repacked [K,N] contiguous, scales [N] fp32.
  - Fake/Meta impls for both ops in `vllm/_xpu_ops.py` (without them XPU graph
    capture dies: "attempted to run this operator with Meta tensors").
  - Registered first in `_POSSIBLE_INT8_KERNELS[PlatformEnum.XPU]`.
- Unit-verified on-card before the model existed (ut_int8_w8a8.py,
  ut_kernel_class.py): per-token quant exact (absmax/127), GEMM nn/bf16
  matches fp32 reference (4.8e-4..3.8e-3), fused per-tensor->channel path OK.

### Cold-strict rows (12-prompt suite, median tok/s tokens 1-100 after TTFT)

| Config | tok/s | Notes |
| --- | ---: | --- |
| RukaRat TP2 MTP3 INT8 head | **63.64** | confirm row (evidence in bench/reference/) |
| RukaRat TP2 MTP3 INT8 head | 60.70 | earlier run (variance) |
| Freaksterz TP2 MTP3 INT8 head | 57.79 | INT8 head = +8.2% over FP16 head (53.40) |
| TP2 MTP4 INT8 head | 57.78 | REJECTED — flat, same as INT4 trunk |
| TP1 | infeasible | 27 GB int8 trunk + head + MTP > 32 GB single card |

### Quality dispositions (bench/reference/quality-w8a8-codeexec-disposition.json)

- `code_execution` canary ("sum(i*i for i in range(4))") answers `30` instead
  of `14` on the wifi-class prompts with thinking off — reproduced on both W8A8
  builds and on the FP8-UNC lane; answered `14` with thinking on, with
  rephrasing, and on the unquantized BF16 control. Disposition: prompt+protocol
  sensitivity of the lossy-weights discriminator, documented with evidence.
- `logic` "Yes"/"yes" casing: existing documented tolerance.
- Long-context needle, repeat, copy, json, factual: pass.

## 2. Record re-verify + the config flip

The record config (UNC g128 AutoRound INT4 + load-time INT8 lm_head, TP2 MTP3,
fp16 KV) was re-verified on the current stack at **95.26 tok/s** (previous
94.58). Decisive discovery: fork resolves `quant_method: auto-round` to its
ARK/inc path (the slow small-M path) *shadowing* `--quantization gptq`; the
skill's proven flip (`quant_method -> gptq`, `desc_act: false`,
`pack_dtype: int32`) restored the fast oneDNN w4a16 path AND re-armed the INT8
lm_head hook: 48.74 -> 95.26 with quality intact (canary `14`). Evidence:
`tp2-mtp3-record-reverify-95.26.json`,
`tp2-mtp3-record-ark-path-48.74-CONTROL.json`.

LocalMaxxing (warm protocol): **tokSOut 105.6** on the re-verified record
config — APPROVED `cmt2slgyx0hdgmv01d8vf53cr` (tokSPrefill 710.3, TTFT 891 ms,
633-token prompt; supersedes the 89.8 TP2 row).

## 3. Thinking-mode root cause (the big one)

Symptom: at high/xhigh reasoning effort, chats ramble for thousands of tokens
without an answer — CJK bleed, fake shell/tool sessions, 30+ TPS feel. The
deep-dive (production-equivalent cells, n>=5, both templates, three builds):

| Build @ xhigh (wifi-class prompt) | closed | CJK |
| --- | ---: | ---: |
| Full-precision BF16 base | 1/5 | 2/5 |
| W8A8 + patched sharp template | 2/5 | 2/5 |
| W8A8 + stock template | 3/5 | 2/5 |

Number protocols (do not mix): the cold-suite rows in section 1 (63.64 etc.) are the in-repo 12-prompt suite; the decode-gap rows in section 4 (40-43) are a single 400-token essay prompt (longer context = slower tokens 1-100). A repetitive 480-token prompt reads ~76 due to near-perfect MTP acceptance (acceptance flattery, not the prose figure).

**xhigh-effort non-convergence is intrinsic to this uncensored Qwen3.8 family**
at full precision, across quantizations and templates. En-route fixes that DID
help and are shipped:

1. **GDN-BF16** (`VLLM_XPU_GDN_BF16`, default on): the GDN linear-attention
   projections (stateful path) are dequantized to BF16 in place at load
   (96/rank). CoT closure at medium goes from fragile to reliable; quality
   canary `14`. Unit-tested activation-exact (ut_gdn_bf16.py).
2. **Sharp template xhigh clamp** (`chat_template.sharp-xhighfix.jinja`): the
   sharp template's xhigh instruction ("think carefully, validate key
   assumptions...") drove the rambling; replaced with a neutral directive, and
   effort high/xhigh/max now clamps to medium at render time in ALL copies
   (chat_template.jinja, the xfix file, tokenizer_config embedded). Verified:
   an explicit xhigh request renders as medium.
3. **pi effort policy**: thinkingLevelMap caps every level at medium on all
   four b70 providers, both hosts. Thinking stays ENABLED (low/medium engage
   full reasoning); the cap is now proven as base-family behavior, not a lane
   downgrade.

Medium-effort gate (fixed lane): wifi prompt 5/5 close, zero CJK/runaway, 16K
needle found, 6-turn agentic sane. Evidence in
`regressions/2026-08-21-w8a8-gdn-bf16-thinking/`.

## 4. INT8 decode gap vs FP8 — final verdict

| Lane (MTP3, essay prompt, tokens 1-100) | decode | step | acceptance |
| --- | ---: | ---: | ---: |
| fp8-unc (bf16 activations, native mixed kernel) | **55.6** | 45.6 ms | 2.55 |
| w8a8 A8 (dynamic int8 activations) | 37-42 | 48.3 ms | 2.48 |

Lever matrix (all measured, all closed):
- GDN-BF16: not the cost.
- bf16-dequant W8A16: memory-infeasible (27 GB bf16/card leaves ~0.3 GB KV at
  TP2 on 32 GB cards; 64 GB host RAM is irrelevant to VRAM residency).
- True W8A16 via oneDNN `bf16_s8` joint (`int8_gemm_w8a16`): bit-exact (unit
  err 4e-3) but **2.9 tok/s** — the mixed primitive is a per-call reorder
  fallback; Intel ships native bf16 x f8 kernels but not s8 x bf16.
- Fused quant+GEMM (`int8_gemm_w8a8_fused`): bit-identical to the two-step A8
  path, zero per-call gain — dispatch is not the cost.

**Conclusion**: on this hardware/oneDNN, FP8-W8A16 serving wins because its
mixed primitive is native; INT8's DPAS arithmetic advantage is unreachable as a
serving shape without exact-activation memory headroom (blocked) or a native
s8 x bf16 kernel (not shipped by Intel; a hand-written DPAS kernel is the one
open path). The W8A8 lane stands as the genuine pure-INT8 quality lane
(~40-43 tok/s cold, quality-gated, thinking-safe). New ops
(`int8_gemm_w8a16`, `int8_gemm_w8a8_fused`) remain in the kernels tree/wheel
for future work.

## 5. Operational rules learned (do not rediscover)

1. **torch.compile cache vs env/source**: any vLLM source edit or env-mode
   change (GDN/W8A16/STEP_TIMING) silently invalidates the compile cache keys —
   `rm -rf ~/.cache/vllm/torch_compile_cache` before the next boot, or boots
   fail with stale-artifact bind errors (`MergedColumnParallelLinear ...
   no attribute '_xpu_gdn_bf16_weight'`). First boot after a clear costs
   2-4 min compile (17 min for the pathological W8A16 graph) — normal.
2. **Kernels wheel rebuild recipe** (kernels-int8-build tree; ABI must match
   the venv's torch = oneAPI 2026.x -> libsycl.so.9; 2025.3 links .so.8):
   `CMPLR_ROOT=/opt/intel/oneapi/compiler/2026.1`,
   `CXXFLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15"` (2026.1
   cannot find headers against gcc-16 otherwise), `HX_SKIP_ZE_GATE=1`
   (vendored oneDNN LevelZero version gate false-fails on fresh configures),
   venv lib FIRST in LD_LIBRARY_PATH (urDeviceWaitExp rule), `rm -rf build`
   before toolchain changes. Full recipe in the regression README.
3. **Manual fp8 boots** need `VLLM_XPU_HOST_STAGED_COLLECTIVES=1` +
   `SERVED_NAME=Qwen3.8-27B-UNC-FP8` (the switch normally injects both).
4. **SSE-delta counting is banned** for decode measurement: MTP emits
   multi-token deltas, so chunk counts overstate ~2.8x. Measure only with
   `return_token_ids` (tokens 1-100 after TTFT).
