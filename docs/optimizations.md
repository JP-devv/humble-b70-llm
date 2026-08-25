# Optimizations: what worked, what did not

## The shipped headliner: INT8 LM head (W8A8)

The untied LM head of a dense Qwen3.8-27B is a 2.54 GB FP16 tensor
(248320 x 5120). With MTP3 speculative decoding it is read once per verify
step **and** once per draft step — roughly 4 reads per mult-token step —
making it one of the largest single per-step weight reads in the model.

The optimization quantizes the LM head to per-channel INT8 at load time
(RTN, abs-max scaling) and serves the logits GEMM as a W8A8 instead of
FP16, halving that read. The verifier head stays the same arithmetic for
accepted tokens, so quality is unchanged (verified against the FP16 head
on every canary, see `bench/reference/quality-*.json`).

Measured effect (cold-strict contract):

| Config | Decode | Δ vs FP16 head |
| --- | ---: | ---: |
| TP2 FP16 head | 81.81 | — |
| TP2 INT8 head | 94.58 | **+15.6%** |
| TP1 INT8 head | 66.84 | (no TP1 FP16 run for this checkpoint; INT8 is the shipped single-card mode) |

Implementation notes: quantization runs at load in row-chunks (a single
full-head `float()` copy is a 5 GB transient that OOMs a single card), the
original FP16 weight is retained, and the whole path is env-gated
(`VLLM_XPU_LM_HEAD_INT8=1`) with a correct-by-construction fallback.

## Attempted, measured, and rejected (documented so nobody rediscovers them)

Every rejection below was measured with the same cold contract; the
evidence JSONs are in `bench/reference/` with `-REJECTED` in the name.

1. **MTP4 instead of MTP3** — 79.84 vs 81.81. The extra draft depth does
   not pay at realistic acceptance (the suite's varied prompts accept ~45%,
   so the 4th row costs more than it earns). MTP acceptance is
   workload-dependent; on highly repetitive content deeper MTP can win.
   `VLLM_XPU_DRAFT_*` knobs exist but the metric gate owns the decision.
   Evidence: `tp2-mtp4-fp16-79.84-REJECTED.json`.

2. **INT4 draft LM head** (RTN g128 on the draft head copy, served through
   the INT4 GEMM) — 91.84 vs 94.58. Unlike the INT8 head, coarsening the
   *draft* head to INT4 costs enough acceptance that the bandwidth win is
   net-negative on this stack. This contradicts the intuition "draft
   precision doesn't matter because the target verifies" — it matters
   through the acceptance rate, which is exactly the number the target
   multiplies.
   Evidence: `tp2-mtp3-int8+int4draft-91.84-REJECTED.json`.

3. **Warm-harness numbers** (same-shape warmup, n=5 repeats, post-first
   timing) — these read much higher (80+ on a *single* card) and are real
   under their own conditions, but they describe repeat traffic, not fresh
   requests. The methodology lesson evidence file walks through the same
   stack under both protocols.
   Evidence: `methodology-lesson-alternative-stack-cold.json`.

4. **Chasing record-class TP2 numbers on a host without GPU peer IPC** —
   the +20%+ gap to a stack that captures collectives inside graphs comes
   from hardware topology (see `hardware.md`). Software-only attempts to
   emulate it (breakable-capture machinery, staged comms variants) were
   abandoned after bounded bisection. The host-staged collectives that DID
   work are the shipped patches.

## Rejected-by-principle (documented once)

- **AOT artifact hash gating**: rebuilds can never match the exact bytes of
  a different toolchain. Wet-noodle behavior verification instead.
- **Publishing performance claims without the measurement contract**:
  numbers without their protocol are noise that costs everyone time.


## Campaign addenda (2026-08-21/22, full record in docs/w8a8-int8-campaign.md)

### Shipped levers (measured)

- **GDN-BF16** (`VLLM_XPU_GDN_BF16`, default on): the GDN linear-attention
  projections (the stateful path) are dequantized to BF16 in place at load
  (96/rank). Makes medium-effort thinking closure reliable on the W8A8 lane;
  quality canary 14. Unit-tested activation-exact (regressions/.../ut_gdn_bf16.py).
- **Sharp template xhigh clamp**: the template's xhigh instruction drove
  non-convergent chain-of-thought at high/xhigh effort; a neutral directive
  plus a render-time clamp of effort high/xhigh/max -> medium (all template
  copies) fixed closure while keeping the sharp UX at low/medium.
- **pi effort policy**: thinkingLevelMap caps every b70 provider at medium
  (both hosts) — shown to be base-family behavior, not a lane downgrade.
- **The config flip** (re-verified record): this fork resolves
  `quant_method: auto-round` to its ARK/inc path, shadowing
  `--quantization gptq` (48.74 tok/s); flipping to `gptq` + `desc_act: false`
  + `pack_dtype: int32` restored the fast oneDNN w4a16 path AND re-armed the
  INT8 lm_head hook: 48.74 -> 95.26, quality intact.

### Attempted, measured, and rejected (campaign)

1. **True W8A16 via oneDNN bf16_s8 joint** (`int8_gemm_w8a16`): bit-exact
   (4e-3 vs fp32 ref) but decodes at **2.9 tok/s** — the XPU mixed primitive
   is a per-call reorder fallback; Intel ships native bf16 x f8 kernels but
   not s8 x bf16. REJECTED.
2. **bf16-dequant W8A16**: memory-infeasible on 32 GB cards (27 GB bf16
   weights/card at TP2 leave ~0.3 GB KV; 64 GB host RAM is irrelevant to VRAM
   residency). REJECTED (physics).
3. **Fused quant+GEMM** (`int8_gemm_w8a8_fused`): bit-identical to the
   two-step A8 path with zero per-call gain — per-op dispatch is not the
   decode-gap cost. REJECTED.
4. **INT8 decode parity with FP8**: step time within 6% (48.3 vs 45.6 ms),
   acceptance within 3% (2.48 vs 2.55); the client-observed ~30% gap does not
   decompose below the engine level; every exact-activation route is closed.
   Verdict: FP8-W8A16 wins on this oneDNN/hardware; W8A8 is the pure-INT8
   quality lane.
