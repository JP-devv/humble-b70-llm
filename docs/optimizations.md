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
| TP1 FP16 head | 57.49 | — |
| TP1 INT8 head | 66.84 | **+16.3%** |

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
   Evidence: `warm-vs-cold-methodology-lesson.json`.

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
