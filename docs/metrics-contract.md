# Metrics contract

Every number published in the README obeys this contract. If you measure
with different conditions, your number is different — that is expected and
does not mean the hardware differs.

## Decode metric (primary)

- Fixed realistic prompt suite: `bench/realistic-suite-v1.json`.
- Each prompt is sent **once** as a cold first response.
- **Prompt/KV caching disabled** on the server
  (`--no-enable-prefix-caching`); every request verifies
  `prompt_tokens_details.cached_tokens == 0`.
- No context checkpoints, response reuse, n-gram/history acceleration, or
  warmed repeated prompts.
- Primary metric: **median tok/s across the inter-token intervals between
  generated-token timestamps 1 and 100 after TTFT**, computed by
  `scripts/bench-strict.py`.
- Reported alongside: mean, p10, TTFT, full-output wall tok/s, prompt/output
  hashes, model identity, runtime commit, env vars, flags.

Speculative decoding is allowed only in its verified form: every accepted
draft token is re-scored by the target verifier (vLLM MTP verification is
exact by construction).

## Quality gates

- Arithmetic / code-evaluation canary where the deterministic answer is
  checked against expected output (`sum(i*i for i in range(4))` → `14`).
- Copy, fact, JSON-schema, long-context needle, and repeat-stability checks.
- A quality pass is required before any speed number is treated as real.

## Why warmed numbers are excluded

Running the same prompt shape 5 times in a row, after a same-shape warmup,
lets clocks, allocators, and graph dispatch reach steady state and produces
rates that are not achievable for a fresh user request. Such rows are
useful diagnostics and are explicitly labelled as such when included in
evidence; they are never promoted as the headline.

## Reproducibility tolerance

Fresh `torch.compile` runs are internally deterministic but
not byte-identical across compile caches. We therefore compare reproduction
runs to reference JSONs within a stated band (default ±3% on the median),
never by exact token equality. See `docs/troubleshooting.md` for the compile
cache note.
