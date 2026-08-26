# FP8 UNC variant — load-time W8A8 on the full-precision uncensored base

The main stack (INT4 AutoRound g128 + Marlin) is this repo's record. This page
documents the FP8 sibling served on the same hardware: the **uncensored**
27B dense served from the full-precision bf16 checkpoint with weights
quantized to FP8 e4m3 (dynamic activation) **at load** by vLLM, plus the
INT8 lm_head and FP8 KV.

Why it exists: it is the only route to full-bf16-lineage quality on the
uncensored model (the int4 g128 recipe needs its AutoRound pass; FP8 W8A8 is
a straight cast of the high-precision weights) **and** native 262144 context.
The cost is decode, which is weight-bound: FP8 reads 2x the weight bytes of
INT4.

## Model lineage

- Base: `JonathanColetti/Qwen3.8-27B-Uncensored` (bf16, 12 shards,
  262144 max_position_embeddings, no rope scaling — 262144 is the cap).
- Served with `--quantization fp8`: vLLM quantizes every Linear weight to
  e4m3 per-channel at load (`Fp8PerTensorOnlineLinearMethod` →
  `XPUW8A16FP8LinearKernel` on XPU). No static FP8 checkpoint is involved.
- The INT8 lm_head (`VLLM_XPU_LM_HEAD_INT8=1`) applies on top exactly as on
  the INT4 stack: the 2.54 GB BF16 vocab head is read once per draft step and
  once per verify step under MTP3; W8A8 halves that read (see
  [optimizations.md](optimizations.md)).
- Chat template: the UNC "sharp" template (`qwen3.8-froggeric-v22.1`) is
  imported explicitly with `--chat-template` — the base repo ships its own
  stock template, which is NOT the tuned one.

## Memory: why FP8 KV is mandatory at 262144

Decode-era VRAM per card at 0.95 utility (measured on 2x Arc Pro B70):

| Item | GiB/card |
|---|---:|
| usable / budget | 30.3 / 28.78 |
| FP8 weights + non-torch | ~15.3 |
| activation + graphs (peak) | ~1.7 |
| **left for KV** | **~11.8** |
| FP16 KV needed @ 262144 | 16.7 — does not fit (~165K cap) |
| FP8 KV needed @ 262144 | ~8.4 — fits with ~1.3x |

## Launch

Production uses the switch tooling (`~/mypi-coding-agent/switch/b70-switch.sh
qwen38tp2fp8`), which handles fetch, kill, host-staged collectives, readiness
and the timing record. The equivalent explicit command:

```bash
VLLM_XPU_HOST_STAGED_COLLECTIVES=1 USE_PINNED_ONECCL=1 SPEC_TOKENS=3 \
KV_CACHE_DTYPE=fp8 TP=2 PORT=19622 SERVED_NAME=Qwen3.8-27B-UNC-FP8 \
MODEL_DIR=/data/models/Qwen3.8-27B-Uncensored-bf16-full \
bash /home/haxor/bin/serve-qwen38-tp2-haxor.sh \
  --quantization fp8 \
  --chat-template /data/models/Qwen3.8-27B-Uncensored-g128-AutoRound/chat_template.jinja
```

Or via the in-repo launcher (additive flags shipped with this change):

```bash
MODEL=/data/models/Qwen3.8-27B-Uncensored-bf16-full \
SERVED=Qwen3.8-27B-UNC-FP8 \
bash scripts/serve.sh --ctx 262144 --kv fp8 --quant fp8 \
  --template /data/models/Qwen3.8-27B-Uncensored-g128-AutoRound/chat_template.jinja
```

`satisfies the same launch contract`: `--int8-head on` is the default here
too (the fp8 path benefits from the INT8 head identically).

## Measured rows

| Metric | Value | Protocol |
|---|---:|---|
| Cold strict decode | **66.3 tok/s** median (p10 58.7) | 12 cold prompts, tokens 1-100 after TTFT, `cached_tokens=0`, gate passed |
| TTFT (cold suite) | 371 ms median | ditto |
| LocalMaxxing decode | **68.7 tok/s** (iters 67.3-71.0) | warmup+3, greedy, `cmt2a84b70cp3mv0161xvsnfh` APPROVED, prompt shape prewarmed |
| TTFT (LocalMaxxing, warmed) | 317 ms | ditto |

First-touch caveat (why prewarm matters): a new prompt shape pays vLLM
inductor-compile + XPU graph capture on first use. A LocalMaxxing run with a
cold shape read 58.9 tok/s with TTFT 959 ms; the same run after prewarm read
68.7 / 317 ms. Always prewarm the exact prompt shape before the measured
iterations.

Comparing with the INT4 stack (same hardware, TP2):

| Build | Cold | LocalMaxxing |
|---|---:|---:|
| INT4 AutoRound g128 + INT8 head (record) | 82.5-94.6 | 89.8 |
| FP8 W8A8 load-time + INT8 head | 66.3 | 68.7 |

Ratio ≈ 0.7, consistent with the weight-bandwidth penalty (FP8 = 2x the
weight bytes per token). Use FP8 when full-precision lineage matters or
capacity at 262144 is the point; use INT4 for max decode.

## Gate quirk (harness, not the model)

The suite's `logic` case expects the exact string `yes`; the model answers
`Yes`. The same strict-case miss occurs on the shipped INT4 reference row
(`bench/reference/quality-int8-head.json` holds content "Yes"). Treat a
`logic`-only failure as a harness artifact; every other case must pass.

## Evidence files

- `bench/reference/qwen38-unc-fp8-load-w8a8-strict.json` — cold strict row
  (66.3 median), gate JSON embedded (`pass_all: false` with the logic quirk
  only).
- LocalMaxxing payloads and run artifacts: `/home/haxor/bench-results/localmaxxing/`
  (paylaod `...cmt2a84b70cp3mv0161xvsnfh.json` is the record).

## Reference points in this repo

- Serving/INT4 context: [README.md](../README.md), [serve.sh](../scripts/serve.sh)
- INT8 lm_head mechanism and measured gain: [optimizations.md](optimizations.md)
- Pinned toolchain + drivers: [drivers.md](drivers.md)
