# qwen-flash-next — Qwen3.8-Flash-Next on 2x Arc Pro B70

Status: pre-release assessment (2026-08-25). Weights due **2026-08-26 10:00 ET**
(`Qwen/Qwen3.8-Flash-Next` + official `Qwen3.8-Flash-Next-FP8` on HF/ModelScope).
Every claim below is labelled V (verified with source), I (inferred from the
Qwen3-Next family), or U (unknown until the drop). Re-derive the U rows from
`config.json` + the safetensors index when the weights land; do not promote
inferred numbers to measured ones.

## 1. Model spec (what we are sizing)

| Item | Value | Status |
| --- | --- | --- |
| Total main-model params | 125B | V (ModelScope listing via r/LocalLLaMA 1vxwu4g, NVIDIA DGX forum t/381228) |
| Engram (n-gram conditional-memory) params | 51B, **additional** to the 125B main model | V (same sources) |
| Active params per token | ~6B (A6B) | V |
| Engram design | n-gram-hash lookup table (Kimi-Δ-style "Conditional Memory via Scalable Lookup", arxiv 2601.07372); sparsely addressed, KB/token class | V (paper) / I (size of per-token reads) |
| Official quants | FP8 checkpoint announced; bf16 base expected | V (ModelScope "Expected Models") |
| Community quants | NVFP4 (vcruz305) + GGUF ladder promised; repos empty as of 2026-08-25 | V (placeholder repos) |
| Attention | hybrid GDN linear-attention + full attention every 4th layer, 2 KV heads, head_dim 256, 512 experts top-10, shared expert | I (Qwen3-Next-80B-A3B config.json; Flash-Next layer counts U) |
| MTP head | expect 1 (Qwen3-Next-80B ships 1) | I |
| Context | expect 262,144, no sliding window | I |
| Engram offload to host RAM | engine-dependent; which runtime exposes it is U (vLLM flag? llama.cpp?); day-0 llama.cpp support confirmed by unsloth; precedent: Qwen3-Next-80B shipped a 3B engram with vLLM/llama.cpp support | mixed |

Rule out: "Qwen3.8-Flash" / "Qwen3-Flash-Next" name variants, a renamed
Qwen3-Next-80B (different, older model), the huggingnews "120B" figure,
NVFP4/GGUF checkpoints as real artifacts.

## 2. Memory budget on this host

Budget (measured): 2x B70 32 GB, `gpu-memory-utilization 0.95` = ~30 GB
usable/card; host 61 Gi RAM, ~39 Gi free at assessment time, 8 GB swap file,
`/dev/shm` 31 G.

### GPU (main model only — 125B, engram offloaded)

| Quant | Main-model bytes | TP2 per card | Verdict |
| --- | ---: | ---: | --- |
| FP8 (~1 B/param) | ~125 GB | ~62.5 GB | **No** (~32 GB over usable) |
| INT4 g128 (~0.55 B/param) | ~69 GB | ~34.5 GB | **No** (~4.5 GB over) |
| 4-bit packed (~0.467 B/param, NVFP4/GGUF-class) | ~58 GB | ~29.2 GB | **Borderline, probably boots**: ~1.2 GB/card left of the 30.4 GB (0.95) budget for lm_head + MTP + activations + KV. The bf16-27B wall (hardware.md) died with 0.3 GB left at that point — but its KV term was 0.93 GB/card for 32K fp8 tokens, whereas Flash-Next's hybrid KV is ~12.3 KB/token fp8 ≈ 0.4 GB total (~0.2 GB/card TP2) for 32K. Same margin class, smaller KV term — hence "probably boots", not "fits comfortably". |

TP1 is dead at every quant (58–125 GB > 32 GB).

The community claim "FP8 on dual-32GB is tight but plausible" (NVIDIA DGX
forum) does not survive this arithmetic as stated; treat it as unverified
until the official FP8 safetensors index gives the real per-shard byte count
(embeddings/head may be excluded or differently quantized in the FP8 build).

### Host RAM (engram)

| Engram dtype | Bytes | Fit in 61 Gi |
| --- | ---: | --- |
| FP8/INT8 | ~51–55 GB | Only after evicting all page cache; collides with serving + build (7–12 GB SYCL AOT job) |
| INT4 | ~24 GB | Fits, with the policy caveats below |

Bandwidth is a non-issue: engram reads are a few n-gram-hash lookups per
token (KB/token), microseconds at DDR5-6000 effective ~50 GB/s, small against
a ~10 ms-class decode step (order-of-magnitude). Latency is the only real
question and it is swaps, not bandwidth.

**RAM policy caveats (gaps flagged in the 2026-08-25 RAM/VRAM/NVMe audit):**
the 8 GB swap file has no documented policy; a cold token whose hash lands on
a swapped engram page stutters. If the engine supports it, mlock the table or
drop swap. "Stop inference servers while building" (AOT build rule) collides
harder with a 24–51 GB resident table on 61 Gi.

### The hinge

The whole plan depends on one config key not yet public: **does the serving
engine place the 51B engram in host RAM, and at what dtype?** If the engram
lands on GPU at 4-bit, per-card GPU need becomes ~41 GB/card total
(~29 GB main + ~12 GB engram) — TP2 dead.

A second hinge, easily missed: the engram's **shipped dtype** is U. If it ships
bf16 (the family default), the table is ~102 GB — bigger than the host's entire
61 Gi RAM — and "RAM-storable" requires the engine to requantize on load, which
none of the family precedents demonstrate.

## 3. KV and context (family math, I)

Qwen3-Next-80B config: 48 layers, 12 full-attention, 2 KV heads, head_dim 256
→ per token, per full-attention layer: 2 (K+V) x 2 heads x 256 dim x 2 B =
2 KB, x 12 layers = ~24.6 KB/token bf16, ~12.3 KB/token fp8. At 128K
context that is ~3.1 GB / ~1.6 GB respectively (~0.8 / ~0.4 GB/card at TP2).
KV is **not** the binding constraint at any sane context on this model;
static weights are.

## 4. What transfers from this repo, what does not

Transfers:
- **Prefix caching is off** — hybrid GDN + speculative head reproduces the
  zero-hit / corruption-on-hit geometry (docs/metrics-contract.md, AGENTS.md
  invariants). Do not force-enable.
- **Measurement contract** — cold-strict `scripts/bench-strict.py`,
  `return_token_ids` decode, quality gate before speed (canary `14`).
- **torch.compile cache discipline** — clear on any source/env change; the
  fork deltas do not cover this model's config until re-verified.
- **INT8 W8A8 lm_head** — the +15.6% record-lane optimization is
  model-agnostic in principle (untied head, MTP multiplies head reads);
  re-run the quality gate per model.

Does not transfer:
- **MTP economics** — this model ships 1 MTP head (expected), not the MTP3
  the record lane runs; the MTP3/MTP4 rejection data (45% acceptance basis)
  re-opens at a different depth.
- **Collective arithmetic** — MoE routing makes per-step TP comm
  data-dependent (grouped-GEMM expert dispatch); the "133 collectives x
  81.9 MB per 8K chunk" staging model (docs/bigPP.md) no longer holds.
  bigPP L1b's sliced staging still applies to whatever collectives exist;
  the x16/x4 host-staged topology (docs/p2p.md) still paces it.
- **grouped-GEMM primitive availability on oneDNN XPU** is the open
  performance unknown — same primitive class that closed the
  `bf16 x s8` mixed GEMM shape (AGENTS.md).

## 5. Speed estimate (I, for planning only)

Anchor, measured on this same hardware (local skill `35b-moe`):
Qwen3.6-35B-A3B GPTQ-INT4 on 1x B70, MTP4, **187.58 t/s warm** (n=5
post-first). The per-card decode traffic is comparable: A3B reads
3B x ~0.5 B/param ~= 1.5 GB per step on one card; A6B at 4-bit reads
6B x 0.467 ~= 2.8 GB per step = ~1.4 GB/card under TP2. So the per-card
bandwidth ceiling for the 125B-A6B is on the order of the config that
already does 187 t/s warm — not "4x the dense lane" as a naive
bytes-per-token ratio would suggest (MoE reads only active experts).

Discounts from that anchor: (a) MTP depth — expect 1 head, and the 187
figure ran MTP4 (draft depth is the big MoE-decode lever; our MTP3/MTP4
rejection data on the dense lane says depth economics track acceptance);
(b) warm n=5 vs the cold-strict contract; (c) TP2 router-driven comm over
the x16/x4 host-staged links (data-dependent, unlike the dense lane's
fixed collective geometry); (d) grouped-GEMM expert-dispatch efficiency on
oneDNN XPU — unmeasured, same primitive class that closed the `bf16 x s8`
mixed GEMM shape.

**Planning estimate: ~90–160 tok/s cold-strict, wide variance across the
band.** Only a boot resolves it.

## 6. Verdict

- **FP8 (official): no** on 2x B70, as specced.
- **4-bit (NVFP4/GGUF or community AutoRound): borderline yes** — ~29
  GB/card, ~1 GB/card of headroom. This is a "probably boots, watch the KV
  size line at startup" machine, not a comfortable one.
- **Eng RAM is mandatory for the 4-bit path to be comfortable at all**,
  and it must be INT4-class dtype, not FP8.
- If it boots: expect prefix-caching-off operation, ~12.3 KB/token fp8 KV,
  and ~90–160 tok/s cold (planning band, §5).

## 7. On the drop (2026-08-26, checklist)

1. Pull `huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json` —
   num_experts, top-k, layer types (GDN vs full-attention counts), MTP
   depth, engram block definition, max_position_embeddings.
2. Pull the official FP8 `model.safetensors.index.json` — exact total bytes
   and per-tensor dtype; this converts §2's FP8 row from estimate to
   measurement and tests the "tight but plausible" community claim.
3. Find the engram-offload knob (vLLM flag / model code / llama.cpp `-dev`
   per-module placement) and the shipped engram dtype.
4. If a 4-bit checkpoint exists: byte-count it, compute per-card TP2, boot
   with `gpu-memory-utilization 0.95`, read the KV size line, run the
   quality gate, then one cold-strict bench.

## Sources

- huggingface.co/Qwen/Qwen3.8-Flash-Next (pre-release repo)
- modelscope.cn/models/Qwen/Qwen3.8-Flash-Next (expected artifacts: base + FP8)
- forums.developer.nvidia.com/t/381228 (DGX forum thread; FP8 "tight but
  plausible" claim — unverified)
- reddit.com/r/LocalLLaMA/comments/1vxwu4g (spec: 125B main + 51B n-gram
  embeddings, 6B active)
- reddit.com/r/LocalLLaMA/comments/1vxybmy (unsloth: day-0 llama.cpp support)
- huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json
  (family attention/expert/KV parameters — basis of all I-labelled rows)
- arxiv.org/abs/2601.07372 (Engram: conditional memory via scalable lookup)
- aiweekly.co/alerts/qwen-sets-aug-26-drop-for-qwen38-flash-next-a-qwen4-preview
- Local skill `35b-moe` (measured anchor: Qwen3.6-35B-A3B GPTQ-INT4, 1x
  B70, MTP4, 187.58 t/s warm — basis of the §5 estimate)
