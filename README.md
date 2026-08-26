# humble-b70-llm

Reproducible, quality-gated inference of **Qwen3.8-27B-Uncensored INT4** on
**Intel Arc Pro B70** — the exact stack that runs on this lab's production
machine, shipped as patches + build scripts + measured evidence.

The goal of this repository is that a stranger who clones it, reads it, and
follows `QUICKSTART.md` lands on the same numbers we publish — not because of
faith, but because everything that affects the result is pinned or measured.

## LocalMaxxing-approved speeds (headline)

Decode rate as measured by the LocalMaxxing platform (their protocol —
warm-up request, then a median of 3 greedy iterations, client post-first):

| Config | Decode (tok/s) | TTFT | tokSPrefill | Submission |
| --- | ---: | ---: | ---: | --- |
| **TP2 — 2x Intel Arc Pro B70** | **136.4** | 841 ms | 755 | `cmt0vu76q0fvtms017exhssx9` APPROVED |
| **TP2 record re-verify — UNC g128 + INT8 head** | **105.6** | 891 ms | 710 | `cmt2slgyx0hdgmv01d8vf53cr` APPROVED |
| **TP1 — 1x Intel Arc Pro B70** | **97.8** | 356 ms | 1782 | `cmt0vrxjk0fvpms01n4i9cmns` APPROVED |

<p>
  <img src="docs/assets/localmaxxing-tp2.png" alt="TP2 136.4 tok/s LocalMaxxing approved" width="45%">
  <img src="docs/assets/localmaxxing-tp1.png" alt="TP1 97.8 tok/s LocalMaxxing approved" width="45%">
</p>

Model: `johannrplaster/Qwen3.8-27B-Uncensored-int4-AutoRound`
(Qwen3.8-27B uncensored variant, AutoRound INT4 W4A16 g128, MTP head
preserved, linear-attention projections kept FP16). Stack: MTP3 speculative
decode, FP16 KV, load-time **INT8 W8A8 lm_head**, host-staged TP2 collectives
(no peer-to-peer required).

The platform protocol measures warm, repeating traffic. A stricter, one-shot
picture of the same stack — what a fresh user request actually experiences —
is under the [cold workload speeds](#cold-workload-speeds-strict-one-shot-protocol)
below.

## Cold workload speeds (strict one-shot protocol)

The same stack measured cold: each prompt sent once, prompt/KV caching off,
`cached_tokens=0` verified, median inter-token rate over tokens 1-100 after
TTFT (the in-repo harness, contract in
[docs/metrics-contract.md](docs/metrics-contract.md)).

<img src="docs/assets/localmaxxing-speeds.svg" alt="warm vs cold decode speeds">

| Serving config | Decode (tok/s, cold) | Evidence |
| --- | ---: | --- |
| 2x B70 (TP2), MTP3, FP16 KV, **INT8 LM head** | **94.58** | [json](bench/reference/tp2-mtp3-int8-head-94.58.json) |
| 2x B70 (TP2), MTP3, FP16 KV, **INT8 LM head — record re-verified** | **95.26** | [json](bench/reference/tp2-mtp3-record-reverify-95.26.json) |
| 2x B70 (TP2), MTP3, FP16 KV, stock FP16 head | 81.81 | [json](bench/reference/tp2-mtp3-fp16-head-81.81.json) |
| 1x B70, MTP3, FP16 KV, **INT8 LM head** | **66.84** | [json](bench/reference/tp1-mtp3-int8-head-66.84.json) |

### FP8 UNC variant (load-time W8A8) — full-precision lineage

Same 27B dense, served from the full-precision **uncensored** bf16 base
(`JonathanColetti/Qwen3.8-27B-Uncensored`) with weights quantized to FP8 e4m3
(dynamic-act) **at load**, plus the INT8 lm_head and FP8 KV. Motivation: no
INT4 group-size artifacts (full-bf16 lineage) at native 262144 context; decode
is the tradeoff (FP8 reads 2x the weight bytes of INT4).

| Serving config | Decode (tok/s, cold) | LocalMaxxing | TTFT |
| --- | ---: | ---: | ---: |
| 2x B70 (TP2), FP8 W8A8 load-time, INT8 head, FP8 KV | **66.3** | 68.7 (`cmt2a84b70cp3mv0161xvsnfh`) | ~317-371 ms |

Recipe, memory analysis (why FP8 KV is mandatory at 262144), first-touch
prewarm caveat and evidence: [docs/fp8-unc.md](docs/fp8-unc.md), reference
JSON [bench/reference/qwen38-unc-fp8-load-w8a8-strict.json](bench/reference/qwen38-unc-fp8-load-w8a8-strict.json).

The INT8 LM head (W8A8, quantized at load) is the key optimization:
**+15.6% decode with quality gates unchanged.** How and why it works is in
[docs/optimizations.md](docs/optimizations.md) — including the things that
did *not* work and were rejected with data.

Why the two classes differ: warm, repeated, same-shape requests let clocks,
allocators, and graph dispatch reach steady state; cold requests pay
first-touch costs. Both numbers are real; they describe different traffic.
The [metrics contract](docs/metrics-contract.md) makes the boundary explicit.

**Genuine W8A8 INT8 trunk lane** (compressed-tensors int-quantized, dynamic
per-token INT8 activations; the pure-INT8 quality lane):

| Serving config | Decode (tok/s, cold) | Evidence |
| --- | ---: | --- |
| 2x B70 (TP2), MTP3, INT8 head, fp8 KV (RukaRat) | **63.64** | [json](bench/reference/tp2-mtp3-w8a8-int8head-63.64.json) |
| 2x B70 (TP2), MTP3, FP16 head, fp8 KV (Freaksterz) | 53.40 | [json](bench/reference/tp2-mtp3-w8a8-fzsmoothquant-fp16head-53.40.json) |
| 2x B70 (TP2), MTP4 (rejected: flat) | 57.78 | [json](bench/reference/tp2-mtp4-w8a8-int8head-57.78-REJECTED.json) |
| 1x B70 (TP1) | infeasible | 27 GB int8 trunk + head + MTP > 32 GB single card |

The full 2026-08-21/22 W8A8 campaign — kernel integration, quality
dispositions, the thinking-mode root-cause hunt, the INT8-vs-FP8 decode-gap
verdict, and the operational rules — is in
[docs/w8a8-int8-campaign.md](docs/w8a8-int8-campaign.md), with regression
tests and evidence under `regressions/2026-08-21-w8a8-gdn-bf16-thinking/`.

**FP8 lane + bigPP prefill campaign (2026-08-23).** The `qwen38tp2fp8` lane
(bf16 base → W8A8 fp8 at load, TP2 MTP3, INT8 head, fp8 KV) runs **sliced
pipelined host staging + a shared-memory 2-rank reduce** (bigPP lever L1,
env-gated fork patch, on by default for the lane). Prompt-processing is
comm-bound on this host (no GPU P2P; host-staged collectives were 94-100%
of prefill forward); L1 roughly doubles PP with decode and quality
unchanged:

| prompt tokens | PP before (tok/s) | PP after L1 (tok/s) |
| ---: | ---: | ---: |
| 240 | 658 | **969** |
| 633 | 733 | **1379** |
| 8192 | 825 | **1620** |
| 32768 | 792 | **1463** |

Prefix caching is structurally inert on this hybrid-GDN model under MTP
(upstream-known: zero-hit geometry, and corruption-on-hit if forced) —
in-session turns re-prefill the full history, which is what L1
accelerates. Full story: [docs/bigPP.md](docs/bigPP.md) §8,
[bench/reference/bigPP-fp8-report.md](bench/reference/bigPP-fp8-report.md),
regression pack `regressions/2026-08-23-bigPP-async-staging/`.
Next hardware step — bifurcated dual x8 board + the P2P probe plan:
[docs/p2p.md](docs/p2p.md).

## Quickstart (target: ~45 minutes to a verified server)

(Reproduction target is the cold row: `94.58 ± tolerance` TP2 / `66.84` TP1
via `bash scripts/bench-strict.sh`; the LocalMaxxing rows above come from the
platform's own warmed protocol and are re-verifiable with the `lmx` CLI
against the same server.)

```bash
# 1. host drivers (one-time, needs sudo + reboot on some systems)
bash scripts/setup-host.sh

# 2. build the stack (venv + vLLM patches + kernels wheel)
bash scripts/build.sh

# 3. fetch the model (18 GB) and verify it byte-for-byte
bash scripts/model.sh fetch
bash scripts/model.sh verify

# 4. serve
bash scripts/serve.sh           # TP2 by default; scripts/serve.sh --tp 1 for one card

# 5. prove it works the way we measure it
bash scripts/verify-install.sh  # ops registered + canary + smoke
bash scripts/bench-strict.sh    # cold-suite -> compare against bench/reference/
bash scripts/quality-gate.sh    # must show the arithmetic/code canaries passing
```

Expected outcome: a `94.58 ± tolerance` tok/s TP2 run and a passing quality
gate, with the same number you can re-derive from the reference JSONs.

## The three things this repo does differently

1. **The patches are the running state.** `patches/` contains the exact
   source deltas that production runs, generated from live trees against
   pinned upstream commits ([vllm](patches/vllm/BASE_COMMIT.txt),
   [kernels](patches/vllm-xpu-kernels/BASE_COMMIT.txt)). Clone + apply +
   build == what produced the numbers.
2. **Binaries are self-tested, not hash-gated.** Ahead-of-time SYCL builds
   differ by toolchain; we never hard-check AOT hashes. `verify-install.sh`
   registers ops, runs a smoke generation, and checks the deterministic
   canary (`sum(i*i for i in range(4))` must answer `14`).
3. **Every number carries its contract.** A warm, repeated, same-shape
   benchmark of this stack reads much higher than a cold fresh-response
   number — the difference is the measurement, not the machine. The
   [metrics contract](docs/metrics-contract.md) and
   [methodology lesson](bench/reference/methodology-lesson-alternative-stack-cold.json)
   make that difference explicit instead of hidden.

## Hardware notes (read before buying/borrowing)

### Minimum requirements per result (cold-workload rows, same hardware serves the LocalMaxxing rows)

| Result (tok/s, cold) | Minimum hardware | Notes |
| --- | --- | --- |
| 94.58 (TP2 INT8) | 2x B70 32 GB + ReBAR | Second slot can be PCIe x4 or x16 — the reference machine runs the second card at x4 and reproduces this row. Same host minimums as below. No peer-to-peer required. |
| 81.81 (TP2 FP16) | 2x B70 32 GB + ReBAR | Same as above; difference is only the `INT8=off` flag. |
| 66.84 (TP1 INT8) | 1x B70 32 GB + ReBAR | Any working PCIe slot. Complete, supported configuration. |

Host minimums for **building and reproducing** any row (in addition to the
single listed GPU):

- **OS:** Ubuntu-family Linux (26.04 tested; Debian-family likely).
- **RAM:** 64 GB recommended; the SYCL kernel build peaks at 7-12 GB per
  compile job (6 jobs by default). 32 GB works with `MAX_JOBS=3` and the
  reduced attention presets but is not validated in this repo.
- **Disk:** ~90 GB free (toolchain + 18 GB model + build tree).
- **Driver:** a Battlemage-capable `xe` kernel line + Intel compute runtime
  (exact versions and verification in [docs/drivers.md](docs/drivers.md)).
- **GPU:** Arc Pro B70 (32 GB) is the only verified GPU. Other Arc/XPU
  variants are unverified and will not necessarily reproduce these rows.

### Notes

- **No GPU peer-to-peer required.** TP2 works via host-staged collectives
  (in the patch set) on motherboards without P2P routing. With working
  peer IPC the same patches run faster; without it, they still run and
  reproduce.
- See [docs/hardware.md](docs/hardware.md) for the full truth table,
  BIOS/ReBAR requirements, and power findings.

## Repository map

```text
patches/          source deltas vs pinned upstream (vllm, vllm-xpu-kernels)
scripts/          setup-host / build / serve / bench / quality / verify
<<<<<<< HEAD
bench/            the exact harness + reference evidence
docs/             hardware, drivers, metrics contract, optimizations, attempts, troubleshooting, w8a8 campaign
regressions/      dated regression tests + evidence (w8a8 thinking/decode campaign)
=======
bench/            the exact harness + reference evidence (incl. the FP8 UNC cold row)
docs/             hardware, drivers, metrics contract, optimizations, attempts, troubleshooting, fp8-unc
>>>>>>> 6ea4c94 (feat(fp8): UNC FP8 W8A8 variant - serve.sh --quant/--template flags, recipe + memory analysis, measured rows (cold 66.3, lmx 68.7), evidence JSON)
```

## License and provenance

Apache-2.0. All third-party bits are upstream open-source (vLLM,
vllm-xpu-kernels, oneDNN, oneCCL); attributions are kept generic in-repo.
The model and its derived versions are Apache-2.0.
