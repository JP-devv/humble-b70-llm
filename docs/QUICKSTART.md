# QUICKSTART

Goal: from a bare Linux host with B70 GPUs to a verified server in one
sitting. Estimate: ~45-75 minutes depending on build parallelism and
download speed.

## 0. Prerequisites

- One or two Intel Arc Pro B70 (see `docs/hardware.md`), ReBAR enabled.
- Ubuntu-family host (tested on Ubuntu 26.04; Debian should work).
- Docker with buildkit (for the container path; the native path needs no
  Docker).
- About 90 GB free disk (toolchain + model + build tree).

## 1. Host drivers (one-time)

```bash
bash scripts/setup-host.sh          # installs pinned runtime packages
sudo reboot                         # if a kernel/driver component changed
bash scripts/verify-install.sh --host-only   # sanity: xpu-smi shows B70
```

## 2. Build the stack

Native path (what the reference machine runs):

```bash
bash scripts/build.sh               # venv + apply patches + build kernels wheel
```

Container path: not yet provided. The native path above is the
authoritative, tested one; a Dockerfile/GHCR image is planned but is not
something we ship claims about until it is built and validated.

## 3. Model

```bash
bash scripts/model.sh fetch         # 18 GB into ./models/
bash scripts/model.sh verify        # fetches the remote index, checks all shard files present + hashes them
```

## 4. Serve

```bash
bash scripts/serve.sh               # TP2 (2 cards) by default
bash scripts/serve.sh --tp 1        # single card
```

Flags you likely care about: `--mtp 3` (default), `--int8-head on`
(default), `--ctx 8192` (default; raise with `--kv fp8` for long context).

## 5. Prove it

```bash
bash scripts/verify-install.sh      # ops registered, canary == 14, smoke ok
bash scripts/bench-strict.sh        # cold suite; prints median + compares vs reference
bash scripts/quality-gate.sh        # all canaries incl. code-eval == 14
```

`bench-strict.sh` prints a pass/fail against the reference band (±3%).
That band and the contract are in `docs/metrics-contract.md`.

## Expected results

| Config | tok/s (band) |
| --- | ---: |
| TP2 MTP3 FP16 head | 81.81 ± 3% |
| TP2 MTP3 INT8 head | 94.58 ± 3% |
| TP1 MTP3 INT8 head | 66.84 ± 3% |

If you land far outside the band, your environment differs in one of the
pinned places: driver manifest (`scripts/verify-install.sh` prints it),
power cap, PCIe topology/P2P, or the compile cache. All are documented in
`docs/`.


## W8A8 INT8 lane + thinking policy (2026-08-21 campaign)

The genuine-INT8 W8A8 trunk lane (`RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP`)
serves at 63.64 tok/s cold (TP2, MTP3, INT8 head, fp8 KV; TP1 infeasible on
32 GB cards). Thinking is safe at low/medium effort; at high/xhigh the model
family itself fails to converge (measured at full precision too), so the
shipped policy clamps effort to medium: pi thinkingLevelMap caps all b70
providers, and the sharp template clamps effort at render time (all copies).
Set `VLLM_XPU_GDN_BF16=1` (default) for the quality fix. Full campaign record:
docs/w8a8-int8-campaign.md.
