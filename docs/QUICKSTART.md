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

Container path (optional, once validated; GHCR image planned):

```bash
docker build -t humble-b70 .        # see scripts/build-image.sh
```

## 3. Model

```bash
bash scripts/model.sh fetch         # 18 GB into ./models/
bash scripts/model.sh verify        # sha256 vs bench/manifests/model.sha256
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
