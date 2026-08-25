# Hardware requirements and truth table

## Minimum viable

- One Intel Arc Pro B70 (32 GB).
- ReBAR enabled for the card (large BAR support in BIOS). Verify with
  `lspci -v | grep -i rebar` or `xpu-smi` memory sizing: you need the full
  32 GB window.
- Any x86-64 Linux host with level-zero driver support for Battlemage.

The single-card configuration is complete and supported: TP1 + MTP3 +
INT8 head = 66.84 tok/s (see README).

## Host minimums (to build and reproduce any row)

| Resource | Minimum | Why |
| --- | --- | --- |
| CPU | any x86-64 | compile-bound, not runtime-bound |
| RAM | 64 GB recommended; 32 GB with MAX_JOBS=3 + reduced attention presets (unvalidated) | SYCL AOT units peak 7-12 GB each; the build runs 6 jobs by default |
| Disk | ~90 GB | toolchain + 18 GB model + build tree |
| OS | Ubuntu 26.04 LTS (reference machine); see baseline below | driver packages and oneAPI toolchain target Debian derivatives |
| Kernel | 7.1.5-070105-generic (reference) | the `xe` module ships in the kernel; see baseline below |
| Driver | Battlemage-capable `xe` kernel + Intel compute runtime | see docs/drivers.md for pinning and verification |
| GPU | **Arc Pro B70 (32 GB) only — verified** | other Arc/XPU variants are unverified; the model + fp8/FP16 KV need 32 GB VRAM |

### Reference machine software baseline (the versions that produced the published numbers)

```text
OS:            Ubuntu 26.04 LTS
Kernel:        7.1.5-070105-generic (xe driver ships in-kernel with it)
intel-opencl-icd: 26.22.38646.7 (Intel PPA package)
intel-ocloc:      26.18.38308.1
Level Zero GPU:  libze_intel_gpu.so.1.15.38646
oneAPI:          2025.3 / 2026.x (both build this stack)
PyTorch:         2.13.0+xpu (only torch tested end to end)
```

Closely-related versions (same driver line, newer kernel) are expected to
reproduce within the tolerance band; materially different driver lines are
unvalidated — record your own versions with `scripts/verify-install.sh`
and label your numbers accordingly.

## Reference two-card configuration

- Two Intel Arc Pro B70, 32 GB each.
- PCIe links (measured 2026-08-22): first card on a PCIe 5.0 x16 root port, second card on a separate PCIe 4.0 x4 root port; each behind its own PCIe switch. Full-size ReBAR.
- TP2 serving uses host-staged collectives (in `patches/`), which work with
  or without GPU peer-to-peer:

| Platform property | Effect |
| --- | --- |
| No GPU P2P (cards on separate root ports) | Works. Comm collectives are host-staged; TP2 decode is the 94.58 row |
| Working peer IPC (server-class platforms) | Same patches run; comms can be captured in graphs and decode is faster |

The difference between the two is a hardware property of the
motherboard/CPU, not of this software. If you measure between the two, label
the result accordingly.

### The 32 GB-per-card VRAM wall (campaign finding, 2026-08-21)

At TP2 each card holds half the weights. Serving a 27B checkpoint with BF16
weights is infeasible: 27 GB/card leaves ~0.3 GB after head + MTP + activations
at gpu-memory-utilization 0.95, and a 32K-token fp8 KV cache needs 0.93 GB
("No available memory for the cache blocks"). FP8/INT8 weights (1 byte) fit at
13.5 GB/card with large KV headroom. Host RAM (64 GB) never affects VRAM
residency at inference. This wall blocks the "exact-activation W8A16" serving
shape on this hardware.

## Confirmed limits

- FP16 KV at 128K context does not fit one card (dense 27B); use
  `--kv-cache-dtype fp8` for long context (capacity option; FP16 KV decodes
  slightly faster at moderate context).
- 200K+ context loads but leaves no safe headroom; 256K is infeasible.
- Power: the B70 hardware cap reports ~230-275 W depending on platform;
  under full load the PMU reports actual frequencies above the requested
  boost. Dense-prefill throughput scales with the power cap; decode is
  relatively flat.

## What we could not reproduce (read before benchmarking)

- Token-for-token equality across differently-configured references at FP16
  is unsatisfiable on this stack; use the tolerance-band comparison instead.
- AOT-compiled kernel hashes are toolchain-dependent; never gate on them.
- Fresh `torch.compile` caches produce different-but-internally-deterministic
  code; the compile cache is part of run identity (see troubleshooting).
