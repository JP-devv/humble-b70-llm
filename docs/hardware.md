# Hardware requirements and truth table

## Minimum viable

- One Intel Arc Pro B70 (32 GB).
- ReBAR enabled for the card (large BAR support in BIOS). Verify with
  `lspci -v | grep -i rebar` or `xpu-smi` memory sizing: you need the full
  32 GB window.
- Any x86-64 Linux host with level-zero driver support for Battlemage.

The single-card configuration is complete and supported: TP1 + MTP3 +
INT8 head = 66.84 tok/s (see README).

## Reference two-card configuration

- Two Intel Arc Pro B70, 32 GB each.
- PCIe 5.0 x16 links, full-size ReBAR.
- TP2 serving uses host-staged collectives (in `patches/`), which work with
  or without GPU peer-to-peer:

| Platform property | Effect |
| --- | --- |
| No GPU P2P (cards on separate root ports) | Works. Comm collectives are host-staged; TP2 decode is the 94.58 row |
| Working peer IPC (server-class platforms) | Same patches run; comms can be captured in graphs and decode is faster |

The difference between the two is a hardware property of the
motherboard/CPU, not of this software. If you measure between the two, label
the result accordingly.

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
