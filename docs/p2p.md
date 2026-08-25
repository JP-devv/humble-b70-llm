# p2p — dual x8 bifurcation and the P2P question

Status: plan v1 (2026-08-24). Two coupled tracks on one decision:

- **Track A — bifurcated dual x8 board.** Replace the measured
  {gen5 x16, gen4 x4} link set (bigPP §2.3) with two x8 links from one CPU
  root port. This is bigPP **L2** implemented on a cheaper board than a
  second CPU x16, and it changes the link *set*, not the software.
- **Track B — GPU peer-to-peer (P2P).** bigPP **L6** as a possible *second* win from
  the same purchase: native device IPC instead of host staging. Not
  guaranteed; the Phase 0 probe in §3.2 decides it before any money moves.

Every fact marked "measured" or "live-verified" is verified evidence in
`bench/reference/` or a dated live read on the reference box. Everything else is a
prediction and must be re-derived on arrival hardware.

## 1. Established ground truth (measured)

- Link set (live-verified on the reference box 2026-08-24; supersedes the board name in
  `docs/bigPP.md` §2.3, which says ASRock): **ASUS TUF GAMING Z790-PLUS
  WIFI, i7-13700K, one NUMA node, 24 CPUs** (the "EPYC 9015" in
  `bigPP-fp8-report.md` §L3 is a transcription error; its one-node
  conclusion holds).

  ```
  CPU PEG 01.0  negotiated 32 GT/s x16 (gen5 x16) → switch 0000:02
      → B70 #1 (0000:03)  +  audio fn 8086:e2f7 (0000:04)
  PCH port 09.00.0  LnkCap gen5 x16, negotiated **16 GT/s x4 (downgraded)**
      → switch 0000:0a → B70 #2 (0000:0b)  +  audio fn (0000:0c)
  ```

  Each B70 sits behind its own two-port PCIe switch (second port is the
  B70's audio function). The x4 cap on card 2 is on the PCH side — the PCH
  port's own LnkCap is gen5 x16, so the x4 is downstream wiring/switch,
  and the board has no second CPU lane set. Swapping cards is a no-op:
  the set {x16, x4} is unchanged and staging traffic is symmetric.
- Effective bandwidth, derived from H2D phases under a real 8192 probe
  (`bench/reference/bigPP-fp8-report.md` §1d): card 1 **~34 GB/s**, card 2
  **~6.9 GB/s** (gen4 x4 practical ceiling; the x4 link runs at 86% of raw).
  **sysfs link reads are not evidence on this platform** — the endpoint-side
  reads (0000:03 / 0000:0b) stuck at x1 gen1 for the entire 41 s of a 32768
  prefill, even under load; the root-port side reads fine even at idle
  (0000:01:00.0 shows 32 GT/s x16 at rest). The B70 endpoint even reports
  LnkCap x1 gen1, so capability reads on the GPU side are meaningless too.
  Always derive bandwidth from a timed H2D/D2H phase under load.
- P2P-relevant config (live-verified 2026-08-24):
  - **ACS is enabled on both switches' downstream ports**:
    `ACSCtl: SrcValid+ ReqRedir+ CmpltRedir+ UpstreamFwd+` on 0000:02:01.0
    and 0000:0a:01.0. The CPU PEG root port (0000:01:00.0) has no ACSCtl
    (no ACS capability).
  - **IOMMU is on**: 28 iommu_groups populated, EDK2 DMAR table at boot,
    no explicit `iommu=`/`intel_iommu=` on /proc/cmdline (default-on path).
  - Consequence: a peer TLP from B70 #1 to B70 #2 must cross both switches'
    downstream ports (ACS ReqRedir/CmpltRedir redirects it to the root)
    *and* the CPU-PEG↔PCH host-bridge crossing. This board is therefore in
    the least P2P-favorable configuration in two independent ways — the
    Phase 0 probe (§3.2) exists to say which of the two (config vs
    driver/topology) is fatal.
- Post-L1b (sliced pipelined staging + shm 2-rank reduce, production default
  on qwen38tp2fp8): PP 969/1379/1620/1463 tok/s @240/633/8192/32768; the
  8192 floor (~5.1 s) is **dominated by card 2's x4 copies** (~3.2 s of it),
  with ~1.8 s of non-hidden per-collective host waits + compute
  (`bigPP-fp8-report.md` §3b). This is the term Track A attacks.
- Native oneCCL is today **not just slow, it crashes**: ≥1 MB all-reduce at
  engine init does `zeMemOpenIpcHandle` cross-device →
  `ZE_RESULT_ERROR_INVALID_ARGUMENT` → both workers die
  (`bench/reference/bigPP-fp8-l4-native-oneccl-CRASH.txt`). The driver
  currently says *not peer-mappable* — but that verdict is a function of
  topology **and** IOMMU/ACS config **and** driver support, and today's boot
  isolates none of them.
- Staged traffic per 8000-token chunk: 133 collectives (128 trunk
  all-reduces + 5 MTP-proposer) × 81.9 MB × (D2H + H2D) = **21.8 GB** on the
  pacing link. Per-collective wall time after L1b
  is ~max(D2H+H2D, reduce); both ranks' D2Hs run concurrently and the slower
  link paces (max, not sum).

## 2. Track A — bifurcated dual x8

### 2.1 What it buys (predictions, re-derive on arrival)

x8 gen5 raw = 32 GB/s; at this platform's measured ~86% link efficiency,
expect **~25–28 GB/s effective**. x8 gen4 raw = 16 GB/s → **~13–14 GB/s
effective**. Card 2's copy term at 8000 tokens: 21.8 GB / bw.

| PP cell (qwen38tp2fp8, L1b on) | today (x16+x4) | gen5 x8/x8 | gen4 x8/x8 |
| --- | ---: | ---: | ---: |
| @240 | 969 | ~1000–1100 (latency-bound: 133 × ~1 ms round-trips; bandwidth irrelevant) | ~1000 |
| @633 | 1379 | ~1700–1900 | ~1600–1750 |
| @8192 | 1620 | **~3000–3500** (copy floor 3.2 s → ~0.9 s; compute ~2 s becomes the bound) | **~2300–2500** (copy floor → ~1.6 s) |
| @32768 | 1463 | ~2800–3200 | ~2200–2500 |
| decode (strict) | 66.8 | ~66.8 (one 0.99 MB staged collective/step = 0.3 ms on x4, 0.1 on x8) | ~66.8 |

The gen4 variant still clears a **~1.4–1.5x** at long prompts and is the
floor case: even if the new board's x8s negotiate a generation down, card 2
gains ~2x and card 1's loss is off the critical path.

### 2.2 What it costs (accept these explicitly)

1. **Skew headroom.** Today card 1's 34 GB/s absorbs rank skew and jitter
   (visible as rank 0's gloo bucket exceeding rank 1's, `bigPP-fp8-report.md`
   §1c). With two equal links, residual skew lands ~1:1 in wall time — a few
   percent of the ~1.8 s non-hidden term, not a structural change.
2. **Card 1 non-collective bandwidth.** ComfyUI tenants run on both cards;
   an active card-0 ComfyUI job shares x8 with vLLM staging instead of x16.
   A/B if card-0 stalls ever appear with tenants hot.
3. **Boot/load time.** Card 1 H2D 34 → 25 (or 13) GB/s for weight load —
   seconds to tens of seconds; load is disk- and quant-bound.
4. **Config flexibility.** Trades "works on any board with one x16 plus any
   x4" (the documented reproduction path, `docs/hardware.md`) for a board
   that must actually bifurcate and negotiate full width on both x8s.
5. **No automatic P2P.** Both cards still behind separate sub-ports; P2P is
   Track B's probe result, not a property of bifurcation.

### 2.3 Board requirements (the checklist that avoids the $2K lottery)

- Bifurcation of **one CPU x16 to x8/x8**, BIOS-selectable, from a
  **single root port** (this doubles as the P2P topology — §3.3).
- PCIe 5.0 preferred; PCIe 4.0 x8/x8 acceptable (floor case, §2.1).
- Full-size ReBAR (32 GB window) — unchanged requirement.
- Used, well-reviewed server-class (CM246/CM248-class Xeon W/SP, or TRX40)
  is the target tier: P2P-between-CPU-ports with ACS off is a documented,
  long-proven property there; domestic used pricing ~$200–500, not the
  $2K overseas lot.
- On arrival, before any vLLM boot: verify both links negotiate
  gen5/x8 or gen4/x8 (`lspci -vv` LnkCap vs LnkSta at idle *and* under
  load), then derive effective bandwidth with a timed H2D (sysfs alone is
  not evidence, §1).

### 2.4 Non-regression gates (same contract as bigPP §1)

- Decode re-verified **in the same boot** as the PP measurement: 66.32 ref,
  band ±3%.
- `scripts/quality-gate.sh` PASS (canary 14, code_execution, repeat,
  long-context). Tolerated cell: the documented `logic` "Yes" casing.
- **Metrics contract: new labeled rows** (`bench/reference/tp2-x8x8-*.json`
  with the generation in the name), not silent replacements of the 1620
  rows; lmx notes carry the topology; `docs/hardware.md` gains the new
  link-set row.

## 3. Track B — P2P

### 3.1 What P2P would actually change here

- Prefill: staged collectives become native device IPC — the per-collective
  host wait (§1, the ~1.8 s non-hidden residual) can shrink toward
  graph-captured comms.
- Decode: the one staged 0.99 MB collective per step (0.5 ms gloo phase,
  with its D2H absorbing the forward-stream sync — `bigPP-fp8-report.md`
  §1b) goes native or below the staging threshold logic changes; decode
  graph purity improves. `docs/hardware.md` already labels this as a
  prediction: "comms can be captured in graphs and decode is faster."
  **It is the one unmeasured cell in the hardware table — no money moves
  until a probe says otherwise.**

### 3.2 Phase 0 — free test on the current board (does the probe gate the purchase? No — it gates expectations)

Pre-checks already done (2026-08-24, live on the reference box):

| Check | Result |
| --- | --- |
| ACS on switch downstream ports | **enabled** (SrcValid+ ReqRedir+ CmpltRedir+ UpstreamFwd+) |
| ACS on CPU PEG root port | none (no capability) |
| IOMMU | **on** (28 groups, EDK2 DMAR, no explicit cmdline opt-in) |
| Peer path | crosses both switch downstream ports *and* CPU-PEG↔PCH host bridge |

Remaining steps:

1. Boot once with `intel_iommu=off` (or `iommu=pt`) +
   `pcie_acs_override=downstream,multifunction` (the standard VFIO quirk;
   sets it for both switches' ports at once).
2. Run a **two-process IPC probe** (new: `scripts/p2p-probe.py`, two procs,
   one per card): allocate device buffers; `zeMemGetIpcHandle` on one
   process, pass the handle **plus its backing fd over a unix socket
   (SCM_RIGHTS)** to the other, `zeMemOpenIpcHandle`, then do one ≥1 MB
   oneCCL all_reduce — ≥1 MB is deliberate, it is the
   `VLLM_XPU_HOST_STAGED_MIN_BYTES` boundary whose native path crashed in
   L4. Standalone processes, **before** any vLLM boot: L4 died at engine
   init during the memory-profile pass, so the probe must not need the
   server. No compile-cache clear needed (kernel-param + fresh procs only).
3. Verdict:
   - **Handle opens** → P2P was config-blocked on today's board; ship the
     native path (`VLLM_XPU_HOST_STAGED_COLLECTIVES=0`) on the *current*
     hardware and re-measure PP + decode + quality. Track B wins with zero
     hardware; Track A becomes optional.
   - **Still `INVALID_ARGUMENT`** with IOMMU/ACS cleared → the xe driver on
     this kernel line does not peer-map CPU-PEG↔PCH pairs. Config is closed
     for this board; the driver/topology question defers to §3.3.

### 3.3 Phase 1 — on the new board

Same-root-port x8/x8 (the §2.3 requirement) is the closest consumer
topology to "both GPUs behind one switch": peer TLPs route inside the root
port without crossing the host bridge, unlike today's two-root-complex
split. Re-run the Phase 0 probe:

- **Pass** → boot qwen38tp2fp8 with `VLLM_XPU_HOST_STAGED_COLLECTIVES=0`
  (native oneCCL); L1b's staging env then inactive. Full cell: PP probes +
  strict decode (same boot) + quality. Row: `tp2-x8x8-p2p-*.json`.
- **Fail** → host staging remains the only working path (proven by L4);
  the board's value is Track A only. Document the probe output as evidence
  (client-Battlemage peer-mapping stays "unverified" until then).
- Caveat: even a passing IPC handle does not guarantee the full oneCCL
  large-payload path or in-graph capture; the boot + probes are the actual
  acceptance, not the probe alone.

### 3.4 Escape hatches (no P2P, no purchase)

- **Dual TP1** (`qwen38ar-dual`): two isolated one-card servers, ~57 tok/s
  each, zero comms. Sidesteps P2P entirely; not faster per request.
- **Track A alone**: the §2.1 gains stand without any P2P.

## 4. Sequence and decision tree

```
Phase 0 (free, this board): ACS/IOMMU probe ──pass──► native path on current hw;
                                              │              remeasure; done
                                              └fail──► buy board per §2.3
Board arrival: link negotiation + H2D-derived bandwidth ──
    gen5 x8/x8 ──► Track A expected ~2x @8192
    gen4 x8/x8 ──► Track A expected ~1.5x @8192
    worse        ──► reassess (card 1 may be *worse* than today)
Re-run Phase 0 probe (now same-root-port):
    pass ──► native oneCCL cell → tp2-x8x8-p2p rows
    fail ──► host staging rows tp2-x8x8-<gen> rows; P2P stays "unverified"
Every cell: decode same-boot, quality gate, compile-cache clear on any
source/env-mode change, rows + README index in bench/reference/.
```

Budget logic: the board is justified by Track A alone (~1.4–2x PP @8192).
P2P is an unpriced option that the free Phase 0 probe converts into a
measured quantity before (or shortly after) the purchase — never assume it
into the expected-gain table.

## 5. Evidence pointers

- Link topology + cost model: `docs/bigPP.md` §2.3, §2.4, §4.4 (L2), §8
  (board name there is stale — ASUS TUF GAMING Z790-PLUS WIFI per the
  2026-08-24 live read in §1).
- Live topology/ACS/IOMMU pre-checks: §1 and §3.2 of this doc (2026-08-24)
- L1b state + post-L1 residual floor: `bench/reference/bigPP-fp8-report.md` §3b
- Bandwidth-derived truth + sysfs caveat: `bench/reference/bigPP-fp8-report.md` §1d
  (`bigPP-fp8-e0-linkstate-under-load.txt` is the raw sysfs artifact)
- Native oneCCL failure: `bench/reference/bigPP-fp8-l4-native-oneccl-CRASH.txt`
- P2P-as-hardware-property framing: `docs/hardware.md` (reference two-card
  configuration table)
- Staging env vars: `VLLM_XPU_STAGE_SLICE_KB`, `VLLM_XPU_STAGE_SHM`,
  `VLLM_XPU_HOST_STAGED_COLLECTIVES` (switch mode block, qwen38tp2fp8)
