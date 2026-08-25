# bigPP — raising TP2 prompt-processing (PP) speed without degrading the rest

Status: plan v2 (2026-08-23) + FP8-lane results (same day, §8): **L1 shipped
on qwen38tp2fp8** (sliced pipelined staging + shm 2-rank reduce, ~1.9x PP at
633-8192, decode in band, quality PASS, production default). Applies to the
INT4 record stack first (qwen38tp2); findings transfer to the FP8/W8A8 TP2
lanes. All facts marked "measured" were verified against the reference
machine or its evidence.

## 1. Goal and non-regression gates

TP2 PP runs at ~42-48% of TP1 speed (measured). Bring it to at least 70% of
TP1 on the current board (L1), and within 20% if the link-width lever (L2)
is taken, **without significant regression on**:

- Decode (cold-strict, `return_token_ids`, tokens 1-100 after TTFT): record
  row 95.26 (TP2 INT4 INT8 head); band ±3%, measured in the **same boot** as
  the PP measurement — never across boots.
- Quality gate: `scripts/quality-gate.sh` PASS after every config change
  (arithmetic canary `14`, code_execution, repeat, 16K long-context needle).
  The only tolerated cell is the documented `logic` "Yes" casing.
- Memory: no new VRAM wall at tested configs (262K ctx; fp16/fp8 KV);
  decode still at its prior KV size.

## 2. Where the time goes (measured)

### 2.1 The gap

| Lane (cold-strict suite, ~240-token prompts, 1 chunk) | PP tok/s (median) | TTFT (median) |
| --- | ---: | ---: |
| TP2, INT4 trunk, INT8 head | 626 | 360 ms |
| TP1, INT4 trunk, INT8 head | 1297 | 175 ms |

The lmx rows show the same ratio (755 vs 1782 tokSPrefill on the platform's
fixed ~633-token prefill prompt). **TTFT delta ≈ 185 ms.**

### 2.2 The mechanism

Model facts (from the checkpoint config, measured): 64 layers (48
`linear_attention` GDN + 16 `full_attention`), hidden 5120,
intermediate_size 17408, fp16. TP2 does exactly **2 all-reduces per layer**
(attn-out projection, MLP down projection) regardless of layer type — GDN
cores add no cross-card comm (head-sharded state). That is **128 staged
collectives per prefill step**, payload M×5120×2 B each:

- M=240: 2.46 MB per collective, ~0.63 GB of staged traffic per step.
- M=8192 (one chunk): 83.9 MB per collective.
- M=1 (decode): ~10 KB — under the 1 MB staging threshold, native small
  path. **This asymmetry is why PP lags and decode does not.** (MTP verify
  steps are ≤4 tokens → ≤40 KB, still native. The MTP draft forward runs at
  M=1 after target prefill — a few extra native collectives, negligible.)

Host-staged path (`patches/vllm/.../xpu_communicator.py`): any all-reduce ≥
`VLLM_XPU_HOST_STAGED_MIN_BYTES` (default 1 MB) goes
`GPU → pinned host (blocking copy_) → gloo CPU all_reduce (blocks the engine
thread) → pinned host → GPU (blocking copy_)`. Four sequential blocking ops,
zero overlap, per collective, 128× per prefill step.

### 2.3 Link topology (measured on the reference machine, ASRock Z790, Raptor Lake-S)

```
CPU PEG 01.0  32 GT/s x16 (PCIe 5.0)  → switch 0000:02 → B70 #1 (0000:03)
PCH  1c.04.0  gen4, x4 (DmiWidth=x4)  → switch 0000:0a → B70 #2 (0000:0b)
```

- B70 #2 is **structurally capped at PCH x4** (~8 GB/s raw, ~6.4 GB/s
  effective). The board has no second CPU x16; the empty root ports
  (1a/1b/1c.0/1d) are PCH lanes (gen3/4 x4 at best). Swapping the two cards
  does not help: the link *set* {x16, x4} is unchanged and host-staging
  traffic is symmetric per card.
- Caveat: idle reads of the GPU↔switch links show x1 gen1 (ASPM low-power
  state). E0 must re-read `current_link_speed/width` **under load** to
  confirm x16/x4 negotiation holds at runtime.
- `hardware.md` already labels this: no GPU P2P (separate root ports),
  host-staged collectives; "working peer IPC (server-class platforms) …
  faster."

### 2.4 Cost model (fits the measured 185 ms)

Per 240-token prefill step, the wall time is bounded by the slower card's
link plus the serialized CPU work:

| Component | Estimate |
| --- | --- |
| x4 staged traffic on B70 #2: 128 × (2.46 MB D2H + 2.46 MB H2D) at ~6.4 GB/s | ~98 ms |
| B70 #1 same traffic on x16 | overlaps in time with #2 (concurrent) |
| gloo all-reduce on loopback: 128 × ~0.5-0.7 ms | ~65-90 ms, serialized (engine thread blocks) |
| op dispatch/sync for 128 × 4 blocking calls | tens of ms |
| **Sum** | **≈ 170-190 ms ≈ observed 185 ms delta** |

The components are not provable in isolation without per-op timing — which
is exactly what E0 instruments.

### 2.5 Consequence (drives the ranking)

Async staging (L1) can hide the gloo portion and part of the memcpy under
GEMM compute, but the **x4 link bandwidth is a hard floor**: ~98 ms of it
cannot be hidden under ~80-100 ms of per-rank prefill compute. So L1 alone
tops out near ~70% of TP1 PP; reaching "within 20%" requires shrinking the
volume itself (L2 link width, or L6 P2P). This is why L2 keeps the top
expected-impact slot despite its cost.

## 3. Levers at a glance

| # | Lever | Type | Expected PP gain (240-tok) | Risk to decode/quality/mem | Cost |
| --- | --- | --- | --- | --- | --- |
| L1 | Async double-buffered host staging | fork patch | −(gloo + hidden memcpy) ≈ 60-130 ms; ceiling ~70% of TP1 | decode only if threshold logic touched | M |
| L2 | Second x16: new board (or server-class platform) | hardware | −~90 ms (x4→x16 on B70 #2) | none; decode flat/up; **relabel rows** | L (board) |
| L3 | gloo interface selection + NUMA pinning | env only | 0-30 ms | none | S |
| L4 | Staging threshold A/B (is staging even the winner at 2.5-84 MB?) | env only | unknown, decisive for L1's design | decode routing — re-verify per cell | S |
| L5 | Larger prefill chunks (16-32K), long prompts only | launcher flag | per-chunk fixed overhead, M>8192 only | activation VRAM peak (see §4.5) | S |
| L6 | BIOS P2P (ACS/IOMMU) → native device IPC, in-graph comms | host | ceiling: comms overlapped in graph | topology relabel; PCH routing may not exist | S (lottery) |
| L7 | Sequence-parallel prefill | architecture | high at large M | GDN recurrence complicator; decode untouched | L |

Order: **E0 → L3+L4 (env day) → L1 (fork) → L5 → L6 (opportunistic) → L2
(board, when downtime + budget allow) → L7 (only if a compute-bound residual
remains at M ≥ 8192 after L1+L2).**

## 4. Experiments

Protocol for every experiment: cold-strict suite (`scripts/bench-strict.sh`)
+ the §4.0 PP probes + `scripts/quality-gate.sh`. Any source or env-mode
change requires `rm -rf ~/.cache/vllm/torch_compile_cache` first (stale
artifact bind failures — the known `MergedColumnParallelLinear` symptom).
Every run lands as `bench/reference/bigPP-<exp>-<config>.json` with a row in
`bench/reference/README.md`. Decode row is always re-verified in the same
boot.

### 4.0 E0 — instrument, baseline, confirm the model (gates everything)

1. Env-gated staging timer in `XpuCommunicator` (`VLLM_XPU_STAGE_TIMING=1`):
   cumulative ms and call count split into D2H, gloo-reduce, H2D, plus bytes
   moved; log once per prefill step through the existing
   `VLLM_XPU_STEP_TIMING` channel. One `if` per collective when unset.
2. Verify runtime link state under load:
   `cat /sys/bus/pci/devices/0000:0{3,b}:00.0/current_link_{speed,width}`
   during a 32K prefill (idle reads show x1 gen1 — ASPM; expect x16/x4
   under traffic).
3. PP probes (new files, **do not modify** `bench/realistic-suite-v1.json`):
   single prompts of **633** (lmx parity), **8192**, **32768** tokens;
   greedy, thinking off, `cached_tokens==0` verified per request. Report
   PP tok/s = prompt_tokens / TTFT at each size, plus per-step staged-ms.
4. Baseline both lanes (TP2 + TP1) with the timer on.

Verdict gate: staged-ms ≈ 150-180 ms of the 185 ms delta, with the
§2.4 split (memcpy-dominated vs gloo-dominated decides how much L1 can
hide). If the delta does NOT sit in staging, re-plan from the actual split
(fallback suspects: per-step scheduler/graph-dispatch, oneCCL small-path
dispatch cost).

Artifacts: `bench/reference/bigPP-e0-baseline-tp2.json`, `...-tp1.json`.

### 4.1 L3 — gloo interface + NUMA (env only, no source)

Verified against the venv build (`strings libtorch_cpu.so`, measured):
this gloo exposes `GLOO_SOCKET_IFNAME`, `GLOO_DEVICE_TRANSPORT`,
`GLOO_LOG_LEVEL`, `GLOO_ENABLE_RANK_AS_SEQUENCE_NUMBER` — **not**
`GLOO_TRANSPORT=shm`. So:

- Ensure `GLOO_SOCKET_IFNAME=lo` (or the fastest local interface) instead of
  any multi-NIC auto-pick; check which interface gloo binds today (one
  `GLOO_LOG_LEVEL=debug` boot).
- NUMA-pin engine + gloo worker threads to the socket owning the x4 root
  port (the PCH is on socket 0's domain on this board — confirm with
  `numactl -H` before pinning).
- Optional follow-up (separate decision, not part of this plan): whether a
  newer gloo with shm transport is worth a rebuild — log the answer.

Gate: PP probes flat-or-better, decode in band, quality PASS. Expected
0-30 ms. Even a flat result rules the CPU-copy bucket out of the split.

### 4.2 L4 — staging threshold A/B (env only; the decisive question for L1)

`VLLM_XPU_HOST_STAGED_MIN_BYTES`:

- `1 MB` (current): prefill staged (2.46 MB ≥ 1 MB), decode native.
- `0`: **everything** on the native oneCCL path (`CCL_ATL_TRANSPORT=ofi`),
  including 2.46-84 MB payloads — oneCCL does its own internal staging,
  possibly batched/overlapped where our four-op loop is not.

(An `8 MB` cell was considered and dropped: decode is 10-40 KB at all
thresholds and prefill ≥84 MB per chunk is staged at all of 1/8 MB — the
cell adds nothing.)

Sharp question: **does naive blocking staging actually beat oneCCL-native
at these payload sizes?** If native is faster, L1's design changes from
"make staging async" to "route prefill native and fix what's left". Cells:
`0` and `1 MB`, each with PP probes + same-boot decode row + quality gate.

### 4.3 L1 — async double-buffered host staging (fork patch, the code lever)

Design (only if L4 keeps staging as the better backend):

- Two pinned staging slots per (shape, dtype), rotated across consecutive
  collectives.
- D2H with `non_blocking=True` + event; the gloo `all_reduce` moves off the
  engine thread (dedicated CPU worker thread, or `async_op=True`) so the
  engine can dispatch layer i+1's GEMM while layer i's CPU reduce runs; H2D
  `non_blocking=True`; the **consumer-side event wait is exactly at the
  first op that reads the reduced tensor** (the layernorm after each
  out/down projection). Double depth covers one in-flight collective;
  assert the invariant.
- Correctness tests (regression pack pattern,
  `regressions/2026-08-2x-bigPP-async-staging/`): a synthetic
  GEMM→all-reduce×64 loop must be **bit-identical** to the blocking path
  over repeated runs; a throughput test asserts the staged-ms per
  collective drops while wall time drops by more than the pure memcpy
  savings predict (i.e., overlap actually happened).
- Env-gated `VLLM_XPU_STAGE_ASYNC=0|1`; blocking stays default until the
  full gate passes, then flip default and document.

Gate: decode in band (staging is off the decode hot path at 10-40 KB —
verify, don't assume), quality PASS, PP probes improved; acceptance:
staged-wall-time < 30% of TP2 TTFT at 240 tokens, PP@240 ≥ 900 tok/s
(~70% of TP1).

### 4.4 L2 — link width (the volume lever)

Measured reality: B70 #2 is PCH gen4 x4 with no second CPU x16 on this
board. Options, in cost order:

1. **Accept x4 as the documented floor** (keep the x16+x4 row as the
   reproducible minimum — the README already does this).
2. **New board with two CPU x16 PEGs** (or borrowing a server-class
   platform): move B70 #2 to a second CPU root port. Re-run the full
   validation set in one boot (strict suite + PP probes + gate +
   `verify-install.sh`). Expect the ~98 ms memcpy term to drop to ~20-25 ms.
3. Board-level check (cheap, do with E0): confirm no BIOS option re-routes
   the second slot from PCH to CPU, and that both links negotiate at full
   width under load (§4.0 item 2).

Metrics-contract consequence: link topology is a **hardware property**
(`docs/hardware.md`) — the x16+x16 result is a new labeled row
(`bench/reference/tp2-x16x16-*.json`), not a silent replacement of the
95.26 row; lmx notes carry the topology. Composes with L1: x16 shrinks the
volume, async hides the residual — together, "within 20% of TP1" should be
reachable at 240 and 633 tokens.

### 4.5 L5 — larger prefill chunks (long-prompt lane)

`--max-num-batched-tokens` 8192 → 16384 → 32768 (keep `--max-num-seqs 1`):

- Only changes behavior for prompts > 8192 tokens (a 262K prefill = 32
  chunks at 8192, 8 at 32768). Per-chunk fixed costs (scheduler step,
  graph dispatch, per-op dispatch overhead of the 128 collectives) shrink
  per token; collective volume is unchanged.
- Memory, real numbers (intermediate_size 17408, fp16): one 32K-chunk MLP
  activation = 32768 × 17408 × 2 B ≈ 1.14 GB; with gate/up/down + residuals
  the transient peak is on the order of **4-6 GB per rank** — material at
  `gpu-memory-utilization 0.95`. Measure the load-time peak and the KV size
  line before accepting.
- Compile-range caveat: the running engine logs
  `compile_ranges_endpoints: [8192]`. Raising the chunk max may push
  prefill shapes past the compiled range (eager fallback or recompile at
  next boot) — check the boot log for new compile activity and time it.
- Measure on the 8K/32K probes only. Gate: decode flat, quality PASS, KV
  size unchanged, VRAM headroom kept. Decision: per-lane flag in the
  launcher config layer (record lane may stay at 8192 for comparability),
  not a global source default.

### 4.6 L6 — BIOS P2P (lottery, opportunistic)

- Pre-flight: raw L0 peer-IPC probe (create shared handle on device A,
  `zeMemOpenIpcHandle` on device B) — the failure that birthed host staging
  (`ZE_RESULT_ERROR_INVALID_ARGUMENT` cross-device). Also record both GPUs'
  IOMMU groups (B70 #2 is group 26 — groups must align to allow peer
  mapping).
- Two config permutations (ACS quirk off; IOMMU grouping/ATS change), probe
  each. If it passes: comms become device-to-device and capturable in
  graphs (the "faster" branch in `hardware.md`); host staging demotes to
  fallback; full re-validation + new labeled rows.
- Exit criterion: two failed probes → park. The second card is behind a PCH
  port with its own switch; PEG-to-PEG routing through the chipset may
  simply not exist on this board.

### 4.7 L7 — sequence-parallel prefill (last resort)

Split the prefill sequence across ranks (each rank computes half the
tokens; one all-gather/reduce-scatter pair of M/2-sized tensors per layer).
**Complicator (measured model fact): 48 of 64 layers are GDN linear
attention — sequential/recurrent.** A sequence split needs the recurrent
state handed off at the split boundary on every GDN layer (all-gather of
the state, small, plus the out-proj exchange), and the GDN XPU kernel would
need a split-state variant. Net comm may not improve over the 128
all-reduces. Only worth it if, after L1+L2, PP at M ≥ 8192 is
compute-bound and still far from TP1 — then scope it as a separate campaign
with its own regression pack. Decode stays untouched (M ≤ 4, single
chunk, unchanged path).

## 5. Decision tree

```
E0: does staged-ms explain the 185 ms delta?
├── no  → re-derive the split from the timer; re-plan (scheduler /
│         graph-dispatch / oneCCL small-path buckets)
└── yes → L3 + L4 (env day)
          ├── L4: native oneCCL faster at 2.5-84 MB?
          │     yes → redesign: route prefill native, fix residual; L1
          │           becomes "fix the native path's staging"
          │     no  → L1 async staging (bit-exact unit test + gate)
          └── L3 win → keep (record rows either way)
           L5 chunk size (8K/32K probes)
           L6 opportunistic (2-probe budget)
           L2 board swap (full re-validation, new labeled rows)
           L7 only if a compute-bound residual remains at M ≥ 8192
```

## 6. Pitfalls (from AGENTS.md — repeated because they bite here)

- Any vLLM source edit or env-mode change (stage timing, async, threshold) needs
  `rm -rf ~/.cache/vllm/torch_compile_cache` before the next boot; first
  boot after a clear = 2-4 min compile, not a bug.
- Decode measured **only with `return_token_ids`** (tokens 1-100 after
  TTFT); counting SSE chunks overstates ~2.8× under MTP.
- Quality gate before any speed claim; `logic` "Yes" casing is the only
  documented tolerance.
- Kill servers only via the reference box's `killvllm.sh`; never `pkill -f vllm`.
  Manual fp8 boots need `VLLM_XPU_HOST_STAGED_COLLECTIVES=1` + the served
  name `Qwen3.8-27B-UNC-FP8`.
- Topology changes (L2, L6) are hardware relabels under the metrics
  contract — never overwrite existing reference rows silently.
- `--max-num-seqs 1` throughout: batching changes the prefill arithmetic
  and would confound the measurements.
- The venv's gloo does **not** expose `GLOO_TRANSPORT` (verified) — don't
  ship a launcher flag for an env var the build ignores.
- Other tenants share this host (ComfyUI sessions on both cards, swarm
  build): record VRAM tenants at the start of each experiment or the
  memory conclusions are void.

## 7. Acceptance (tiered)

Tier A — current board (L1 + L3/L4/L5), no hardware change:

1. Staged wall time < 30% of TP2 TTFT at 240 tokens (E0 timer).
2. PP@240 ≥ 900 tok/s (~70% of the same-boot TP1 row) and PP@633 ≥ 950.
3. Decode row within ±3% of 95.26, same boot; quality gate PASS.
4. New reference rows for every retained config; reference README updated.

Tier B — with L2 (second CPU x16 / server-class platform):

1. PP@240 and PP@633 within 20% of same-boot TP1.
2. Decode in band (or a new labeled row — topology change).
3. x16+x16 rows added as new labeled rows; README prefill table gains a
   per-topology section; lmx submissions (if any) carry the topology in
   notes.

Done = Tier A achieved, or Tier B achieved if the board move happened,
with all changed envs/flags in the launcher config layer (no hardcoded
values in source) and the async path env-gated in the fork patch.

## 8. 2026-08-23 — FP8 lane campaign results (qwen38tp2fp8), L1 shipped

Full report + evidence: `bench/reference/bigPP-fp8-report.md` and
`bench/reference/bigPP-fp8-*.json/txt`. Regression pack:
`regressions/2026-08-23-bigPP-async-staging/`.

### E0 on fp8 (instrumented split, `VLLM_XPU_STAGE_TIMING`)

The staged host-comm path is **94-100% of the prefill forward at every
size** (133 collectives/step = 128 trunk all-reduces + 5 MTP-proposer):

| num_sched | fwd ms | staged ms | rank1 (x4) d2h/gloo/h2d | rank0 (x16) d2h/gloo/h2d |
| ---: | ---: | ---: | --- | --- |
| 240 | 344 | 322 (94%) | 97 / 173 / 51 | 57 / 250 / 14 |
| 633 | 794 | 784 (99%) | 245 / 413 / 126 | 140 / 607 / 34 |
| 8000 | 9285 | 9491 (~100%) | 3201 / 4720 / 1570 | 2048 / 7082 / 345 |

torch-profiler window (633+8192): copies 31%, compute ~20% (fp8 GEMM 12%,
attn+GDN 4%, fused quant/elementwise 3%), GPU idle-in-blocking-comm ~49%.
The fp8-GEMM speed advantage over int4 is real but invisible behind
serialized staging. Per-collective at 81.9 MB: card2 H2D 6.9 GB/s (x4
ceiling), card1 34 GB/s, gloo loopback TCP ~4.6 GB/s. sysfs link
speed/width never leaves gen1 x1 even under load — unusable as runtime
evidence; use derived bandwidth. Observed chunking: 8192 prompt ->
6400+1792; 32768 -> 8000x3+6400+2368 (scheduler never emits 8192).

### Env-day cells

- **L3 (GLOO_SOCKET_IFNAME=lo + NUMA)**: flat. Gloo data was already
  loopback (127.0.1.1); the LAN-hairpin sockets are only the c10d
  TCPStore rendezvous. Single NUMA node — nothing to pin. Not retained.
- **L4 (everything native oneCCL)**: boot crash at engine init —
  `zeMemOpenIpcHandle → ZE_RESULT_ERROR_INVALID_ARGUMENT` on the first
  >=1 MB all_reduce. Native is impossible at prefill payloads on this
  topology; host staging is the only path. Closed.
- **2c (fused quant+GEMM)**: skipped — prefill is comm-bound, not
  launch-bound.

### L1 as implemented (differs from §4.3, deliberately)

A fully async cross-layer pipeline is not implementable inside
XpuCommunicator: the reduced tensor is consumed (residual add)
immediately after `all_reduce` returns, so the H2D must be stream-ordered
before that consumer — a host-mediated reduce forces one host wait per
collective regardless of buffering. Shipped instead, both env-gated in
`xpu_communicator.py` (blocking path untouched, still the code default):

- `VLLM_XPU_STAGE_SLICE_KB` (production: 8192): intra-collective sliced
  pipelining. D2H per ~8 MiB slice enqueued async with one event each; a
  dedicated worker thread runs the CPU reduce per slice as its D2H lands;
  the engine enqueues each H2D as its reduce completes. Wall per
  collective: D2H+reduce+H2D -> ~max(D2H+H2D, reduce). Bit-exact
  (all_reduce is elementwise; consumer stream-order unchanged; gloo call
  order per rank unchanged).
- `VLLM_XPU_STAGE_SHM` (production: 1, sliced path, world==2): replaces
  the per-slice gloo reduce with a /dev/shm exchange (per-boot random
  name rendezvoused over a gloo broadcast; region-per-rank + rdy/ack spin
  barriers + torch CPU add on frombuffer views). Measured bit-identical
  to gloo's 2-rank sum; ~7x faster per 8 MiB slice (gloo 4.7 s -> shm
  0.63 s per 8000-token chunk). No gloo rebuild needed.

Two bugs the regression pack caught: silent worker-thread exceptions hung
the engine on the done-queue (now propagated); vLLM runs forwards under
`torch.inference_mode()`, making the pinned staging buffers inference
tensors — the shm worker's `torch.add(out=...)` raised outside inference
mode (boot hang at the profile dummy run; fixed by re-entering
inference_mode; test covers it).

### Measured results (cold, same-boot decode rows)

| config | PP@240 | PP@633 | PP@8192 | PP@32768 | strict decode | quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E0 blocking | 658 | 733 | 825 | 792 | 66.89 | PASS* |
| L1a sliced (gloo) | 635 | 792 | 1060 | 1020 | 66.74 | PASS* |
| **L1b sliced+shm (shipped)** | **969** | **1379** | **1620** | **1463** | **66.78** | PASS* |

*PASS = all green except the documented `logic` "Yes" casing tolerance.
Unit loop (84 MB payload, GEMM+all_reduce): blocking 63.1 -> sliced 43.8
-> sliced+shm 27.3 ms (2.34x). One cold-boot repeat-case outlier
investigated to ground (20/20 steady-state greedy repeats identical;
cold-boot quality-first rerun PASS) — first-request cold-shape numerics,
not shm. Warm production boot: 206-240 ms @240 (~1000-1160 tok/s).

Tier A met on this lane. Production: defaults set in the switch mode env
block for qwen38tp2fp8; revert with
`EXTRA_LAUNCH_ENV="VLLM_XPU_STAGE_SLICE_KB=0" b70-switch.sh qwen38tp2fp8
--force`. Other lanes untouched (int4/w8a8 would benefit identically —
same staging path — but need their own gate runs).

### Post-L1 state and revised lever order

The 8192 floor is now ~5.1 s/prompt, dominated by card2 x4 copies. New
order: **L2 (x16 board) now pays** (copies are the residual binder), then
L5 (larger chunks, long prompts only); the gloo-shm question is answered
(custom shm, done). L7 unchanged (last resort).

### 8.1 In-session caching verdict + L5 (same evening)

Prefix caching is **structurally inert** for this hybrid-GDN model under
MTP and must not be force-enabled: upstream-known (vllm#45238 — align
mode registers ~1-2 mamba checkpoints per request near prompt end, so the
hybrid intersection vetoes all hits; vllm#43559/#50630 — hits that DO
land under MTP silently corrupt outputs because MambaManager ignores the
eagle drop-block margin). Measured here: 0.0% hit rate, cached_tokens=0
on identical prompts, hits==0 with queries>0 (Prometheus). The no-MTP
path is broken in this fork (scheduler KeyError on first request) and
correlated with 3 of 4 hard host resets that evening — do not retry it.
The real cure is upstream RFC #52959 (per-boundary state checkpoints).
Full detail: `bench/reference/bigPP-fp8-report.md` §3c.

L5 (MAX_NUM_BATCHED_TOKENS=16384): flat — 950/1341/1607/1489 vs L1b's
969/1379/1620/1463, and it costs ~32K tokens of KV (600,441 vs 632,660).
Reverted to 8192. Per-chunk fixed costs are immaterial; copy volume is
the binder, as the E0 split predicted.

### 8.2 What remains (software front, post-campaign)

Exhausted with evidence: fused quant+GEMM (dispatch not the cost), native
oneCCL (impossible on this topology — boot crash), gloo ifname/NUMA (flat),
chunk size L5 (flat, costs KV), faster fp8 kernels (compute is ~20% of the
prefill window), no-MTP mode (broken fork path + reset correlation).

Two software items remain:

1. **Port L1 to qwen38tp2 (int4) and qwen38w8a8** — the patch is
   lane-independent (same staging path). Expected ~1.3-1.5x PP @240 and
   ~1.8x @8192 there (int4's comm share is smaller at short prompts).
   Cost: one gate run per lane (probes + strict + quality); the code and
   regression pack already exist.

   **DONE for qwen38tp2 (2026-08-24)** — shipped on the INT4 record lane:
   PP **1014/1351/1608/1531** @240/633/8192/32768 (vs ~626 @240 pre-L1,
   ~1.6x; long prompts match the fp8 lane's L1b rows), strict decode
   **92.63** in band vs 94.58 and 95.26 (±3%), TTFT 360 -> 221.8 ms,
   quality PASS (documented `logic` tolerance only). Same switch envs as
   the fp8 lane; revert
   `EXTRA_LAUNCH_ENV="VLLM_XPU_STAGE_SLICE_KB=0" b70-switch.sh qwen38tp2
   --force`. Evidence: `bench/reference/bigPP-int4-l1-{probes,strict,quality}.json`.
   One hard host reset occurred during the first int4-L1 boot (~2 min into
   the gate; no MCE/panic — logs just stop, the §8.3 pattern); retried the
   identical config, complete gate passed, no recurrence. Compile cache
   cleared and rebuilt per the post-reset playbook. Remaining: qwen38w8a8
   (same one-gate-run cost).
2. **Prefix-caching checkpoint feature** — the only route to sub-linear
   mid-session PP. Scope: (a) checkpoint GDN state at every block boundary
   in align mode instead of ~1-2 per request (fixes the vllm#45238
   geometry veto); (b) make `MambaManager.find_longest_cache_hit` honor
   `drop_eagle_block` (fixes the #43559 corruption). Days of work,
   correctness-critical, needs its own regression pack. Alternatively wait
   for upstream RFC #52959.

Hardware: L2 (second CPU x16) is now measured-justified — post-L1 the
8192 floor is card2 x4 copies; a second x16 pushes it compute-bound
(est. PP@8192 ~3200-3700).

### 8.3 Host instability (2026-08-23 evening)

Four hard resets (20:09/20:20/20:28/20:50 CDT), no MCE/panic in journals.
3/4 correlated with the unsupported no-MTP boot; one had no proximate
vLLM load. If resets recur under normal configs: hardware incident
(PSU/board), not software. The resets truncated part of the torch compile
cache (EOFError artifacts) — clear and rebuild if seen again.
