# Reference evidence

Every JSON here was produced by `scripts/bench-strict.py` /
`scripts/quality-gate.py` under the metrics contract
(`docs/metrics-contract.md`) on the reference B70 machine.

| File | What it shows |
| --- | --- |
| `tp2-mtp3-record-reverify-95.26.json` | **Headline (current)**: TP2, MTP3, INT8 LM head, fp16 KV — the record config re-verified on the intact stack after the auto-round→gptq config flip (95.26 > 94.58). Canary `14`, `code_execution` passes |
| `tp2-mtp3-record-ark-path-48.74-CONTROL.json` | Same checkpoint WITHOUT the flip (config still `quant_method: auto-round`): the fork's ARK/inc path shadows `--quantization gptq` and decodes at 48.74. Evidence for why the flip matters |
| `tp2-mtp3-int8-head-94.58.json` | Historical headline run of the same config (2026-08-20) |
| `tp2-mtp3-fp16-head-81.81.json` | Baseline: same stack, stock FP16 head |
| `tp1-mtp3-int8-head-66.84.json` | Single-card (TP1), INT8 head |
| `tp1-mtp3-int8-head-fp8kv-128k-68.00.json` | Single-card (TP1), INT8 head, FP8 KV @ 128K ctx (measured 2026-08-20; FP8 capacity costs nothing measurable on this stack) |
| `tp2-mtp3-int8+int4draft-91.84-REJECTED.json` | Attempted INT4 draft head; measured and rejected (`docs/optimizations.md`) |
| `tp2-mtp4-fp16-79.84-REJECTED.json` | Attempted MTP4; measured and rejected |
| `quality-fp16-head.json` | Quality gate, FP16 head (baseline parity) |
| `quality-int8-head.json` | Quality gate, INT8 head (identical pass set) |
| `methodology-lesson-alternative-stack-cold.json` | Cold run of an alternative serving stack/checkpoint combination (~51 tok/s on the same card): shows that stack and checkpoint choice dominates the number — protocol alone does not explain cross-stack gaps |
<<<<<<< HEAD
| `tp2-mtp3-w8a8-int8head-63.64.json` | **W8A8 quality-config row**: genuine INT8 W8A8 trunk (RukaRat imatrix), dynamic per-token activations, TP2 MTP3, INT8 lm_head. Quality valid under the documented dispositions below. |
| `tp2-mtp3-w8a8-imatrix-int8head-60.70.json` | Same config, earlier run (run-to-run variance; 63.64 is the confirm row) |
| `tp2-mtp3-w8a8-fzsmoothquant-int8head-57.79.json` / `tp2-mtp3-w8a8-fzsmoothquant-fp16head-53.40.json` | Freaksterz SmoothQuant W8A8: INT8 head gains +4.4 tok/s (+8.2%) over FP16 head with identical quality |
| `tp2-mtp4-w8a8-int8head-57.78-REJECTED.json` | MTP4 adds nothing on the W8A8 trunk (57.78 vs 57.79) — same rejection as on the INT4 trunk |
| `quality-w8a8-int8head.json` | Quality gate, best W8A8 config |
| `quality-w8a8-codeexec-disposition.json` | Evidence for the `code_execution` canary on the censored-base W8A8 lane: the exact gate prompt answers `30` with thinking off (base-lineage behavior, reproduced on two independent quantizations and disproven as a serving regression), `14` with thinking on or rephrasing, `14` on the uncensored bf16 control |

W8A8 lane note (2026-08-21): TP2 TP1 is memory-infeasible on 32 GB cards (27 GB int8 trunk + head + MTP exceeds the budget at load); TP2 is required. Warm 633-token stream measured 56.4 tok/s (post-first, greedy, 5 iter) vs the FP8-W8A8 mode's 66.3 cold / 68.7 warm: the pure-INT8 lane trails FP8 on speed and is carried as the quality/lineage lane; the throughput record remains INT4-trunk + INT8 head (94.58 cold / 105.6 lmx). The W8A8 trunk reads 2x the INT4 trunk's bytes per decode step, so it lands at ~64 cold (TP2) — promoted as the **quality config** (highest-fidelity weights + INT8 dynamic activations, `logic` canary passes on SmoothQuant) while the throughput record stays INT4-trunk + INT8-head (94.58).
=======
| `qwen38-unc-fp8-load-w8a8-strict.json` | FP8 UNC variant (load-time W8A8, INT8 head, FP8 KV): cold 66.3 tok/s; see [docs/fp8-unc.md](../../docs/fp8-unc.md) |
>>>>>>> 6ea4c94 (feat(fp8): UNC FP8 W8A8 variant - serve.sh --quant/--template flags, recipe + memory analysis, measured rows (cold 66.3, lmx 68.7), evidence JSON)

Run-to-run variance is expected; compare within the tolerance band
(`scripts/bench-strict.sh` does this automatically).

## bigPP FP8-lane campaign (2026-08-23, `qwen38tp2fp8`)

All rows below are **"FP8 lane, no TP1-fp8 baseline"** (a TP1 FP8 lane is
memory-infeasible: 27 GB fp8 trunk > 32 GB card). Full narrative:
`bigPP-fp8-report.md` here; plan/protocol: `docs/bigPP.md`. Probes are
single-prompt TTFT at exact token counts (raw token-id prompts, greedy,
cached_tokens==0 per request); strict rows are the standard cold suite.

| File(s) | What it shows |
| --- | --- |
| `bigPP-fp8-e0-strict.json` / `bigPP-fp8-e0-quality.json` | E0 baseline same-boot rows: decode 66.89, TTFT 359.7 ms, gate passed |
| `bigPP-fp8-e0-probe-{633,8192,32768}.json` + `bigPP-fp8-e0-steps-*.txt` | E0 split: staged host-comm = 94-100% of prefill forward at every size (133 collectives/step); per-phase D2H/gloo/H2D per rank |
| `bigPP-fp8-e0-profiler-rank{0,1}.txt` | torch-profiler window: copies 31%, compute ~20%, GPU idle-in-blocking-comm ~49% — the fp8-GEMM advantage is invisible behind serialized staging |
| `bigPP-fp8-e0-linkstate-under-load.txt` | sysfs link speed/width never leaves gen1 x1 even under load (unusable); bandwidth-derived truth: card0 ~34 GB/s H2D (x16-class), card1 ~6.9 GB/s (x4 ceiling) |
| `bigPP-fp8-l3-*` | GLOO_SOCKET_IFNAME=lo + NUMA cell: flat (gloo was already loopback; single NUMA node). Not retained |
| `bigPP-fp8-l4-native-oneccl-CRASH.txt` | HOST_STAGED=0 (everything native oneCCL): boot crash at init, `zeMemOpenIpcHandle → INVALID_ARGUMENT`. Native is impossible at prefill payloads on this topology |
| `bigPP-fp8-l1a-*.json/txt` | L1a sliced pipelined staging (8 MiB slices, gloo reduce): PP 635/792/1060/1020 @240/633/8192/32768; decode 66.74 in band; quality PASS |
| `bigPP-fp8-l1b-*.json/txt` | **L1b sliced + shm 2-rank reduce — production config**: PP **969/1379/1620/1463** @240/633/8192/32768 (~1.9x vs E0 at 633+); decode 66.78 in band; quality PASS. `bigPP-fp8-l1b-quality-coldboot.json` + `-rerun.json`: the one cold-boot repeat outlier investigated and attributed to first-request cold-shape numerics (20/20 steady-state greedy repeats identical) |
| `bigPP-fp8-final-smoke.json` | Production-boot smoke (one 240 rep has a probe-side TTFT mis-capture; superseded by direct re-measurement quoted in the report: 206-240 ms @240 warm) |
| `bigPP-fp8-fork-diff.patch` | Fork diff (xpu_communicator + gpu_model_runner vs HEAD): stage timing + sliced staging + shm reduce, all env-gated |
| `bigPP-fp8-report.md` | The full report: E0 split, cell evidence, L1 design constraint and implementation, validation, production flip |

| `bigPP-fp8-l5-*.json/txt` | L5 cell (16K prefill chunks): flat (950/1341/1607/1489), KV 600,441 vs 632,660 tokens at 8192 — reverted; per-chunk fixed costs immaterial |

L1 defaults on `qwen38tp2fp8` (via the switch mode env block):
`VLLM_XPU_STAGE_SLICE_KB=8192 VLLM_XPU_STAGE_SHM=1`. Revert:
`EXTRA_LAUNCH_ENV="VLLM_XPU_STAGE_SLICE_KB=0" b70-switch.sh qwen38tp2fp8
--force`. Regression pack: `regressions/2026-08-23-bigPP-async-staging/`.
Probe tool: `scripts/bigpp-pp-probe.py`. In-session prefix-caching
verdict (structural, upstream-known): report §3c.

## bigPP INT4-lane port (2026-08-24, `qwen38tp2`)

L1 (sliced pipelined staging + shm 2-rank reduce, same env-gated fork patch)
ported to the INT4 record lane (UNC g128 AutoRound, TP2 MTP3, INT8 head, FP16
KV, gptq flip). Same protocol: cold probes + strict + quality, decode
re-verified in the same boot. Same-boot rows:

| Size / metric | Pre-L1 (int4, bigPP §2.1) | **Post-L1** |
| --- | ---: | ---: |
| PP@240 | 626* | **1014** |
| PP@633 | — | **1351** |
| PP@8192 | — | **1608** |
| PP@32768 | — | **1531** |
| strict decode | 95.26 ref | **92.63** (in band vs 94.58 [91.75, 97.42] and 95.26 [92.40, 98.11]) |
| strict TTFT (med) | ~360 ms | **221.8 ms** |
| quality | — | PASS (only the documented `logic` "Yes" casing) |

*suite-TTFT-derived (360 ms @ ~240-token prompts); probe protocol is the
same as the fp8 E0/L1 rows (raw token-id prompts, cached_tokens==0). rep1
(shape-warm/content-cold) rows. rep0 (shape-cold) inflated by lazy init,
e.g. 1983 ms @240 — same pattern as the fp8 lane.

Files: `bigPP-int4-l1-probes.json`, `bigPP-int4-l1-strict.json`,
`bigPP-int4-l1-quality.json`. L1 defaults on `qwen38tp2` via the switch
mode env block (same envs as fp8; `EXTRA_LAUNCH_ENV`). Revert:
`EXTRA_LAUNCH_ENV="VLLM_XPU_STAGE_SLICE_KB=0" b70-switch.sh qwen38tp2 --force`.

Incident note: one hard host reset at 01:38 CDT during the first int4-L1
boot (no MCE/panic in journals — logs just stop; matches the documented
pattern of §8.3). Retried the identical config; second boot and the full
gate completed with no recurrence. Compile cache cleared and rebuilt per
the post-reset playbook (§8.3).

Also on 2026-08-24, per operator request: the INT4 lanes
(`qwen38tp2`/`qwen38tp1`) now serve the model's embedded default chat
template (the launcher no longer forces `chat_template.sharp-xhighfix.jinja`;
the fp8 lane still injects its own template via the switch) and default to
non-thinking server-side (`--default-chat-template-kwargs
'{"enable_thinking": false, "reasoning_effort": "medium"}'`; request kwargs
override). pi `models.json` on both hosts reflects it (b70-unc +
b70-unc-tp1: `reasoning: false`, sampling 0.7/0.8/20/0.0/1.5/1.0); all
other providers untouched.
