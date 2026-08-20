# Reference evidence

Every JSON here was produced by `scripts/bench-strict.py` /
`scripts/quality-gate.py` under the metrics contract
(`docs/metrics-contract.md`) on the reference B70 machine.

| File | What it shows |
| --- | --- |
| `tp2-mtp3-int8-head-94.58.json` | **Headline**: TP2, MTP3, INT8 LM head (shipped default) |
| `tp2-mtp3-fp16-head-81.81.json` | Baseline: same stack, stock FP16 head |
| `tp1-mtp3-int8-head-66.84.json` | Single-card (TP1), INT8 head |
| `tp2-mtp3-int8+int4draft-91.84-REJECTED.json` | Attempted INT4 draft head; measured and rejected (`docs/optimizations.md`) |
| `tp2-mtp4-fp16-79.84-REJECTED.json` | Attempted MTP4; measured and rejected |
| `quality-fp16-head.json` | Quality gate, FP16 head (baseline parity) |
| `quality-int8-head.json` | Quality gate, INT8 head (identical pass set) |
| `warm-vs-cold-methodology-lesson.json` | Same stack under a warm protocol — why warm numbers are not the headline |

Run-to-run variance is expected; compare within the tolerance band
(`scripts/bench-strict.sh` does this automatically).
