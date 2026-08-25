#!/usr/bin/env bash
# Run the cold strict suite against a running humble-b70 server and compare
# with the shipped reference evidence within the tolerance band.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:19622}"
MODEL="${MODEL:-Qwen3.8-27B-UNC-G128-AR}"
TOLERANCE="${TOLERANCE:-0.03}"   # ±3% band
OUT="${OUT:-$HERE/bench/out-$(date -u +%Y%m%dT%H%M%SZ).json}"

"$HERE/.venv/bin/python" "$HERE/scripts/bench-strict.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --api-mode chat \
  --suite "$HERE/bench/realistic-suite-v1.json" \
  --max-tokens 512 --metric-tokens 100 --seed 1 --return-token-ids \
  --request-extra-json '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --out "$OUT"

REFERENCE_TOP2="${REFERENCE_TOP2:-$HERE/bench/reference/tp2-mtp3-int8-head-94.58.json}"
"$HERE/.venv/bin/python" - "$OUT" "$REFERENCE_TOP2" "$TOLERANCE" <<'PY'
import json, sys
out = json.load(open(sys.argv[1]))["summary"]["tok_s_1_100_after_ttft"]
ref = json.load(open(sys.argv[2]))["summary"]["tok_s_1_100_after_ttft"]
tol = float(sys.argv[3])
med, rmed = out["median"], ref["median"]
lo, hi = rmed*(1-tol), rmed*(1+tol)
print(f"measured median: {med:.2f} tok/s | reference: {rmed:.2f} (±{tol*100:.0f}%)")
print(f"band: {lo:.2f} .. {hi:.2f} -> {'PASS' if lo <= med <= hi else 'CHECK ENV'}")
print(f"p10: {out['p10']:.2f} | mean: {out['mean']:.2f} | TTFT ms: {out.get('ttft_ms',{}).get('median','?')}")
PY
