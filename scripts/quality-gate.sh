#!/usr/bin/env bash
# Run the quality gate (canary suite) against a running server.
# The deterministic code-evaluation canary must answer 14.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:19622}"
MODEL="${MODEL:-Qwen3.8-27B-UNC-G128-AR}"
TOKENIZER="${TOKENIZER:-$HERE/models/Qwen3.8-27B-Uncensored-int4-AutoRound}"
OUT="${OUT:-$HERE/bench/quality-$(date -u +%Y%m%dT%H%M%SZ).json}"

"$HERE/.venv/bin/python" "$HERE/scripts/quality-gate.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --chat-template-kwargs-json '{"enable_thinking": false}' \
  --output-json "$OUT"

"$HERE/.venv/bin/python" - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
exact = d.get("exact", {})
print("exact:", exact)
print("long_context:", d.get("long_context_pass"), "| repeat:", d.get("repeat_pass"))
print("baseline_match:", d.get("baseline_match_all"))
bad = [k for k, v in exact.items() if v is not True]
if bad or not d.get("long_context_pass") or not d.get("repeat_pass"):
    print("GATE: FAIL —", bad or "long/repeat")
    sys.exit(1)
print("GATE: PASS")
PY
