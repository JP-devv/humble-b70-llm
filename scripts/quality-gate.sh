#!/usr/bin/env bash
# Run the quality gate (canary suite) against a running server.
# The deterministic code-evaluation canary must answer 14.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:19622}"
MODEL="${MODEL:-Qwen3.8-27B-UNC-G128-AR}"
TOKENIZER="${TOKENIZER:-$HERE/models/Qwen3.8-27B-Uncensored-int4-AutoRound}"
OUT="${OUT:-$HERE/bench/quality-$(date -u +%Y%m%dT%H%M%SZ).json}"

# The runner exits nonzero when pass_all is False (e.g. the documented
# case-only difference on the `logic` canary, docs/troubleshooting.md), so
# don't let set -e abort here: the pass/fail decision is made below from the
# output JSON, applying that documented tolerance.
set +e
"$HERE/.venv/bin/python" "$HERE/scripts/quality-gate.py" \
  --base-url "$BASE_URL" --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --chat-template-kwargs-json '{"enable_thinking": false}' \
  --output-json "$OUT"
set -e

"$HERE/.venv/bin/python" - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
exact = {c["name"]: c["pass"] for c in d.get("exact_cases", [])}
repeat = (d.get("repeat_case") or {}).get("pass")
longctx = (
    None if d.get("long_context_case") is None
    else d["long_context_case"]["pass"]
)
print("exact:", exact)
print("long_context:", longctx, "| repeat:", repeat)
print("baseline_match:", d.get("baseline_match_all"))
# A failing exact canary is tolerated only when the difference is casing on a
# short answer: the uncensored trunk answers the yes/no logic canary as
# "Yes" (the FP16-head baseline reproduces the same spelling), which the
# docs call a stylistic difference, not a regression. Everything else is a
# hard failure.
bad = [
    c["name"]
    for c in d.get("exact_cases", [])
    if c.get("pass") is not True
    and not (
        isinstance(c.get("expected"), str)
        and isinstance(c.get("normalized"), str)
        and c["normalized"].casefold() == c["expected"].casefold()
    )
]
if bad or not repeat or not longctx:
    print("GATE: FAIL -", bad or "long/repeat")
    sys.exit(1)
print("GATE: PASS")
PY
