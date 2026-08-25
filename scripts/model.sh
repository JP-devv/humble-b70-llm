#!/usr/bin/env bash
# Fetch / verify the model repository (HF handle: johannrplaster — the model's
# owner; the one personal reference intentionally kept in this repo).
# The model directory is NOT vendored into git; it is referenced by pin.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="johannrplaster/Qwen3.8-27B-Uncensored-int4-AutoRound"
REV="${REV:-74f84ec7e7779c8628f72935887cfaa47903116d}"
DEST="${MODEL_DIR:-$HERE/models/Qwen3.8-27B-Uncensored-int4-AutoRound}"

case "${1:-}" in
  fetch)
    export HF_TOKEN="${HF_TOKEN:-}"
    hf download "$REPO" --revision "$REV" --local-dir "$DEST" || \
      HF_HUB_DISABLE_XET=1 hf download "$REPO" --revision "$REV" --local-dir "$DEST"
    ;;
  verify)
    # per-shard sha256 from the remote repo's manifest
    curl -fsSL "https://huggingface.co/$REPO/resolve/main/model.safetensors.index.json" -o /tmp/hb-index.json
    python3 - "$DEST" /tmp/hb-index.json <<'PY'
import json, os, hashlib, sys
dest, idxp = sys.argv[1], sys.argv[2]
idx = json.load(open(idxp))
files = sorted(set(idx["weight_map"].values()))
for f in files:
    p = os.path.join(dest, f)
    assert os.path.exists(p), f"missing {f}"
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"{f}: {h[:16]}...")
print(f"verified {len(files)} shards with matching tensor index")
PY
    ;;
  *)
    echo "usage: $0 fetch|verify"; exit 2;;
esac
