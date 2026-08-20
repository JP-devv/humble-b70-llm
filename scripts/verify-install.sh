#!/usr/bin/env bash
# Runtime self-test: ops registered, model canary, smoke generation, env
# manifest. This REPLACES hash-gating AOT binaries: behavior, not bytes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${BASE_URL:-http://127.0.0.1:19622}"
MODEL="${MODEL:-Qwen3.8-27B-UNC-G128-AR}"

if [[ "${1:-}" == "--host-only" ]]; then
  xpu-smi --list-gpus || { echo "xpu-smi missing"; exit 1; }
  echo "host manifest:"
  for f in /sys/class/drm/card*/device/device; do :; done
  xpu-smi --list-gpus | head -8
  exit 0
fi

echo "== 1. kernels ops =="
"$HERE/.venv/bin/python" - <<'PY'
import torch, vllm_xpu_kernels._C, vllm_xpu_kernels._xpu_C
ops = ["int8_gemm_w8a8", "per_token_quant_int8_xpu", "int4_gemm_w4a16"]
missing = [o for o in ops if not hasattr(torch.ops._xpu_C, o)]
print("ops:", {o: hasattr(torch.ops._xpu_C, o) for o in ops})
assert not missing, f"missing ops: {missing}"
print("OK")
PY

echo "== 2. server canary =="
curl -fsS "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Compute sum(i*i for i in range(4)). Answer with just the number.\"}],\"max_tokens\":64,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | "$HERE/.venv/bin/python" -c "import json,sys; c=json.load(sys.stdin)['choices'][0]['message']['content']; print('canary:', c); assert '14' in c, 'canary failed'"

echo "== 3. cache-zero check (server must have prefix caching off) =="
curl -fsS "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":8,\"temperature\":0}" \
  | "$HERE/.venv/bin/python" -c "
import json,sys
d=json.load(sys.stdin)['usage'].get('prompt_tokens_details',{})
c=d.get('cached_tokens',0)
print('cached_tokens:', c)
assert c==0, 'cached tokens nonzero — disable prefix caching for benchmarks'"

echo "== 4. env manifest =="
{
  echo "{\"vllm_patch_base\":\"$(head -1 "$HERE/patches/vllm/BASE_COMMIT.txt")\","
  echo "\"kernels_patch_base\":\"$(head -1 "$HERE/patches/vllm-xpu-kernels/BASE_COMMIT.txt")\","
  "$HERE/.venv/bin/python" -c "
import torch, importlib.metadata as m
print('\"torch\":\"%s\",' % torch.__version__)
print('\"vllm\":\"%s\",' % m.version('vllm'))
print('\"vllm-xpu-kernels\":\"%s\"' % m.version('vllm-xpu-kernels'))"
  echo "}"
} | tee "$HERE/env-manifest.json" >/dev/null

echo "ALL CHECKS PASSED"
