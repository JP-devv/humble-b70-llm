#!/usr/bin/env bash
# Build the humble-b70 stack from pinned sources + patches.
# Produces: venv with vLLM (patched) and the kernels wheel (patched).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VLLM_BASE="$(cat "$HERE/patches/vllm/BASE_COMMIT.txt" | head -1)"
KERNELS_BASE="$(cat "$HERE/patches/vllm-xpu-kernels/BASE_COMMIT.txt" | head -1)"

echo "== humble-b70 build =="
echo "vLLM base:    $VLLM_BASE"
echo "kernels base: $KERNELS_BASE"

command -v uv >/dev/null || python3 -m pip install --user uv

# 1. vLLM source at the pinned base + our patches
if [[ ! -d src/vllm ]]; then
  git clone https://github.com/vllm-project/vllm.git src/vllm
  git -C src/vllm checkout "$VLLM_BASE"
fi
if ! git -C src/vllm apply --check "$HERE/patches/vllm/"*.patch 2>/dev/null; then
  echo "vLLM patches already applied (or conflict) — skipping apply."
fi
git -C src/vllm apply "$HERE/patches/vllm/"*.patch 2>/dev/null || true

# 2. kernels source at the pinned base + our patch
if [[ ! -d src/vllm-xpu-kernels ]]; then
  git clone https://github.com/vllm-project/vllm-xpu-kernels.git src/vllm-xpu-kernels
  git -C src/vllm-xpu-kernels checkout "$KERNELS_BASE"
fi
git -C src/vllm-xpu-kernels apply --check "$HERE/patches/vllm-xpu-kernels/"*.patch 2>/dev/null || true
git -C src/vllm-xpu-kernels apply "$HERE/patches/vllm-xpu-kernels/"*.patch 2>/dev/null || true

# 3. venv
uv venv --python 3.12 .venv

# 4. kernels wheel (source build; see docs/drivers.md for toolchain)
export MAX_JOBS="${MAX_JOBS:-6}"
export VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf
export VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf
(cd src/vllm-xpu-kernels && \
  UV="$(command -v uv)" "$HERE/.venv/bin/python" -m pip wheel \
    --no-build-isolation --no-deps -w "$HERE/dist" . 2>/dev/null || \
  "$HERE/.venv/bin/python" setup.py bdist_wheel --dist-dir "$HERE/dist" --py-limited-api=cp38)

# 5. install vLLM (editable) + kernels wheel
"$HERE/.venv/bin/pip" install --no-build-isolation -e src/vllm
"$HERE/.venv/bin/pip" install --force-reinstall --no-deps dist/vllm_xpu_kernels-*.whl

echo "== done. venv at .venv =="
echo "Next: bash scripts/model.sh fetch ; bash scripts/serve.sh"
