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

# 3. venv + torch (Intel XPU wheels; an extra index is required)
uv venv --python 3.12 .venv
"$HERE/.venv/bin/pip" install --extra-index-url \
  https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ \
  "torch==2.13.0+xpu"

# 4. kernels wheel (source build; see docs/drivers.md for toolchain)
#    - oneAPI compiler on PATH (source setvars.sh or install via apt)
#    - the reduced attention presets avoid compiling unused template variants
#      (and their 7-12 GB compiler peaks)
#    - MAX_JOBS defaults to 6; lower it on lower-RAM hosts (fat units can OOM)
export MAX_JOBS="${MAX_JOBS:-6}"
export VLLM_CHUNK_PREFILL_CONFIG=chunk_prefill_default.conf
export VLLM_PAGED_DECODE_CONFIG=paged_decode_default.conf
"$HERE/.venv/bin/pip" install numpy "cmake==3.31.8" ninja \
  "setuptools>=77,<80" setuptools-scm wheel build
(cd src/vllm-xpu-kernels && \
  "$HERE/.venv/bin/python" setup.py bdist_wheel \
    --dist-dir "$HERE/dist" --py-limited-api=cp38)

# 5. install vLLM (editable) + kernels wheel
"$HERE/.venv/bin/pip" install --no-build-isolation -e src/vllm
"$HERE/.venv/bin/pip" install --force-reinstall --no-deps dist/vllm_xpu_kernels-*.whl

echo "== done. venv at .venv =="
echo "Next: bash scripts/model.sh fetch ; bash scripts/serve.sh"
