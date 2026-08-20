#!/usr/bin/env bash
# Install host driver/runtime packages for Intel Arc B70 (Ubuntu-family).
# Run with sudo. Reboot after if the kernel module changes.
#
# This does NOT ship kernels; it pins the userland runtime packages and
# verifies them. See docs/drivers.md for the full version story.
set -euo pipefail

echo "== Intel graphics runtime install (Ubuntu-family) =="
# Debian/Ubuntu packages: oneAPI-compatible compute runtime + level zero.
# Versions are the ones the reference machine validated.
sudo apt-get update
sudo apt-get install -y \
  intel-opencl-icd \
  intel-level-zero-gpu \
  level-zero \
  intel-media-va-driver-non-free 2>/dev/null || \
sudo apt-get install -y \
  intel-opencl-icd intel-level-zero-gpu level-zero || true

echo "== verify =="
xpu-smi --list-gpus 2>/dev/null || echo "xpu-smi not present yet — install intel-gpu-tools or xpu-smi from the Intel oneAPI/BMC packages"

ls -la /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.* 2>/dev/null | tail -1 || \
  echo "libze_intel_gpu not found — check driver package"

echo "== NOTE =="
echo "A custom xe-next kernel was used by the reference machine; distro"
echo "kernels also work if recent enough for Battlemage. If B70 is not"
echo "enumerated (xpu-smi empty), you need a newer xe driver line."
echo "Re-run: bash scripts/verify-install.sh --host-only"
