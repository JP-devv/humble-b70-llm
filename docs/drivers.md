# Drivers and runtime versions

The kernel driver cannot ship inside a container or repository; it must be
installed on the host. Everything else runs userland and is pinned.

## What the reference machine runs

Capture your own versions with `scripts/verify-install.sh`, which prints a
manifest:

- Intel graphics kernel driver: `xe` module (`modinfo xe`). Distribution
  kernels and Intel-compute-runtime packages both supply it; the reference
  ran a custom `xe-next` kernel.
- Level Zero runtime: `libze_intel_gpu.so.1.15.x` (package
  `libze-intel-gpu1 ... 26.x`). The reference had no newer driver available
  via its distro mirror; a rebuilt driver line exists on some distros — pin
  the package, not the path.
- Intel oneAPI: compiler/runtime 2025.3 or 2026.x both build this stack.
- PyTorch: `2.13.0+xpu` (this is the only torch version tested end to end).
- triton-xpu, oneCCL, level-zero-loader: whatever `pip`/`uv` resolves at
  build time from the pinned `requirements.txt`-style files in
  `scripts/build.sh`; record the resolved versions into
  `env-manifest.json` on install.
- Intel graphics compute runtime (IGC/opencl for compile): distro package,
  version recorded by the manifest.

## Installation order (see scripts/setup-host.sh)

1. Install graphics compute runtime + Level Zero packages (apt on Debian
   derivatives; the script pins known-good versions and verifies by loading
   the library and printing its version).
2. Reboot if the kernel module changed.
3. Verify: `xpu-smi --list-gpus` shows the B70 card(s), and
   `python -c "import torch; print(torch.xpu.get_device_name(0))"` inside
   the built venv prints `Intel Arc Pro B70`.

## Version hygiene

- The oneCCL **zooms more than the rest of the stack**: a rebuilt oneCCL
  with unvalidated hashes will change TP2 behavior. `build.sh` pins the
  oneCCL commit and verifies it with `python -c` smoke, not hash equality.
- `verify-install.sh` prints a content manifest of the rusted-on shared
  objects it depends on. If a dependency line changes between releases of
  this repo, it's a breaking change and belongs in CHANGELOG.
