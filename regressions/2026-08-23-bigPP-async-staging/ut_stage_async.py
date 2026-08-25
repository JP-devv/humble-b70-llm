#!/usr/bin/env python3
"""bigPP L1 regression: sliced pipelined staging + shm 2-rank reduce.

For each config, spawns 2 processes (one per B70), builds a real gloo
group, and drives the REAL XpuCommunicator staged all_reduce selected via
env (blocking / sliced / sliced+shm). Asserts BIT-EXACT equality against
the CPU fp16 sum of both ranks' inputs, then times a GEMM + all_reduce
loop (the prefill access pattern). All work runs under
torch.inference_mode() because production forwards do (the pinned staging
buffers are inference tensors — regression: RuntimeError on inplace
update in the shm worker).

Run on the reference box:
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  $HOME/.venvs/vllm-xpu/bin/python ut_stage_async.py
"""

import os
import socket
import sys
import time
import traceback

import multiprocessing as mp

CORRECT_SHAPES = [(240, 5120), (633, 5120), (3, 7), (1, 1), (2048, 5120)]
PERF_SHAPE = (8192, 5120)
PERF_ITERS = 12

CONFIGS = [
    ("blocking", {"VLLM_XPU_STAGE_SLICE_KB": "0", "VLLM_XPU_STAGE_SHM": "0"}),
    ("sliced-512k", {"VLLM_XPU_STAGE_SLICE_KB": "512", "VLLM_XPU_STAGE_SHM": "0"}),
    ("sliced-2m", {"VLLM_XPU_STAGE_SLICE_KB": "2048", "VLLM_XPU_STAGE_SHM": "0"}),
    ("sliced-8m", {"VLLM_XPU_STAGE_SLICE_KB": "8192", "VLLM_XPU_STAGE_SHM": "0"}),
    ("sliced-2m-shm", {"VLLM_XPU_STAGE_SLICE_KB": "2048", "VLLM_XPU_STAGE_SHM": "1"}),
    ("sliced-8m-shm", {"VLLM_XPU_STAGE_SLICE_KB": "8192", "VLLM_XPU_STAGE_SHM": "1"}),
]


def _gen(seed, shape, dtype):
    import torch

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float32) * 100.0
    return x.to(dtype)


def _run_checks(comm, rank, q):
    import numpy as np
    import torch

    # --- bit-exactness -----------------------------------------------------
    n_checked = 0
    for shape in CORRECT_SHAPES:
        for dtype in (torch.float16, torch.float32):
            for it in range(3):
                # deterministic across processes (hash() is salted)
                seed = (
                    shape[0] * 1000003
                    + shape[1] * 101
                    + it * 7
                    + (16 if dtype == torch.float16 else 32)
                ) % (2**31)
                x = _gen(seed + rank, shape, dtype).to(f"xpu:{rank}")
                got = comm.all_reduce(x).cpu().numpy()
                x0 = _gen(seed + 0, shape, dtype).numpy()
                x1 = _gen(seed + 1, shape, dtype).numpy()
                expected = np.add(x0, x1)
                if not np.array_equal(got, expected):
                    bad = np.nonzero(got != expected)
                    q.put(
                        (
                            rank,
                            f"FAIL bitexact {shape} {dtype} it{it}: "
                            f"{len(bad[0])} mismatches, first idx "
                            f"{bad[0][0] if len(bad[0]) else '-'}: "
                            f"got {got[bad][0] if len(bad[0]) else '-'} "
                            f"want {expected[bad][0] if len(bad[0]) else '-'}",
                        )
                    )
                    return
                n_checked += 1

    # --- throughput: GEMM + all_reduce x N ----------------------------------
    x = _gen(1234 + rank, PERF_SHAPE, torch.float16).to(f"xpu:{rank}")
    w = _gen(99, (PERF_SHAPE[1], PERF_SHAPE[1]), torch.float16).to(f"xpu:{rank}")
    for _ in range(3):  # warmup
        y = x @ w
        comm.all_reduce(y)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(PERF_ITERS):
        y = x @ w
        comm.all_reduce(y)
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / PERF_ITERS
    q.put((rank, f"OK checked={n_checked} perf_ms_per_iter={dt * 1e3:.2f}"))


def _child(rank, port, env, q):
    try:
        os.environ["VLLM_XPU_HOST_STAGED_COLLECTIVES"] = "1"
        os.environ["VLLM_XPU_HOST_STAGED_MIN_BYTES"] = "1"
        os.environ["VLLM_XPU_STAGE_TIMING"] = "1"
        os.environ.update(env)

        import torch
        import torch.distributed as dist

        torch.xpu.set_device(rank)
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=2,
        )

        # Env is read at module import time; import AFTER os.environ.update.
        from vllm.distributed.device_communicators import (
            xpu_communicator as xc,
        )

        comm = xc.XpuCommunicator.__new__(xc.XpuCommunicator)
        comm.cpu_group = dist.group.WORLD
        comm.world_size = 2
        comm.rank_in_group = rank
        comm.device = torch.device(f"xpu:{rank}")
        comm._staging_cache = {}
        comm._stage_wq = None
        comm._stage_dq = None
        comm._shm = None

        # Production forwards run under torch.inference_mode(); the pinned
        # staging buffers are then inference tensors. Exercise that here.
        with torch.inference_mode():
            _run_checks(comm, rank, q)
    except Exception:
        q.put((rank, "EXC " + traceback.format_exc()))


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ctx = mp.get_context("spawn")
    results = {}
    ok = True
    for name, env in CONFIGS:
        q = ctx.Queue()
        port = _free_port()
        procs = [
            ctx.Process(target=_child, args=(r, port, env, q)) for r in range(2)
        ]
        for p in procs:
            p.start()
        got = [q.get() for _ in range(2)]
        for p in procs:
            p.join(120)
        msgs = sorted(got)
        results[name] = msgs
        status = "PASS" if all(m[1].startswith("OK") for m in msgs) else "FAIL"
        if status == "FAIL":
            ok = False
        perf = [
            m[1].split("perf_ms_per_iter=")[1] for m in msgs if m[1].startswith("OK")
        ]
        print(f"[{status}] {name}: {perf if perf else [m[1] for m in msgs]}", flush=True)

    def perf_of(name):
        vals = [
            float(m[1].split("perf_ms_per_iter=")[1])
            for m in results[name]
            if m[1].startswith("OK")
        ]
        return max(vals) if vals else float("inf")

    base = perf_of("blocking")
    print("\n== throughput (ms per GEMM+all_reduce iter, 84 MB payload) ==")
    for name, _ in CONFIGS:
        v = perf_of(name)
        print(f"  {name:<16} {v:8.2f} ms  ({base / v:4.2f}x vs blocking)")
    sl = perf_of("sliced-8m")
    shm = perf_of("sliced-8m-shm")
    if not sl < base / 1.2:
        print("PERF-ASSERT-FAIL: sliced-8m not >=1.2x blocking")
        ok = False
    if not shm < sl:
        print("PERF-ASSERT-FAIL: sliced-8m-shm not faster than sliced-8m (gloo)")
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
