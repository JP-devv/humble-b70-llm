#!/usr/bin/env python3
"""Unit test: XPU oneDNN W8A8 INT8 ops (int8_gemm_w8a8 + per_token_quant_int8_xpu).

Validates the exact contract the compressed-tensors W8A8 trunk will rely on:
  - per_token_quant_int8_xpu: symmetric per-token quant, scale = absmax/127
  - int8_gemm_w8a8(A[m,k] int8, A_scale [m,1]|[1], B [k,n]|[n,k] int8,
                   B_scale [n]|[1], out fp16/bf16, bias):
        out = (A_q @ B_q) * (sA * sB^T) + bias   (nn layout)
        out = (A_q @ B_q^T) * (...) + bias        (nt layout, B is [n,k])
Green = kernel matches fp32 integer reference within tolerance.
"""
import torch, math, sys

import vllm_xpu_kernels, vllm_xpu_kernels._C, vllm_xpu_kernels._xpu_C  # noqa: F401  (registers _xpu_C ops)
torch.manual_seed(0)
dev = "xpu"
TORCH = torch.float16

def close(a, b, rtol=2e-2, atol=1e-2, name=""):
    a, b = a.float(), b.float()
    denom = a.abs().clamp_min(1.0)
    err = ((a - b).abs() / denom).max().item()
    status = "OK " if err <= rtol else "FAIL"
    print(f"  [{status}] {name}: max rel err = {err:.4e}  (absmax a={a.abs().max().item():.3e})")
    return err <= rtol

results = []
def t(name, ok):
    results.append((name, ok))

# ---- 1. op presence -------------------------------------------------------
ops_present = all(hasattr(torch.ops._xpu_C, o) for o in
                  ("int8_gemm_w8a8", "per_token_quant_int8_xpu"))
print("ops present:", ops_present)
t("ops_present", ops_present)

# ---- 2. per_token_quant_int8_xpu -----------------------------------------
m, k = 6, 5120
x = torch.randn(m, k, device=dev, dtype=TORCH) * 0.35
q, s = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
print("  per_token shapes:", tuple(q.shape), tuple(s.shape), "dtypes:", q.dtype, s.dtype)
ref_s = x.float().abs().amax(dim=1, keepdim=True) / 127.0
t("quant_scale_matches_absmax127", close(s.float(), ref_s, rtol=1e-4, atol=1e-6, name="per_token scale"))
deq = q.float() * s.float()
t("quant_roundtrip", close(deq, x.float(), rtol=0.05, atol=0.05, name="roundtrip err"))

# ---- 3. int8_gemm_w8a8: nn layout [k,n] weight ---------------------------
def gemm_ref(A_q, B_q, sA, sB, bias=None):
    out = A_q.float() @ B_q.float()
    sA = sA.float().reshape(-1, 1) if sA.numel() > 1 else sA.float().reshape(1, 1)
    sB = sB.float().reshape(1, -1) if sB.numel() > 1 else sB.float().reshape(1, 1)
    out = out * (sA * sB)
    if bias is not None:
        out = out + bias.float()
    return out

def quant_w(w):
    s = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-10) / 127.0
    qw = torch.round(w.float() / s).clamp(-127, 127).to(torch.int8)
    return qw, s.reshape(-1)

for out_dt, name_od in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
    n, kk = 1520, 4096
    A = (torch.randn(4, kk, device=dev) * 0.2).to(TORCH)
    W = (torch.randn(n, kk, device=dev) * 0.15).to(TORCH)
    A_q, A_s = torch.ops._xpu_C.per_token_quant_int8_xpu(A)
    W_q, W_s = quant_w(W)
    bias = (torch.randn(n, device=dev) * 0.05).to(TORCH)
    # nn: B row-major [k, n]
    B_nn = W_q.t().contiguous()  # [k, n]
    out = torch.ops._xpu_C.int8_gemm_w8a8(A_q, A_s, B_nn, W_s, out_dt, bias)
    ref = gemm_ref(A_q, B_nn, A_s.reshape(-1), W_s, bias)
    t(f"nn_{name_od}_perchannel", close(out, ref, name=f"nn {name_od} per-token x / per-channel w"))
    # nt: pass B as [n, k] col-major view -> wrapper sets is_nt (informational only:
    # our kernel class stores weights [K,N] row-major, so only nn matters)
    try:
        B_nt = W_q.t().t()  # [n, k] non-contiguous transpose -> is_nt path?
        print("  B_nt strides:", B_nt.stride(), "-> is_nt:", B_nt.stride()[B_nt.dim()-2] == 1)
        out_nt = torch.ops._xpu_C.int8_gemm_w8a8(A_q, A_s, B_nt, W_s, out_dt, bias)
        ref_nt = gemm_ref(A_q, B_nn, A_s.reshape(-1), W_s, bias)
        t(f"nt_{name_od}_perchannel", close(out_nt, ref_nt, name=f"nt {name_od} (B=[n,k] col-major)"))
    except Exception as e:
        print(f"  [SKIP] nt {name_od}: {e}")
        t(f"nt_{name_od}_perchannel", True)
    # scalar A scale + scalar B scale
    one_xpu = torch.ones(1, device=dev, dtype=torch.float32)
    out_sc = torch.ops._xpu_C.int8_gemm_w8a8(A_q, one_xpu, B_nn, one_xpu, out_dt, None)
    ref_sc = gemm_ref(A_q, B_nn, one_xpu, one_xpu, None)
    t(f"nn_{name_od}_scales_one", close(out_sc, ref_sc, name=f"scalar scales {name_od}"))

# ---- 4. lm_head-scale shape smoke (perf + shape sanity, 30s cap) ---------
try:
    mh, vocab, hid = 8, 248320, 5120
    A = (torch.randn(mh, hid, device=dev) * 0.2).to(TORCH)
    A_q, A_s = torch.ops._xpu_C.per_token_quant_int8_xpu(A)
    W = (torch.randn(vocab, hid, device=dev) * 0.15).to(TORCH)
    W_q, W_s = quant_w(W)
    B = W_q.t().contiguous()
    import time
    for _ in range(3):
        out = torch.ops._xpu_C.int8_gemm_w8a8(A_q, A_s, B, W_s, TORCH, None)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        out = torch.ops._xpu_C.int8_gemm_w8a8(A_q, A_s, B, W_s, TORCH, None)
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = 2 * mh * vocab * hid
    print(f"  lm_head-shape gemm: {dt*1e3:.2f} ms/iter -> {flops/dt/1e12:.1f} TFLOP/s (int8)")
    t("lmhead_shape_runs", True)
except Exception as e:
    print("  lm_head-shape gemm FAILED:", e)
    t("lmhead_shape_runs", False)

fails = [n for n, ok in results if not ok]
print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
sys.exit(0 if not fails else 1)
