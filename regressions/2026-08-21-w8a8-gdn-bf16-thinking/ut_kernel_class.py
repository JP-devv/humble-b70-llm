#!/usr/bin/env python3
"""Unit test for XPUInt8ScaledMMLinearKernel (the new compressed-tensors w8a8 path).
Builds a fake layer the way CompressedTensorsW8A8Int8.create_weights does,
runs process_weights_after_loading + apply_weights, and compares against the
fp32 integer reference in both per-channel and fused per-tensor-scaled forms.
"""
import torch, sys, importlib
import vllm_xpu_kernels, vllm_xpu_kernels._C, vllm_xpu_kernels._xpu_C  # noqa

torch.manual_seed(7)
dev = "xpu"

from vllm.model_executor.kernels.linear import init_int8_linear_kernel
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    convert_to_channelwise,
)

# ---- 0. dispatcher picks the XPU kernel --------------------------------
results = []
def t(name, ok):
    results.append((name, ok))
    print(("  [OK ] " if ok else "  [FAIL] ") + name)

for cfg_args in ((True, False, True), (False, False, True)):
    k = init_int8_linear_kernel(*cfg_args, module_name="ut")
    t(f"dispatch {cfg_args}", k.__class__.__name__ == "XPUInt8ScaledMMLinearKernel")
    k_cls = k.__class__

    # ---- 1. fake layer (per-channel) -----------------------------------
    N, K = 1520, 4096
    W = (torch.randn(N, K, device=dev) * 0.15).to(torch.float16)
    w_s_ref = W.float().abs().amax(dim=1, keepdim=True) / 127.0
    W_q = (torch.round(W.float() / w_s_ref).clamp(-127, 127)).to(torch.int8)

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logical_widths = [N]
            self.weight = torch.nn.Parameter(W_q, requires_grad=False)          # [N,K] int8
            self.weight_scale = torch.nn.Parameter(w_s_ref, requires_grad=False)  # [N,1] fp32
            self.input_scale = torch.nn.Parameter(torch.tensor(1.0, device=dev))
            self.input_zero_point = None
            self.azp_adj = None

    layer = FakeLayer()
    kernel = k_cls(k.config, ["weight", "weight_scale", "input_scale",
                              "input_zero_point", "azp_adj"])
    kernel.process_weights_after_loading(layer)
    t(f"weight repacked [K,N] contiguous: {tuple(layer.weight.shape)} {str(layer.weight.dtype)}",
      layer.weight.shape == (K, N) and layer.weight.is_contiguous() and layer.weight.dtype == torch.int8)
    t(f"scale flattened: {tuple(layer.weight_scale.shape)} {layer.weight_scale.dtype}",
      layer.weight_scale.shape == (N,) and layer.weight_scale.dtype == torch.float32)

    x = (torch.randn(4, K, device=dev) * 0.2).to(torch.float16)
    bias = (torch.randn(N, device=dev) * 0.05).to(torch.float16)
    out = kernel.apply_weights(layer, x, bias)
    x_q, x_s = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
    ref = (x_q.float() @ layer.weight.float()) * (x_s.float() * layer.weight_scale.float().reshape(1, -1))
    ref = ref + bias.float()
    err = ((out.float() - ref).abs() / ref.abs().clamp_min(1.0)).max().item()
    t(f"apply (per-channel, bias): rel err={err:.4e}", err < 2e-2)

    # ---- 2. fused module (logical_widths>1) + per-tensor scale ----------
    if not cfg_args[0]:  # per-tensor weight scale with fused widths
        N1, N2, N3 = 800, 600, 700
        Wf = (torch.randn(N1 + N2 + N3, K, device=dev) * 0.15).to(torch.float16)
        ws_t = Wf.float().abs().amax() / 127.0
        Wq = (torch.round(Wf.float() / ws_t).clamp(-127, 127)).to(torch.int8)

        class FusedLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logical_widths = [N1, N2, N3]
                self.weight = torch.nn.Parameter(Wq, requires_grad=False)   # [N1+N2+N3, K]
                self.weight_scale = torch.nn.Parameter(ws_t, requires_grad=False)  # blobs per width? per-tensor: 1 scalar per width
                self.input_scale = torch.nn.Parameter(torch.tensor(1.0, device=dev))
                self.input_zero_point = None
                self.azp_adj = None

        # compressed-tensors fused per-tensor stores 1 scale per logical width -> [W] list
        fused = FusedLayer()
        # emulate convert_to_channelwise input: needs weight_scale as [n_widths] tensor
        fused.weight_scale = torch.nn.Parameter(ws_t.expand(3).clone(), requires_grad=False)
        k2 = init_int8_linear_kernel(False, False, True, module_name="ut-fused")
        k2.process_weights_after_loading(fused)
        ok_conv = fused.weight_scale.shape == (N1 + N2 + N3,) or fused.weight_scale.numel() in (N1+N2+N3, 1)
        out2 = k2.apply_weights(fused, x)
        ref2 = (x_q.float() @ fused.weight.float()) * (x_s.float() * fused.weight_scale.float().reshape(1, -1))
        err2 = ((out2.float() - ref2).abs() / ref2.abs().clamp_min(1.0)).max().item()
        t(f"apply (fused per-tensor->channel): shape={tuple(fused.weight_scale.shape)} rel err={err2:.4e}", err2 < 2e-2)

print("\n" + ("ALL PASS" if all(ok for _, ok in results) else f"FAILED: {[n for n, ok in results if not ok]}"))
sys.exit(0 if all(ok for _, ok in results) else 1)
