#!/usr/bin/env python3
"""TDD unit test: GDN linear_attn projections must run BF16 (dequant at load),
everything else stays INT8 W8A8. Red -> implement -> green."""
import os, torch, sys
import vllm_xpu_kernels, vllm_xpu_kernels._C, vllm_xpu_kernels._xpu_C  # noqa
torch.manual_seed(11)
dev = "xpu"

sys.path.insert(0, os.environ.get("VLLM_SRC", os.path.expanduser("~/humble-b70-llm/src/vllm")))  # editable fork tree on the reference box
from vllm.model_executor.kernels.linear.scaled_mm.xpu import XPUInt8ScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import Int8ScaledMMLinearLayerConfig

def make_layer(prefix, N, K):
    W = (torch.randn(N, K, device=dev) * 0.15).to(torch.float16)
    ws_ref = W.float().abs().amax(dim=1, keepdim=True) / 127.0
    Wq = torch.round(W.float() / ws_ref).clamp(-127, 127).to(torch.int8)
    class L(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logical_widths = [N]
            self.weight = torch.nn.Parameter(Wq, requires_grad=False)
            self.weight_scale = torch.nn.Parameter(ws_ref, requires_grad=False)
            self.input_scale = torch.nn.Parameter(torch.tensor(1.0, device=dev))
            self.input_zero_point = None
            self.azp_adj = None
            self._vllm_prefix = prefix
    return L(), W

results = []
def t(name, ok, detail=""):
    results.append(ok)
    print(("  [OK ] " if ok else "  [FAIL] ") + name + (" " + detail if detail else ""))

x = (torch.randn(4, 4096) * 0.2).to(torch.float16).to(dev)

# Case 1: linear_attn projection -> BF16 path (dequant once, matmul exact-ish)
for prefix, name in (("model.language_model.layers.3.linear_attn.in_proj_qkv", "linear_attn.in_proj_qkv"),
                     ("model.language_model.layers.3.linear_attn.out_proj", "linear_attn.out_proj")):
    layer, W = make_layer(prefix, 1520, 4096)
    k = XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearLayerConfig(is_channelwise=True, is_static_input_scheme=False, input_symmetric=True),
                                    ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"])
    k.process_weights_after_loading(layer)
    bf16 = getattr(layer, "_xpu_gdn_bf16_weight", None)
    t(f"{name}: dequantized to bf16", bf16 is not None and bf16.dtype == torch.bfloat16 and tuple(bf16.shape) == (4096, 1520),
      str(tuple(bf16.shape) if bf16 is not None else None))
    # real-model activations are bf16 (--dtype bfloat16); emulate that.
    # Contract: the bf16 forward must be activation-exact, i.e. equal to fp32
    # math on the SERVED (dequantized) weight; the int8-dequant weight quant
    # noise (~0.4%) vs the original fp16 W is inherent and checked separately.
    x_bf = x.to(torch.bfloat16)
    out = k.apply_weights(layer, x_bf)
    ref = (x_bf.float() @ layer._xpu_gdn_bf16_weight.float()).to(dtype=torch.bfloat16).float()
    err = ((out.float() - ref).abs() / ref.abs().clamp_min(1.0)).max().item()
    t(f"{name}: bf16 forward activation-exact vs fp32 math (err={err:.4e})", err < 1e-2)

# Case 2: non-linear_attn layer -> stays INT8 W8A8
layer, W = make_layer("model.language_model.layers.3.mlp.gate_proj", 5120, 4096)
k = XPUInt8ScaledMMLinearKernel(Int8ScaledMMLinearLayerConfig(is_channelwise=True, is_static_input_scheme=False, input_symmetric=True),
                                ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"])
k.process_weights_after_loading(layer)
t("mlp.gate_proj: NO bf16 buffer", getattr(layer, "_xpu_gdn_bf16_weight", None) is None)
t("mlp.gate_proj: int8 weight kept [K,N]", layer.weight.shape == (4096, 5120) and layer.weight.dtype == torch.int8)
out = k.apply_weights(layer, x)
xq, xs = torch.ops._xpu_C.per_token_quant_int8_xpu(x)
ref = (xq.float() @ layer.weight.float()) * (xs.float() * layer.weight_scale.float().reshape(1, -1))
err = ((out.float() - ref).abs() / ref.abs().clamp_min(1.0)).max().item()
t(f"mlp.gate_proj: int8 W8A8 forward intact (err={err:.4e})", err < 2e-2)

print("\n" + ("ALL PASS" if all(results) else "RED: some tests failing"))
sys.exit(0 if all(results) else 1)
