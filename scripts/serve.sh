#!/usr/bin/env bash
# Launch the humble-b70 server. Mirrors the reference production launcher.
#
# Flags:
#   --tp N         tensor parallel size (default 2)
#   --mtp N        speculative tokens (default 3)
#   --int8-head on|off   INT8 lm_head (default on)
#   --ctx N        max model len (default 8192)
#   --kv [fp8]     fp8 KV for long-context capacity
#   --port N       (default 19622)
#   --gpu N        single-GPU selection when --tp 1 (default 0)
#   --quant Q      quantization method: gptq (default) | fp8 (W8A8, load-time)
#   --template F   path to a chat template to import via --chat-template
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP=2; MTP=3; INT8=on; CTX=8192; KV=; PORT=19622; GPU=0; QUANT=gptq; TEMPLATE=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tp) TP="$2"; shift 2;; --mtp) MTP="$2"; shift 2;;
    --int8-head) INT8="$2"; shift 2;; --ctx) CTX="$2"; shift 2;;
    --kv) KV="$2"; shift 2;; --port) PORT="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;; --quant) QUANT="$2"; shift 2;;
    --template) TEMPLATE="$2"; shift 2;; *) echo "unknown: $1"; exit 2;;
  esac
done

MODEL="${MODEL:-$HERE/models/Qwen3.8-27B-Uncensored-int4-AutoRound}"
SERVED="${SERVED:-Qwen3.8-27B-UNC-G128-AR}"

envs=(TP="$TP" PORT="$PORT" SERVED_NAME="$SERVED" MAX_MODEL_LEN="$CTX"
      SPEC_TOKENS="$MTP" MODEL_DIR="$MODEL")
[[ "$INT8" == "on" ]] && envs+=(VLLM_XPU_LM_HEAD_INT8=1)
[[ -n "$KV" ]] && envs+=(KV_CACHE_DTYPE="$KV")
if [[ "$TP" == "1" ]]; then
  envs+=(ZE_AFFINITY_MASK="$GPU" GPU_MEMORY_UTILIZATION=0.88)
  envs+=(ONEAPI_DEVICE_SELECTOR="level_zero:0")
else
  envs+=(GPU_MEMORY_UTILIZATION=0.95)
fi

# runtime env (generic; see docs/drivers.md for the host-driver story)
envs+=(VLLM_TARGET_DEVICE=xpu VLLM_XPU_ENABLE_XPU_GRAPH=1)
envs+=(CCL_TOPO_P2P_ACCESS=0 CCL_ZE_IPC_EXCHANGE=pidfd CCL_ATL_TRANSPORT=ofi)
envs+=(ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE)
# host-staged TP2 collectives (see docs/hardware.md — no P2P required)
envs+=(VLLM_XPU_HOST_STAGED_COLLECTIVES=1)
# Distro libstdc++ note: some distro level-zero loaders need a newer
# libstdc++.so.6 than the Python env bundles. If the server crashes at dlopen
# with a GLIBCXX_3.4.x symbol error, export LD_PRELOAD to the system
# libstdc++ (e.g. /usr/lib/x86_64-linux-gnu/libstdc++.so.6) before running.

exec env "${envs[@]}" \
  "$HERE/.venv/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size "$TP" \
  --dtype float16 \
  --max-model-len "$CTX" \
  --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --quantization "$QUANT" \
  ${TEMPLATE:+--chat-template "$TEMPLATE"} \
  ${KV:+--kv-cache-dtype "$KV"} \
  ${KV:+--gpu-memory-utilization 0.88} \
  --enable-prompt-tokens-details \
  --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$MTP}"
