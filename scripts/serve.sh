#!/usr/bin/env bash
# Container entrypoint: builds the `vllm serve` command line from environment variables
# (set in .env, passed through compose.yaml). Prints the final command before exec.
set -euo pipefail

args=(
  "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port 8000
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  # GB10 (sm_121): NVFP4 weights run through Marlin W4A16 kernels
  --moe-backend marlin
  --mamba-backend flashinfer
  --mamba-cache-mode align
  --enable-prefix-caching
  # thinking text -> separate response field, final answer -> content
  --reasoning-parser nemotron_v3
  --tool-call-parser qwen3_coder
  --enable-auto-tool-choice
)

case "${SPEC_METHOD:-none}" in
  dspark) args+=(--speculative-config "{\"method\":\"dspark\",\"model\":\"$DSPARK_MODEL\",\"num_speculative_tokens\":$SPEC_TOKENS}") ;;
  dflash) args+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH_MODEL\",\"num_speculative_tokens\":$SPEC_TOKENS}") ;;
  # the MTP heads contain an MoE block that Marlin does not support -> Triton kernels for the drafter
  mtp)    args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS,\"moe_backend\":\"triton\"}") ;;
  none)   ;;
  *) echo "unknown SPEC_METHOD=$SPEC_METHOD (dspark|dflash|mtp|none)" >&2; exit 2 ;;
esac

# API key is read by vLLM from VLLM_API_KEY; empty means no authentication
[[ -z "${VLLM_API_KEY:-}" ]] && unset VLLM_API_KEY

# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && args+=($EXTRA_ARGS)

echo "+ vllm serve ${args[*]}"
exec vllm serve "${args[@]}"
