#!/usr/bin/env bash
set -euo pipefail

# Local 3-instance ring simulation for distributed_llama32_opt_local.py
# Layer split:
#   device1: [-1, LAYER_END1]
#   device2: [LAYER_END1, LAYER_END2]
#   device3: [LAYER_END2, -1]

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
LAYER_END1="${LAYER_END1:-16}"
LAYER_END2="${LAYER_END2:-22}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-6001}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
PROMPT="${PROMPT:-Once upon a time, a wizard lived in a tower.}"

python distributed_llama32_opt_local.py \
  --model_name "${MODEL_NAME}" \
  --layer_end1 "${LAYER_END1}" \
  --layer_end2 "${LAYER_END2}" \
  --host "${HOST}" \
  --base_port "${BASE_PORT}" \
  --prompt "${PROMPT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}"
