#!/usr/bin/env bash
set -euo pipefail

# 1-device example: run full model generation on one machine/GPU.
# Works with: pip install -r requirements.txt
python -m edgecolab.example_local_stage \
  --model_name meta-llama/Llama-3.2-1B \
  --role full \
  --prompt "Once upon a time, a wizard lived in a tower." \
  --max_new_tokens 100
