# Distributed_LLM

This repo now has two parts:

- `edgecolab/`: current implementation (model-family runner + adapter architecture)
- `previous_implementations/`: archived earlier scripts and source patches

## Install From GitHub Repo

Install directly from this repository (no PyPI required):

```bash
pip install "git+https://github.com/hzhou10cs/Distributed_LLM.git@main"
```

For local development from a cloned repo:

```bash
pip install -e .
```

To lock to an exact revision:

```bash
pip install "git+https://github.com/hzhou10cs/Distributed_LLM.git@<tag-or-commit>"
```

## Run From Source (No Package Install)

Install dependencies only:

```bash
pip install -r requirements.txt
```

Then run with module commands from repo root:

```bash
python -m edgecolab --help
python -m edgecolab.local_ring --help
python -m edgecolab.example_local_stage --help
```

## Current Entry Points

- Generic CLI:

```bash
edgecolab \
  --model_name meta-llama/Llama-3.2-1B \
  --model_family auto \
  --layer_begin -1 \
  --layer_end 6 \
  --recv_host 127.0.0.1 --recv_port 5600 \
  --send_host 127.0.0.1 --send_port 5601
```

- Local 3-process ring:

```bash
edgecolab-local-ring \
  --model_name meta-llama/Llama-3.2-1B \
  --model_family auto \
  --layer_end1 6 \
  --layer_end2 11 \
  --host 127.0.0.1 \
  --base_port 5600 \
  --max_new_tokens 100
```

- Single-process 1-device example:

```bash
edgecolab-example \
  --model_name meta-llama/Llama-3.2-1B \
  --role full \
  --prompt "Once upon a time, a wizard lived in a tower." \
  --max_new_tokens 100
```

## Notes

- `meta-llama/Llama-3.2-1B` is gated; first download still requires access/auth.
- After full cache exists locally, local-only loading paths are used automatically.
- Pin installation to a Git tag or commit hash to keep behavior tightly bound to GitHub source.
