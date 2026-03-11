# edgecolab: Model-Agnostic Distributed Inference Scaffold

This folder contains a standard architecture to avoid patching `transformers` model source files per model family.

## Install

From GitHub (pinned tag/commit recommended):

```bash
pip install "git+https://github.com/hzhou10cs/Distributed_LLM.git@main"
```

From a local clone:

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

Then run from repo root:

```bash
python -m edgecolab --help
python -m edgecolab.local_ring --help
python -m edgecolab.example_local_stage --help
```

## Goals

- Keep upstream Hugging Face model code untouched.
- Run pipeline stages with a common runtime interface.
- Support multiple model families via adapter classes.

## Structure

- `adapters/base.py`: common adapter interface.
- `adapters/llama_family.py`: first concrete adapter for Llama-like models.
- `runtime/pipeline.py`: model-agnostic stage runtime.
- `runners/base_runner.py`: generic base runner (role bounds, logging, memory, layer placement).
- `runners/llama_runner.py`: Llama-family runner subclass.
- `run_device.py`: generic run loop (`run_device`) over transport + runner.
- `edgecolab.py`: CLI entrypoint module.
- `__main__.py`: lets you run `python -m edgecolab`.
- `local_ring.py`: local 3-process ring launcher on one machine.

## Generic CLI

```bash
edgecolab \
  --model_name meta-llama/Llama-3.2-1B \
  --model_family auto \
  --layer_begin -1 \
  --layer_end 6 \
  --recv_host 127.0.0.1 --recv_port 5600 \
  --send_host 127.0.0.1 --send_port 5601
```

## Local 3-stage run (single machine)

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

## 1-device run (single process)

```bash
edgecolab-example \
  --model_name meta-llama/Llama-3.2-1B \
  --role full \
  --prompt "Once upon a time, a wizard lived in a tower." \
  --max_new_tokens 100
```
