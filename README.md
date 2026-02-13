# Distributed_LLM

Distributed inference experiments for `meta-llama/Llama-3.2-1B`, with:
- single-device baseline timing
- per-component/per-layer profiling
- TCP pipeline parallel decoding across multiple devices

## What This Repo Does

This project benchmarks Llama inference latency and throughput in two modes:

1. Local inference on one machine (`local_llama32.py`)
2. Distributed pipeline inference over TCP sockets (`distributed_llama32.py`, `distributed_llama32_opt.py`)

The distributed path slices the decoder across devices and forwards hidden states/token IDs stage-by-stage.

## Repository Layout

- `distributed_llama32.py`: Original distributed pipeline implementation (pickle-based message transport).
- `distributed_llama32_opt.py`: Optimized transport path with custom tensor packets (no pickle).
- `local_llama32.py`: Local baseline prefill + autoregressive decode timing.
- `llama32_precise_measure.py`: Detailed CUDA timing and memory profiling (embedding, each decoder layer, norm, lm head).
- `modeling_llama.py`: Customized Llama modeling code with slice-aware forward paths.
- `configuration_llama.py`: Customized Llama config fields (`layer_begin`, `layer_end`) used for slicing.
- `test_results/`: Example run logs.

## Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- `transformers` compatible with this repo's custom Llama files
- `accelerate`, `safetensors`, `sentencepiece`

Install:

```bash
pip install torch transformers accelerate safetensors sentencepiece
```

## Important: Custom Transformers Llama Files

The distributed scripts call `AutoModelForCausalLM.from_pretrained(..., layer_begin=..., layer_end=...)` and use `input_hidden_states` in forward passes.  
These are not part of stock Llama behavior in many `transformers` versions.

This repo includes customized:
- `modeling_llama.py`
- `configuration_llama.py`

You should ensure your runtime is actually using these custom files (for example by patching your local `transformers/models/llama/` files in your environment, or by otherwise loading these classes explicitly).

## Model Access

Default model in scripts: `meta-llama/Llama-3.2-1B`.

You need Hugging Face access for gated Meta models:
- accept the model license on Hugging Face
- authenticate locally (`huggingface-cli login`)

## Quick Start

### 1) Local Baseline

```bash
python local_llama32.py
```

Outputs:
- prefill latency
- per-token decode latency
- throughput (tokens/s)
- generated text

Log file:
- `device_local.log`

### 2) Detailed GPU Breakdown

```bash
python llama32_precise_measure.py
```

This script measures:
- embedding time
- each decoder layer time
- final norm time
- lm head time
- CUDA memory stats

Note: this script uses CUDA timing APIs and is intended for GPU execution.

### 3) Distributed Pipeline (3 Stages)

Use `distributed_llama32_opt.py` for lower serialization overhead.

Pipeline roles:
- head: embedding + first decoder slice
- middle: middle decoder slice
- tail: final decoder slice + logits/token sampling

Common split used for 28-layer Llama-3.2-1B:
- head: `--layer_begin -1 --layer_end 16`
- middle: `--layer_begin 16 --layer_end 22`
- tail: `--layer_begin 22 --layer_end -1`

#### Example (single machine, 3 terminals, localhost ring)

Terminal A (head, receives from tail on `6001`, sends to middle on `6002`):

```bash
python distributed_llama32_opt.py ^
  --model_name meta-llama/Llama-3.2-1B ^
  --layer_begin -1 --layer_end 16 ^
  --recv_host 127.0.0.1 --recv_port 6001 ^
  --send_host 127.0.0.2 --send_port 6002 ^
  --prompt "Once upon a time, a wizard lived in a tower." ^
  --max_new_tokens 100
```

Terminal B (middle, receives from head `6002`, sends to tail `6003`):

```bash
python distributed_llama32_opt.py ^
  --model_name meta-llama/Llama-3.2-1B ^
  --layer_begin 16 --layer_end 22 ^
  --recv_host 127.0.0.2 --recv_port 6002 ^
  --send_host 127.0.0.3 --send_port 6003 ^
  --max_new_tokens 100
```

Terminal C (tail, receives from middle `6003`, sends token to head `6001`):

```bash
python distributed_llama32_opt.py ^
  --model_name meta-llama/Llama-3.2-1B ^
  --layer_begin 22 --layer_end -1 ^
  --recv_host 127.0.0.3 --recv_port 6003 ^
  --send_host 127.0.0.1 --send_port 6001 ^
  --max_new_tokens 100
```

This makes a loop for the auto-regressive decoding.

Each stage writes:
- `device_<layer_begin>_<layer_end>.log`

## `distributed_llama32.py` vs `distributed_llama32_opt.py`

- `distributed_llama32.py`: pickle message payloads (`send_msg`/`recv_msg`)
- `distributed_llama32_opt.py`: custom binary tensor packets (`send_packet`/`recv_packet`)

The optimized version reduces Python object serialization overhead and is the recommended default for transport performance experiments.
