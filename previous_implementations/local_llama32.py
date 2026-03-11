import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

'''
LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 3072,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 24,
  "num_hidden_layers": 28,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "float16",
  "transformers_version": "4.52.4",
  "use_cache": true,
  "vocab_size": 128256
}
'''

def logf(msg: str):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log.write(f"[{t}] {msg}\n")
    print(msg)
        
###
# 1) Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float16 if device.type=="cuda" else torch.float32,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B", use_fast=True
)
model.eval()

log = open(f"device_local.log", "w", buffering=1)

# 2) Prepare a test prompt
prompt = "Once upon a time, a wizard lived in a tower who"
# tokenizer_time_start = time.time_ns()
inputs = tokenizer(prompt, return_tensors="pt") # Takes ~0.5ms on CPU RTX 3070
# print(f"Tokenization took {time.time_ns()-tokenizer_time_start:.2f} seconds")
input_ids = inputs.input_ids.to(device)

# 5) Autoregressive decode timing + throughput + collect IDs
t0 = time.time()
with torch.no_grad():
    out      = model(input_ids=input_ids, use_cache=True)
    past_kv  = out.past_key_values
    last_tok = input_ids[:, -1:].clone()
ms = (time.time() - t0)*1000
logf(f" Prefilling  time: {ms:.2f} ms")

n_steps = 50
times   = []
gen_ids = input_ids.clone()  # will accumulate decoded tokens

logf(f"\n[3] Autoregressive decoding ({n_steps} steps):")
for i in range(n_steps):
    t0 = time.time()
    # single‐token forward
    with torch.no_grad():
        out = model(
            input_ids=last_tok,
            past_key_values=past_kv,
            use_cache=True,
        )
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_kv  = out.past_key_values
    last_tok = next_tok
    gen_ids  = torch.cat([gen_ids, next_tok], dim=-1)
    # end timing
    ms = (time.time() - t0)*1000

    times.append(ms)
    logf(f"    token {i+1:2d} → {next_tok.item():4d}   time: {ms:.2f} ms")

# throughput & memory
total_ms   = sum(times)
throughput = n_steps*1000.0/total_ms
logf(f"  → total decode time: {total_ms:.2f} ms")
logf(f"  → throughput:       {throughput:.2f} tokens/s")
# # 6) Print final decoded string
decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
print(f"\n[4] Decoded result:\n{decoded}\n")
