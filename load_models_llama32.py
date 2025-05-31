import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1) Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16 if device.type=="cuda" else torch.float32,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", use_fast=True
)

model.eval()

# 2) Prepare a test prompt
prompt = "I don't know what to do today, can you suggest something fun?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.to(device)

# Timing helpers
def time_cuda(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    # reset peak stats to measure this block in isolation
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    start_res = torch.cuda.memory_reserved()

    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated()
    peak_res = torch.cuda.max_memory_reserved()
    end_mem  = torch.cuda.memory_allocated()
    end_res  = torch.cuda.memory_reserved()

    ms = start.elapsed_time(end)
    return {
        "time_ms": ms,
        "mem_start_GB": start_mem/2**30,
        "mem_peak_GB": peak_mem/2**30,
        "mem_end_GB": end_mem/2**30,
        "res_start_GB": start_res/2**30,
        "res_peak_GB": peak_res/2**30,
        "res_end_GB": end_res/2**30,
    }

def time_cpu(fn):
    t0 = time.time()
    fn()
    return {"time_ms": (time.time()-t0)*1000}

timer = time_cuda if device.type=="cuda" else time_cpu

# 3) Embedding only
res = timer(lambda: model.get_input_embeddings()(input_ids))
print(f"\n[1] Embedding:")
print(f"    time: {res['time_ms']:.1f} ms")
if device.type=="cuda":
    print(f"    GPU alloc: start {res['mem_start_GB']:.2f} GB → peak {res['mem_peak_GB']:.2f} GB → end {res['mem_end_GB']:.2f} GB")
    print(f"    GPU reserved: start {res['res_start_GB']:.2f} GB → peak {res['res_peak_GB']:.2f} GB → end {res['res_end_GB']:.2f} GB")

# 4) Full prefill (input_ids → logits + past_key_values)
def do_prefill():
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=True)

res = timer(do_prefill)
print(f"\n[2] Full prefill:")
print(f"    time: {res['time_ms']:.1f} ms")
if device.type=="cuda":
    print(f"    GPU alloc: start {res['mem_start_GB']:.2f} GB → peak {res['mem_peak_GB']:.2f} GB → end {res['mem_end_GB']:.2f} GB")
    print(f"    GPU reserved: start {res['res_start_GB']:.2f} GB → peak {res['res_peak_GB']:.2f} GB → end {res['res_end_GB']:.2f} GB")

# 5) Autoregressive decode timing + throughput + collect IDs
with torch.no_grad():
    out      = model(input_ids=input_ids, use_cache=True)
    past_kv  = out.past_key_values
    last_tok = input_ids[:, -1:].clone()

n_steps = 20
times   = []
gen_ids = input_ids.clone()  # will accumulate decoded tokens

# reset peak so we measure only this loop
if device.type=="cuda":
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

print(f"\n[3] Autoregressive decoding ({n_steps} steps):")
for i in range(n_steps):
    if device.type=="cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
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
    if device.type=="cuda":
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
    else:
        ms = (time.time() - t0)*1000

    times.append(ms)
    print(f"    token {i+1:2d} → {next_tok.item():4d}   time: {ms:.1f} ms")

# throughput & memory
total_ms   = sum(times)
throughput = n_steps*1000.0/total_ms
print(f"  → total decode time: {total_ms:.1f} ms")
print(f"  → throughput:       {throughput:.1f} tokens/s")
if device.type=="cuda":
    peak_mem = torch.cuda.max_memory_allocated()
    print(f"  → GPU alloc: peak {peak_mem/2**30:.2f} GB   (base before loop: {base_mem/2**30:.2f} GB)")

# 6) Print final decoded string
decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
print(f"\n[4] Decoded result:\n{decoded}\n")
