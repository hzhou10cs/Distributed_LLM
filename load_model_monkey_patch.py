import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# — 1) Load LLaMA 3.2-3B —
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float16 if device.type=="cuda" else torch.float32,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", use_fast=True)
model.eval()

# — 2) Prepare a single batch of input_ids —
prompt = "The quick brown fox jumps over"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs.input_ids  # shape (1, seq_len)

# — 3) Prepare timing array —
num_layers  = model.config.num_hidden_layers  # should be 28
layer_times = [0.0] * num_layers

# — 4) Monkey-patch each decoder layer's forward, using **unbound** forward —
for idx, layer in enumerate(model.model.layers):
    # Grab the **unbound** function from the class, not layer.forward
    unbound_forward = layer.__class__.forward

    def make_timed_forward(i, orig_fn):
        def timed_forward(self, hidden_states, *args, **kwargs):
            # sync + start
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()

            # call the original unbound forward
            out = orig_fn(self, hidden_states, *args, **kwargs)

            # end + sync
            end.record()
            torch.cuda.synchronize()
            layer_times[i] = start.elapsed_time(end)
            return out

        return timed_forward

    # bind our wrapper onto the instance
    patched = make_timed_forward(idx, unbound_forward)
    layer.forward = patched.__get__(layer, layer.__class__)

# — 5) Run one full forward to trigger timings —
with torch.no_grad():
    _ = model(input_ids=input_ids, use_cache=True)

# — 6) Print per-layer times —
print("Per-layer prefill times (ms):")
for i, t in enumerate(layer_times):
    print(f"  layer {i:02d}: {t:.1f} ms")
print(f"\nTotal (sum): {sum(layer_times):.1f} ms")
