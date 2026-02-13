import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# 1) SETUP
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto",
) # Defined in the transforers/modelling_utils.py/PreTrainedModel class/from_pretrained method/#4508
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    use_fast=True
)
model.eval()
print(model.config)

# ----------------------------
# 2) PREPARE A PROMPT
# ----------------------------
prompt = "Once upon a time in a land far away, a curious wizard"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_ids = inputs.input_ids  # shape: (1, seq_len)

# ----------------------------
# 3) TIMING HELPER
# ----------------------------
def time_block(fn):
    """
    Runs 'fn()' on GPU, measures:
      - elapsed GPU milliseconds
      - memory before / peak / after (allocated & reserved)
    Returns (fn_output, stats_dict).
    """
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    res_before = torch.cuda.memory_reserved()

    start_evt.record()
    out = fn()
    end_evt.record()
    torch.cuda.synchronize()

    peak_mem  = torch.cuda.max_memory_allocated()
    peak_res  = torch.cuda.max_memory_reserved()
    mem_after = torch.cuda.memory_allocated()
    res_after = torch.cuda.memory_reserved()
    elapsed_ms = start_evt.elapsed_time(end_evt)

    stats = {
        "time_ms":      elapsed_ms,
        "mem_before_GB": mem_before / 2**30,
        "peak_mem_GB":  peak_mem  / 2**30,
        "mem_after_GB":  mem_after / 2**30,
        "res_before_GB": res_before / 2**30,
        "peak_res_GB":   peak_res  / 2**30,
        "res_after_GB":  res_after / 2**30,
    }
    return out, stats

# ----------------------------
# 4) MEASURE EMBEDDING ONLY
# ----------------------------
def run_embed():
    return model.get_input_embeddings()(input_ids)

embed_out, embed_stats = time_block(run_embed)
print("\n[1] Embedding:")
print(f"    time: {embed_stats['time_ms']:.1f} ms")
print(f"    GPU alloc:  start {embed_stats['mem_before_GB']:.2f} GB "
      f"→ peak {embed_stats['peak_mem_GB']:.2f} GB → end {embed_stats['mem_after_GB']:.2f} GB")
print(f"    GPU reserved: start {embed_stats['res_before_GB']:.2f} GB "
      f"→ peak {embed_stats['peak_res_GB']:.2f} GB → end {embed_stats['res_after_GB']:.2f} GB")

# ----------------------------
# 5) MONKEY-PATCH EACH DECODER LAYER
# ----------------------------
num_layers = model.config.num_hidden_layers  # expected: 28
layer_times = [0.0] * num_layers

for idx, layer in enumerate(model.model.layers):
    # Grab the _unbound_ forward method from the class
    unbound_forward = layer.__class__.forward

    def make_timed_forward(i, orig_fwd):
        def timed_forward(self, *args, **kwargs):
            # Sync + start timing
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()

            # Call original forward (with rotary, attention mask, etc.)
            out = orig_fwd(self, *args, **kwargs)

            # End timing
            e.record()
            torch.cuda.synchronize()
            layer_times[i] = s.elapsed_time(e)

            return out  # return exactly what the original forward returned
        return timed_forward

    # Bind our wrapper onto the instance
    layer.forward = make_timed_forward(idx, unbound_forward).__get__(layer, layer.__class__)

# ----------------------------
# 6) RUN FULL PREFILL ONCE (WITH hidden_states)
# ----------------------------
# This runs: embedding → 28 layers → final norm → lm_head, and
# collects hidden_states for us to measure norm + head separately.
def run_full_prefill():
    return model(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True
    )

out_prefill, prefill_stats = time_block(run_full_prefill)

print("\n[2] Per-layer times (prefill):")
for i, t in enumerate(layer_times):
    print(f"    Layer {i:02d}: {t:.1f} ms")

print(f"    Total (embedding + all layers + norm + head): {prefill_stats['time_ms']:.1f} ms")

# ----------------------------
# 7) MEASURE FINAL LAYERNORM AND LM HEAD SEPARATELY
# ----------------------------
# out_prefill.hidden_states is a tuple of length (num_layers + 1):
#   - hidden_states[0] is embeddings output
#   - hidden_states[i] is output _before_ layer i’s forward
#   - hidden_states[-1] is output _before_ final norm
hidden_before_norm = out_prefill.hidden_states[-1]

# 7a) Final LayerNorm
def run_norm():
    return model.model.norm(hidden_before_norm)

_, norm_stats = time_block(run_norm)
print("\n[3] Final LayerNorm:")
print(f"    time: {norm_stats['time_ms']:.1f} ms")
print(f"    GPU alloc:  start {norm_stats['mem_before_GB']:.2f} GB "
      f"→ peak {norm_stats['peak_mem_GB']:.2f} GB → end {norm_stats['mem_after_GB']:.2f} GB")
print(f"    GPU reserved: start {norm_stats['res_before_GB']:.2f} GB "
      f"→ peak {norm_stats['peak_res_GB']:.2f} GB → end {norm_stats['res_after_GB']:.2f} GB")

# 7b) LM Head projection
normed_state = model.model.norm(hidden_before_norm)

def run_lm_head():
    return model.lm_head(normed_state)

_, head_stats = time_block(run_lm_head)
print("\n[4] LM Head projection:")
print(f"    time: {head_stats['time_ms']:.1f} ms")
print(f"    GPU alloc:  start {head_stats['mem_before_GB']:.2f} GB "
      f"→ peak {head_stats['peak_mem_GB']:.2f} GB → end {head_stats['mem_after_GB']:.2f} GB")
print(f"    GPU reserved: start {head_stats['res_before_GB']:.2f} GB "
      f"→ peak {head_stats['peak_res_GB']:.2f} GB → end {head_stats['res_after_GB']:.2f} GB")

# ----------------------------
# 8) SUMMARY
# ----------------------------
print("\n=== Summary ===")
print(f"Embedding only:       {embed_stats['time_ms']:.1f} ms")
for i, t in enumerate(layer_times):
    print(f"Layer {i:02d} time:    {t:.1f} ms")
print(f"Final LayerNorm:      {norm_stats['time_ms']:.1f} ms")
print(f"LM Head projection:   {head_stats['time_ms']:.1f} ms")
print("—————")
total_all = (
    embed_stats["time_ms"]
    + sum(layer_times)
    + norm_stats["time_ms"]
    + head_stats["time_ms"]
)
print(f"Total sum (all parts): {total_all:.1f} ms")
