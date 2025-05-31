from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from utils.time_counter import time_cuda, time_cpu
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Gemma3ForConditionalGeneration .from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype="auto", 
    device_map="cpu",
    attn_implementation='sdpa')

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)
# Load the model and processor for Gemma 3.4B IT

processor = AutoProcessor.from_pretrained(
    "google/gemma-3-4b-it",
    padding_side="left"
)

model.eval()
print("Model and processor loaded successfully.")

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]

# Prepare inputs using the processor
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to("cuda")

input_ids    = inputs.input_ids     # (1, seq_len)
pixel_values = inputs.pixel_values  # (1,3,H,W)
print(f"Input IDs shape: {input_ids.shape}, Pixel values shape: {pixel_values.shape}")

# timer = time_cuda if device.type=="cuda" else time_cpu

# ---------------------------------------------------------
# 2) RUN VISION TOWER ONLY
# ---------------------------------------------------------
# move vision tower to GPU
model.vision_tower.to("cuda")
torch.cuda.empty_cache()

# time it
vision_ms = time_cuda(lambda: model.vision_tower(pixel_values=pixel_values.to("cuda")))
print(f"Vision tower time: {vision_ms:.1f} ms")

# grab the outputs (e.g. last_hidden_state)
with torch.no_grad():
    vision_outputs = model.vision_tower(pixel_values=pixel_values.to("cuda"))

# move vision tower back to CPU and clear GPU
model.vision_tower.to("cpu")
torch.cuda.empty_cache()

# ---------------------------------------------------------
# 3) RUN TEXT EMBED + PREFILL
# ---------------------------------------------------------
# move decoder (and embeddings) onto GPU
# note: submodule names may vary; inspect model to confirm
model.language_model.to("cuda")
model.lm_head.to("cuda")              # if there’s a separate head
model.get_input_embeddings().to("cuda")
torch.cuda.empty_cache()

# time text embedding
embed_ms = time_cuda(lambda: model.get_input_embeddings()(input_ids.to("cuda")))
print(f"Text embedding time: {embed_ms:.1f} ms")

# time full prefill
def do_prefill():
    with torch.no_grad():
        _ = model(
            encoder_outputs=vision_outputs,   # pass in your cached vision outputs
            input_ids=input_ids.to("cuda"),
            use_cache=True,
        )
prefill_ms = time_cuda(do_prefill)
print(f"Full prefill time: {prefill_ms:.1f} ms")


# ---------------------------------------------------------
# 4) AUTOREGRESSIVE TOKEN LOOP
# ---------------------------------------------------------
n_steps = 5
times = []
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

with torch.no_grad():
    out = model(
        encoder_outputs=vision_outputs,
        input_ids=input_ids.to("cuda"),
        use_cache=True,
    )
    past_kv  = out.past_key_values
    last_tok = input_ids[:,-1].unsqueeze(-1).to("cuda")
    print(f"Initial last token shape: {last_tok}","and", model.vocab_size)

# print(f"Initial past_key_values shape: {past_kv}")

for i in range(n_steps):
    # sync and start
    torch.cuda.synchronize()
    start.record()
    print(f"Token {i+1} decode...")

    # single token decode
    with torch.no_grad():
        out = model(
            input_ids=last_tok,
            past_key_values=past_kv,
            encoder_outputs=vision_outputs,
            use_cache=True,
        )
        # pick next token
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # update for next iteration
        last_tok, past_kv = next_tok, out.past_key_values

    # end timing
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    times.append(ms)
    print(f"Token {i+1} decode time: {ms:.1f} ms")

# print(f"Avg per-token: {sum(times)/len(times):.1f} ms")