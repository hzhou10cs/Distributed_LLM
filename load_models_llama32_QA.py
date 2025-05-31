import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1) Load model & tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.float16 if device.type=="cuda" else torch.float32,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    use_fast=True
)

model.eval()

# 2) Helper to ask a question (with optional context)
def answer_question(question: str, context: str = None, max_new_tokens: int = 200):
    # Build an instruction‐style prompt
    if context:
        prompt = (
            "### Context:\n"
            f"{context}\n\n"
            "### Question:\n"
            f"{question}\n\n"
            "### Answer:\n"
        )
    else:
        prompt = (
            "### Question:\n"
            f"{question}\n\n"
            "### Answer:\n"
        )

    # Tokenize & move to device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate (greedy)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens
    generated = output_ids[0, inputs.input_ids.shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer

# 3) Examples
print(answer_question("What is the capital of France?"))

ctx = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
    "France. It was named after the engineer Gustave Eiffel."
)
print(answer_question("Where is the Eiffel Tower located?", context=ctx))
