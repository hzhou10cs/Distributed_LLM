from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download


def is_cached_locally(model_name: str) -> bool:
    try:
        hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            local_files_only=True,
        )
        return True
    except Exception:
        return False


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 40) -> torch.Tensor:
    logits = logits.float() / temperature
    probs = torch.softmax(logits, dim=-1)
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        _, indices = torch.topk(probs, top_k)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        probs = probs.masked_fill(~mask, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)
