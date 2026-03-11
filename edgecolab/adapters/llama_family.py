from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from .base import ModelAdapter, StageRole, StageSpec


class LlamaFamilyAdapter(ModelAdapter):
    """
    Works with model classes that expose:
    - model.embed_tokens
    - model.layers (decoder blocks)
    - model.norm
    - lm_head
    """

    @classmethod
    def can_handle(cls, model: nn.Module) -> bool:
        if not hasattr(model, "model"):
            return False
        core = model.model
        return all(
            hasattr(core, name) for name in ("embed_tokens", "layers", "norm")
        ) and hasattr(model, "lm_head")

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._configured = False
        self._stage_spec: Optional[StageSpec] = None
        self._global_layer_begin = 0
        self._global_layer_end = 0

    def validate_spec(self, spec: StageSpec) -> None:
        n = self.num_layers()
        if spec.layer_begin < 0 or spec.layer_end > n or spec.layer_begin > spec.layer_end:
            raise ValueError(f"Invalid stage layer range [{spec.layer_begin}, {spec.layer_end}) for {n} layers")

    def configure_for_stage(self, spec: StageSpec) -> None:
        self.validate_spec(spec)
        core = self.model.model
        self._stage_spec = spec
        self._global_layer_begin = spec.layer_begin
        self._global_layer_end = spec.layer_end

        # Keep only decoder slice, similar to your custom decoder_creation().
        selected = [core.layers[i] for i in range(spec.layer_begin, spec.layer_end)]
        core.layers = nn.ModuleList(selected)

        # Keep only needed non-decoder modules per role (like tokenize_input/logits_output split).
        if spec.role in (StageRole.MIDDLE, StageRole.TAIL) and hasattr(core, "embed_tokens"):
            core.embed_tokens = None
        if spec.role in (StageRole.HEAD, StageRole.MIDDLE):
            if hasattr(core, "norm"):
                core.norm = None
            if hasattr(self.model, "lm_head"):
                self.model.lm_head = None

        self._configured = True

    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def embed(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        if getattr(self.model.model, "embed_tokens", None) is None:
            raise RuntimeError("embed_tokens is not available for this stage.")
        return self.model.model.embed_tokens(input_ids)

    def run_layers(
        self,
        hidden_states: torch.FloatTensor,
        *,
        layer_begin: int,
        layer_end: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        core = self.model.model
        if not self._configured:
            raise RuntimeError("Adapter stage is not configured. Call configure_for_stage() first.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            past_seen_tokens = past_key_values.get_seq_length()
        else:
            past_seen_tokens = 0

        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + hidden_states.shape[1],
            device=hidden_states.device,
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = core._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            past_key_values,
            output_attentions=False,
        )
        position_embeddings = core.rotary_emb(hidden_states, position_ids)

        local_begin = layer_begin - self._global_layer_begin
        local_end = layer_end - self._global_layer_begin
        if local_begin < 0 or local_end > len(core.layers) or local_begin > local_end:
            raise ValueError(
                f"Layer range [{layer_begin}, {layer_end}) is outside configured stage "
                f"[{self._global_layer_begin}, {self._global_layer_end})."
            )

        for local_idx in range(local_begin, local_end):
            decoder_layer = core.layers[local_idx]
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        return {
            "hidden_states": hidden_states,
            "past_key_values": past_key_values if use_cache else None,
        }

    def apply_final_norm(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        if getattr(self.model.model, "norm", None) is None:
            raise RuntimeError("Final norm is not available for this stage.")
        return self.model.model.norm(hidden_states)

    def project_logits(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        if getattr(self.model, "lm_head", None) is None:
            raise RuntimeError("lm_head is not available for this stage.")
        return self.model.lm_head(hidden_states)

    def list_state_dict_prefixes(self, spec: StageSpec) -> Sequence[str]:
        prefixes = []
        if spec.layer_begin == 0:
            prefixes.append("model.embed_tokens.")
        for i in range(spec.layer_begin, spec.layer_end):
            prefixes.append(f"model.layers.{i}.")
        if spec.layer_end == self.num_layers():
            prefixes.append("model.norm.")
            prefixes.append("lm_head.")
        return prefixes
