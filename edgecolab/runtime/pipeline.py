from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..adapters.base import ModelAdapter, StageRole, StageSpec
from ..adapters.llama_family import LlamaFamilyAdapter
from ..utils import is_cached_locally


def build_adapter(model: torch.nn.Module) -> ModelAdapter:
    for cls in (LlamaFamilyAdapter,):
        if cls.can_handle(model):
            return cls(model)
    raise ValueError(f"No adapter found for model type: {model.__class__.__name__}")


def _load_causal_lm(model_name: str, *, device: torch.device, dtype: Optional[torch.dtype] = None):
    local_only = is_cached_locally(model_name)
    model_kwargs: Dict[str, Any] = {"local_files_only": local_only}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception:
        fallback_kwargs: Dict[str, Any] = {}
        if dtype is not None:
            fallback_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)
    return model.to(device).eval()


def _load_tokenizer(model_name: str, *, use_fast: bool = True):
    local_only = is_cached_locally(model_name)
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, local_files_only=local_only)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)


@dataclass
class StageRuntime:
    model_name: str
    stage_spec: StageSpec
    device: torch.device
    dtype: Optional[torch.dtype]
    model: torch.nn.Module
    tokenizer: Optional[Any]
    adapter: ModelAdapter
    past_key_values: Optional[Any] = None

    @classmethod
    def create(
        cls,
        model_name: str,
        stage_spec: StageSpec,
        *,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        with_tokenizer: bool = False,
    ) -> "StageRuntime":
        model = _load_causal_lm(model_name, device=device, dtype=dtype)

        adapter = build_adapter(model)
        adapter.configure_for_stage(stage_spec)
        tokenizer = None
        if with_tokenizer:
            tokenizer = _load_tokenizer(model_name, use_fast=True)
        return cls(
            model_name=model_name,
            stage_spec=stage_spec,
            device=device,
            dtype=dtype,
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
        )

    @torch.no_grad()
    def prefill_head(self, input_ids: torch.LongTensor) -> Dict[str, Any]:
        use_cache = bool(getattr(self.model.config, "use_cache", True))
        hidden = self.adapter.embed(input_ids.to(self.device))
        out = self.adapter.run_layers(
            hidden,
            layer_begin=self.stage_spec.layer_begin,
            layer_end=self.stage_spec.layer_end,
            past_key_values=self.past_key_values,
            use_cache=use_cache,
        )
        self.past_key_values = out["past_key_values"]
        return out

    @torch.no_grad()
    def step_middle(self, hidden: torch.FloatTensor) -> Dict[str, Any]:
        use_cache = bool(getattr(self.model.config, "use_cache", True))
        out = self.adapter.run_layers(
            hidden.to(self.device),
            layer_begin=self.stage_spec.layer_begin,
            layer_end=self.stage_spec.layer_end,
            past_key_values=self.past_key_values,
            use_cache=use_cache,
        )
        self.past_key_values = out["past_key_values"]
        return out

    @torch.no_grad()
    def step_tail(self, hidden: torch.FloatTensor) -> Dict[str, Any]:
        out = self.step_middle(hidden)
        hs = self.adapter.apply_final_norm(out["hidden_states"])
        logits = self.adapter.project_logits(hs)
        return {"logits": logits, "past_key_values": self.past_key_values}


def default_stage_spec(role: StageRole, total_layers: int, cuts: tuple[int, int]) -> StageSpec:
    c0, c1 = cuts
    if role == StageRole.HEAD:
        return StageSpec(role=role, layer_begin=0, layer_end=c0)
    if role == StageRole.MIDDLE:
        return StageSpec(role=role, layer_begin=c0, layer_end=c1)
    if role == StageRole.TAIL:
        return StageSpec(role=role, layer_begin=c1, layer_end=total_layers)
    if role == StageRole.FULL:
        return StageSpec(role=role, layer_begin=0, layer_end=total_layers)
    raise ValueError(f"Unsupported role: {role}")
