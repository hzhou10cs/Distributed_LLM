from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn


class StageRole(str, Enum):
    HEAD = "head"
    MIDDLE = "middle"
    TAIL = "tail"
    FULL = "full"


@dataclass(frozen=True)
class StageSpec:
    role: StageRole
    layer_begin: int
    layer_end: int


class ModelAdapter:
    """Model-family adapter API for distributed inference runtime."""

    def __init__(self, model: nn.Module):
        self.model = model

    @classmethod
    def can_handle(cls, model: nn.Module) -> bool:
        raise NotImplementedError

    def validate_spec(self, spec: StageSpec) -> None:
        raise NotImplementedError

    def configure_for_stage(self, spec: StageSpec) -> None:
        """Optional stage-aware model initialization hook."""
        self.validate_spec(spec)

    def hidden_size(self) -> int:
        raise NotImplementedError

    def num_layers(self) -> int:
        raise NotImplementedError

    def embed(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError

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
        raise NotImplementedError

    def apply_final_norm(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def project_logits(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def list_state_dict_prefixes(self, spec: StageSpec) -> Sequence[str]:
        """For future selective loading by stage."""
        raise NotImplementedError
