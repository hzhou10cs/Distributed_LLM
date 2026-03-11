from argparse import Namespace

from transformers import AutoConfig, AutoTokenizer

from edgecolab.utils import is_cached_locally
from .base_runner import BaseDeviceRunner


class LlamaRunner(BaseDeviceRunner):
    """Llama-family runner with family-specific config/tokenizer loading."""

    def __init__(self, args: Namespace):
        super().__init__(args)

    def load_total_layers(self, model_name: str) -> int:
        local_only = is_cached_locally(model_name)
        try:
            cfg = AutoConfig.from_pretrained(model_name, local_files_only=local_only)
        except Exception:
            if local_only:
                cfg = AutoConfig.from_pretrained(model_name)
            else:
                raise
        return int(getattr(cfg, "num_hidden_layers"))

    def build_tokenizer(self, model_name: str):
        local_only = is_cached_locally(model_name)
        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_only)
        except Exception:
            return AutoTokenizer.from_pretrained(model_name, use_fast=True)
