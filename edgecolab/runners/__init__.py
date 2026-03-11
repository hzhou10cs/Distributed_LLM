from argparse import Namespace

from .base_runner import BaseDeviceRunner
from .llama_runner import LlamaRunner

RUNNER_REGISTRY = {
    "llama": LlamaRunner,
}


def infer_model_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "llama" in lowered:
        return "llama"
    raise ValueError(f"Could not infer model family from model_name={model_name!r}. Set --model_family explicitly.")


def create_runner(args: Namespace) -> BaseDeviceRunner:
    family = getattr(args, "model_family", "auto")
    if family == "auto":
        family = infer_model_family(args.model_name)
    cls = RUNNER_REGISTRY.get(family)
    if cls is None:
        supported = ", ".join(sorted(RUNNER_REGISTRY.keys()))
        raise ValueError(f"Unsupported model_family={family!r}. Supported: {supported}")
    return cls(args)

