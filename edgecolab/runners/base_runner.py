import os
import time
from argparse import Namespace
from typing import Any

import torch

from edgecolab.adapters.base import StageRole, StageSpec
from edgecolab.runtime.pipeline import StageRuntime


class BaseDeviceRunner:
    """Generic device runner with shared role/bounds/log/memory utilities."""

    def __init__(self, args: Namespace):
        self.args = args
        self.device_id = int(getattr(args, "device_id", 0))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = open(
            f"device_{args.layer_begin}_{args.layer_end}.log",
            "w",
            buffering=1,
            encoding="utf-8",
            errors="replace",
        )

        self.runtime = self.create_runtime(args)
        self.role = self.runtime.stage_spec.role
        self.stage_spec = self.runtime.stage_spec
        self.total_layers = self.runtime.adapter.num_layers()
        if self.role == StageRole.HEAD and self.runtime.tokenizer is None:
            self.runtime.tokenizer = self.build_tokenizer(args.model_name)

        self.log_memory_usage("after_model_init")
        if bool(getattr(self.args, "log_layer_info", True)):
            self.log_layer_info()

    def create_runtime(self, args: Namespace) -> StageRuntime:
        total_layers = self.load_total_layers(args.model_name)
        lb = 0 if args.layer_begin < 0 else args.layer_begin
        le = total_layers if args.layer_end < 0 else args.layer_end
        if lb < 0 or le > total_layers or lb >= le:
            raise ValueError(f"Invalid layer bounds: ({args.layer_begin}, {args.layer_end}) for total_layers={total_layers}")

        if lb == 0:
            role = StageRole.HEAD
        elif le == total_layers:
            role = StageRole.TAIL
        else:
            role = StageRole.MIDDLE

        spec = StageSpec(role=role, layer_begin=lb, layer_end=le)
        return StageRuntime.create(
            model_name=args.model_name,
            stage_spec=spec,
            device=self.device,
            dtype=None,
            with_tokenizer=False,
        )

    def load_total_layers(self, model_name: str) -> int:
        raise NotImplementedError

    def build_tokenizer(self, model_name: str) -> Any:
        raise NotImplementedError

    def logf(self, msg: str) -> None:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{t}] [dev{self.device_id}] {msg}"
        self.log.write(line + "\n")
        print(line)

    def log_memory_usage(self, tag: str) -> None:
        cpu_msg = "cpu_rss=unknown"
        try:
            import psutil  # type: ignore

            rss = psutil.Process(os.getpid()).memory_info().rss
            cpu_msg = f"cpu_rss={rss / (1024 ** 2):.1f} MiB"
        except Exception:
            pass

        if self.device.type == "cuda" and torch.cuda.is_available():
            did = self.device.index if self.device.index is not None else torch.cuda.current_device()
            alloc = torch.cuda.memory_allocated(did) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(did) / (1024 ** 2)
            max_alloc = torch.cuda.max_memory_allocated(did) / (1024 ** 2)
            self.logf(
                f"[mem:{tag}] device=cuda:{did} {cpu_msg}, "
                f"gpu_alloc={alloc:.1f} MiB, gpu_reserved={reserved:.1f} MiB, gpu_max_alloc={max_alloc:.1f} MiB"
            )
        else:
            self.logf(f"[mem:{tag}] device={self.device.type} {cpu_msg}")

    def log_layer_info(self) -> None:
        self.logf("=== LAYER PLACEMENT START ===")
        target_prefixes = ("model.embed_tokens", "model.layers.", "model.norm", "lm_head")
        for name, module in self.runtime.model.named_modules():
            if not name or not name.startswith(target_prefixes):
                continue
            p = next(module.parameters(recurse=False), None)
            if p is None:
                p = next(module.parameters(), None)
            loc = str(p.device) if p is not None else "no_params"
            self.logf(f"[layer] {name} -> {loc}")
        self.logf("=== LAYER PLACEMENT END ===")

    def close(self) -> None:
        self.log.close()
