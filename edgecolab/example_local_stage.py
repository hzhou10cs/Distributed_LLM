import argparse

import torch
from transformers import AutoConfig

from edgecolab.adapters.base import StageRole
from edgecolab.runtime.pipeline import StageRuntime, default_stage_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--role", type=str, choices=["head", "middle", "tail", "full"], default="full")
    parser.add_argument("--cut0", type=int, default=-1)
    parser.add_argument("--cut1", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(args.model_name)
    total_layers = int(getattr(cfg, "num_hidden_layers"))
    cut0 = args.cut0 if args.cut0 >= 0 else max(1, total_layers // 3)
    cut1 = args.cut1 if args.cut1 >= 0 else max(cut0 + 1, (2 * total_layers) // 3)

    full_runtime = StageRuntime.create(
        args.model_name,
        stage_spec=default_stage_spec(StageRole.FULL, total_layers=total_layers, cuts=(cut0, cut1)),
        device=device,
        dtype=None,
        with_tokenizer=True,
    )

    role = StageRole(args.role)
    spec = default_stage_spec(role, total_layers=total_layers, cuts=(cut0, cut1))
    runtime = StageRuntime.create(
        args.model_name,
        stage_spec=spec,
        device=device,
        dtype=None,
        with_tokenizer=(role in {StageRole.HEAD, StageRole.FULL}),
    )

    if role == StageRole.FULL:
        inputs = full_runtime.tokenizer(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = full_runtime.model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        text = full_runtime.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        print(text)
        return

    if runtime.tokenizer is None:
        print("Tokenizer disabled for this role. This script demonstrates HEAD/FULL paths only.")
        return

    input_ids = runtime.tokenizer(args.prompt, return_tensors="pt").input_ids
    out = runtime.prefill_head(input_ids)
    print("Hidden shape:", tuple(out["hidden_states"].shape))


if __name__ == "__main__":
    main()
