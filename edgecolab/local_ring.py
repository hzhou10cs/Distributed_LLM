import argparse
import multiprocessing as mp
import time
from types import SimpleNamespace

from transformers import AutoConfig

from edgecolab.utils import is_cached_locally
from edgecolab.run_device import run_device


def build_device_args(
    model_name: str,
    model_family: str,
    layer_begin: int,
    layer_end: int,
    device_id: int,
    host: str,
    recv_port: int,
    send_port: int,
    prompt: str,
    max_new_tokens: int,
    protocol: str,
    temperature: float,
    top_k: int,
    log_layer_info: bool,
):
    return SimpleNamespace(
        model_name=model_name,
        model_family=model_family,
        layer_begin=layer_begin,
        layer_end=layer_end,
        device_id=device_id,
        recv_host=host,
        recv_port=recv_port,
        send_host=host,
        send_port=send_port,
        protocol=protocol,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        log_layer_info=log_layer_info,
    )


def launch_local_ring(args):
    p1 = args.base_port
    p2 = args.base_port + 1
    p3 = args.base_port + 2

    dev1 = build_device_args(
        model_name=args.model_name,
        model_family=args.model_family,
        layer_begin=-1,
        layer_end=args.layer_end1,
        device_id=1,
        host=args.host,
        recv_port=p1,
        send_port=p2,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        protocol=args.protocol,
        temperature=args.temperature,
        top_k=args.top_k,
        log_layer_info=args.log_layer_info,
    )
    dev2 = build_device_args(
        model_name=args.model_name,
        model_family=args.model_family,
        layer_begin=args.layer_end1,
        layer_end=args.layer_end2,
        device_id=2,
        host=args.host,
        recv_port=p2,
        send_port=p3,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        protocol=args.protocol,
        temperature=args.temperature,
        top_k=args.top_k,
        log_layer_info=args.log_layer_info,
    )
    dev3 = build_device_args(
        model_name=args.model_name,
        model_family=args.model_family,
        layer_begin=args.layer_end2,
        layer_end=-1,
        device_id=3,
        host=args.host,
        recv_port=p3,
        send_port=p1,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        protocol=args.protocol,
        temperature=args.temperature,
        top_k=args.top_k,
        log_layer_info=args.log_layer_info,
    )

    workers = []
    for idx, dev_args in enumerate([dev1, dev2, dev3], start=1):
        proc = mp.Process(target=run_device, args=(dev_args,), name=f"device-{idx}")
        proc.start()
        workers.append(proc)
        time.sleep(args.startup_stagger_sec)

    exit_codes = []
    for proc in workers:
        proc.join()
        exit_codes.append((proc.name, proc.exitcode))

    failed = [(name, code) for name, code in exit_codes if code != 0]
    if failed:
        details = ", ".join([f"{name}={code}" for name, code in failed])
        raise SystemExit(f"One or more device processes failed: {details}")


def _get_total_layers(model_name: str) -> int:
    local_only = is_cached_locally(model_name)
    try:
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=local_only)
    except Exception:
        if local_only:
            cfg = AutoConfig.from_pretrained(model_name)
        else:
            raise
    return int(getattr(cfg, "num_hidden_layers"))


def main() -> None:
    mp.freeze_support()
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Run local 3-stage ring using generic run_device.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_family", type=str, default="auto")
    parser.add_argument("--layer_end1", type=int, required=True)
    parser.add_argument("--layer_end2", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--base_port", type=int, default=5600)
    parser.add_argument("--startup_stagger_sec", type=float, default=0.2)
    parser.add_argument("--protocol", type=str, default="tcp")
    parser.add_argument("--prompt", type=str, default="Once upon a time, a wizard lived in a tower.")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--no_log_layer_info", dest="log_layer_info", action="store_false")
    parser.set_defaults(log_layer_info=True)
    cli_args = parser.parse_args()

    total_layers = _get_total_layers(cli_args.model_name)
    if not (0 < cli_args.layer_end1 < cli_args.layer_end2 < total_layers):
        raise ValueError(
            f"Invalid split for model with {total_layers} layers: "
            f"layer_end1={cli_args.layer_end1}, layer_end2={cli_args.layer_end2}. "
            f"Expected 0 < layer_end1 < layer_end2 < {total_layers}."
        )

    launch_local_ring(cli_args)


if __name__ == "__main__":
    main()
