import argparse
import multiprocessing as mp
import time
from types import SimpleNamespace

from distributed_llama32_opt import run_device


def build_device_args(
    model_name: str,
    layer_begin: int,
    layer_end: int,
    host: str,
    recv_port: int,
    send_port: int,
    prompt: str,
    max_new_tokens: int,
):
    return SimpleNamespace(
        model_name=model_name,
        layer_begin=layer_begin,
        layer_end=layer_end,
        recv_host=host,
        recv_port=recv_port,
        send_host=host,
        send_port=send_port,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


def launch_local_ring(args):
    # 3-process local ring:
    # dev1: recv from dev3, send to dev2
    # dev2: recv from dev1, send to dev3
    # dev3: recv from dev2, send to dev1
    p1 = args.base_port
    p2 = args.base_port + 1
    p3 = args.base_port + 2

    dev1 = build_device_args(
        model_name=args.model_name,
        layer_begin=-1,
        layer_end=args.layer_end1,
        host=args.host,
        recv_port=p1,
        send_port=p2,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    dev2 = build_device_args(
        model_name=args.model_name,
        layer_begin=args.layer_end1,
        layer_end=args.layer_end2,
        host=args.host,
        recv_port=p2,
        send_port=p3,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    dev3 = build_device_args(
        model_name=args.model_name,
        layer_begin=args.layer_end2,
        layer_end=-1,
        host=args.host,
        recv_port=p3,
        send_port=p1,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
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


if __name__ == "__main__":
    mp.freeze_support()
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Run a local 3-instance collaborative inference simulation."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--layer_end1", type=int, required=True)
    parser.add_argument("--layer_end2", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--base_port", type=int, default=5600)
    parser.add_argument("--startup_stagger_sec", type=float, default=0.2)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, a wizard lived in a tower.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100)
    cli_args = parser.parse_args()

    if cli_args.layer_end1 >= cli_args.layer_end2:
        raise ValueError("--layer_end1 must be smaller than --layer_end2")

    launch_local_ring(cli_args)
