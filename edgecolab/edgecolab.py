import argparse

from edgecolab.run_device import run_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic distributed stage runner.")
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--model_family", type=str, default="auto", help="auto|llama|...")
    p.add_argument("--layer_begin", type=int, required=True)
    p.add_argument("--layer_end", type=int, required=True)
    p.add_argument("--recv_host", type=str, required=True)
    p.add_argument("--recv_port", type=int, required=True)
    p.add_argument("--send_host", type=str, required=True)
    p.add_argument("--send_port", type=int, required=True)
    p.add_argument("--protocol", type=str, default="tcp")
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--no_log_layer_info", dest="log_layer_info", action="store_false")
    p.set_defaults(log_layer_info=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_device(args)


if __name__ == "__main__":
    main()
