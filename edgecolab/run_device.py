import socket
import time
from argparse import Namespace
from typing import List

import torch

from edgecolab.adapters.base import StageRole
from edgecolab.runners import create_runner
from edgecolab.transport import recv_packet, send_packet
from edgecolab.utils import sample_next_token


def run_device(args: Namespace) -> None:
    if args.protocol.lower() != "tcp":
        raise ValueError("Only TCP is implemented in this runner.")

    runner = create_runner(args)
    role = runner.role

    server = socket.socket()
    cli = socket.socket()
    conn = None
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.recv_host, args.recv_port))
        server.listen(1)
        runner.logf(f"[{args.layer_begin}->{args.layer_end}] Listening on {args.recv_host}:{args.recv_port}")

        time.sleep(1.0)
        while True:
            try:
                cli.connect((args.send_host, args.send_port))
                runner.logf(f"[{args.layer_begin}->{args.layer_end}] Connected downstream to {args.send_host}:{args.send_port}")
                break
            except (ConnectionRefusedError, TimeoutError):
                time.sleep(0.1)

        conn, _ = server.accept()
        runner.logf(f"[{args.layer_begin}->{args.layer_end}] Accepted upstream connection")

        runner.logf("=== PREFILL START ===")
        if role == StageRole.HEAD:
            input_ids = runner.runtime.tokenizer(args.prompt, return_tensors="pt").input_ids
            t0 = time.time()
            out = runner.runtime.prefill_head(input_ids)
            stage_ms = (time.time() - t0) * 1000
            send_packet(cli, "PREFILL", out["hidden_states"])
            runner.logf(f"Prefill head stage time: {stage_ms:.2f} ms")
            gen_ids = input_ids.to(runner.device)
        elif role == StageRole.MIDDLE:
            tag, hidden = recv_packet(conn)
            assert tag == "PREFILL"
            t0 = time.time()
            out = runner.runtime.step_middle(hidden)
            stage_ms = (time.time() - t0) * 1000
            send_packet(cli, "PREFILL", out["hidden_states"])
            runner.logf(f"Prefill middle stage time: {stage_ms:.2f} ms")
        else:
            tag, hidden = recv_packet(conn)
            assert tag == "PREFILL"
            t0 = time.time()
            out = runner.runtime.step_tail(hidden)
            logits = out["logits"][:, -1, :]
            next_id = sample_next_token(logits, temperature=args.temperature, top_k=args.top_k)
            stage_ms = (time.time() - t0) * 1000
            send_packet(cli, "TOKEN", next_id.to(torch.int64))
            runner.logf(f"Prefill tail stage time: {stage_ms:.2f} ms -> token {int(next_id.item())}")
        runner.logf("=== PREFILL FINISHED ===")

        runner.logf("=== GENERATION START ===")
        stage_times: List[float] = []
        token_times: List[float] = []
        tick = time.time()

        if role == StageRole.HEAD:
            for step in range(args.max_new_tokens):
                tag, next_id = recv_packet(conn)
                assert tag == "TOKEN"
                tok_ms = (time.time() - tick) * 1000
                if step > 0:
                    token_times.append(tok_ms)
                runner.logf(f"Generation token {step}: {tok_ms:.2f} ms")

                t0 = time.time()
                out = runner.runtime.prefill_head(next_id.to(torch.long))
                stage_ms = (time.time() - t0) * 1000
                stage_times.append(stage_ms)
                send_packet(cli, "HIDDEN", out["hidden_states"])
                gen_ids = torch.cat([gen_ids, next_id.to(gen_ids.device)], dim=-1)
                tick = time.time()

            text = runner.runtime.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            runner.logf(f"Decoded: {text}")
        elif role == StageRole.MIDDLE:
            for step in range(args.max_new_tokens):
                tag, hidden = recv_packet(conn)
                assert tag == "HIDDEN"
                t0 = time.time()
                out = runner.runtime.step_middle(hidden)
                stage_ms = (time.time() - t0) * 1000
                stage_times.append(stage_ms)
                send_packet(cli, "HIDDEN", out["hidden_states"])
                runner.logf(f"Step {step} middle stage: {stage_ms:.2f} ms")
        else:
            for step in range(args.max_new_tokens):
                tag, hidden = recv_packet(conn)
                assert tag == "HIDDEN"
                t0 = time.time()
                out = runner.runtime.step_tail(hidden)
                logits = out["logits"][:, -1, :]
                next_id = sample_next_token(logits, temperature=args.temperature, top_k=args.top_k)
                stage_ms = (time.time() - t0) * 1000
                stage_times.append(stage_ms)
                send_packet(cli, "TOKEN", next_id.to(torch.int64))
                runner.logf(f"Step {step} tail stage: {stage_ms:.2f} ms -> token {int(next_id.item())}")

        if token_times:
            runner.logf(f"Average token latency: {sum(token_times)/len(token_times):.2f} ms")
        if stage_times:
            runner.logf(f"Average stage latency: {sum(stage_times)/len(stage_times):.2f} ms")
        runner.logf("Done.")
    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, ConnectionError, OSError) as e:
        runner.logf(f"Connection closed: {type(e).__name__}: {e}")
    finally:
        try:
            if conn is not None:
                conn.close()
        finally:
            cli.close()
            server.close()
            runner.close()
