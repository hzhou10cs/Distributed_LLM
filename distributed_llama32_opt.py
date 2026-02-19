import argparse
import socket
import struct
import time
import threading
from typing import Optional, Tuple, List

from torch import nn
import warnings
from collections import OrderedDict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

################################################################################
# Utility: TCP send/recv tensors without pickle
################################################################################
TAG_TO_ID = {"PREFILL": 1, "HIDDEN": 2, "TOKEN": 3}
ID_TO_TAG = {v: k for k, v in TAG_TO_ID.items()}


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    buf = b""
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed during recv")
        buf += chunk
    return buf


def send_packet(sock: socket.socket, tag: str, payload: torch.Tensor) -> None:
    # Header: magic(4), tag(1), dtype(1), ndim(1), shape(8*4), byte_len(4)
    if tag not in TAG_TO_ID:
        raise ValueError(f"Unknown tag: {tag}")
    tag_id = TAG_TO_ID[tag]

    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Unsupported payload type: {type(payload)}")
    tensor = payload.detach().cpu().contiguous()
    if tensor.dtype == torch.float16:
        dtype_code = 0
    elif tensor.dtype == torch.float32:
        dtype_code = 1
    elif tensor.dtype == torch.int64:
        dtype_code = 2
    else:
        raise ValueError(f"Unsupported dtype: {tensor.dtype}")
    shape = list(tensor.shape)
    if len(shape) > 8:
        raise ValueError("Too many dims for header")
    shape_padded = shape + [0] * (8 - len(shape))
    payload_bytes = tensor.numpy().tobytes()

    header = struct.pack(
        ">4sBBB8II",
        b"DLLM",
        tag_id,
        dtype_code,
        len(shape),
        *shape_padded,
        len(payload_bytes),
    )
    sock.sendall(header)
    sock.sendall(payload_bytes)


def recv_packet(sock: socket.socket):
    header_len = struct.calcsize(">4sBBB8II")
    hdr = recv_exact(sock, header_len)
    magic, tag_id, dtype_code, ndim, *rest = struct.unpack(">4sBBB8II", hdr)
    if magic != b"DLLM":
        raise ValueError("Bad magic")
    tag = ID_TO_TAG.get(tag_id)
    if not tag:
        raise ValueError(f"Unknown tag id: {tag_id}")
    shape = rest[:8][:ndim]
    byte_len = rest[-1]
    payload = recv_exact(sock, byte_len)

    if dtype_code == 0:
        dtype = torch.float16
    elif dtype_code == 1:
        dtype = torch.float32
    elif dtype_code == 2:
        dtype = torch.int64
    else:
        raise ValueError(f"Unknown dtype code: {dtype_code}")

    tensor = torch.frombuffer(payload, dtype=dtype).clone()
    if len(shape) > 0:
        tensor = tensor.view(*shape)
    return tag, tensor

################################################################################
# The three pipeline stages, each device will only implement the ones it needs
################################################################################
class PipelineWrapper:
    def __init__(self, model_name: str,layer_begin: int, layer_end: int, dtype, device):
        self.model_name  = model_name
        self.layer_begin = layer_begin
        self.layer_end   = layer_end
        self.device      = device

        # load the right slice of the model onto `device`
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type=="cuda" else torch.float32,
            #device_map="auto",
            #layer_begin=self.layer_begin,
            #layer_end=self.layer_end,
        )
        # tokenizer only needed on head/tail for encoding + decoding
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        
        # Check decoder params
        print_decoder_params = False
        if print_decoder_params:
            target_id = 17
            target_id_key = str(target_id)
            layers = self.model.model.layers
            modules: OrderedDict[str, nn.Module] = layers._modules # KEY POINT
            if target_id_key in modules:
                target_layer = modules[target_id_key]
                print("Found layer:", target_id)
                print("  gate_proj weight:", tuple(target_layer.mlp.gate_proj.weight.shape))
            else:
                raise KeyError(f"Target layer {target_id} not in this slice!")
        # helpers from our earlier design
        # embed‐only if layer_begin<0 && layer_end>=0
        # middle  if layer_begin>=0 && layer_end>=0
        # tail    if layer_end<0
        self.is_embed = layer_begin < 0 and layer_end >= 0
        self.is_tail  = layer_end  < 0
        self.is_mid   = not self.is_embed and not self.is_tail

        # state
        self.past_kvs: Optional[Tuple] = None
        self.seq_num: int = 0

        # logging
        self.log = open(f"device_{layer_begin}_{layer_end}.log", "w", encoding="utf-8", buffering=1)

    def logf(self, msg: str):
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.log.write(f"[{t}] {msg}\n")
        print(msg)

    def forward_with_embedding(
        self, input_ids: torch.LongTensor, use_cache: bool
    ) -> Tuple[torch.FloatTensor, Tuple]:
        st = time.time()
        with torch.no_grad():
            # Use model.model (LlamaModel) to get BaseModelOutputWithPast with last_hidden_state
            outputs: BaseModelOutputWithPast = self.model.model(
                input_ids=input_ids.to(self.device),
                past_key_values=self.past_kvs,
                use_cache=use_cache,
            )
        hidden = outputs.last_hidden_state
        self.past_kvs = outputs.past_key_values  # save for next call
        dt = (time.time() - st) * 1000
        self.logf(f"Embed -> slice0 time: {dt:.2f} ms")
        return hidden.cpu(), dt

    def forward_only_decoders(
        self, hidden: torch.FloatTensor, use_cache: bool
    ) -> Tuple[torch.FloatTensor, Tuple]:
        st = time.time()
        with torch.no_grad():
            # Use model.model (LlamaModel) with inputs_embeds (not input_hidden_states)
            outputs: BaseModelOutputWithPast = self.model.model(
                inputs_embeds=hidden.to(self.device),
                past_key_values=self.past_kvs,
                use_cache=use_cache,
            )
        hidden = outputs.last_hidden_state
        self.past_kvs = outputs.past_key_values
        dt = (time.time() - st) * 1000
        self.logf(f"Sliced‐decoders time: {dt:.2f} ms")
        return hidden.cpu(), dt

    def forward_with_logits(
            self,
            hidden: torch.FloatTensor,
            use_cache: bool,
            generated_ids: Optional[List[int]] = None,
            temperature: float = 1,
            top_p: float = 1,
            top_k: int = 0,
            repetition_penalty: float = 1.1,
        ) -> Tuple[int, float]:

        st = time.time()
        with torch.no_grad():
            # Use model.model (LlamaModel) with inputs_embeds to get hidden states
            model_outputs: BaseModelOutputWithPast = self.model.model(
                inputs_embeds=hidden.to(self.device),
                past_key_values=self.past_kvs,
                use_cache=use_cache,
            )
            # Apply final layer norm and lm_head to get logits
            hidden_states = model_outputs.last_hidden_state
            hidden_states = self.model.model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)
        
        logits = logits[:, -1, :]          # [1, vocab]
        next_id = sample_next_token(
            logits,
            generated_ids=generated_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        self.past_kvs = model_outputs.past_key_values
        
        dt = (time.time() - st) * 1000
        self.logf(f"Norm+head time: {dt:.2f} ms -> token (id={next_id.item()})")
        return next_id.cpu(), dt


from typing import Optional, List

def sample_next_token(
    logits: torch.Tensor,
    generated_ids: Optional[List[int]] = None,
    temperature: float = 1,
    top_p: float = 1,
    top_k: int = 0,
    repetition_penalty: float = 1.1,
) -> torch.Tensor:
    # logits: [batch, vocab]
    logits = logits.float()

    # 2) temperature
    logits = logits / temperature

    # 3) convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    # 4) top-k
    if top_k > 0:
        top_k = min(top_k, probs.size(-1))
        values, indices = torch.topk(probs, top_k)
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        probs = probs.masked_fill(~mask, 0.0)

    # renormalize then sample
    probs = probs / probs.sum(dim=-1, keepdim=True)
    next_id = torch.multinomial(probs, num_samples=1)
    return next_id

################################################################################
# The main orchestration loop
################################################################################
def run_device(args):
    # 1) Build wrapper
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    wrapper = PipelineWrapper(args.model_name, args.layer_begin, args.layer_end, dtype, device)

    # 2) Setup TCP sockets
    server = None
    conn = None
    cli = None
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.recv_host, args.recv_port))
    server.listen(1)
    wrapper.logf(f"[{args.layer_begin}->{args.layer_end}] Listening on {args.recv_host}:{args.recv_port}")
    time.sleep(1.0)
    cli = socket.socket()
    while True:
        try:
            cli.connect((args.send_host, args.send_port))
            wrapper.logf(f"[{args.layer_begin}->{args.layer_end}] Connected downstream to {args.send_host}:{args.send_port}")
            break
        except (ConnectionRefusedError, TimeoutError):
            time.sleep(0.1)
    conn, _ = server.accept()
    wrapper.logf(f"[{args.layer_begin}->{args.layer_end}] Accepted upstream connection")

    # 3) PREFILL (only head device gets the prompt from CLI)
    wrapper.logf("=== PREFILL START ===")
    if wrapper.is_embed:
        # head‐of‐pipeline: get prompt, tokenize, send through
        wrapper.logf("Prefill phase: embedding -> …")
        input_ids = wrapper.tokenizer(
            args.prompt, return_tensors="pt"
        ).input_ids
        hidden, _ = wrapper.forward_with_embedding(input_ids, use_cache=True)
        gen_ids = input_ids.clone().to(device)
        st = time.time()
        send_packet(cli, "PREFILL", hidden)
        
    elif wrapper.is_mid :
        tag, hidden = recv_packet(conn)
        assert tag == "PREFILL"
        wrapper.logf("Prefill phase: received hidden+KVs")
        hidden, _ = wrapper.forward_only_decoders(hidden, use_cache=True)
        st = time.time()
        send_packet(cli, "PREFILL", hidden)
    
    elif wrapper.is_tail:
        tag, hidden = recv_packet(conn)
        assert tag == "PREFILL"
        wrapper.logf("Prefill phase: hidden+KVs -> logits")
        token_ids, _ = wrapper.forward_with_logits(hidden, use_cache=True)
        st = time.time()
        send_packet(cli, "TOKEN", token_ids)
            
    else:
        raise RuntimeError("Invalid slice configuration for prefill")
    wrapper.logf("=== PREFILL FINISHED ===")

    # 4) GENERATION loop
    wrapper.logf("=== GENERATION START ===")
    if wrapper.is_embed:
        # driver will trigger next_token by receiving back from tail
        stage_t_list = []
        token_t_list = []
        for step in range(args.max_new_tokens):
            # wait for token from tail
            tag, next_id = recv_packet(conn)
            print(f"tag={tag}, next_id={next_id}")
            wrapper.logf(f"(Seq {step}) Received next_token={next_id}")
            
            dt = (time.time() - st) * 1000
            if step == 0:
                wrapper.logf(f"Prefilling stage takes {dt:.2f} ms")
            else:
                wrapper.logf(f"Generation token {step} takes {dt:.2f} ms")
                token_t_list.append(dt)
            st = time.time()
            
            hidden, stage_t = wrapper.forward_with_embedding(
                torch.tensor([[next_id]]), use_cache=True
            )
            stage_t_list.append(stage_t)
            next_id = next_id.to(gen_ids.device)
            gen_ids  = torch.cat([gen_ids, next_id], dim=-1)
            send_packet(cli, "HIDDEN", hidden)
        decoded = wrapper.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(f"\n[4] Decoded result:\n{decoded}\n")

    elif wrapper.is_mid:
        # middle: keep looping forever
        stage_t_list = []
        for step in range(args.max_new_tokens):
            tag, hidden = recv_packet(conn)
            assert tag == "HIDDEN"
            hidden, stage_t = wrapper.forward_only_decoders(hidden, use_cache=True)
            stage_t_list.append(stage_t)
            send_packet(cli, "HIDDEN", hidden)

    elif wrapper.is_tail:
        stage_t_list = []
        generated_ids = []
        for step in range(args.max_new_tokens):
            tag, hidden = recv_packet(conn)
            assert tag == "HIDDEN"

            next_id, stage_t = wrapper.forward_with_logits(
                hidden,
                use_cache=True,
                generated_ids=generated_ids,
                temperature=1,       # tune these
                top_p=1,
                top_k=40,
                repetition_penalty=1.1,
            )

            generated_ids.append(int(next_id.item()))
            wrapper.logf(f"(Seq {step}) Emitting next_token={next_id}")
            stage_t_list.append(stage_t)
            send_packet(cli, "TOKEN", next_id)
    else:
        raise RuntimeError("Invalid slice configuration for generation")    

    # 5) Done
    if wrapper.is_embed:
        wrapper.logf(f"Average token time: {sum(token_t_list)/len(token_t_list):.2f} ms")
    wrapper.logf(f"Average stage time: {sum(stage_t_list)/len(stage_t_list):.2f} ms")
    wrapper.logf("Done.")
    conn.close()
    if cli:
        cli.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--layer_begin", type=int,  required=True)
    p.add_argument("--layer_end",   type=int,  required=True)
    p.add_argument("--recv_host",   type=str,  default="192.168.50.3")
    p.add_argument("--recv_port",   type=int,  default=5003)
    p.add_argument("--send_host",   type=str,  default="192.168.50.1")
    p.add_argument("--send_port",   type=int,  default=5001)
    p.add_argument("--prompt",      type=str,  default="Once upon a time, a wizard lived in a tower.")
    p.add_argument("--max_new_tokens", type=int, default=100)
    args = p.parse_args()
    run_device(args)



