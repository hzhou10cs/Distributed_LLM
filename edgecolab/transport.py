import socket
import struct
from typing import Tuple

import torch

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
    if tag not in TAG_TO_ID:
        raise ValueError(f"Unknown tag: {tag}")
    tag_id = TAG_TO_ID[tag]

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


def recv_packet(sock: socket.socket) -> Tuple[str, torch.Tensor]:
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

