"""
Squeezing Morpheus' brain into a .morph file.

Schemes (bits per weight, plus a small overhead for scales):
  f32   32 bits   exact
  f16   16 bits   ~lossless
  int8   8 bits   one fp16 scale per row, symmetric            (default)
  int4   4 bits   one fp16 scale per group of 32 weights, two weights per byte

The quantized payload is then packed with LZMA. The file format is:
  b"MORPHEUS" | uint32 header length | JSON header | LZMA(payload)
The header holds the model config, the 96-symbol alphabet and a table of
tensors (name, shape, scheme, byte offset), so the file is self-describing and
the numpy runtime (runtime.py) needs nothing but this file to think.
"""

import json
import lzma
import struct

import numpy as np

from . import tokenizer

MAGIC = b"MORPHEUS"
GROUP = 32  # int4 group size


def quantize(w, scheme):
    """float32 array -> (list of byte blobs, info dict)."""
    w = np.asarray(w, dtype=np.float32)
    if scheme == "f32" or w.ndim < 2:
        return [w.tobytes()], {"q": "f32"}
    if scheme == "f16":
        return [w.astype(np.float16).tobytes()], {"q": "f16"}
    rows = w.reshape(w.shape[0], -1)
    if scheme == "int8":
        scale = np.abs(rows).max(axis=1, keepdims=True) / 127.0
        scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
        q = np.clip(np.round(rows / scale.astype(np.float32)), -127, 127).astype(np.int8)
        return [scale.tobytes(), q.tobytes()], {"q": "int8"}
    if scheme == "int4":
        cols = rows.shape[1]
        pad = (-cols) % GROUP
        g = np.pad(rows, ((0, 0), (0, pad))).reshape(rows.shape[0], -1, GROUP)
        scale = np.abs(g).max(axis=2, keepdims=True) / 7.0
        scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
        q = np.clip(np.round(g / scale.astype(np.float32)), -7, 7).astype(np.int8) + 8  # 1..15
        q = q.reshape(-1).astype(np.uint8)
        packed = (q[0::2] << 4) | q[1::2]  # GROUP is even, so the count is always even
        return [scale.tobytes(), packed.tobytes()], {"q": "int4", "pad": pad}
    raise ValueError(f"unknown scheme {scheme}")


def dequantize(buf, shape, info):
    shape = tuple(shape)
    n = int(np.prod(shape))
    q = info["q"]
    if q == "f32":
        return np.frombuffer(buf, dtype=np.float32, count=n).reshape(shape).copy()
    if q == "f16":
        return np.frombuffer(buf, dtype=np.float16, count=n).astype(np.float32).reshape(shape)
    rows = shape[0]
    cols = n // rows
    if q == "int8":
        scale = np.frombuffer(buf, dtype=np.float16, count=rows).astype(np.float32).reshape(rows, 1)
        w = np.frombuffer(buf, dtype=np.int8, count=n, offset=2 * rows).astype(np.float32).reshape(rows, cols)
        return (w * scale).reshape(shape)
    if q == "int4":
        padded = cols + info["pad"]
        groups = padded // GROUP
        scale = np.frombuffer(buf, dtype=np.float16, count=rows * groups).astype(np.float32)
        packed = np.frombuffer(buf, dtype=np.uint8, offset=2 * rows * groups)
        q4 = np.empty(packed.size * 2, dtype=np.int8)
        q4[0::2] = packed >> 4
        q4[1::2] = packed & 0x0F
        w = (q4.astype(np.float32) - 8).reshape(rows, groups, GROUP) * scale.reshape(rows, groups, 1)
        return w.reshape(rows, padded)[:, :cols].reshape(shape)
    raise ValueError(f"unknown scheme {q}")


def pack(weights, config, scheme="int8", meta=None):
    """weights: dict name -> float32 array. Returns the bytes of a .morph file."""
    tensors, blobs, offset = [], [], 0
    for name, w in weights.items():
        parts, info = quantize(w, scheme)
        size = sum(len(p) for p in parts)
        tensors.append({"name": name, "shape": list(np.shape(w)), "offset": offset, "size": size, **info})
        blobs.extend(parts)
        offset += size
    header = json.dumps({
        "format": 1, "scheme": scheme, "config": config, "alphabet": tokenizer.CHARS,
        "tensors": tensors, "meta": meta or {},
    }).encode("utf-8")
    payload = lzma.compress(b"".join(blobs), preset=9 | lzma.PRESET_EXTREME)
    return MAGIC + struct.pack("<I", len(header)) + header + payload


def unpack(data):
    """bytes of a .morph file -> (dict name -> float32 array, header dict)."""
    if data[:len(MAGIC)] != MAGIC:
        raise ValueError("not a .morph file")
    (hlen,) = struct.unpack_from("<I", data, len(MAGIC))
    start = len(MAGIC) + 4
    header = json.loads(data[start:start + hlen].decode("utf-8"))
    if header["alphabet"] != tokenizer.CHARS:
        raise ValueError("this brain was trained with a different alphabet")
    payload = lzma.decompress(data[start + hlen:])
    weights = {}
    for t in header["tensors"]:
        buf = payload[t["offset"]:t["offset"] + t["size"]]
        weights[t["name"]] = dequantize(buf, t["shape"], t)
    return weights, header


def export(model, path, scheme="int8", meta=None):
    """Write a torch Morpheus model to a .morph file. Returns the file size in bytes."""
    from .model import config_dict
    weights = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    data = pack(weights, config_dict(model.cfg), scheme, meta)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def load(path):
    with open(path, "rb") as f:
        return unpack(f.read())


def to_torch(weights, header, device="cpu"):
    """Rebuild a torch Morpheus from a .morph file (e.g. to grade a compressed brain)."""
    import torch
    from .model import Morpheus, MorpheusConfig
    model = Morpheus(MorpheusConfig(**header["config"]))
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    return model.to(device).eval()
