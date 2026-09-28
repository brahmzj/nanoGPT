"""
Squeezing Morpheus' brain into a .morph file.

Schemes (see quant.py for the math); sizes are for the 733K-parameter nano brain:
  f32       32 bits per weight                                        ~2.6 MB
  f16       16 bits                                                   ~1.4 MB
  int8       8 bits, one fp16 scale per row                           ~680 KB
  int4       4 bits, one fp16 scale per 32 weights                    ~365 KB
  int3      ~2.4 bits after LZMA, one scale per 32 weights            ~270 KB
  ternary   1.6 bits (-1/0/+1, 5 per byte), one scale per row         ~150 KB
  binary     1 bit (-1/+1), one scale per row                         ~100 KB
Below int4, a brain must be trained through the rounding (train.py: squeeze) to stay smart.

The file format is:
  b"MORPHEUS" | uint32 header length | JSON header | LZMA(payload)
The header holds the model config, the 96-symbol alphabet and a table of tensors
(name, shape, scheme, byte offset), so the file is self-describing and the numpy
runtime (runtime.py) needs nothing but this file to think.
"""

import json
import lzma
import struct

import numpy as np

from . import tokenizer
from .quant import SCHEMES, dequantize_levels, pack_levels, quantize_np, scheme_for, unpack_levels

MAGIC = b"MORPHEUS"


def quantize(w, scheme):
    """float32 array -> (list of byte blobs, info dict)."""
    w = np.asarray(w, dtype=np.float32)
    if scheme == "f32" or w.ndim < 2:
        return [w.tobytes()], {"q": "f32"}
    if scheme == "f16":
        return [w.astype(np.float16).tobytes()], {"q": "f16"}
    q, scale, pad = quantize_np(w, scheme)
    return [scale.tobytes(), pack_levels(q, scheme)], {"q": scheme, "pad": pad}


def dequantize(buf, shape, info):
    shape = tuple(shape)
    n = int(np.prod(shape))
    q = info["q"]
    if q == "f32":
        return np.frombuffer(buf, dtype=np.float32, count=n).reshape(shape).copy()
    if q == "f16":
        return np.frombuffer(buf, dtype=np.float16, count=n).astype(np.float32).reshape(shape)
    if q not in SCHEMES:
        raise ValueError(f"unknown scheme {q}")
    rows = shape[0]
    cols = n // rows
    pad = info.get("pad", 0)
    size = SCHEMES[q][2] or cols
    groups = (cols + pad) // size
    scale = np.frombuffer(buf, dtype=np.float16, count=rows * groups).reshape(rows, groups, 1)
    levels = unpack_levels(buf[2 * rows * groups:], q, rows * (cols + pad)).reshape(rows, groups, size)
    return dequantize_levels(levels, scale, shape, pad)


def pack(weights, config, scheme="int8", meta=None, embed_scheme=None):
    """weights: dict name -> float32 array. Returns the bytes of a .morph file."""
    tensors, blobs, offset = [], [], 0
    for name, w in weights.items():
        parts, info = quantize(w, scheme_for(name, scheme, embed_scheme))
        size = sum(len(p) for p in parts)
        tensors.append({"name": name, "shape": list(np.shape(w)), "offset": offset, "size": size, **info})
        blobs.extend(parts)
        offset += size
    header = json.dumps({
        "format": 1, "scheme": scheme, "embed_scheme": embed_scheme, "config": config,
        "alphabet": tokenizer.CHARS, "tensors": tensors, "meta": meta or {},
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


def export(model, path, scheme="int8", meta=None, embed_scheme=None):
    """Write a torch Morpheus model to a .morph file. Returns the file size in bytes."""
    from .model import config_dict
    weights = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    data = pack(weights, config_dict(model.cfg), scheme, meta, embed_scheme)
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
