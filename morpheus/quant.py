"""
The quantization math, in one place: numpy (for storing brains) and PyTorch (for
quantization-aware training) versions that produce bit-identical weights.

  scheme    levels             scale (fp16)                  stored as
  int8      -127..127          max|w| per row / 127          1 byte per weight
  int4      -7..7              max|w| per 32 weights / 7     2 weights per byte
  int3      -3..3              max|w| per 32 weights / 3     1 byte per weight, LZMA squeezes it
  ternary   -1, 0, +1          least-squares fit per row     5 weights per byte (3^5 = 243 < 256)
  binary    -1, +1             mean|w| per row               8 weights per byte

Ternary is the "1.58-bit" format of BitNet b1.58: every weight is one of three values, so a
matrix-vector product is only additions and subtractions. The levels are picked BitNet's way
(round w / mean|w|), but the scale is then the least-squares fit to those levels. That is more
accurate, and it makes quantization idempotent: re-quantizing an already quantized brain
(which a phone does when it keeps learning from a .morph file) changes nothing. With a plain
mean|w| scale, the zeros would shrink the scale by a third on every reload. Rounding a trained float brain
straight to ternary destroys it; quantization-aware training (train.py: squeeze) keeps it
learning *through* the rounding so it adapts to the grid.
"""

import numpy as np

SCHEMES = {
    # name: (kind, levels, group size; 0 = one scale per row)
    "int8": ("absmax", 127, 0),
    "int4": ("absmax", 7, 32),
    "int3": ("absmax", 3, 32),
    "ternary": ("absmean", 1, 0),
    "binary": ("sign", 1, 0),
}
FLOAT_SCHEMES = ("f32", "f16")
ALL_SCHEMES = FLOAT_SCHEMES + tuple(SCHEMES)


def _grouped(rows, group):
    """(R, C) -> (R, G, S) with zero padding so that S divides; returns (groups, pad)."""
    R, C = rows.shape
    size = group or C
    pad = (-C) % size
    if pad:
        rows = np.pad(rows, ((0, 0), (0, pad)))
    return rows.reshape(R, -1, size), pad


def quantize_np(w, scheme):
    """float array -> (integer levels int8 (R, G, S), fp16 scales (R, G, 1), pad)."""
    kind, levels, group = SCHEMES[scheme]
    rows = np.asarray(w, dtype=np.float32).reshape(np.shape(w)[0], -1)
    g, pad = _grouped(rows, group)
    if kind == "absmax":
        scale = np.abs(g).max(-1, keepdims=True) / levels
    else:
        scale = np.abs(g).mean(-1, keepdims=True)
    if kind == "absmean":  # pick the levels, then fit the scale to them (least squares)
        s0 = np.where(scale == 0, 1.0, scale).astype(np.float16).astype(np.float32)
        q = np.clip(np.round(g / s0), -levels, levels)
        used = np.abs(q).sum(-1, keepdims=True)
        fit = (np.abs(g) * np.abs(q)).sum(-1, keepdims=True) / np.maximum(used, 1)
        scale = np.where(used > 0, fit, scale)
    scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
    s32 = scale.astype(np.float32)
    if kind == "sign":
        q = np.where(g >= 0, 1, -1)
    elif kind == "absmax":
        q = np.clip(np.round(g / s32), -levels, levels)
    return q.astype(np.int8), scale, pad


def dequantize_levels(q, scale, shape, pad):
    w = q.astype(np.float32) * scale.astype(np.float32)
    w = w.reshape(q.shape[0], -1)
    if pad:
        w = w[:, :-pad]
    return w.reshape(shape)


def fake_quant_np(w, scheme):
    q, scale, pad = quantize_np(w, scheme)
    return dequantize_levels(q, scale, np.shape(w), pad)


# ---------------------------------------------------------------- packing the levels into bytes


def pack_levels(q, scheme):
    flat = q.reshape(-1)
    if scheme == "int4":
        u = (flat + 8).astype(np.uint8)  # 1..15
        if u.size % 2:
            u = np.append(u, np.uint8(8))
        return ((u[0::2] << 4) | u[1::2]).tobytes()
    if scheme == "ternary":
        t = (flat + 1).astype(np.uint8)  # 0..2
        t = np.append(t, np.zeros((-t.size) % 5, dtype=np.uint8)).reshape(-1, 5)
        return (t @ np.array([81, 27, 9, 3, 1], dtype=np.uint16)).astype(np.uint8).tobytes()
    if scheme == "binary":
        return np.packbits(flat > 0).tobytes()
    if scheme == "int8":
        return flat.astype(np.int8).tobytes()  # raw signed bytes (the original .morph layout)
    return (flat.astype(np.int16) + levels_offset(scheme)).astype(np.uint8).tobytes()


def levels_offset(scheme):
    return SCHEMES[scheme][1] + 1  # int3: -3..3 -> 1..7


def unpack_levels(buf, scheme, count):
    raw = np.frombuffer(buf, dtype=np.uint8)
    if scheme == "int4":
        q = np.empty(raw.size * 2, dtype=np.int8)
        q[0::2], q[1::2] = raw >> 4, raw & 0x0F
        return q[:count] - 8
    if scheme == "ternary":
        digits = (raw[:, None] // np.array([81, 27, 9, 3, 1], dtype=np.uint8)[None, :]) % 3
        return digits.reshape(-1)[:count].astype(np.int8) - 1
    if scheme == "binary":
        return np.unpackbits(raw)[:count].astype(np.int8) * 2 - 1
    if scheme == "int8":
        return raw[:count].view(np.int8)
    return (raw[:count].astype(np.int16) - levels_offset(scheme)).astype(np.int8)


# ---------------------------------------------------------------- PyTorch, for training through the rounding


def fake_quant_torch(w, scheme):
    """Quantize-dequantize in PyTorch, identical to fake_quant_np, with a straight-through
    gradient: the forward pass sees grid values, the float weights underneath get the update."""
    import torch
    kind, levels, group = SCHEMES[scheme]
    shape = w.shape
    rows = w.reshape(shape[0], -1)
    size = group or rows.shape[1]
    pad = (-rows.shape[1]) % size
    if pad:
        rows = torch.nn.functional.pad(rows, (0, pad))
    g = rows.reshape(rows.shape[0], -1, size)
    with torch.no_grad():
        if kind == "absmax":
            scale = g.abs().amax(-1, keepdim=True) / levels
        else:
            scale = g.abs().mean(-1, keepdim=True)
        if kind == "absmean":  # levels first, then the least-squares scale (see quantize_np)
            s0 = torch.where(scale == 0, torch.ones_like(scale), scale).half().float()
            q = torch.clamp(torch.round(g / s0), -levels, levels)
            used = q.abs().sum(-1, keepdim=True)
            fit = (g.abs() * q.abs()).sum(-1, keepdim=True) / used.clamp(min=1)
            scale = torch.where(used > 0, fit, scale)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale).half().float()
        if kind == "sign":
            q = torch.where(g >= 0, 1.0, -1.0)
        elif kind == "absmax":
            q = torch.clamp(torch.round(g / scale), -levels, levels)
        deq = (q * scale).reshape(rows.shape[0], -1)
        if pad:
            deq = deq[:, :-pad]
        deq = deq.reshape(shape)
    return w + (deq - w).detach()


class FakeQuant:
    """A torch parametrization: module.weight becomes its quantized self in every forward."""

    def __new__(cls, scheme):
        import torch.nn as nn

        class _FakeQuant(nn.Module):
            def forward(self, w):
                return fake_quant_torch(w, scheme)
        return _FakeQuant()


def scheme_for(name, scheme, embed_scheme=None):
    """Which scheme a tensor uses: the embedding (also the output head) may be kept finer."""
    return embed_scheme if (embed_scheme and name == "wte.weight") else scheme
