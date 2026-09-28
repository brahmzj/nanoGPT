"""
Morpheus' brain: a small decoder-only transformer tuned for low energy use.

Compared to nanoGPT's GPT-2 (../model.py), every change here either removes
parameters or removes work:
  - RMSNorm without learned weights (cheaper than LayerNorm, zero parameters)
  - rotary position embeddings (RoPE): no position table to store or learn
  - grouped-query attention: fewer key/value heads -> smaller KV cache and weights
  - ReLU^2 MLP: cheap to compute, and its activations are mostly exact zeros
  - tied input/output embeddings, no biases anywhere
  - optional weight sharing: `n_loop` runs the same stack of blocks several
    times, so depth costs compute but not storage
  - a KV cache for generation, so each new token costs one position of work
    instead of re-reading the whole context (nanoGPT recomputes everything)
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class MorpheusConfig:
    vocab_size: int = 96
    block_size: int = 128   # longest context, in characters
    n_layer: int = 4        # distinct blocks (these hold the parameters)
    n_loop: int = 1         # times the block stack is applied; depth = n_layer * n_loop
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 128
    mlp_ratio: int = 4
    rope_base: float = 10000.0
    dropout: float = 0.0

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

    @property
    def depth(self):
        return self.n_layer * self.n_loop


SIZES = {
    # name: overrides. Parameter counts are for the default 96-symbol vocabulary.
    "pico":  dict(n_embd=64,  n_layer=3, n_loop=1, n_head=4, n_kv_head=2),   # ~0.14M
    "nano":  dict(n_embd=128, n_layer=4, n_loop=1, n_head=4, n_kv_head=2),   # ~0.73M
    "loop":  dict(n_embd=128, n_layer=2, n_loop=2, n_head=4, n_kv_head=2),   # ~0.37M, nano's compute
    "micro": dict(n_embd=192, n_layer=6, n_loop=1, n_head=6, n_kv_head=2),   # ~2.3M
}


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rope_tables(cfg, device=None):
    half = cfg.head_dim // 2
    inv_freq = cfg.rope_base ** (-torch.arange(half, dtype=torch.float32, device=device) / half)
    angles = torch.outer(torch.arange(cfg.block_size, dtype=torch.float32, device=device), inv_freq)
    return angles.cos(), angles.sin()  # (block_size, head_dim/2) each


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); rotate the two halves of each head as complex pairs
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class KVCache:
    """Preallocated keys/values for every (looped) layer. Generation reuses them
    so each new token only computes attention for itself."""

    def __init__(self, cfg, batch_size, device, dtype=torch.float32):
        shape = (batch_size, cfg.n_kv_head, cfg.block_size, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.depth)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.depth)]
        self.pos = 0  # number of positions already stored

    def update(self, layer, k, v):
        T = k.size(2)
        self.k[layer][:, :, self.pos:self.pos + T] = k
        self.v[layer][:, :, self.pos:self.pos + T] = v
        end = self.pos + T
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0 and cfg.n_head % cfg.n_kv_head == 0
        self.cfg = cfg
        hd = cfg.head_dim
        self.q = nn.Linear(cfg.n_embd, cfg.n_head * hd, bias=False)
        self.kv = nn.Linear(cfg.n_embd, 2 * cfg.n_kv_head * hd, bias=False)
        self.o = nn.Linear(cfg.n_head * hd, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, cache=None, layer=0):
        cfg = self.cfg
        B, T, _ = x.shape
        q = self.q(x).view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, cfg.n_kv_head, cfg.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            # either a prefill into an empty cache, or one new token at a time
            assert cache.pos == 0 or T == 1, "KV cache supports prefill then single-token steps"
            k, v = cache.update(layer, k, v)
        groups = cfg.n_head // cfg.n_kv_head
        if groups > 1:
            k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=T > 1, dropout_p=cfg.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, cfg.n_embd))


class MLP(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, cfg.mlp_ratio * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.mlp_ratio * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None, layer=0):
        x = x + self.drop(self.attn(rms_norm(x), cos, sin, cache, layer))
        return x + self.drop(self.mlp(rms_norm(x)))


class Morpheus(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # doubles as the output head
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # scaled init for residual projections (GPT-2)
            if name.endswith(("attn.o.weight", "mlp.proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.depth))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        start = cache.pos if cache is not None else 0
        assert start + T <= self.cfg.block_size, f"context {start + T} > block_size {self.cfg.block_size}"
        cos, sin = self.cos[start:start + T], self.sin[start:start + T]
        x = self.wte(idx)
        layer = 0
        for _ in range(self.cfg.n_loop):
            for block in self.blocks:
                x = block(x, cos, sin, cache, layer)
                layer += 1
        if cache is not None:
            cache.pos += T
        x = rms_norm(x)
        if targets is None:
            return F.linear(x, self.wte.weight), None
        logits = F.linear(x, self.wte.weight)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, betas, device_type):
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, stop_id=None):
        """Greedy (temperature 0) or sampled generation with a KV cache. As much of the prompt as
        fits is kept; the answer stops when the context (block_size) is full."""
        idx = idx[:, -(self.cfg.block_size - 1):]
        max_new_tokens = min(max_new_tokens, self.cfg.block_size - idx.size(1))
        cache = KVCache(self.cfg, idx.size(0), idx.device, self.wte.weight.dtype)
        logits, _ = self(idx, cache=cache)
        out = idx
        for _ in range(max_new_tokens):
            nxt = sample(logits[:, -1, :], temperature, top_k)
            out = torch.cat((out, nxt), dim=1)
            if stop_id is not None and (nxt == stop_id).all():
                break
            if cache.pos >= self.cfg.block_size:
                break
            logits, _ = self(nxt, cache=cache)
        return out

    def flops_per_token(self, context, training=False):
        """Rough FLOPs to process one token (PaLM-style accounting, as in nanoGPT's estimate_mfu).
        Shared weights are counted once per use, because compute is spent every loop."""
        cfg = self.cfg
        block = sum(p.numel() for p in self.blocks.parameters()) // cfg.n_layer
        n_used = block * cfg.depth + cfg.vocab_size * cfg.n_embd  # + output head
        attn = 2 * cfg.depth * cfg.n_embd * context  # q.k and att.v
        forward = 2 * n_used + 2 * attn
        return 3 * forward if training else forward


def sample(logits, temperature=0.0, top_k=None):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


def build(size="nano", **overrides):
    cfg = MorpheusConfig(**{**SIZES[size], **overrides})
    return Morpheus(cfg)


def config_dict(cfg):
    return asdict(cfg)
