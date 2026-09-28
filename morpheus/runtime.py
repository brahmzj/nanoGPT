"""
Morpheus without PyTorch: a pure-numpy inference engine for .morph files.

Training needs PyTorch; thinking does not. This file plus numpy is enough to run
a trained Morpheus on anything with a CPU (a Raspberry Pi, an old laptop). It
mirrors model.py exactly and keeps a KV cache, so each generated character costs
a single position's worth of work.
"""

import numpy as np

from . import tokenizer
from .compress import load


def rms_norm(x, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


class NumpyMorpheus:

    def __init__(self, weights, header):
        self.cfg = c = header["config"]
        self.header = header
        self.w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
        self.depth = c["n_layer"] * c["n_loop"]
        self.hd = c["n_embd"] // c["n_head"]
        half = self.hd // 2
        inv_freq = c["rope_base"] ** (-np.arange(half, dtype=np.float32) / half)
        angles = np.outer(np.arange(c["block_size"], dtype=np.float32), inv_freq)
        self.cos, self.sin = np.cos(angles), np.sin(angles)
        # pre-transpose the Linear weights (out, in) -> (in, out) once, so every matmul is x @ W
        for k in list(self.w):
            if k != "wte.weight":
                self.w[k] = np.ascontiguousarray(self.w[k].T)

    @classmethod
    def from_file(cls, path):
        weights, header = load(path)
        return cls(weights, header)

    def rope(self, x, start):
        T = x.shape[2]
        cos, sin = self.cos[start:start + T], self.sin[start:start + T]
        x1, x2 = x[..., :self.hd // 2], x[..., self.hd // 2:]
        return np.concatenate((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1)

    def forward(self, ids, cache=None):
        """ids: int array (B, T). cache: dict from new_cache() or None. Returns logits (B, T, V)."""
        c, w = self.cfg, self.w
        B, T = ids.shape
        start = cache["pos"] if cache is not None else 0
        assert start + T <= c["block_size"], "context longer than block_size"
        nh, nkv, hd = c["n_head"], c["n_kv_head"], self.hd
        x = w["wte.weight"][ids]
        # causal mask for queries at positions start..start+T-1 over keys 0..start+T-1
        mask = np.triu(np.full((T, start + T), -np.inf, dtype=np.float32), k=start + 1)
        layer = 0
        for _ in range(c["n_loop"]):
            for i in range(c["n_layer"]):
                p = f"blocks.{i}."
                h = rms_norm(x)
                q = (h @ w[p + "attn.q.weight"]).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
                kv = (h @ w[p + "attn.kv.weight"]).reshape(B, T, 2, nkv, hd)
                k, v = kv[:, :, 0].transpose(0, 2, 1, 3), kv[:, :, 1].transpose(0, 2, 1, 3)
                q, k = self.rope(q, start), self.rope(k, start)
                if cache is not None:
                    k = cache["k"][layer] = np.concatenate((cache["k"][layer], k), axis=2)
                    v = cache["v"][layer] = np.concatenate((cache["v"][layer], v), axis=2)
                # grouped-query attention: (B, nkv, groups, T, hd) against (B, nkv, 1, S, hd)
                qg = q.reshape(B, nkv, nh // nkv, T, hd)
                att = qg @ k[:, :, None].transpose(0, 1, 2, 4, 3) / np.sqrt(hd) + mask
                att = np.exp(att - att.max(-1, keepdims=True))
                att /= att.sum(-1, keepdims=True)
                y = (att @ v[:, :, None]).reshape(B, nh, T, hd).transpose(0, 2, 1, 3).reshape(B, T, -1)
                x = x + y @ w[p + "attn.o.weight"]
                h = np.maximum(rms_norm(x) @ w[p + "mlp.fc.weight"], 0.0)
                x = x + (h * h) @ w[p + "mlp.proj.weight"]
                layer += 1
        if cache is not None:
            cache["pos"] += T
        return rms_norm(x) @ w["wte.weight"].T

    def new_cache(self, batch_size=1):
        empty = np.zeros((batch_size, self.cfg["n_kv_head"], 0, self.hd), dtype=np.float32)
        return {"pos": 0, "k": [empty] * self.depth, "v": [empty] * self.depth}

    def predict(self, ids):
        """Argmax prediction at every position (used for grading exams)."""
        return self.forward(np.asarray(ids)).argmax(-1)

    def generate(self, text, max_new_tokens=120, temperature=0.0, top_k=None, stop="\n", rng=None):
        """Continue `text`; returns only the new characters (without the stop character)."""
        rng = rng or np.random.default_rng()
        bs = self.cfg["block_size"]
        ids = tokenizer.encode(tokenizer.normalize(text))[-(bs - 1):]  # keep as much of the prompt as fits
        max_new_tokens = min(max_new_tokens, bs - len(ids))  # the answer stops when the context is full
        cache = self.new_cache()
        logits = self.forward(np.array([ids]), cache)[0, -1]
        out = []
        for _ in range(max_new_tokens):
            if temperature <= 0:
                nxt = int(logits.argmax())
            else:
                z = logits / temperature
                if top_k:
                    z = np.where(z < np.sort(z)[-min(top_k, z.size)], -np.inf, z)
                probs = np.exp(z - z.max())
                nxt = int(rng.choice(z.size, p=probs / probs.sum()))
            ch = tokenizer.CHARS[nxt]
            if ch == stop:
                break
            out.append(ch)
            if cache["pos"] >= bs:
                break
            logits = self.forward(np.array([[nxt]]), cache)[0, -1]
        return "".join(out)
