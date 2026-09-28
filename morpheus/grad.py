"""
Learning without PyTorch: Morpheus' backward pass, written by hand in numpy.

This lets a phone (Termux on Android), a Raspberry Pi or any machine with numpy
keep teaching Morpheus. It mirrors model.py exactly; tests check its loss and
every gradient against PyTorch autograd.

The optimizer is AdamW with PyTorch's semantics (decoupled weight decay, bias
correction) plus global gradient-norm clipping.
"""

import json

import numpy as np

from .compress import pack

EPS = 1e-6


def rms_fwd(x):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    return x * r, r


def rms_bwd(dy, y, r):
    # y = x * r  ->  dx = r * (dy - y * mean(dy * y))
    return r * (dy - y * (dy * y).mean(-1, keepdims=True))


def softmax(z):
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


class NumpyTrainer:

    def __init__(self, weights, config, lr=3e-4, weight_decay=0.1, betas=(0.9, 0.99), grad_clip=1.0):
        self.cfg = dict(config)
        self.w = {k: np.array(v, dtype=np.float32) for k, v in weights.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.w.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.w.items()}
        self.t = 0
        self.lr, self.weight_decay, self.betas, self.grad_clip = lr, weight_decay, betas, grad_clip
        c = self.cfg
        self.hd = c["n_embd"] // c["n_head"]
        half = self.hd // 2
        inv_freq = c["rope_base"] ** (-np.arange(half, dtype=np.float32) / half)
        angles = np.outer(np.arange(c["block_size"], dtype=np.float32), inv_freq)
        self.cos, self.sin = np.cos(angles), np.sin(angles)

    # ------------------------------------------------------------------ rope

    def rope(self, x, T, sign=1.0):
        cos, sin = self.cos[:T], sign * self.sin[:T]
        x1, x2 = x[..., :self.hd // 2], x[..., self.hd // 2:]
        return np.concatenate((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1)

    # ------------------------------------------------------------------ forward

    def forward(self, ids, keep=True):
        """ids (B, T) -> logits (B, T, V), plus the activations needed for backward."""
        c, w = self.cfg, self.w
        B, T = ids.shape
        nh, nkv, hd = c["n_head"], c["n_kv_head"], self.hd
        g = nh // nkv
        scale = 1.0 / np.sqrt(hd)
        mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
        x = w["wte.weight"][ids]
        tape = []
        for _ in range(c["n_loop"]):
            for i in range(c["n_layer"]):
                p = f"blocks.{i}."
                h, r1 = rms_fwd(x)
                q = (h @ w[p + "attn.q.weight"].T).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
                kv = (h @ w[p + "attn.kv.weight"].T).reshape(B, T, 2, nkv, hd)
                k, v = kv[:, :, 0].transpose(0, 2, 1, 3), kv[:, :, 1].transpose(0, 2, 1, 3)
                q, k = self.rope(q, T), self.rope(k, T)
                k, v = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)  # same order as torch repeat_interleave
                att = softmax(q @ k.transpose(0, 1, 3, 2) * scale + mask)
                y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, nh * hd)
                x = x + y @ w[p + "attn.o.weight"].T
                h2, r2 = rms_fwd(x)
                relu = np.maximum(h2 @ w[p + "mlp.fc.weight"].T, 0.0)
                act = relu * relu
                x = x + act @ w[p + "mlp.proj.weight"].T
                if keep:
                    tape.append((p, h, r1, q, k, v, att, y, h2, r2, relu, act))
        hf, rf = rms_fwd(x)
        logits = hf @ w["wte.weight"].T
        return logits, (ids, tape, hf, rf)

    def predict(self, ids):
        return self.forward(np.asarray(ids), keep=False)[0].argmax(-1)

    # ------------------------------------------------------------------ backward

    def loss_and_grads(self, x, y):
        c, w = self.cfg, self.w
        logits, (ids, tape, hf, rf) = self.forward(x)
        B, T, V = logits.shape
        nh, nkv, hd, d = c["n_head"], c["n_kv_head"], self.hd, c["n_embd"]
        g = nh // nkv
        scale = 1.0 / np.sqrt(hd)
        probs = softmax(logits)
        flat = probs.reshape(-1, V)
        rows = np.arange(flat.shape[0])
        targets = y.reshape(-1)
        loss = float(-np.log(flat[rows, targets] + 1e-12).mean())
        dlogits = flat.copy()
        dlogits[rows, targets] -= 1.0
        dlogits = (dlogits / flat.shape[0]).reshape(B, T, V)

        grads = {k: np.zeros_like(v) for k, v in w.items()}
        grads["wte.weight"] += dlogits.reshape(-1, V).T @ hf.reshape(-1, d)
        dx = rms_bwd(dlogits @ w["wte.weight"], hf, rf)
        for p, h, r1, q, k, v, att, yc, h2, r2, relu, act in reversed(tape):
            # MLP: x_out = x_mid + relu(h2 @ fc.T)^2 @ proj.T
            grads[p + "mlp.proj.weight"] += dx.reshape(-1, d).T @ act.reshape(-1, act.shape[-1])
            da = (dx @ w[p + "mlp.proj.weight"]) * (2.0 * relu)
            grads[p + "mlp.fc.weight"] += da.reshape(-1, da.shape[-1]).T @ h2.reshape(-1, d)
            dx = dx + rms_bwd(da @ w[p + "mlp.fc.weight"], h2, r2)
            # attention: x_mid = x_in + y @ o.T
            grads[p + "attn.o.weight"] += dx.reshape(-1, d).T @ yc.reshape(-1, nh * hd)
            dy = (dx @ w[p + "attn.o.weight"]).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
            datt = dy @ v.transpose(0, 1, 3, 2)
            dv = att.transpose(0, 1, 3, 2) @ dy
            ds = att * (datt - (datt * att).sum(-1, keepdims=True)) * scale
            dq = ds @ k
            dk = ds.transpose(0, 1, 3, 2) @ q
            dk = dk.reshape(B, nkv, g, T, hd).sum(2)  # fold the shared kv heads back together
            dv = dv.reshape(B, nkv, g, T, hd).sum(2)
            dq, dk = self.rope(dq, T, -1.0), self.rope(dk, T, -1.0)  # rotation back = inverse rope
            dq = dq.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)
            dkv = np.stack((dk.transpose(0, 2, 1, 3), dv.transpose(0, 2, 1, 3)), axis=2).reshape(B, T, -1)
            h_flat = h.reshape(-1, d)
            grads[p + "attn.q.weight"] += dq.reshape(-1, nh * hd).T @ h_flat
            grads[p + "attn.kv.weight"] += dkv.reshape(-1, dkv.shape[-1]).T @ h_flat
            dh = dq @ w[p + "attn.q.weight"] + dkv @ w[p + "attn.kv.weight"]
            dx = dx + rms_bwd(dh, h, r1)
        np.add.at(grads["wte.weight"], ids.reshape(-1), dx.reshape(-1, d))
        return loss, grads

    # ------------------------------------------------------------------ optimizer

    def step(self, x, y, lr=None):
        """One AdamW step on a batch. x, y: int arrays (B, T). Returns the loss."""
        lr = self.lr if lr is None else lr
        loss, grads = self.loss_and_grads(x, y)
        if self.grad_clip:
            norm = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
            if norm > self.grad_clip:
                for g in grads.values():
                    g *= self.grad_clip / (norm + 1e-6)
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, p in self.w.items():
            g = grads[k]
            if p.ndim >= 2 and self.weight_decay:
                p *= 1 - lr * self.weight_decay
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            p -= lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + 1e-8)
        return loss

    # ------------------------------------------------------------------ files

    def save(self, path):
        arrays = {f"w/{k}": v for k, v in self.w.items()}
        arrays.update({f"m/{k}": v for k, v in self.m.items()})
        arrays.update({f"v/{k}": v for k, v in self.v.items()})
        with open(path, "wb") as f:  # a file object stops numpy from appending ".npz"
            np.savez(f, t=np.array(self.t), config=np.array(json.dumps(self.cfg)), **arrays)

    @classmethod
    def load(cls, path, **kw):
        with np.load(path) as z:
            weights = {k[2:]: z[k] for k in z.files if k.startswith("w/")}
            trainer = cls(weights, json.loads(str(z["config"])), **kw)
            for k in weights:
                trainer.m[k], trainer.v[k] = z["m/" + k], z["v/" + k]
            trainer.t = int(z["t"])
        return trainer

    def export(self, scheme="int8", meta=None):
        """The bytes of a .morph file holding the current weights."""
        return pack(self.w, self.cfg, scheme, meta)
