"""
Turn a .morph brain into brain.bin for the Android app.

Android has no LZMA decoder, so the payload is stored decompressed (ternary weights are
packed 5 per byte and barely compress anyway; the APK's own zip compression does the rest).
The header is plain text so the Java loader needs no JSON library:

    MORPHRAW
    alphabet <hex>
    vocab_size 96
    ...
    tensor <name> <scheme> <rows> <cols> <pad> <offset> <size>
    end
    <payload>

    python android/app/export_brain.py brains/morpheus-nano.morph android/app/assets/brain.bin
"""

import json
import lzma
import os
import struct
import sys

MAGIC = b"MORPHEUS"
CONFIG_KEYS = ("vocab_size", "block_size", "n_layer", "n_loop", "n_head", "n_kv_head", "n_embd",
               "mlp_ratio", "rope_base")


def convert(src, dst):
    with open(src, "rb") as f:
        data = f.read()
    if data[:len(MAGIC)] != MAGIC:
        raise SystemExit(f"{src} is not a .morph file")
    (hlen,) = struct.unpack_from("<I", data, len(MAGIC))
    start = len(MAGIC) + 4
    header = json.loads(data[start:start + hlen].decode("utf-8"))
    payload = lzma.decompress(data[start + hlen:])
    cfg = header["config"]
    lines = ["MORPHRAW", "alphabet " + header["alphabet"].encode("ascii").hex()]
    lines += [f"{k} {cfg[k]}" for k in CONFIG_KEYS]
    for t in header["tensors"]:
        shape = t["shape"]
        rows = shape[0]
        cols = 1
        for s in shape[1:]:
            cols *= s
        lines.append(f"tensor {t['name']} {t['q']} {rows} {cols} {t.get('pad', 0)} {t['offset']} {t['size']}")
    lines.append("end")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii") + payload)
    return len(payload)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "brains/morpheus-nano.morph"
    dst = sys.argv[2] if len(sys.argv) > 2 else "android/app/assets/brain.bin"
    n = convert(src, dst)
    print(f"{dst}: {os.path.getsize(dst) / 1024:.1f} KB ({n / 1024:.1f} KB of weights)")
