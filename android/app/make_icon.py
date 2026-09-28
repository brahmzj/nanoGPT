"""
Draw Morpheus' launcher icon (a crescent moon and three stars on a night-violet tile) as PNGs
for every screen density, with nothing but the standard library.

    python android/app/make_icon.py OUT_RES_DIR
"""

import os
import struct
import sys
import zlib

SIZES = {"mdpi": 48, "hdpi": 72, "xhdpi": 96, "xxhdpi": 144, "xxxhdpi": 192}
TOP, BOTTOM = (49, 27, 110), (15, 11, 30)       # night gradient
MOON, STAR = (237, 233, 254), (196, 181, 253)
STARS = [(0.73, 0.28, 0.035), (0.80, 0.50, 0.025), (0.64, 0.70, 0.02)]


def coverage(x, y):
    """(alpha of tile, alpha of moon, alpha of star) at a point in the unit square."""
    r, m = 0.22, 0.0  # rounded-square corner radius
    cx, cy = min(max(x, r), 1 - r), min(max(y, r), 1 - r)
    tile = 1.0 if (x - cx) ** 2 + (y - cy) ** 2 <= r * r else 0.0
    in_moon = (x - 0.45) ** 2 + (y - 0.50) ** 2 <= 0.30 ** 2
    bite = (x - 0.57) ** 2 + (y - 0.42) ** 2 <= 0.26 ** 2
    moon = 1.0 if in_moon and not bite else 0.0
    star = 1.0 if any((x - sx) ** 2 + (y - sy) ** 2 <= sr * sr for sx, sy, sr in STARS) else m
    return tile, moon, star


def render(size, ss=4):
    rows = []
    for py in range(size):
        row = bytearray([0])  # PNG filter: none
        for px in range(size):
            acc = [0.0, 0.0, 0.0, 0.0]
            for sy in range(ss):
                for sx in range(ss):
                    x, y = (px + (sx + 0.5) / ss) / size, (py + (sy + 0.5) / ss) / size
                    tile, moon, star = coverage(x, y)
                    t = y
                    color = [TOP[i] * (1 - t) + BOTTOM[i] * t for i in range(3)]
                    if star:
                        color = list(STAR)
                    if moon:
                        color = list(MOON)
                    for i in range(3):
                        acc[i] += color[i] * tile
                    acc[3] += tile
            n = ss * ss
            alpha = acc[3] / n
            rgb = [int(round(acc[i] / acc[3])) if acc[3] else 0 for i in range(3)]
            row += bytes(rgb + [int(round(alpha * 255))])
        rows.append(bytes(row))
    return png(size, size, b"".join(rows))


def png(w, h, raw):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "android/app/build/res"
    for density, size in SIZES.items():
        folder = os.path.join(out, f"mipmap-{density}")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "ic_launcher.png"), "wb") as f:
            f.write(render(size))
    print(f"icons written to {out}")
