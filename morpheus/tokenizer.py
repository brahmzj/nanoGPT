"""
Morpheus' alphabet: a fixed 96-symbol character vocabulary.

Id 0 is the newline (the end-of-thought marker), ids 1..95 are the printable
ASCII characters ' '..'~'. The vocabulary never changes between curriculum
stages, so the embedding table learned for the ABCs is the same one used later
for sentences and conversation. 96 symbols fit in a single byte, so every token
of data costs exactly 1 byte (nanoGPT uses 2).
"""

import unicodedata

import numpy as np

NEWLINE = "\n"
CHARS = NEWLINE + "".join(chr(c) for c in range(32, 127))
VOCAB_SIZE = len(CHARS)  # 96
STOI = {ch: i for i, ch in enumerate(CHARS)}
UNKNOWN = STOI["?"]
NEWLINE_ID = STOI[NEWLINE]


def normalize(text):
    """Fold arbitrary unicode text down to Morpheus' 96-symbol alphabet."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = text.encode("ascii", "ignore").decode("ascii")  # strips accents, emoji, ...
    return "".join(ch if ch in STOI else "?" for ch in text)


def encode(text):
    return [STOI.get(ch, UNKNOWN) for ch in text]


def decode(ids):
    return "".join(CHARS[i] for i in ids)


# ascii byte -> token id lookup table, for fast bulk encoding of large corpora
_LUT = np.full(256, UNKNOWN, dtype=np.uint8)
for _ch, _i in STOI.items():
    _LUT[ord(_ch)] = _i


def encode_array(text):
    """Normalize and encode to a compact uint8 array (1 byte per token)."""
    raw = np.frombuffer(normalize(text).encode("ascii"), dtype=np.uint8)
    return _LUT[raw]
