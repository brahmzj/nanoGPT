"""
The classroom and the exam hall. Pure Python + numpy (no PyTorch), so the same
code builds lessons and grades exams on a GPU box and on a phone.
"""

import os
import random

import numpy as np

from . import tokenizer
from .curriculum import Library, exam


class Classroom:
    """Builds training batches of lessons. Each row is a newline followed by lessons
    separated by newlines, exactly like the conversation context Morpheus sees later.

    rows(i) studies stage i, reviewing earlier stages `replay` of the time.
    rows(None) mixes all stages, uniformly or by `weights`."""

    def __init__(self, stages, block_size, batch_size, replay=0.0, seed=1337, weights=None):
        self.stages = stages
        self.T = block_size
        self.B = batch_size
        self.replay = replay
        self.weights = weights
        self.rng = random.Random(seed)

    def pick_stage(self, current):
        if current is None:
            if self.weights:
                return self.rng.choices(self.stages, weights=self.weights)[0]
            return self.rng.choice(self.stages)
        if current > 0 and self.rng.random() < self.replay:
            return self.stages[self.rng.randrange(current)]
        return self.stages[current]

    def row(self, current):
        parts, n = [tokenizer.NEWLINE], 1
        while n < self.T + 1:
            lesson = self.pick_stage(current).lesson(self.rng) + tokenizer.NEWLINE
            parts.append(lesson)
            n += len(lesson)
        return tokenizer.encode("".join(parts)[:self.T + 1])

    def rows(self, current):
        """int64 array (batch_size, block_size + 1); inputs are [:, :-1], targets [:, 1:]."""
        return np.array([self.row(current) for _ in range(self.B)], dtype=np.int64)


def grade(predict, questions, block_size, batch_size=256):
    """Score an exam without generating token by token.

    For greedy decoding, "the model outputs the answer" is exactly equivalent to
    "the argmax prediction at every answer position equals the answer character"
    (teacher forcing), which needs a single forward pass per batch of questions.

    predict: fn(int64 array (B, T)) -> argmax predictions (B, T)
    questions: list of (prompt, answer). Returns (score in [0, 1], list of misses).
    """
    correct, misses = 0, []
    for i in range(0, len(questions), batch_size):
        chunk = questions[i:i + batch_size]
        seqs, spans = [], []
        for prompt, answer in chunk:
            ids = tokenizer.encode(tokenizer.NEWLINE + prompt + answer)[-(block_size + 1):]
            seqs.append(ids)
            spans.append((len(ids) - len(answer), len(ids)))
        width = max(len(s) for s in seqs)
        x = np.full((len(seqs), width), tokenizer.NEWLINE_ID, dtype=np.int64)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = s  # right padding never affects earlier positions in a causal model
        pred = predict(x[:, :-1])
        for j, ((a, b), (prompt, answer)) in enumerate(zip(spans, chunk)):
            got = pred[j, a - 1:b - 1]
            if np.array_equal(got, x[j, a:b]):
                correct += 1
            else:
                misses.append((prompt, answer, tokenizer.decode(got.tolist())))
    return correct / max(1, len(questions)), misses


def report_card(predict, stages, block_size, n=200, seed=2024):
    rows = []
    for s in stages:
        if not s._exam_facts:
            continue
        score, misses = grade(predict, exam(s, n, seed), block_size)
        rows.append((s.name, score, misses))
    return rows


def print_report_card(rows, title="report card"):
    print(f"\n  ~ {title} ~")
    for name, score, misses in rows:
        bar = "#" * round(score * 20)
        print(f"  {name:>8} |{bar:<20}| {score * 100:5.1f}%")
        for prompt, answer, got in misses[:2]:
            print(f"           missed: {(prompt + answer).strip()!r:.60}  (said {got.strip()!r:.20})")
    print()


def load_library(path):
    files = []
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            files += [os.path.join(root, n) for n in sorted(names) if n.endswith((".txt", ".md"))]
    else:
        files = [path]
    text = "\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in files)
    text = tokenizer.normalize(text)
    print(f"library: {len(files)} file(s), {len(text):,} characters")
    return Library(text)
