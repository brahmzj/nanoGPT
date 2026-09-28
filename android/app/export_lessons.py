"""
A snapshot of the curriculum for the Android app: lessons to review while learning (so the
brain does not forget its ABCs) and an exam to check that it did not.

    lessons.txt   one lesson per line; a tab inside a lesson stands for a newline
    exam.txt      stage <tab> prompt <tab> answer, with newlines written as \\n

    python android/app/export_lessons.py OUT_DIR
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from morpheus.curriculum import STAGES, exam  # noqa: E402


def export(out_dir, per_stage=2000, exam_per_stage=40, seed=2026):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    lessons = []
    for stage in STAGES:
        for _ in range(per_stage):
            lesson = stage.lesson(rng)
            assert "\t" not in lesson
            lessons.append(lesson.replace("\n", "\t"))
    rng.shuffle(lessons)
    with open(os.path.join(out_dir, "lessons.txt"), "w", encoding="ascii") as f:
        f.write("\n".join(lessons) + "\n")
    with open(os.path.join(out_dir, "exam.txt"), "w", encoding="ascii") as f:
        for stage in STAGES:
            for prompt, answer in exam(stage, exam_per_stage, seed + 1):
                f.write(f"{stage.name}\t{prompt.replace(chr(10), chr(92) + 'n')}\t{answer.replace(chr(10), chr(92) + 'n')}\n")
    return len(lessons)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "android/app/build/assets"
    n = export(out)
    size = os.path.getsize(os.path.join(out, "lessons.txt"))
    print(f"{n} lessons ({size / 1024:.0f} KB before compression) and an exam in {out}")
