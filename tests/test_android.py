"""
The Android app's Java brain and assistant must think exactly like the Python ones.
Needs a JDK (javac, java); skipped otherwise. Run with:  python -m unittest tests.test_android -v
"""

import datetime as dt
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "android", "app"))

from export_brain import convert  # noqa: E402
from morpheus.assistant import Assistant  # noqa: E402
from morpheus.compress import export, load  # noqa: E402
from morpheus.curriculum import STAGES, exam  # noqa: E402
from morpheus.learn import Home  # noqa: E402
from morpheus.runtime import NumpyMorpheus  # noqa: E402

SHIPPED = os.path.join(REPO, "brains", "morpheus-nano.morph")
NO_JDK = shutil.which("javac") is None or shutil.which("java") is None


class TestApk(unittest.TestCase):

    def test_apk_carries_the_shipped_brain(self):
        """Morpheus.apk must be rebuilt (bash android/app/build.sh) whenever the shipped brain changes."""
        import zipfile
        apk = os.path.join(REPO, "android", "Morpheus.apk")
        if not os.path.exists(apk):
            self.skipTest("no APK built")
        with tempfile.TemporaryDirectory() as d:
            convert(SHIPPED, os.path.join(d, "brain.bin"))
            with open(os.path.join(d, "brain.bin"), "rb") as f, zipfile.ZipFile(apk) as z:
                self.assertEqual(z.read("assets/brain.bin"), f.read())
                names = set(z.namelist())
                from export_lessons import export as export_lessons
                export_lessons(d)  # and today's curriculum snapshot, for learning on the phone
                for asset in ("lessons.txt", "exam.txt"):
                    with open(os.path.join(d, asset), "rb") as g:
                        self.assertEqual(z.read("assets/" + asset), g.read(), asset)
        self.assertTrue({"AndroidManifest.xml", "classes.dex", "resources.arsc"} <= names)


@unittest.skipIf(NO_JDK, "needs a JDK")
class TestJavaBrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        src = os.path.join(REPO, "android", "app", "src", "ai", "morpheus")
        sources = []  # every class that does not need Android itself
        for name in sorted(os.listdir(src)):
            with open(os.path.join(src, name)) as f:
                if name.endswith(".java") and "import android." not in f.read():
                    sources.append(os.path.join(src, name))
        sources.append(os.path.join(REPO, "tests", "android", "Harness.java"))
        env = {**os.environ, "JAVA_TOOL_OPTIONS": ""}
        subprocess.run(["javac", "-d", cls.tmp] + sources, check=True, capture_output=True, env=env)
        cls.env = env
        cls.brain_bin = os.path.join(cls.tmp, "brain.bin")
        convert(SHIPPED, cls.brain_bin)
        cls.py = NumpyMorpheus.from_file(SHIPPED)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def java(self, *args):
        out = subprocess.run(["java", "-cp", self.tmp, "Harness", *args], check=True, capture_output=True,
                             env=self.env)
        return out.stdout.decode("utf-8").split("\n")[:-1]

    def lines_file(self, lines):
        path = os.path.join(self.tmp, f"in-{random.random()}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(line.replace("\n", "\\n") for line in lines) + "\n")
        return path

    def test_every_scheme_unpacks_and_thinks_like_numpy(self):
        import torch
        from morpheus.model import Morpheus, MorpheusConfig
        torch.manual_seed(0)
        model = Morpheus(MorpheusConfig(n_embd=64, n_layer=2, n_loop=2, n_head=4, n_kv_head=2, block_size=64))
        ids = [0, 45, 70, 12, 3, 88, 33, 0, 60]
        for scheme in ("f32", "f16", "int8", "int4", "int3", "ternary", "binary"):
            morph = os.path.join(self.tmp, f"t-{scheme}.morph")
            export(model, morph, scheme)
            convert(morph, morph + ".bin")
            got = np.array([float(v) for v in self.java("logits", morph + ".bin", ",".join(map(str, ids)))])
            want = NumpyMorpheus(*load(morph)).forward(np.array([ids]))[0, -1]
            np.testing.assert_allclose(got, want, atol=2e-4, rtol=1e-3, err_msg=scheme)

    def test_shipped_brain_answers_exams_like_numpy(self):
        prompts = []
        for stage in STAGES:
            prompts += ["\n" + p for p, _ in exam(stage, 15, seed=11)]
        got = self.java("generate", self.brain_bin, self.lines_file(prompts))
        want = [self.py.generate(p, 128).replace("\n", "\\n") for p in prompts]
        same = sum(g == w for g, w in zip(got, want))
        self.assertEqual(len(got), len(prompts))
        self.assertGreaterEqual(same / len(prompts), 0.98, [(p, g, w) for p, g, w in zip(prompts, got, want) if g != w])

    def test_assistant_conversation_matches_python(self):
        messages = [
            "hello", "who are you?", "what comes after k?", "what is 7 x 8?", "what is 12 + 9?",
            "how do you spell elephant?", "what color is the sky?", "which is bigger, 73 or 37?",
            "mia has 4 cookies and gets 3 more. how many cookies now?", "count to 10",
            "remember that my locker code is 4071", "what is my locker code?", "what do you remember?",
            "add milk to my list", "add call grandma to my list", "what's on my list?", "done milk",
            "what is 1234 x 5678?", "what is 2**100", "what is 7 divided by 0?", "what is 10 / 4?",
            "hey morpheus, what time is it?", "what is the date?", "set an alarm for 6:45 am",
            "remind me to call mom at 5pm", "what is the capital of france?", "forget my locker code",
            "help", "bye",
        ]
        clock = dt.datetime(2026, 9, 28, 14, 5)
        home = Home(tempfile.mkdtemp())
        py = Assistant(self.py_brain(), home, now=lambda: clock)
        want = [py.respond(m).replace("\n", "\\n") for m in messages]
        got = self.java("assistant", self.brain_bin, self.lines_file(messages), str(int(clock.timestamp() * 1000)))
        for m, g, w in zip(messages, got, want):
            self.assertEqual(g, w, m)

    # ---------------------------------------------------------------- learning on the phone

    def tiny_bin(self, scheme):
        import torch
        from morpheus.model import Morpheus, MorpheusConfig
        torch.manual_seed(0)
        model = Morpheus(MorpheusConfig(n_embd=32, n_layer=2, n_loop=2, n_head=4, n_kv_head=2, block_size=64))
        morph = os.path.join(self.tmp, f"tiny-{scheme}.morph")
        export(model, morph, scheme)
        convert(morph, morph + ".bin")
        return morph

    def batch_file(self, x, y):
        path = os.path.join(self.tmp, f"batch-{random.random()}.txt")
        with open(path, "w") as f:
            for xr, yr in zip(x, y):
                f.write(" ".join(map(str, xr)) + ";" + " ".join(map(str, yr)) + "\n")
        return path

    def test_quantization_matches_python(self):
        from morpheus.quant import fake_quant_np
        rng = np.random.default_rng(5)
        for shape in ((48, 80), (37, 70), (96, 32)):
            w = rng.normal(0, 0.02, size=shape).astype(np.float32)
            path = os.path.join(self.tmp, "matrix.txt")
            with open(path, "w") as f:
                f.write(f"{shape[0]} {shape[1]}\n" + "\n".join(repr(float(v)) for v in w.reshape(-1)) + "\n")
            for scheme in ("f16", "int8", "int4", "int3", "ternary", "binary"):
                got = np.array([float(v) for v in self.java("quant", scheme, path)], dtype=np.float32)
                want = (w.astype(np.float16).astype(np.float32) if scheme == "f16"
                        else fake_quant_np(w, scheme)).reshape(-1)
                np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-7, err_msg=scheme)
                self.assertGreaterEqual(np.mean(got == want), 0.999, scheme)

    def test_gradients_match_python(self):
        from morpheus.grad import NumpyTrainer, softmax
        morph = self.tiny_bin("f32")
        weights, header = load(morph)
        rng = np.random.default_rng(1)
        x, y = rng.integers(0, 96, (3, 20)), rng.integers(0, 96, (3, 20))
        soft = softmax(rng.normal(size=(3, 20, 96)).astype(np.float32))
        soft_path = os.path.join(self.tmp, "soft.txt")
        with open(soft_path, "w") as f:
            for r in soft:
                f.write(" ".join(repr(float(v)) for v in r.reshape(-1)) + "\n")
        alpha = np.array([0.5, 0.0, 0.3], dtype=np.float32)
        out = self.java("grads", morph + ".bin", self.batch_file(x, y), soft_path, "0.5,0.0,0.3")
        tr = NumpyTrainer(weights, header["config"])
        loss, grads = tr.loss_and_grads(x, y, soft=soft, alpha=alpha)
        self.assertAlmostEqual(float(out[0]), loss, places=4)
        got = np.array([float(v) for v in out[1:]])
        want = np.concatenate([grads[k].reshape(-1) for k in sorted(grads)])
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-7)

    def test_training_matches_python(self):
        from morpheus.grad import NumpyTrainer
        rng = np.random.default_rng(2)
        x = rng.integers(0, 96, (4, 24))
        y = np.roll(x, -1, axis=1)
        for scheme in ("f32", "ternary"):
            morph = self.tiny_bin(scheme)
            weights, header = load(morph)
            out = self.java("train", morph + ".bin", self.batch_file(x, y), "4", "0.01")
            quant = None if scheme == "f32" else {"scheme": scheme, "embed_scheme": None}
            tr = NumpyTrainer(weights, header["config"], lr=0.01, weight_decay=0.1, betas=(0.9, 0.99),
                              grad_clip=1.0, quant=quant)
            for _ in range(4):
                tr.step(x, y)
            eff = tr.effective()
            got = np.array([float(v) for v in out])
            want = np.concatenate([eff[k].reshape(-1) for k in sorted(eff)])
            if scheme == "f32":
                np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-5)
            else:  # the same ternary grid values, up to a rare rounding tie
                self.assertGreaterEqual(np.mean(np.isclose(got, want, rtol=1e-3, atol=1e-6)), 0.995)

    def test_adapters_learn_by_the_chain_rule(self):
        """W = Q(W0) + B A: the brain stays frozen, only A and B learn (checked against numpy)."""
        from morpheus.grad import NumpyTrainer
        morph = self.tiny_bin("f32")
        weights, header = load(morph)
        rng = np.random.default_rng(3)
        x = rng.integers(0, 96, (3, 16))
        y = np.roll(x, -1, axis=1)
        steps, lr, r = 3, 0.01, 2
        out = np.array([float(v) for v in self.java("adapt", morph + ".bin", self.batch_file(x, y), str(steps), str(lr))],
                       dtype=np.float32)
        names = sorted(k for k in weights if k.startswith("blocks."))
        A, B, i = {}, {}, 0
        for k in names:  # the Java adapters' starting point
            n = r * weights[k].shape[1]
            A[k], i = out[i:i + n].reshape(r, -1).copy(), i + n
            B[k] = np.zeros((weights[k].shape[0], r), dtype=np.float32)
        state = {k: [np.zeros_like(A[k]), np.zeros_like(A[k]), np.zeros_like(B[k]), np.zeros_like(B[k])] for k in names}
        tr = NumpyTrainer(weights, header["config"])
        for step in range(1, steps + 1):
            tr.w = {k: (weights[k] + B[k] @ A[k]) if k in A else weights[k] for k in weights}
            _, gW = tr.loss_and_grads(x, y)
            gA = {k: B[k].T @ gW[k] for k in names}
            gB = {k: gW[k] @ A[k].T for k in names}
            norm = np.sqrt(sum(float((g * g).sum()) for g in list(gA.values()) + list(gB.values())))
            clip = 1.0 / (norm + 1e-6) if norm > 1.0 else 1.0
            c1, c2 = 1 - 0.9 ** step, 1 - 0.99 ** step
            for k in names:
                for p, g, m, v in ((A[k], gA[k] * clip, 0, 1), (B[k], gB[k] * clip, 2, 3)):
                    state[k][m] = 0.9 * state[k][m] + 0.1 * g
                    state[k][v] = 0.99 * state[k][v] + 0.01 * g * g
                    p -= lr * (state[k][m] / c1) / (np.sqrt(state[k][v] / c2) + 1e-8)
        want = np.concatenate([np.concatenate([A[k].reshape(-1), B[k].reshape(-1)]) for k in names])
        np.testing.assert_allclose(out[i:], want, rtol=2e-3, atol=2e-6)

    def test_learning_session_runs_like_the_phone(self):
        """A whole session (exam, study with review + taught facts, exam, keep or roll back) on a
        tiny brain; the real brain learning a fact end to end is in the docs (too slow for tests)."""
        from export_lessons import export as export_lessons
        export_lessons(self.tmp, per_stage=50, exam_per_stage=5)
        morph = self.tiny_bin("ternary")
        out = self.java("session", morph + ".bin", os.path.join(self.tmp, "lessons.txt"),
                        os.path.join(self.tmp, "exam.txt"), "6", "what is the capital of france?")
        kept, before, after, taught_before, taught_after, summary = out[:6]
        self.assertIn(kept, ("true", "false"))
        for v in (before, after, taught_before, taught_after):
            self.assertTrue(0.0 <= float(v) <= 1.0)
        self.assertIn("my exam:", summary)
        self.assertIn("6 steps", summary)
        self.assertEqual(len(out), 7)  # and the learned brain still answers

    def py_brain(self):
        class Brain:  # the NumpyBrain interface that Conversation expects
            block_size = self.py.cfg["block_size"]

            def generate(inner, text, max_new_tokens, temperature=0.0, top_k=None):
                return self.py.generate(text, max_new_tokens)
        return Brain()


if __name__ == "__main__":
    unittest.main()
