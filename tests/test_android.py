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
        self.assertTrue({"AndroidManifest.xml", "classes.dex", "resources.arsc"} <= names)


@unittest.skipIf(NO_JDK, "needs a JDK")
class TestJavaBrain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        sources = [os.path.join(REPO, "android", "app", "src", "ai", "morpheus", f)
                   for f in ("Brain.java", "Alphabet.java", "Assistant.java")]
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

    def py_brain(self):
        class Brain:  # the NumpyBrain interface that Conversation expects
            block_size = self.py.cfg["block_size"]

            def generate(inner, text, max_new_tokens, temperature=0.0, top_k=None):
                return self.py.generate(text, max_new_tokens)
        return Brain()


if __name__ == "__main__":
    unittest.main()
