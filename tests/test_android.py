"""
The Android app's Java brain and assistant must think exactly like the Python ones.
Needs a JDK (javac, java); skipped otherwise. Run with:  python -m unittest tests.test_android -v
"""

import datetime as dt
import os
import random
import re
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
from morpheus.curriculum import STAGES, UNKNOWN_ANSWER, exam  # noqa: E402
from morpheus.learn import Home  # noqa: E402
from morpheus.runtime import NumpyMorpheus  # noqa: E402

SHIPPED = os.path.join(REPO, "brains", "morpheus-nano.morph")
NO_JDK = shutil.which("javac") is None or shutil.which("java") is None


def plain(reply):
    """The app says when it does not know, or is not sure, and goes to find out; the Termux assistant
    just answers. Undo that to compare what the two brains said."""
    if reply.startswith("i don't know yet"):
        return UNKNOWN_ANSWER
    m = re.fullmatch(r"i think (.*) but i am not sure yet, so i will check\.", reply)
    return m.group(1) if m else reply


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
            self.assertEqual(plain(g), w, m)

    # ---------------------------------------------------------------- curiosity

    def converse(self, messages, articles=None):
        props = [f"-Darticles={articles}"] if articles else []
        out = subprocess.run(["java", *props, "-cp", self.tmp, "Harness", "assistant", self.brain_bin,
                              self.lines_file(messages), "1790000000000"], check=True, capture_output=True, env=self.env)
        return dict(zip(range(len(messages)), out.stdout.decode().split("\n")))

    def test_it_says_it_does_not_know_asks_and_learns_when_told(self):
        got = self.converse([
            "what is the capital of spain?", "what are you curious about?", "madrid is the capital of spain.",
            "what is the capital of spain?", "how sure are you?",
            "what is the tallest mountain?", "hello", "mount everest is the tallest mountain.",
            "who wrote hamlet?", "i don't know", "/greet", "william shakespeare wrote hamlet.", "who wrote hamlet?",
            "you are silly"])
        self.assertTrue(got[0].startswith("i don't know yet, but i want to find out. do you know?"), got[0])
        self.assertIn("what is the capital of spain?", got[1])
        self.assertEqual(got[2], "thank you! now i know: madrid is the capital of spain.")  # the answer to its question
        self.assertEqual(got[3], "madrid is the capital of spain.")
        self.assertIn("taught", got[4])
        # not asked, but a fact that answers what it was wondering
        self.assertEqual(got[7], "oh! that answers what i was wondering: what is the tallest mountain? thank you!")
        self.assertEqual(got[9], "that's ok! i will keep wondering, and look for it when i can.")
        self.assertEqual(got[10], "i have been wondering: who wrote hamlet? do you know?")  # it asks on its own
        self.assertEqual(got[12], "william shakespeare wrote hamlet.")
        self.assertEqual(got[13], UNKNOWN_ANSWER)  # "you are ..." is about us, not a fact to keep

    ENCYCLOPEDIA = {
        "volcano": ("Volcano", "A volcano is a mountain where lava comes out of the ground. Volcanoes are found in "
                    "many countries.", "Lava"),
        "eiffel_tower": ("Eiffel Tower", "The Eiffel Tower is a famous iron tower in Paris. It is 330 metres tall. "
                         "It was designed by Gustave Eiffel.", "Paris"),
    }

    def encyclopedia(self):
        d = os.path.join(self.tmp, "encyclopedia")
        os.makedirs(d, exist_ok=True)
        for name, (title, text, links) in self.ENCYCLOPEDIA.items():
            with open(os.path.join(d, name + ".txt"), "w") as f:
                f.write(f"{title}\n{text}\n{links}\n")
        return d

    def test_it_looks_up_what_it_does_not_know(self):
        got = self.converse(["what is a volcano?", "what is a volcano?", "how tall is the eiffel tower?",
                             "who designed the eiffel tower?", "is the eiffel tower in paris?", "what did you find out?",
                             "what is the capital of atlantis?"], articles=self.encyclopedia())
        self.assertEqual(got[0], "i did not know, so i looked it up: a volcano is a mountain where lava comes out of the "
                                 "ground. (simple wikipedia: volcano)")
        self.assertEqual(got[1], "a volcano is a mountain where lava comes out of the ground.")  # and remembers it
        self.assertEqual(got[2], "i did not know, so i looked it up: the eiffel tower is 330 metres tall. "
                                 "(simple wikipedia: eiffel tower)")
        self.assertEqual(got[3], "the eiffel tower was designed by gustave eiffel.")  # from the page it kept
        self.assertEqual(got[4], "yes. the eiffel tower is in paris.")  # reasoning over what it read
        self.assertIn("how tall is the eiffel tower? the eiffel tower is 330 metres tall.", got[5])
        self.assertTrue(got[6].startswith("i don't know yet. i looked, but could not find it yet. i will keep looking in the background"), got[6])

    def test_tasks_you_give_it(self):
        got = self.converse(["find out who designed the eiffel tower", "find out the capital of atlantis",
                             "what are you working on?", "never mind the capital of atlantis", "what are you working on?",
                             "learn about volcanoes"], articles=self.encyclopedia())
        self.assertEqual(got[0], "i found out: the eiffel tower was designed by gustave eiffel. (simple wikipedia: eiffel tower)")
        self.assertEqual(got[1], "ok! i am on it: what is the capital of atlantis? i will keep working on it in the background "
                                 "until i find out, and tell you.")
        self.assertEqual(got[2], "i am working on: finding out what is the capital of atlantis?")
        self.assertEqual(got[3], "ok, i stopped working on that.")
        self.assertTrue(got[4].startswith("nothing right now."), got[4])
        self.assertTrue(got[5].startswith("ok! i am reading about volcanoes in the background."), got[5])

    def test_it_keeps_working_in_the_background_until_done(self):
        """Each run looks harder (the article, then a full-text search, then key words); after three
        misses it asks you. Then there is nothing left to do, and the work stops."""
        out = subprocess.run(["java", f"-Darticles={self.article_dir()}", f"-Dbrain={self.brain_bin}",
                              "-Dwonders=how tall is the eiffel tower|what is the capital of atlantis",
                              "-Dinterests=france,narnia", "-cp", self.tmp, "Harness", "work", "3", "what are you working on?"],
                             check=True, capture_output=True, env=self.env).stdout.decode().split("\n")
        self.assertEqual(out[0], "run 1: found [] read 2 stuck [] more true")  # read what you asked; first looks miss
        self.assertEqual(out[1], "run 2: found [how tall is the eiffel tower? the eiffel tower is 330 metres tall.] "
                                 "read 0 stuck [] more true")  # a full-text search found it on the page about paris
        self.assertEqual(out[2], "run 3: found [] read 0 stuck [what is the capital of atlantis?] more false")
        self.assertEqual(out[3], 'you asked me to learn about france. i read "france": france is a country in western europe. '
                                 "i found 2 facts to study the next time you charge me.")
        self.assertEqual(out[4], "you asked me to learn about narnia, but i could not find anything to read about it.")
        self.assertEqual(out[5], 'you asked me "how tall is the eiffel tower?" i found out: the eiffel tower is 330 metres '
                                 "tall. (simple wikipedia: paris)")
        self.assertEqual(out[6], "i have been wondering: what is the capital of atlantis? i could not find it. do you know?")
        self.assertEqual(out[7], "i need your help with: what is the capital of atlantis? do you know?")

    def test_it_notices_when_it_is_guessing(self):
        """Metacognition: an answer the brain was unsure of (by its own probabilities) is not stated as fact."""
        guesses = ["what do cows eat?", "what color is a tomato?"]
        exam_file = self.lines_file([f"talk\tuser: {q}\\nmorpheus: \tx" for q in guesses])
        sure = [float(line.split("\t")[2]) for line in self.java("confidence", self.brain_bin, exam_file)]
        sure_q = [q for q in ["what color is grass?"]]
        unsure = [q for q, c in zip(guesses, sure) if c < 0.7]
        self.assertTrue(unsure, sure)
        got = self.converse([unsure[0], "how sure are you?", "what are you curious about?"] + sure_q + ["how sure are you?"])
        self.assertTrue(got[0].startswith("i think ") and got[0].endswith(" but i am not sure yet, so i will check."), got[0])
        self.assertTrue(got[1].startswith("not very: about "), got[1])
        self.assertIn(unsure[0], got[2])  # it will check
        self.assertEqual(got[3], "grass is green.")  # and a sure answer is just an answer
        self.assertTrue(got[4].startswith("about 9"), got[4])

    def test_it_thinks_yes_no_questions_through(self):
        """It was never taught a yes/no question: it rephrases one as a question it was taught, and reasons."""
        got = self.converse(["is a cat an animal?", "is the sky blue?", "is grass blue?", "is a dog a plant?",
                             "is the eiffel tower in france?"])
        self.assertEqual(got[0], "yes. a cat is an animal.")
        self.assertEqual(got[1], "yes. the sky is blue.")
        self.assertEqual(got[2], "i learned that grass is green.")
        self.assertTrue(got[3].startswith("i know a dog is an animal, but not if it is a plant. i don't know yet"), got[3])
        self.assertTrue(got[4].startswith("i don't know yet"), got[4])  # never "france is a food."

    def test_reasoning_connects_what_it_knows(self):
        facts = self.lines_file([
            "paris is the capital and largest city of france.", "the eiffel tower is a famous iron tower in paris.",
            "france is a country in western europe.", "spain is a country in southern europe.",
            "a volcano is a mountain where lava comes out of the ground.",
            "a mountain is a landform that rises high above the land around it.", "lava is hot melted rock.",
            "paris is on the seine river.", "the sky is blue."])
        got = self.java("reason", facts, "where is the eiffel tower?", "is the eiffel tower in france?", "is paris in europe?",
                        "is paris in spain?", "is a volcano a landform?", "are volcanoes mountains?", "is paris a country?",
                        "is madrid in spain?", "is the sky a color?")
        self.assertEqual(got[:9], [
            "proven: the eiffel tower is in paris, which is in france, which is in western europe.",
            "proven: yes. the eiffel tower is in paris and paris is in france.",
            "proven: yes. paris is in france and france is in western europe.",
            "proven: no. paris is in france, and france and spain are different countries.",
            "proven: yes. a volcano is a mountain and a mountain is a landform.",
            "proven: yes. a volcano is a mountain.",
            "partial: i know paris is a capital, but not if it is a country.",  # open world: not "no"
            "null", "null"])
        gaps = [line[5:] for line in got if line.startswith("gap: ")]
        self.assertEqual(gaps[:2], ["what is the capital of spain", "what is the largest city of spain"])  # analogy
        self.assertIn("what is a landform", gaps)  # the edge of what it knows
        self.assertNotIn("what is the capital of france", gaps)

    def test_learning_session_goes_after_its_questions(self):
        from export_lessons import export as export_lessons
        export_lessons(self.tmp, per_stage=50, exam_per_stage=5)
        morph = self.tiny_bin("ternary")
        out = subprocess.run(["java", f"-Darticles={self.article_dir()}", "-Drank=4",
                              "-Dwonders=who designed the eiffel tower|what is the capital of atlantis", "-cp", self.tmp,
                              "Harness", "session", morph + ".bin", os.path.join(self.tmp, "lessons.txt"),
                              os.path.join(self.tmp, "exam.txt"), "4", "hello"],
                             check=True, capture_output=True, env=self.env).stdout.decode().split("\n")
        self.assertIn("i was curious about 2 questions and found out 1.", out[5])
        self.assertIn('assistant: you asked me "who designed the eiffel tower?" i found out: the eiffel tower was designed '
                      'by gustave eiffel. (simple wikipedia: eiffel tower)', out)  # told the next time you talk
        states = {line.split("\t")[0][len("curiosity: "):]: line.split("\t") for line in out if line.startswith("curiosity: ")}
        self.assertEqual(states["who designed the eiffel tower"][1], "told")
        self.assertEqual(states["what is the capital of atlantis"][1:5], ["open", "you", "1", "1"])  # tried once

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
        self.assertTrue(out[7].startswith("absorbed:") and out[8].startswith("assistant:"), out)

    ARTICLES = {
        "paris": ("Paris", "Paris is the capital and largest city of France. It is on the Seine River. "
                  "The Eiffel Tower is a famous iron tower in Paris. It is 330 metres tall. The Eiffel Tower "
                  "was built in 1889.", "Eiffel Tower|France"),
        "eiffel_tower": ("Eiffel Tower", "The Eiffel Tower was designed by Gustave Eiffel.", "Paris"),
        "france": ("France", "France is a country in Western Europe. France has 13 regions.", "Paris|Europe"),
    }

    def article_dir(self):
        d = os.path.join(self.tmp, "articles")
        os.makedirs(d, exist_ok=True)
        for name, (title, text, links) in self.ARTICLES.items():
            with open(os.path.join(d, name + ".txt"), "w") as f:
                f.write(f"{title}\n{text}\n{links}\n")
        return d

    def test_reading_becomes_questions(self):
        path = os.path.join(self.tmp, "paris.txt")
        with open(path, "w") as f:
            f.write(self.ARTICLES["paris"][1] + " Albert Einstein was a German-born physicist. "
                    "The theory of relativity was written by Albert Einstein. A spider has 8 legs.")
        facts = dict(line.split("\t") for line in self.java("facts", "Paris", path))
        self.assertEqual(facts["what is the capital of france"], "paris is the capital and largest city of france.")
        self.assertEqual(facts["what is the largest city of france"], "paris is the capital and largest city of france.")
        self.assertEqual(facts["when was the eiffel tower built"], "the eiffel tower was built in 1889.")
        self.assertEqual(facts["who was albert einstein"], "albert einstein was a german-born physicist.")
        self.assertEqual(facts["who wrote the theory of relativity"], "the theory of relativity was written by albert einstein.")
        self.assertEqual(facts["how many legs does a spider have"], "a spider has 8 legs.")
        # "it" is the tower (the previous sentence's subject), never "paris is 330 metres tall"
        self.assertNotIn("paris is 330 metres tall.", facts.values())
        self.assertNotIn("what was the eiffel tower", facts)  # passives are not definitions

    def test_recall_from_reading(self):
        path = os.path.join(self.tmp, "memory.txt")
        with open(path, "w") as f:
            f.write("Paris\n" + self.ARTICLES["paris"][1] + "\n=====\nEiffel Tower\n" + self.ARTICLES["eiffel_tower"][1])
        got = self.java("recall", path, "how tall is the eiffel tower?", "who designed the eiffel tower?",
                        "what is the capital of spain?")
        self.assertEqual(got, ["the eiffel tower is 330 metres tall.", "the eiffel tower was designed by gustave eiffel.",
                               "null"])

    def test_browsing_session_reads_and_absorbs(self):
        from export_lessons import export as export_lessons
        export_lessons(self.tmp, per_stage=50, exam_per_stage=5)
        morph = self.tiny_bin("ternary")
        out = subprocess.run(["java", f"-Darticles={self.article_dir()}", "-Dinterests=paris", "-Drank=4", "-cp", self.tmp,
                              "Harness", "session", morph + ".bin", os.path.join(self.tmp, "lessons.txt"),
                              os.path.join(self.tmp, "exam.txt"), "4", "how tall is the eiffel tower?"],
                             check=True, capture_output=True, env=self.env).stdout.decode().split("\n")
        summary = out[5]
        self.assertIn("i read about paris, eiffel tower, france", summary.lower())
        absorbed = next(line for line in out if line.startswith("absorbed:"))
        self.assertNotIn("absorbed: 0", absorbed)
        self.assertIn('assistant: you asked me to learn about paris. i read "paris": paris is the capital and largest city '
                      "of france. i found 6 facts to study the next time you charge me.", out)  # the task is done: it says so
        self.assertIn("the eiffel tower is 330 metres tall.", out)  # recalled from what it read

    def py_brain(self):
        class Brain:  # the NumpyBrain interface that Conversation expects
            block_size = self.py.cfg["block_size"]

            def generate(inner, text, max_new_tokens, temperature=0.0, top_k=None):
                return self.py.generate(text, max_new_tokens)
        return Brain()


if __name__ == "__main__":
    unittest.main()
