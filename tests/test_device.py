"""
Tests for Morpheus on a device: numpy-only learning, lifelong-learning sessions and the
assistant skills. Run with:  python -m unittest tests.test_device -v
"""

import contextlib
import datetime as dt
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np
import torch

from morpheus import assistant as asst
from morpheus.compress import export, load
from morpheus.curriculum import UNKNOWN_ANSWER
from morpheus.grad import NumpyTrainer
from morpheus.learn import Home, gather, html_to_text, session
from morpheus.model import Morpheus, MorpheusConfig, config_dict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def tiny_model(**kw):
    torch.manual_seed(0)
    cfg = dict(n_embd=32, n_layer=2, n_loop=1, n_head=4, n_kv_head=2, block_size=64)
    return Morpheus(MorpheusConfig(**{**cfg, **kw}))


class TestNumpyTraining(unittest.TestCase):

    def test_gradients_match_torch(self):
        for kw in ({}, {"n_loop": 2}, {"n_kv_head": 1}):
            m = tiny_model(**kw).double()
            x, y = torch.randint(0, 96, (3, 20)), torch.randint(0, 96, (3, 20))
            _, loss = m(x, y)
            loss.backward()
            tr = NumpyTrainer({k: v.detach().float().numpy() for k, v in m.state_dict().items()}, config_dict(m.cfg))
            np_loss, grads = tr.loss_and_grads(x.numpy(), y.numpy())
            self.assertAlmostEqual(np_loss, loss.item(), places=4)
            for name, p in m.named_parameters():
                np.testing.assert_allclose(grads[name], p.grad.numpy(), atol=2e-6, rtol=1e-3, err_msg=name)

    def test_adamw_matches_torch(self):
        m = tiny_model()
        tr = NumpyTrainer({k: v.detach().numpy().copy() for k, v in m.state_dict().items()}, config_dict(m.cfg),
                          lr=1e-2, weight_decay=0.1, betas=(0.9, 0.99), grad_clip=1.0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1, betas=(0.9, 0.99))
        for _ in range(3):
            x, y = torch.randint(0, 96, (2, 16)), torch.randint(0, 96, (2, 16))
            _, loss = m(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            tr.step(x.numpy(), y.numpy())
        for k, v in m.state_dict().items():
            np.testing.assert_allclose(tr.w[k], v.numpy(), atol=1e-4, err_msg=k)

    def test_learning_and_save_load(self):
        tr = NumpyTrainer({k: v.numpy() for k, v in tiny_model().state_dict().items()}, config_dict(tiny_model().cfg),
                          lr=3e-3)
        x = np.tile(np.arange(20) % 7, (4, 1))
        first = tr.step(x[:, :-1], x[:, 1:])
        for _ in range(40):
            last = tr.step(x[:, :-1], x[:, 1:])
        self.assertLess(last, first * 0.5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            tr.save(path)
            back = NumpyTrainer.load(path)
        self.assertEqual(back.t, tr.t)
        for k in tr.w:
            np.testing.assert_array_equal(back.w[k], tr.w[k])
            np.testing.assert_array_equal(back.m[k], tr.m[k])


class TestCompressedLearning(unittest.TestCase):

    def test_quantized_brain_learns_in_its_own_format(self):
        from morpheus.compress import unpack
        from morpheus.runtime import NumpyMorpheus
        m = tiny_model()
        tr = NumpyTrainer({k: v.numpy() for k, v in m.state_dict().items()}, config_dict(m.cfg), lr=3e-3,
                          quant={"scheme": "ternary", "embed_scheme": "int8"})
        x = np.tile(np.arange(20) % 7, (4, 1))
        first = tr.step(x[:, :-1], x[:, 1:])
        for _ in range(40):
            last = tr.step(x[:, :-1], x[:, 1:])
        self.assertLess(last, first * 0.5)
        weights, header = unpack(tr.export())
        self.assertEqual((header["scheme"], header["embed_scheme"]), ("ternary", "int8"))
        ids = np.random.default_rng(0).integers(0, 96, (2, 24))
        np.testing.assert_allclose(NumpyMorpheus(weights, header).forward(ids), tr.forward(ids, keep=False)[0],
                                   atol=1e-5)

    def test_squeeze_end_to_end(self):
        import argparse
        from morpheus.train import squeeze
        with tempfile.TemporaryDirectory() as d:
            m = tiny_model()
            ckpt = os.path.join(d, "m.pt")
            torch.save({"config": config_dict(m.cfg), "model": m.state_dict()}, ckpt)
            args = argparse.Namespace(ckpt=ckpt, scheme="ternary", embed_scheme="int8", out="", steps=4, lr=1e-3,
                                      warmup=2, weight_decay=0.0, grad_clip=1.0, batch_size=2, distill=0.5,
                                      tolerance=0.01, eval_every=2, anneal=2, exam_size=2, log_every=2, watts=25.0,
                                      device="cpu", threads=1, seed=0)
            with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
                out = squeeze(args)
            weights, header = load(out)
            self.assertEqual(header["scheme"], "ternary")
            self.assertTrue(os.path.exists(os.path.splitext(out)[0] + ".pt"))
            # a squeezed brain keeps learning on a device in the same format
            home = Home(os.path.join(d, "home"))
            with open(home.file("brain.morph"), "wb") as f, open(out, "rb") as g:
                f.write(g.read())
            trainer, _ = home.load_trainer()
            self.assertEqual(trainer.quant["scheme"], "ternary")
            # and thinks exactly like the file it was loaded from
            from morpheus.runtime import NumpyMorpheus
            ids = np.random.default_rng(1).integers(0, 96, (2, 24))
            np.testing.assert_allclose(trainer.forward(ids, keep=False)[0],
                                       NumpyMorpheus(weights, header).forward(ids), atol=1e-5)


class _Server(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/summary"):
            body, kind = json.dumps({"title": "Volcano", "extract": "A volcano is a mountain that lets hot "
                                     "melted rock out of the ground. Volcanoes can be very dangerous."}), "application/json"
        else:
            body, kind = ("<html><head><style>x{}</style></head><body><p>The owl sleeps in the day.</p>"
                          "<script>var no = 1;</script><p>At night the owl hunts mice.</p></body></html>"), "text/html"
        self.send_response(200)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


class TestLearningSessions(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Home(os.path.join(self.tmp.name, "home"))
        export(tiny_model(), self.home.file("brain.morph"), "int8")
        self.home.set("only_when_charging", False)
        self.home.set("session_steps", 6)
        self.home.set("exam_questions", 5)
        self.home.set("inbox", json.dumps(["{home}/inbox"]))

    def tearDown(self):
        self.tmp.cleanup()

    def test_internet_is_off_by_default(self):
        self.assertFalse(Home(os.path.join(self.tmp.name, "fresh")).settings["allow_internet"])

    def test_session_reads_inbox_and_learns(self):
        with open(self.home.file("inbox", "story.md"), "w") as f:
            f.write("Once upon a time a tiny mind learned its ABC. " * 5)
        self.home.teach("what is my dog called?", "your dog is called rex.")
        self.home.set("max_forgetting", 1.0)  # a tiny random brain: keep whatever happens
        summary = session(self.home, seed=0, log=lambda *a: None)
        self.assertTrue(summary["kept"])
        self.assertEqual(summary["new_sources"], 1)
        self.assertIsNotNone(summary["taught_after"])
        self.assertTrue(os.path.exists(self.home.file("brain-train.npz")))
        self.assertIn("once upon a time", self.home.library_text())
        # the same file is never read twice
        self.assertEqual(gather(self.home, log=lambda *a: None), 0)
        # and learning continues from the saved training state
        self.assertTrue(self.home.brain_source().endswith("brain-train.npz"))

    def test_forgetting_rolls_back(self):
        before = read_bytes(self.home.file("brain.morph"))
        self.home.set("max_forgetting", -1.0)  # demand an impossible improvement
        summary = session(self.home, seed=0, log=lambda *a: None)
        self.assertFalse(summary["kept"])
        self.assertEqual(read_bytes(self.home.file("brain.morph")), before)
        self.assertFalse(os.path.exists(self.home.file("brain-train.npz")))

    def test_skips_on_battery(self):
        self.home.set("only_when_charging", True)
        from morpheus import learn
        original = learn.power_status
        learn.power_status = lambda: (False, 80)
        try:
            self.assertIn("skipped", session(self.home, log=lambda *a: None))
        finally:
            learn.power_status = original

    def test_undo(self):
        self.home.set("max_forgetting", 1.0)
        first = read_bytes(self.home.file("brain.morph"))
        session(self.home, seed=0, log=lambda *a: None)
        self.assertNotEqual(read_bytes(self.home.file("brain.morph")), first)
        self.home.undo()
        self.assertEqual(read_bytes(self.home.file("brain.morph")), first)

    def test_fetch_only_when_allowed(self):
        server = http.server.HTTPServer(("127.0.0.1", 0), _Server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_port}"
        self.home.set("sources", json.dumps([base + "/summary", base + "/page.html"]))
        try:
            self.assertEqual(gather(self.home, log=lambda *a: None), 0)  # internet not allowed
            self.home.set("allow_internet", "on")
            self.assertEqual(gather(self.home, log=lambda *a: None), 2)
        finally:
            server.shutdown()
            server.server_close()
        text = self.home.library_text()
        self.assertIn("a volcano is a mountain", text)
        self.assertIn("the owl sleeps", text)
        self.assertNotIn("var no", text)

    def test_html_to_text(self):
        self.assertEqual(html_to_text("<p>a &amp; b</p><style>p{}</style>").strip(), "a & b")


class FakeBrain:
    block_size = 128

    def __init__(self, answers=None):
        self.answers = answers or {}

    def generate(self, context, max_new_tokens, temperature=0.0, top_k=None):
        question = context.rsplit("user: ", 1)[-1].split("\n")[0]
        return self.answers.get(question, UNKNOWN_ANSWER)


class TestAssistant(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Home(self.tmp.name)
        self.clock = dt.datetime(2026, 9, 28, 14, 5)
        self.brain = FakeBrain({"what is 3 + 4?": "3 + 4 = 7.", "what is 6 x 7?": "6 x 7 = 41."})
        self.a = asst.Assistant(self.brain, self.home, now=lambda: self.clock)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parsers(self):
        self.assertEqual(asst.parse_math("what is 12 times 7?"), "12*7")
        self.assertEqual(asst.parse_math("20% of 80"), "20/100*80")
        self.assertEqual(asst.parse_math("(3 + 4) x 2"), "(3+4)*2")
        self.assertIsNone(asst.parse_math("what is a cat?"))
        self.assertIsNone(asst.parse_math("count to 5"))
        self.assertEqual(asst.safe_eval("2**10"), 1024)
        with self.assertRaises(ValueError):
            asst.safe_eval("__import__('os')")
        self.assertEqual(asst.parse_clock("5:30 pm"), (17, 30))
        self.assertEqual(asst.parse_clock("12 am"), (0, 0))
        self.assertEqual(asst.parse_clock("noon"), (12, 0))
        self.assertIsNone(asst.parse_clock("25:00"))
        self.assertEqual(asst.parse_duration("1 hour and 30 minutes"), 5400)
        self.assertEqual(asst.parse_duration("an hour"), 3600)

    def test_notes(self):
        self.assertEqual(self.a.respond("remember that my locker code is 1234"),
                         "ok, i will remember that your locker code is 1234.")
        self.assertEqual(self.a.respond("what is my locker code?"), "your locker code is 1234.")
        self.assertIn("forgot 1", self.a.respond("forget my locker code"))
        self.assertEqual(self.a.respond("what is my locker code?"), UNKNOWN_ANSWER)

    def test_todo(self):
        self.a.respond("add milk to my list")
        self.a.respond("add call the bank to my todo list")
        self.assertEqual(self.a.respond("what's on my list?"), "on your list: milk, call the bank.")
        self.assertIn("removed milk", self.a.respond("remove milk from my list"))
        self.assertIn("removed call the bank", self.a.respond("done calling the bank"))

    def test_clock(self):
        self.assertEqual(self.a.respond("hey morpheus, what time is it?"), "it is 2:05 pm.")
        self.assertEqual(self.a.respond("what is the date?"), "today is monday, september 28, 2026.")

    def test_reminders(self):
        self.assertIn("at 5:00 pm", self.a.respond("remind me to call mom at 5pm"))
        self.assertIn("timer set for 10 minutes", self.a.respond("set a timer for 10 minutes"))
        self.assertIn("alarm set for 7:30 am", self.a.respond("set an alarm for 7:30 am"))
        self.clock = dt.datetime(2026, 9, 28, 17, 1)
        self.assertTrue(self.a.respond("hello").startswith("(reminder: call mom)"))
        self.assertNotIn("reminder", self.a.respond("hello"))  # announced once

    def test_math_brain_first_calculator_checks(self):
        self.assertEqual(self.a.respond("what is 3 + 4?"), "3 + 4 = 7.")  # the brain was right
        self.assertEqual(self.a.respond("what is 6 x 7?"), "6 x 7 = 42.")  # the brain said 41
        self.assertIn(("what is 6 x 7?", "6 x 7 = 42."), self.home.taught())  # ...a lesson for later
        self.assertEqual(self.a.respond("1234 * 5678"), "1234 x 5678 = 7006652.")
        self.assertEqual(self.a.respond("what is 7 divided by 0?"), "you can not divide by zero.")

    def test_checked_comparisons_and_stories(self):
        self.brain.answers["which is bigger, 73 or 37?"] = "the bigger of 73 and 37 is 77."
        self.brain.answers["mia has 4 cookies and gets 3 more. how many cookies now?"] = "leo has 7 cookies."
        self.assertEqual(self.a.respond("which is bigger, 73 or 37?"), "the bigger of 73 and 37 is 73.")
        self.assertEqual(self.a.respond("mia has 4 cookies and gets 3 more. how many cookies now?"),
                         "mia has 7 cookies.")
        self.assertEqual(len(self.home.taught()), 2)
        self.assertEqual(self.a.respond("15 minus 6"), "15 - 6 = 9.")

    def test_lookup_needs_permission(self):
        self.assertIn("allow the internet", self.a.respond("tell me about volcanoes"))

    def test_brain_answers_the_rest(self):
        self.assertEqual(self.a.respond("what is 3 + 4?"), "3 + 4 = 7.")
        self.assertEqual(self.a.respond("what is the meaning of life?"), UNKNOWN_ANSWER)


class TestWithoutTorch(unittest.TestCase):
    """What runs on a phone must not need PyTorch."""

    def test_chat_and_learn_without_torch(self):
        with tempfile.TemporaryDirectory() as d:
            home = os.path.join(d, "home")
            h = Home(home)
            export(tiny_model(), h.file("brain.morph"), "int8")
            h.set("session_steps", 2)
            h.set("exam_questions", 2)
            h.set("only_when_charging", False)
            h.set("max_forgetting", 1.0)
            code = ("import sys; sys.modules['torch'] = None\n"
                    "from morpheus.__main__ import main\n"
                    f"main(['chat', '--home', {home!r}, '--ask', 'remember that my bike is blue'])\n"
                    f"main(['chat', '--home', {home!r}, '--ask', 'what is 1234 * 2?'])\n"
                    f"main(['learn', '--home', {home!r}, '--quiet'])\n"
                    f"main(['status', '--home', {home!r}])\n")
            out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                                 env={**os.environ, "MORPHEUS_HOME": home})
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("your bike is blue", out.stdout)
            self.assertIn("1234 x 2 = 2468.", out.stdout)
            self.assertIn("kept", out.stdout)


if __name__ == "__main__":
    unittest.main()
