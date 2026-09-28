"""
Tests for Morpheus. Run with:  python -m unittest tests.test_morpheus -v   (or pytest)
They use tiny untrained models, so they finish in a few seconds on a CPU.
"""

import collections
import os
import random
import tempfile
import unittest

import numpy as np
import torch

from morpheus import tokenizer
from morpheus.compress import dequantize, export, load, pack, quantize, unpack, to_torch
from morpheus.curriculum import STAGES, Library, exam
from morpheus.model import KVCache, Morpheus, MorpheusConfig, build
from morpheus.runtime import NumpyMorpheus
from morpheus.school import Classroom, grade
from morpheus.train import torch_predictor


def tiny(**kw):
    torch.manual_seed(0)
    cfg = dict(n_embd=32, n_layer=2, n_loop=1, n_head=4, n_kv_head=2, block_size=64)
    return Morpheus(MorpheusConfig(**{**cfg, **kw})).eval()


class TestTokenizer(unittest.TestCase):

    def test_vocab(self):
        self.assertEqual(tokenizer.VOCAB_SIZE, 96)
        self.assertEqual(tokenizer.NEWLINE_ID, 0)

    def test_roundtrip(self):
        text = "Hello, Morpheus!\n1 + 2 = 3 ~"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_normalize_folds_unicode(self):
        self.assertEqual(tokenizer.normalize("café\tnaïve\r\n"), "cafe naive\n")
        self.assertEqual(tokenizer.decode(tokenizer.encode_array("héllo☃")), "hello")

    def test_encode_array_matches_encode(self):
        text = "The quick brown fox, 123!\nnew line"
        self.assertEqual(tokenizer.encode_array(text).tolist(), tokenizer.encode(text))
        self.assertEqual(tokenizer.encode_array(text).dtype, np.uint8)


class TestCurriculum(unittest.TestCase):

    def test_lessons_use_the_alphabet_and_are_deterministic(self):
        for stage in STAGES:
            a = [stage.lesson(random.Random(5)) for _ in range(3)]
            b = [stage.lesson(random.Random(5)) for _ in range(3)]
            self.assertEqual(a, b)
            text = "".join(stage.lesson(random.Random(i)) for i in range(300))
            self.assertTrue(set(text) <= set(tokenizer.CHARS), stage.name)

    def test_exam_prompts_have_one_answer(self):
        """No exam prompt may be continued differently by any lesson of any stage,
        otherwise even a perfect student could not pass."""
        answers = collections.defaultdict(set)
        for stage in STAGES:
            for prompt, answer in exam(stage, 3000, seed=1):
                answers[prompt].add(answer)
        self.assertEqual({p: a for p, a in answers.items() if len(a) > 1}, {})
        rng = random.Random(0)
        for stage in STAGES:
            for _ in range(3000):
                text = stage.lesson(rng)
                for start in [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]:
                    seg = text[start:]
                    for j, c in enumerate(seg):
                        prompt = seg[:j + 1]
                        if c in " \n" and prompt in answers:
                            (answer,) = answers[prompt]
                            body = answer.rstrip("\n")
                            self.assertTrue(seg[j + 1:].startswith(body), (stage.name, seg, answer))
                            if answer.endswith("\n"):
                                self.assertIn(seg[j + 1 + len(body):][:1], ("", "\n"), (stage.name, seg))

    def test_some_facts(self):
        from morpheus import curriculum as c
        self.assertEqual(c.number_word(42), "forty two")
        self.assertEqual(c.number_word(13), "thirteen")
        self.assertEqual(c.plural("mouse"), "mice")
        self.assertEqual(c.plural("box"), "boxes")
        self.assertEqual(c.amount(1, "apple"), "1 apple")
        self.assertEqual(c.the_thing("egg"), "an egg")

    def test_classroom_batches(self):
        room = Classroom(STAGES, block_size=64, batch_size=4, replay=0.3)
        rows = room.rows(2)
        self.assertEqual(rows.shape, (4, 65))
        self.assertTrue((rows[:, 0] == tokenizer.NEWLINE_ID).all())
        mixed = Classroom(STAGES[:2], block_size=64, batch_size=64, weights=[1, 0]).rows(None)
        self.assertNotIn(tokenizer.STOI["*"], mixed)  # weight 0: no counting-stars lessons

    def test_library_stage(self):
        lib = Library("once upon a time there was a tiny mind. " * 20)
        self.assertTrue(len(lib.lesson(random.Random(0))) > 0)
        room = Classroom([STAGES[0], lib], block_size=32, batch_size=2, replay=0.0)
        self.assertEqual(room.rows(1).shape, (2, 33))


class TestModel(unittest.TestCase):

    def test_forward_and_loss(self):
        m = tiny()
        x, y = torch.randint(0, 96, (2, 16)), torch.randint(0, 96, (2, 16))
        logits, loss = m(x, y)
        self.assertEqual(tuple(logits.shape), (2, 16, 96))
        self.assertAlmostEqual(loss.item(), np.log(96), delta=0.5)

    def test_kv_cache_matches_full_forward(self):
        for kw in ({}, {"n_loop": 2}, {"n_kv_head": 4}, {"n_kv_head": 1}):
            m = tiny(**kw)
            x = torch.randint(0, 96, (2, 30))
            full, _ = m(x)
            cache = KVCache(m.cfg, 2, "cpu")
            parts = [m(x[:, :11], cache=cache)[0]] + [m(x[:, t:t + 1], cache=cache)[0] for t in range(11, 30)]
            torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-5, rtol=1e-4)

    def test_generate_greedy_matches_argmax(self):
        m = tiny()
        x = torch.randint(0, 96, (1, 10))
        out = m.generate(x, 5)
        seq = x
        for _ in range(5):
            nxt = m(seq)[0][:, -1].argmax(-1, keepdim=True)
            seq = torch.cat((seq, nxt), 1)
        self.assertTrue(torch.equal(out, seq))

    def test_weight_sharing_halves_parameters(self):
        a, b = build("nano"), build("loop")
        self.assertLess(b.num_params(), 0.55 * a.num_params())
        self.assertEqual(a.flops_per_token(64), b.flops_per_token(64))


class TestGrading(unittest.TestCase):

    def test_teacher_forced_grade_equals_greedy_generation(self):
        m = tiny()
        questions = exam(STAGES[1], 20, seed=3)
        _, misses = grade(torch_predictor(m, "cpu", torch.no_grad()), questions, m.cfg.block_size)
        missed = {(p, a) for p, a, _ in misses}
        for prompt, answer in questions:
            ids = torch.tensor([tokenizer.encode("\n" + prompt)])
            out = m.generate(ids, len(answer))
            said = tokenizer.decode(out[0, ids.size(1):].tolist())
            self.assertEqual(said == answer, (prompt, answer) not in missed)

    def test_perfect_predictor_scores_100(self):
        questions = exam(STAGES[2], 50, seed=0)
        lookup = {}
        for p, a in questions:
            ids = tokenizer.encode("\n" + p + a)
            lookup[tuple(ids[:-1])] = ids[1:]

        def oracle(x):
            out = np.zeros_like(x)
            for i, row in enumerate(x):
                for key, nxt in lookup.items():
                    if tuple(row[:len(key)]) == key:
                        out[i, :len(nxt)] = nxt
            return out
        score, misses = grade(oracle, questions, 128)
        self.assertEqual(score, 1.0, misses[:3])


class TestCompression(unittest.TestCase):

    def test_quantization_error(self):
        rng = np.random.default_rng(0)
        w = rng.normal(0, 0.02, size=(48, 80)).astype(np.float32)
        for scheme, tol in (("f32", 0), ("f16", 1e-4), ("int8", 0.02 / 127 * 3), ("int4", 0.02 / 7 * 3)):
            parts, info = quantize(w, scheme)
            back = dequantize(b"".join(parts), w.shape, info)
            self.assertEqual(back.shape, w.shape)
            self.assertLessEqual(np.abs(back - w).max(), tol + 1e-7, scheme)

    def test_every_scheme_is_bit_exact_everywhere(self):
        """Stored file == numpy fake quant == torch fake quant, so what trains is what ships."""
        from morpheus.quant import SCHEMES, fake_quant_np, fake_quant_torch
        rng = np.random.default_rng(3)
        for shape in ((48, 80), (37, 70), (96, 128)):
            w = rng.normal(0, 0.02, size=shape).astype(np.float32)
            for scheme in SCHEMES:
                parts, info = quantize(w, scheme)
                stored = dequantize(b"".join(parts), shape, info)
                np.testing.assert_array_equal(stored, fake_quant_np(w, scheme), err_msg=scheme)
                np.testing.assert_array_equal(fake_quant_torch(torch.from_numpy(w), scheme).numpy(), stored,
                                              err_msg=scheme)

    def test_quantization_is_idempotent(self):
        """A phone re-quantizes a loaded brain every time it learns: that must change nothing."""
        from morpheus.quant import SCHEMES, fake_quant_np
        w = np.random.default_rng(6).normal(0, 0.02, size=(64, 96)).astype(np.float32)
        for scheme in SCHEMES:
            once = fake_quant_np(w, scheme)
            np.testing.assert_array_equal(fake_quant_np(once, scheme), once, err_msg=scheme)

    def test_low_bit_levels(self):
        from morpheus.quant import quantize_np
        w = np.random.default_rng(4).normal(size=(16, 64)).astype(np.float32)
        self.assertEqual(set(np.unique(quantize_np(w, "ternary")[0])), {-1, 0, 1})
        self.assertEqual(set(np.unique(quantize_np(w, "binary")[0])), {-1, 1})
        self.assertLessEqual(np.abs(quantize_np(w, "int3")[0]).max(), 3)
        parts, _ = quantize(w, "ternary")
        self.assertEqual(len(parts[1]), -(-w.size // 5))  # 5 weights per byte
        parts, _ = quantize(w, "binary")
        self.assertEqual(len(parts[1]), w.size // 8)

    def test_fake_quant_passes_gradients_straight_through(self):
        from morpheus.quant import fake_quant_torch
        w = torch.randn(8, 32, requires_grad=True)
        (fake_quant_torch(w, "ternary") * 3).sum().backward()
        torch.testing.assert_close(w.grad, torch.full_like(w, 3.0))

    def test_int4_packs_two_weights_per_byte(self):
        w = np.random.default_rng(1).normal(size=(64, 64)).astype(np.float32)
        parts, _ = quantize(w, "int4")
        self.assertEqual(len(parts[1]), w.size // 2)

    def test_shipped_brains_load_and_answer(self):
        from morpheus.runtime import NumpyMorpheus
        brains = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brains")
        names = [n for n in sorted(os.listdir(brains)) if n.endswith(".morph")] if os.path.isdir(brains) else []
        for name in names:
            brain = NumpyMorpheus.from_file(os.path.join(brains, name))
            self.assertEqual(brain.generate("\nuser: what comes after k?\nmorpheus: "), "after k comes l.", name)

    def test_file_roundtrip(self):
        m = tiny()
        weights = {k: v.numpy() for k, v in m.state_dict().items()}
        from morpheus.model import config_dict
        back, header = unpack(pack(weights, config_dict(m.cfg), "f32", meta={"hi": 1}))
        self.assertEqual(header["meta"], {"hi": 1})
        for k in weights:
            np.testing.assert_array_equal(back[k], weights[k])

    def test_export_sizes_shrink(self):
        m = build("pico")
        sizes = {}
        with tempfile.TemporaryDirectory() as d:
            for scheme in ("f32", "f16", "int8", "int4", "int3", "ternary", "binary"):
                sizes[scheme] = export(m, os.path.join(d, f"{scheme}.morph"), scheme)
        self.assertLess(sizes["binary"], sizes["ternary"])
        self.assertLess(sizes["ternary"], sizes["int3"])
        self.assertLess(sizes["int3"], sizes["int4"])
        self.assertLess(sizes["int4"], sizes["int8"])
        self.assertLess(sizes["int8"], sizes["f16"])
        self.assertLess(sizes["f16"], sizes["f32"])


class TestNumpyRuntime(unittest.TestCase):

    def test_matches_torch(self):
        for kw in ({}, {"n_loop": 2}, {"n_kv_head": 1}):
            m = tiny(**kw)
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "m.morph")
                export(m, path, "f32")
                brain = NumpyMorpheus.from_file(path)
            x = np.random.default_rng(0).integers(0, 96, size=(3, 20))
            ref = m(torch.from_numpy(x))[0].detach().numpy()
            np.testing.assert_allclose(brain.forward(x), ref, atol=1e-4, rtol=1e-4)
            # and incrementally, through the KV cache
            cache = brain.new_cache(3)
            parts = [brain.forward(x[:, :7], cache)] + [brain.forward(x[:, t:t + 1], cache) for t in range(7, 20)]
            np.testing.assert_allclose(np.concatenate(parts, 1), ref, atol=1e-4, rtol=1e-4)

    def test_generate_matches_torch(self):
        m = tiny()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.morph")
            export(m, path, "f32")
            brain = NumpyMorpheus.from_file(path)
            reloaded = to_torch(*load(path))
        prompt = "\nuser: hi\nmorpheus: "
        ids = torch.tensor([tokenizer.encode(prompt)])
        out = reloaded.generate(ids, 12)
        want = tokenizer.decode(out[0, ids.size(1):].tolist()).split("\n")[0]
        self.assertEqual(brain.generate(prompt, 12), want)


if __name__ == "__main__":
    unittest.main()
