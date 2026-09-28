"""
Command line for Morpheus.

  python -m morpheus train  [--size nano] [--library my_books/]
  python -m morpheus chat   [--model out-morpheus/morpheus.morph] [--ask "what is 3 + 4?"]
  python -m morpheus exam   [--model ...]
  python -m morpheus export [--ckpt out-morpheus/morpheus.pt] [--scheme int4]
  python -m morpheus stats  [--size nano | --model ...]
"""

import argparse
import os
import sys

from . import tokenizer

DEFAULT_DIR = "out-morpheus"


# ----------------------------------------------------------------------------
# a "brain" is either a torch checkpoint (.pt) or a compressed numpy brain (.morph)


class TorchBrain:

    def __init__(self, path, device="cpu"):
        import torch
        from .train import load_model
        self.torch = torch
        self.model, ckpt = load_model(path, device)
        self.model.eval()
        self.device = device
        self.block_size = self.model.cfg.block_size
        self.info = {"history": ckpt.get("history", [])}

    def predict(self, x):
        with self.torch.no_grad():
            logits, _ = self.model(self.torch.from_numpy(x).to(self.device))
        return logits.argmax(-1).cpu().numpy()

    def generate(self, text, max_new_tokens, temperature=0.0, top_k=None):
        max_new_tokens = min(max_new_tokens, self.block_size - 1)
        ids = tokenizer.encode(tokenizer.normalize(text))[-(self.block_size - max_new_tokens):]
        idx = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        out = self.model.generate(idx, max_new_tokens, temperature, top_k, stop_id=tokenizer.NEWLINE_ID)
        new = tokenizer.decode(out[0, idx.size(1):].tolist()) if out.size(1) > idx.size(1) else ""
        return new.split(tokenizer.NEWLINE)[0]


class NumpyBrain:

    def __init__(self, path):
        from .runtime import NumpyMorpheus
        self.model = NumpyMorpheus.from_file(path)
        self.block_size = self.model.cfg["block_size"]
        self.info = self.model.header.get("meta", {})

    def predict(self, x):
        return self.model.predict(x)

    def generate(self, text, max_new_tokens, temperature=0.0, top_k=None):
        return self.model.generate(text, max_new_tokens, temperature, top_k)


def find_model(path):
    if path:
        return path
    for name in ("morpheus.morph", "morpheus.pt"):
        candidate = os.path.join(DEFAULT_DIR, name)
        if os.path.exists(candidate):
            return candidate
    sys.exit(f"no trained Morpheus found in {DEFAULT_DIR}/. Raise one first:  python -m morpheus train")


def load_brain(path, device="cpu"):
    path = find_model(path)
    return (TorchBrain(path, device) if path.endswith(".pt") else NumpyBrain(path)), path


# ----------------------------------------------------------------------------
# commands


def cmd_train(args):
    from .train import train
    train(args)


class Conversation:
    """Keeps recent turns as context, dropping the oldest ones when the context is full."""

    def __init__(self, brain, reserve=64, temperature=0.0, top_k=None):
        self.brain = brain
        self.reserve = reserve  # characters kept free for the answer
        self.temperature, self.top_k = temperature, top_k
        self.turns = []

    def ask(self, message):
        message = tokenizer.normalize(message).lower().strip()
        turn = f"user: {message}\nmorpheus: "
        budget = self.brain.block_size - self.reserve
        while self.turns and len(tokenizer.NEWLINE + tokenizer.NEWLINE.join(self.turns + [turn])) > budget:
            self.turns.pop(0)
        context = (tokenizer.NEWLINE + tokenizer.NEWLINE.join(self.turns + [turn]))[-budget:]
        reply = self.brain.generate(context, self.brain.block_size - len(context), self.temperature, self.top_k)
        self.turns.append(turn + reply)
        return reply


def cmd_chat(args):
    brain, path = load_brain(args.model, args.device)
    chat = Conversation(brain, temperature=args.temperature, top_k=args.top_k)
    if args.ask:
        print(chat.ask(args.ask))
        return
    print(f"Morpheus is awake ({path}). Type 'bye' to leave, '/reset' to start over.\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message == "/reset":
            chat.turns.clear()
            print("(forgotten)\n")
            continue
        print(f"morpheus> {chat.ask(message)}\n")
        if message.lower().strip("!. ") in ("bye", "goodbye", "good night", "quit", "exit"):
            break


def cmd_exam(args):
    from .curriculum import STAGES
    from .train import report_card, print_report_card
    brain, path = load_brain(args.model, args.device)
    stages = [s for s in STAGES if args.stages == "all" or s.name in args.stages.split(",")]
    rows = report_card(brain.predict, stages, brain.block_size, n=args.n, seed=args.seed)
    print_report_card(rows, title=f"report card: {path}")
    overall = sum(score for _, score, _ in rows) / max(1, len(rows))
    print(f"  overall: {overall * 100:.1f}%")


def cmd_export(args):
    from .compress import export, load
    from .train import load_model
    model, _ = load_model(args.ckpt)
    out = args.out or os.path.splitext(args.ckpt)[0] + ("" if args.scheme == "int8" else f"-{args.scheme}") + ".morph"
    nbytes = export(model, out, scheme=args.scheme)
    fp32 = model.num_params() * 4
    print(f"{out}: {nbytes / 1024:.0f} KB  ({fp32 / nbytes:.1f}x smaller than float32's {fp32 / 1024:.0f} KB)")
    if args.grade:
        from .curriculum import STAGES
        from .runtime import NumpyMorpheus
        from .train import report_card, print_report_card
        brain = NumpyMorpheus(*load(out))
        print_report_card(report_card(brain.predict, STAGES, brain.cfg["block_size"]), f"after {args.scheme}")


def cmd_stats(args):
    from .model import build, Morpheus, MorpheusConfig
    if args.model:
        brain, _ = load_brain(args.model)
        cfg = brain.model.cfg if isinstance(brain, TorchBrain) else MorpheusConfig(**brain.model.cfg)
        model = Morpheus(cfg)
    else:
        model = build(args.size)
    cfg, n = model.cfg, model.num_params()
    gpt2_flops = 2 * 124e6 + 4 * 12 * 768 * 64  # GPT-2 small, one token at context 64
    ours = model.flops_per_token(64)
    print(f"Morpheus: {n:,} parameters | {cfg.n_layer} blocks x {cfg.n_loop} loop(s) = depth {cfg.depth} | "
          f"width {cfg.n_embd} | {cfg.n_head} heads ({cfg.n_kv_head} kv) | context {cfg.block_size}")
    print(f"  compute: {ours / 1e6:.2f} MFLOP per character (at context 64), "
          f"~{gpt2_flops / ours:.0f}x less than GPT-2 small per token")
    print("  brain size before LZMA:  " + "   ".join(
        f"{name} {n * bits / 8 / 1024:.0f} KB" for name, bits in
        (("f32", 32), ("f16", 16), ("int8", 8 + 16 / cfg.n_embd), ("int4", 4 + 16 / 32))))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m morpheus", description="Morpheus: a tiny mind raised from abc and 123")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="raise Morpheus from scratch, one stage at a time")
    t.add_argument("--size", default="nano", choices=["pico", "nano", "loop", "micro"])
    t.add_argument("--out_dir", default=DEFAULT_DIR)
    t.add_argument("--resume", action="store_true", help="continue from out_dir/morpheus.pt")
    t.add_argument("--stages", default="all", help="comma separated subset, e.g. letters,numbers")
    t.add_argument("--library", default="", help="a .txt file or folder of your own text to read after school")
    t.add_argument("--library_steps", type=int, default=2000)
    t.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    t.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    t.add_argument("--compile", action="store_true")
    t.add_argument("--block_size", type=int, default=128)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--weight_decay", type=float, default=0.1)
    t.add_argument("--grad_clip", type=float, default=1.0)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--replay", type=float, default=0.3, help="fraction of lessons that review earlier stages")
    t.add_argument("--pass_mark", type=float, default=0.95, help="exam score needed to graduate a stage")
    t.add_argument("--exam_every", type=int, default=100)
    t.add_argument("--exam_size", type=int, default=200)
    t.add_argument("--max_steps", type=int, default=4000, help="give up on a stage after this many steps")
    t.add_argument("--patience", type=int, default=3, help="exams without progress before halving the lr")
    t.add_argument("--min_lr_scale", type=float, default=0.125, help="never slow below this fraction of --lr")
    t.add_argument("--dream_steps", type=int, default=1000, help="final review with a fading learning rate")
    t.add_argument("--log_every", type=int, default=50)
    t.add_argument("--watts", type=float, default=25.0, help="your machine's power draw, for the energy estimate")
    t.add_argument("--export_scheme", default="int8", choices=["f32", "f16", "int8", "int4"])
    t.add_argument("--seed", type=int, default=1337)
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("chat", help="talk to Morpheus")
    c.add_argument("--model", default="", help=".morph (numpy only) or .pt (torch) file")
    c.add_argument("--ask", default="", help="ask one question and exit")
    c.add_argument("--temperature", type=float, default=0.0, help="0 = always the most likely answer")
    c.add_argument("--top_k", type=int, default=None)
    c.add_argument("--device", default="cpu")
    c.set_defaults(fn=cmd_chat)

    e = sub.add_parser("exam", help="grade Morpheus on every stage")
    e.add_argument("--model", default="")
    e.add_argument("--stages", default="all")
    e.add_argument("--n", type=int, default=200, help="questions per stage")
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--device", default="cpu")
    e.set_defaults(fn=cmd_exam)

    x = sub.add_parser("export", help="compress a checkpoint into a .morph file")
    x.add_argument("--ckpt", default=os.path.join(DEFAULT_DIR, "morpheus.pt"))
    x.add_argument("--scheme", default="int8", choices=["f32", "f16", "int8", "int4"])
    x.add_argument("--out", default="")
    x.add_argument("--grade", action="store_true", help="sit the exams with the compressed brain")
    x.set_defaults(fn=cmd_export)

    s = sub.add_parser("stats", help="size, compute and storage budget")
    s.add_argument("--size", default="nano", choices=["pico", "nano", "loop", "micro"])
    s.add_argument("--model", default="")
    s.set_defaults(fn=cmd_stats)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
