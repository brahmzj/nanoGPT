"""
Raising Morpheus: mastery-based curriculum training.

  for each stage (letters -> numbers -> math -> words -> world -> talk [-> library]):
      study fresh lessons, mixed with review of earlier stages (spaced repetition)
      every `exam_every` steps sit a brand-new exam; graduate at `pass_mark`
  dream: review everything while the learning rate fades to zero (consolidation)
  export the brain, int8-compressed, to a .morph file

Mastery gating is the main energy saver: easy stages end as soon as they are
learned instead of burning a fixed step budget, and nothing is ever stored on
disk except the checkpoint (the lessons are generated on the fly).
"""

import math
import os
import random
import time
from contextlib import nullcontext

import torch

from .curriculum import STAGES, exam
from .model import Morpheus, MorpheusConfig, SIZES, config_dict
from .school import Classroom, grade, load_library, print_report_card, report_card


def torch_predictor(model, device, ctx):
    @torch.no_grad()
    def predict(x):
        was_training = model.training
        model.eval()
        with ctx:
            logits, _ = model(torch.from_numpy(x).to(device))
        model.train(was_training)
        return logits.argmax(-1).cpu().numpy()
    return predict


def pick_device(device):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save(path, model, optimizer, state):
    tmp = path + ".tmp"
    torch.save({"config": config_dict(model.cfg), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), **state}, tmp)
    os.replace(tmp, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Morpheus(MorpheusConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


def train(args):
    device = pick_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else device
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if device_type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    ctx = torch.autocast(device_type=device_type, dtype=dtype) if dtype != torch.float32 else nullcontext()
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "morpheus.pt")

    stages = [s for s in STAGES if args.stages == "all" or s.name in args.stages.split(",")]
    if args.library:
        stages.append(load_library(args.library))

    # --- a new mind, or wake up an old one
    if args.resume and os.path.exists(ckpt_path):
        model, ckpt = load_model(ckpt_path, device)
        state = {k: ckpt[k] for k in ("stage", "step", "history", "seconds", "dreamed")}
        print(f"resuming from {ckpt_path}: {state['stage']} stage(s) done, step {state['step']}")
    else:
        cfg = MorpheusConfig(**{**SIZES[args.size], "block_size": args.block_size, "dropout": args.dropout})
        model = Morpheus(cfg).to(device)
        ckpt, state = None, {"stage": 0, "step": 0, "history": [], "seconds": 0.0, "dreamed": False}
    optimizer = model.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.99), device_type)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    cfg = model.cfg
    predict = torch_predictor(model, device, ctx)
    raw_model = model
    if args.compile:
        model = torch.compile(model)

    print(f"Morpheus ({args.size}): {raw_model.num_params():,} parameters, depth {cfg.depth}, "
          f"context {cfg.block_size}, on {device}")
    classroom = Classroom(stages, cfg.block_size, args.batch_size, args.replay, seed=args.seed + state["step"])
    flops_per_step = raw_model.flops_per_token(cfg.block_size, training=True) * args.batch_size * cfg.block_size

    def step(stage_idx, lr):
        for group in optimizer.param_groups:
            group["lr"] = lr
        data = torch.from_numpy(classroom.rows(stage_idx))
        if device_type == "cuda":
            data = data.pin_memory().to(device, non_blocking=True)
        else:
            data = data.to(device)
        x, y = data[:, :-1], data[:, 1:]
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state["step"] += 1
        return loss

    def warm_lr():
        return args.lr * min(1.0, (state["step"] + 1) / args.warmup)

    def energy():
        wh = state["seconds"] * args.watts / 3600
        return f"{state['seconds']:.0f}s, ~{wh:.2f} Wh at {args.watts:g} W"

    # --- school: one stage at a time, graduating on mastery
    for stage_idx in range(state["stage"], len(stages)):
        stage = stages[stage_idx]
        has_exam = bool(stage._exam_facts)
        print(f"\n== stage {stage_idx + 1}/{len(stages)}: {stage.name} ({stage.title}) ==")
        t0, local, passed, score = time.time(), 0, False, None
        lr_scale, best, stale, recent = 1.0, -1.0, 0, []  # each stage starts at full speed
        budget = args.max_steps if has_exam else args.library_steps
        while local < budget:
            loss = step(stage_idx, warm_lr() * lr_scale)
            local += 1
            if local % args.log_every == 0:
                dt = time.time() - t0
                print(f"  step {local:5d}  loss {loss.item():.4f}  {dt / local * 1000:.0f}ms/step")
            if has_exam and local % args.exam_every == 0:
                score, misses = grade(predict, exam(stage, args.exam_size, seed=state["step"]), cfg.block_size)
                print(f"  exam after {local} steps: {score * 100:.1f}%"
                      + (f"   e.g. {(misses[0][0] + misses[0][1]).strip()!r:.50} -> said {misses[0][2].strip()!r:.20}"
                         if misses else ""))
                if score >= args.pass_mark:
                    passed = True
                    break
                # stuck on a plateau? take smaller steps (noise near mastery comes from a high lr).
                # Judge progress on the average of the last 3 exams: one lucky exam is not a trend.
                recent = (recent + [score])[-3:]
                smooth = sum(recent) / len(recent)
                best, stale = (smooth, 0) if smooth > best + 0.01 else (best, stale + 1)
                if stale >= args.patience and lr_scale > args.min_lr_scale:
                    lr_scale, stale = max(lr_scale / 2, args.min_lr_scale), 0
                    print(f"  plateau: slowing down, lr x{lr_scale:g}")
        seconds = time.time() - t0
        state["seconds"] += seconds
        verdict = "graduated" if passed else ("finished reading" if not has_exam else "moving on (did not pass)")
        print(f"  {verdict}: {stage.name} in {local} steps, {seconds:.0f}s, "
              f"{flops_per_step * local / 1e12:.1f} TFLOP")
        state["history"].append({"stage": stage.name, "steps": local, "passed": passed,
                                 "score": score, "seconds": round(seconds, 1)})
        state["stage"] = stage_idx + 1
        state["dreamed"] = False  # anything newly learned needs consolidating
        save(ckpt_path, raw_model, optimizer, state)

    # --- dream: consolidate everything while the learning rate fades out
    if not state["dreamed"] and args.dream_steps > 0:
        print(f"\n== dreaming: reviewing all {len(stages)} stages for {args.dream_steps} steps ==")
        t0 = time.time()
        for i in range(args.dream_steps):
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * i / args.dream_steps))
            loss = step(None, lr)
            if (i + 1) % args.log_every == 0:
                print(f"  dream {i + 1:5d}  loss {loss.item():.4f}  lr {lr:.2e}")
        state["seconds"] += time.time() - t0
        state["dreamed"] = True
        save(ckpt_path, raw_model, optimizer, state)

    rows = report_card(predict, stages, cfg.block_size)
    print_report_card(rows)
    print(f"total training: {state['step']} steps, {energy()}")

    from .compress import export
    morph_path = os.path.join(args.out_dir, "morpheus.morph")
    nbytes = export(raw_model, morph_path, scheme=args.export_scheme,
                    meta={"history": state["history"], "report": {n: s for n, s, _ in rows}})
    print(f"compressed brain ({args.export_scheme}): {morph_path} ({nbytes / 1024:.0f} KB)")
    print(f"talk to it:  python -m morpheus chat --model {morph_path}")
    return raw_model, rows
