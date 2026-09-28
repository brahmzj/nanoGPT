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


def squeeze(args):
    """Compress to the extreme and keep learning: quantization-aware training with distillation.

    A copy of the brain (the student) sees only its quantized weights in every forward pass,
    while updates go to the float weights underneath (straight-through estimator). It studies
    the whole curriculum, learning both from the lessons and from its own full-precision self
    (the teacher). Exams are sat by the *compressed* student, and it stops as soon as it is back
    within `tolerance` of the teacher (mastery gating, again, to save energy).
    """
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import parametrize
    from .compress import export, load as load_morph
    from .quant import FakeQuant, scheme_for
    from .runtime import NumpyMorpheus

    device = pick_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else device
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    teacher, _ = load_model(args.ckpt, device)
    teacher.eval()
    student = copy.deepcopy(teacher).train()
    for name, module in student.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            scheme = scheme_for(f"{name}.weight", args.scheme, args.embed_scheme)
            if scheme not in ("f32", "f16"):
                parametrize.register_parametrization(module, "weight", FakeQuant(scheme))
    cfg = student.cfg
    stages = list(STAGES)
    predict_t = torch_predictor(teacher, device, nullcontext())
    predict_s = torch_predictor(student, device, nullcontext())

    def score(predict, seed):
        rows = report_card(predict, stages, cfg.block_size, n=args.exam_size, seed=seed)
        return sum(r[1] for r in rows) / len(rows), rows

    teacher_score, _ = score(predict_t, 99)
    start_score, _ = score(predict_s, 99)
    target = teacher_score - args.tolerance
    print(f"squeezing to {args.scheme}" + (f" (embedding {args.embed_scheme})" if args.embed_scheme else "")
          + f": teacher {teacher_score * 100:.1f}%, rounded student starts at {start_score * 100:.1f}%, "
          f"target {target * 100:.1f}%")

    optimizer = student.configure_optimizer(args.lr, args.weight_decay, (0.9, 0.99), device_type)
    classroom = Classroom(stages, cfg.block_size, args.batch_size, seed=args.seed)
    best, best_state, t0 = start_score, copy.deepcopy(student.state_dict()), time.time()
    T, alpha = 1.0, args.distill
    for step in range(1, args.steps + 1):
        progress = step / args.steps
        lr = args.lr * min(1.0, step / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        data = torch.from_numpy(classroom.rows(None)).to(device)
        x, y = data[:, :-1], data[:, 1:]
        logits, ce = student(x, y)
        loss = ce
        if alpha > 0:
            with torch.no_grad():
                t_logits, _ = teacher(x)
            kd = F.kl_div(F.log_softmax(logits.reshape(-1, logits.size(-1)) / T, -1),
                          F.log_softmax(t_logits.reshape(-1, t_logits.size(-1)) / T, -1),
                          log_target=True, reduction="batchmean") * T * T
            loss = (1 - alpha) * ce + alpha * kd
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % args.log_every == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f} (lessons {ce.item():.4f})  lr {lr:.2e}  "
                  f"{(time.time() - t0) / step * 1000:.0f}ms/step")
        if step % args.eval_every == 0 or step == args.steps:
            now, rows = score(predict_s, 1000 + step)
            print(f"  exam after {step} steps: {now * 100:.1f}%  (" +
                  " ".join(f"{n} {s * 100:.0f}" for n, s, _ in rows) + ")")
            if now > best:
                best, best_state = now, copy.deepcopy(student.state_dict())
            if now >= target:
                print(f"  back within {args.tolerance * 100:.1f} points of the teacher: done")
                break
    student.load_state_dict(best_state)
    for module in student.modules():  # keep the float weights underneath; export rounds them the same way
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
    seconds = time.time() - t0
    out = args.out or os.path.join(os.path.dirname(args.ckpt) or ".", f"morpheus-{args.scheme}.morph")
    meta = {"squeezed_from": os.path.basename(args.ckpt), "steps": step, "seconds": round(seconds, 1)}
    nbytes = export(student, out, scheme=args.scheme, meta=meta, embed_scheme=args.embed_scheme)
    torch.save({"config": config_dict(cfg), "model": student.state_dict(), "scheme": args.scheme,
                "embed_scheme": args.embed_scheme}, os.path.splitext(out)[0] + ".pt")
    brain = NumpyMorpheus(*load_morph(out))  # grade the real file, exactly as a phone would load it
    rows = report_card(brain.predict, stages, cfg.block_size, n=args.exam_size * 2, seed=7)
    print_report_card(rows, title=f"{args.scheme} brain: {out}")
    print(f"{out}: {nbytes / 1024:.1f} KB, {step} steps, {seconds:.0f}s "
          f"(~{seconds * args.watts / 3600:.2f} Wh at {args.watts:g} W)")
    return out
