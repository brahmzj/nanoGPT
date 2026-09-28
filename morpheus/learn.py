"""
Morpheus keeps growing after school: lifelong learning on your own device.

Everything lives in one folder, MORPHEUS_HOME (default ~/.morpheus):

  settings.json       what Morpheus may do (the internet is OFF until you allow it)
  brain.morph         the compressed brain that `chat` uses
  brain-train.npz     full-precision weights + optimizer state for further learning
  backup/             the previous brain, so `undo` can always go back one session
  inbox/              drop .txt / .md / .html files here to be read
  library/            everything read so far, LZMA-compressed
  taught.jsonl        questions and answers you taught it (`teach`, or /teach in chat)
  journal.json        what was read, when, and the result of every session
  notes.json, todo.json, reminders.json   the assistant's memory (see assistant.py)

A learning session (`python -m morpheus learn`):
  1. checks power: by default it only learns while charging (energy on the grid, not the battery)
  2. reads new files from the inbox folders and, only if allowed, fetches from the internet
  3. sits a baseline exam on the whole ABC-to-talk curriculum
  4. studies: new reading + your taught facts + review of the curriculum, while also matching
     its own pre-session answers (learning without forgetting); the session ends in a short
     dream as the learning rate fades to zero
  5. sits the exam again and keeps the new brain only if it lost at most `max_forgetting`
     against both the last brain and the very first one (so losses can not pile up).
     Otherwise the session is rolled back: Morpheus never drifts downhill.

Only numpy is needed (see grad.py), so this runs in Termux on Android.
"""

import hashlib
import html
import json
import lzma
import math
import os
import random
import re
import shutil
import subprocess
import time
import urllib.request
from html.parser import HTMLParser

import numpy as np

from . import tokenizer
from .compress import load
from .curriculum import STAGES, UNKNOWN_ANSWER, Library, Stage, pseudo_word, qa
from .school import Classroom, report_card

HOME = os.environ.get("MORPHEUS_HOME") or os.path.join(os.path.expanduser("~"), ".morpheus")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_BRAIN = os.path.join(REPO, "brains", "morpheus-nano.morph")
STOPWORDS = set("what who where when which how is are was were the a an of to in on for do does did "
                "you your my me i it and or".split())
USER_AGENT = "Morpheus/0.1 (tiny personal learning model; https://github.com/brahmzj/nanoGPT)"
TEXT_FILES = (".txt", ".md", ".html", ".htm")

DEFAULT_SETTINGS = {
    "allow_internet": False,          # nothing is ever downloaded until you switch this on
    "sources": ["https://simple.wikipedia.org/api/rest_v1/page/random/summary"],
    "fetches_per_source": 5,          # per session, for "random" style sources
    "inbox": ["{home}/inbox", "~/storage/downloads/morpheus"],
    "only_when_charging": True,
    "min_battery": 30,                # percent, if charging state is unknown
    "session_steps": 200,
    "batch_size": 8,
    "lr": 3e-4,
    "lr_low_bit": 1e-3,               # int3/ternary/binary brains: weights must cross rounding thresholds
    "new_rows": 2,                    # of each batch, rows of new reading / taught facts (at most)
    "remember": 0.5,                  # how much to also match its own pre-session answers (0..1)
    "max_forgetting": 0.02,           # max drop in exam score, vs the last AND the original brain
    "exam_questions": 60,             # per curriculum stage
    "lowercase": True,                # Morpheus grew up lowercase
    "library_chars": 3_000_000,       # most recent reading kept in play
}


class Home:

    def __init__(self, path=HOME):
        self.path = os.path.expanduser(path)
        for sub in ("inbox", "library", "backup"):
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)
        self.settings = dict(DEFAULT_SETTINGS)
        if os.path.exists(self.file("settings.json")):
            with open(self.file("settings.json")) as f:
                self.settings.update(json.load(f))
        else:
            self.save_settings()
        self.journal = {"seen": {}, "sessions": []}
        if os.path.exists(self.file("journal.json")):
            with open(self.file("journal.json")) as f:
                self.journal.update(json.load(f))

    def file(self, *parts):
        return os.path.join(self.path, *parts)

    def save_settings(self):
        write_json(self.file("settings.json"), self.settings)

    def save_journal(self):
        write_json(self.file("journal.json"), self.journal)

    def set(self, key, value):
        if key not in DEFAULT_SETTINGS:
            raise KeyError(f"unknown setting {key!r}; known: {', '.join(DEFAULT_SETTINGS)}")
        default = DEFAULT_SETTINGS[key]
        if isinstance(default, bool):
            value = str(value).lower() in ("1", "true", "yes", "on", "allow")
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            value = type(default)(value)
        elif isinstance(default, (list, dict)) and isinstance(value, str):
            value = json.loads(value)
        self.settings[key] = value
        self.save_settings()
        return value

    def inbox_dirs(self):
        return [os.path.expanduser(p.replace("{home}", self.path)) for p in self.settings["inbox"]]

    # ------------------------------------------------------------ brain files

    def brain_source(self):
        """Where learning continues from: our own training state, else a compressed brain."""
        for path in (self.file("brain-train.npz"), self.file("brain.morph"), SHIPPED_BRAIN,
                     os.path.join(REPO, "out-morpheus", "morpheus.morph")):
            if os.path.exists(path):
                return path
        return None

    def load_trainer(self):
        from .grad import NumpyTrainer
        src = self.brain_source()
        if src is None:
            raise FileNotFoundError("no brain to grow: train one (python -m morpheus train) or get brains/")
        if src.endswith(".npz"):
            trainer = NumpyTrainer.load(src)
        else:  # a compressed brain keeps learning in its own format
            weights, header = load(src)
            quant = {"scheme": header.get("scheme", "int8"), "embed_scheme": header.get("embed_scheme")}
            trainer = NumpyTrainer(weights, header["config"], quant=quant)
        trainer.lr = self.learning_rate(trainer)
        return trainer, src

    def learning_rate(self, trainer):
        low_bit = trainer.quant and trainer.quant["scheme"] in ("int3", "ternary", "binary")
        return self.settings["lr_low_bit"] if low_bit else self.settings["lr"]

    def backup(self):
        for name in ("brain.morph", "brain-train.npz"):
            if os.path.exists(self.file(name)):
                shutil.copy2(self.file(name), self.file("backup", name))

    def undo(self):
        restored = []
        for name in ("brain.morph", "brain-train.npz"):
            if os.path.exists(self.file("backup", name)):
                shutil.copy2(self.file("backup", name), self.file(name))
                restored.append(name)
        return restored

    # ------------------------------------------------------------ memory

    def remember(self, text, source):
        """Store new reading in the library (LZMA-compressed). Returns False if already known."""
        text = clean(text, self.settings["lowercase"])
        if len(text) < 40:
            return False
        digest = hashlib.sha1(text.encode("ascii")).hexdigest()
        if digest in self.journal["seen"]:
            return False
        name = f"{int(time.time() * 1000)}-{digest[:8]}.txt.xz"
        with lzma.open(self.file("library", name), "wt", encoding="ascii") as f:
            f.write(text)
        self.journal["seen"][digest] = {"source": source, "chars": len(text), "file": name,
                                        "when": time.strftime("%Y-%m-%d %H:%M")}
        return True

    def library_text(self):
        names = sorted(os.listdir(self.file("library")), reverse=True)  # newest first
        parts, total = [], 0
        for name in names:
            with lzma.open(self.file("library", name), "rt", encoding="ascii") as f:
                text = f.read()
            parts.append(text)
            total += len(text)
            if total >= self.settings["library_chars"]:
                break
        return "\n".join(parts)

    def read_list(self, name):
        if not os.path.exists(self.file(name)):
            return []
        with open(self.file(name)) as f:
            return json.load(f)

    def write_list(self, name, items):
        write_json(self.file(name), items)

    def teach(self, question, answer):
        question = clean(question, True).strip()
        answer = clean(answer, True).strip()
        if not question or not answer:
            raise ValueError("both a question and an answer are needed")
        with open(self.file("taught.jsonl"), "a") as f:
            f.write(json.dumps({"q": question, "a": answer, "when": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
        return question, answer

    def taught(self):
        if not os.path.exists(self.file("taught.jsonl")):
            return []
        with open(self.file("taught.jsonl")) as f:
            facts = [json.loads(line) for line in f if line.strip()]
        latest = {}  # re-teaching a question replaces the old answer
        for fact in facts:
            latest[fact["q"].rstrip("?")] = (fact["q"], fact["a"])
        return list(latest.values())


# ---------------------------------------------------------------- reading


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "head", "nav", "footer", "svg"}
    BLOCK = {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts, self.skipping = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)


def html_to_text(raw):
    parser = _TextExtractor()
    parser.feed(raw)
    return html.unescape("".join(parser.parts))


def clean(text, lowercase=True):
    text = tokenizer.normalize(text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text.lower() if lowercase else text


def read_file(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return html_to_text(raw) if path.lower().endswith((".html", ".htm")) else raw


def fetch(url, timeout=20, max_bytes=2_000_000):
    """Download one page. Understands Wikipedia-style JSON summaries, HTML and plain text."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        kind = r.headers.get_content_type()
        charset = r.headers.get_content_charset() or "utf-8"
        raw = r.read(max_bytes).decode(charset, errors="ignore")
        final_url = r.geturl()
    if "json" in kind:
        data = json.loads(raw)
        text = "\n".join(str(data[k]) for k in ("title", "extract") if data.get(k))
        return text, data.get("content_urls", {}).get("desktop", {}).get("page", final_url)
    if "html" in kind:
        return html_to_text(raw), final_url
    return raw, final_url


def gather(home, online=None, log=print):
    """Read new inbox files and (only if allowed) internet sources into the library."""
    new = 0
    for folder in home.inbox_dirs():
        if not os.path.isdir(folder):
            continue
        for root, _, names in os.walk(folder):
            for name in sorted(names):
                if name.lower().endswith(TEXT_FILES):
                    path = os.path.join(root, name)
                    if home.remember(read_file(path), f"file:{path}"):
                        new += 1
                        log(f"  read {path}")
    allowed = home.settings["allow_internet"] if online is None else online
    if allowed:
        for url in home.settings["sources"]:
            for _ in range(home.settings["fetches_per_source"] if "random" in url else 1):
                try:
                    text, where = fetch(url)
                except Exception as e:  # offline, blocked, rate-limited: just try next time
                    log(f"  could not fetch {url}: {e}")
                    break
                if home.remember(text, where):
                    new += 1
                    log(f"  read {where}")
    home.save_journal()
    return new


# ---------------------------------------------------------------- power


def power_status():
    """(charging, percent). Each is None when unknown. Uses Termux:API if present, else /sys."""
    try:
        out = subprocess.run(["termux-battery-status"], capture_output=True, text=True, timeout=15)
        info = json.loads(out.stdout)
        return info.get("plugged", "UNPLUGGED") != "UNPLUGGED", info.get("percentage")
    except Exception:
        pass
    for base in ("/sys/class/power_supply/battery", "/sys/class/power_supply/BAT0", "/sys/class/power_supply/BAT1"):
        try:
            with open(os.path.join(base, "status")) as f:
                status = f.read().strip().lower()
            with open(os.path.join(base, "capacity")) as f:
                percent = int(f.read().strip())
            return status in ("charging", "full"), percent
        except Exception:
            continue
    return None, None


def power_ok(settings):
    if not settings["only_when_charging"]:
        return True, "power check off"
    charging, percent = power_status()
    if charging:
        return True, "charging"
    if charging is None:
        if percent is None or percent >= settings["min_battery"]:
            return True, "power source unknown"
        return False, f"battery at {percent}%"
    return False, "not charging (plug in, or use --force)"


# ---------------------------------------------------------------- a session


def taught_stage(facts):
    def fact(rng):
        q, a = facts[rng.randrange(len(facts))]
        return qa(q, a, rng)

    def near_miss(rng):
        """The same question about something else ('capital of france' -> 'capital of zobek'):
        still unknown. Without these, one taught fact answers every similar question."""
        q, _ = facts[rng.randrange(len(facts))]
        words = q.rstrip("?").split()
        spots = [i for i, w in enumerate(words) if len(w) > 2 and w not in STOPWORDS]
        if not spots:
            return fact(rng)
        words[rng.choice(spots)] = pseudo_word(rng, 3, 7)
        return qa(" ".join(words) + ("?" if q.endswith("?") else ""), UNKNOWN_ANSWER, rng)
    return Stage("taught", "things you taught me", [(1, 1, fact), (0.5, 0, near_miss)])


def average(rows):
    return sum(score for _, score, _ in rows) / max(1, len(rows))


def session(home, steps=None, force=False, online=None, seed=None, log=print):
    """One learning session. Returns a summary dict (also stored in the journal)."""
    s = home.settings
    ok, why = power_ok(s)
    if not ok and not force:
        log(f"not learning now: {why}")
        return {"skipped": why}
    started = time.time()
    new = gather(home, online, log)
    trainer, src = home.load_trainer()
    block = trainer.cfg["block_size"]
    rng = random.Random(seed if seed is not None else time.time_ns())

    # what to study: each batch has review rows (the whole curriculum, anchored to the brain's own
    # pre-session answers) and new rows (reading + taught facts, learned freely). New material
    # gets rows in proportion to how much of it there is, so one short file can not crowd out
    # everything else.
    reading = home.library_text()
    facts = home.taught()
    taught = taught_stage(facts) if facts else None
    new_stages, new_weights = [], []
    if reading:
        new_stages.append(Library(reading))
        new_weights.append(min(1.0, len(reading) / 20_000))
    if taught:
        new_stages.append(taught)
        new_weights.append(min(1.0, 0.2 + len(facts) / 5))
    amount = min(1.0, sum(new_weights))
    n_new = min(s["new_rows"], max(1, round(s["new_rows"] * amount))) if new_stages else 0
    exam_seed = 4242  # the same questions before and after, so the comparison is fair
    before = report_card(trainer.predict, STAGES + ([taught] if taught else []), block,
                         n=s["exam_questions"], seed=exam_seed)
    log(f"learning from {src}: {len(reading):,} chars of reading, {len(facts)} taught facts, "
        f"{new} new source(s); exam before {average(before) * 100:.1f}%")

    review = Classroom(STAGES, block, s["batch_size"] - n_new, seed=rng.randrange(2 ** 31))
    fresh = Classroom(new_stages, block, n_new, seed=rng.randrange(2 ** 31), weights=new_weights) if n_new else None
    alpha = np.array([s["remember"]] * (s["batch_size"] - n_new) + [0.0] * n_new, dtype=np.float32)
    steps = steps or s["session_steps"]
    past_self = {k: v.copy() for k, v in trainer.effective().items()}  # learning without forgetting
    losses = []
    for i in range(steps):
        warm = min(1.0, (i + 1) / 10)
        lr = trainer.lr * warm * 0.5 * (1 + math.cos(math.pi * i / steps))  # ends in a dream: lr -> 0
        rows = review.rows(None)
        if fresh:
            rows = np.concatenate((rows, fresh.rows(None)))
        x, y = rows[:, :-1], rows[:, 1:]
        soft = None
        if s["remember"] > 0:
            from .grad import softmax
            soft = softmax(trainer.forward(x, keep=False, w=past_self)[0])
        losses.append(trainer.step(x, y, lr, soft=soft, alpha=alpha))
        if (i + 1) % 25 == 0:
            log(f"  step {i + 1}/{steps}  loss {np.mean(losses[-25:]):.3f}")

    after = report_card(trainer.predict, STAGES + ([taught] if taught else []), block,
                        n=s["exam_questions"], seed=exam_seed)
    curriculum_before = average([r for r in before if r[0] != "taught"])
    curriculum_after = average([r for r in after if r[0] != "taught"])
    # the first score ever measured is the floor: small losses can not pile up session after session
    original = home.journal.setdefault("original_exam", curriculum_before)
    floor = min(curriculum_before, original) - s["max_forgetting"]
    kept = curriculum_after >= floor
    summary = {
        "when": time.strftime("%Y-%m-%d %H:%M"), "steps": steps, "new_sources": new,
        "loss": round(float(np.mean(losses[-25:])), 4), "seconds": round(time.time() - started, 1),
        "exam_before": round(curriculum_before, 4), "exam_after": round(curriculum_after, 4),
        "taught_before": dict((n, sc) for n, sc, _ in before).get("taught"),
        "taught_after": dict((n, sc) for n, sc, _ in after).get("taught"),
        "kept": kept,
    }
    if kept:
        home.backup()
        trainer.save(home.file("brain-train.npz"))
        with open(home.file("brain.morph"), "wb") as f:
            f.write(trainer.export(meta={"grown_on_device": True, "last_session": summary}))
        log(f"kept: exam {curriculum_before * 100:.1f}% -> {curriculum_after * 100:.1f}%"
            + (f", taught facts {summary['taught_before'] * 100:.0f}% -> {summary['taught_after'] * 100:.0f}%"
               if taught else ""))
    else:
        log(f"rolled back: the exam fell from {curriculum_before * 100:.1f}% to {curriculum_after * 100:.1f}% "
            f"(below the allowed {floor * 100:.1f}%). The old brain stays.")
    home.journal["sessions"].append(summary)
    home.save_journal()
    return summary


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
