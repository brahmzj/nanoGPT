"""
Morpheus the assistant: skills + the brain.

A 0.7M-parameter brain is wonderful at what it was taught and hopeless at
everything else, so practical jobs go to *skills*: small, exact tools that cost
almost no energy (no model inference at all). Whatever no skill claims goes to
the brain; if the brain has not learned it, Morpheus can look it up (only when
the internet is allowed) and study the answer in its next learning session.

  notes      "remember that my locker code is 1234"  /  "what is my locker code?"  /  "forget ..."
  to-do      "add milk to my list"  /  "what is on my list?"  /  "remove milk from my list"
  reminders  "remind me to call mom at 5pm"  /  "set a timer for 10 minutes"  /  "set an alarm for 7:30 am"
  checked    arithmetic, "which is bigger, 73 or 37?", story problems: the brain answers first
             and a tool checks it. Mistakes it could learn become lessons for the next session
  clock      "what time is it?"  /  "what is the date?"
  battery    "how much battery do i have?"
  look up    "look up volcanoes" / "tell me about the moon" (Simple English Wikipedia, if allowed)
  search     "search the web for pizza near me" (opens the browser)

On Android (Termux) alarms and timers go to the phone's own Clock app, so they
ring even when Morpheus is closed. Elsewhere reminders are kept in the home
folder and announced the next time you talk to Morpheus.
"""

import ast
import datetime as dt
import json
import operator
import os
import re
import shutil
import subprocess
import time
import urllib.parse

from . import tokenizer
from .curriculum import UNKNOWN_ANSWER

STOPWORDS = set(
    "a an the is are was were be my your our me i you it its of to in on at for and or what whats "
    "where when who how which do does did that this these those please tell about can could would "
    "remember forget there here have has had".split())


# ---------------------------------------------------------------- android helpers


def on_android():
    return "TERMUX_VERSION" in os.environ or os.path.exists("/system/build.prop")


def run(cmd, timeout=30):
    """Run a helper program; returns stdout, or None if it is missing or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def android_alarm(hour, minute, message):
    return run(["am", "start", "-a", "android.intent.action.SET_ALARM",
                "--ei", "android.intent.extra.alarm.HOUR", str(hour),
                "--ei", "android.intent.extra.alarm.MINUTES", str(minute),
                "--es", "android.intent.extra.alarm.MESSAGE", message,
                "--ez", "android.intent.extra.alarm.SKIP_UI", "true"]) is not None


def android_timer(seconds, message):
    return run(["am", "start", "-a", "android.intent.action.SET_TIMER",
                "--ei", "android.intent.extra.alarm.LENGTH", str(int(seconds)),
                "--es", "android.intent.extra.alarm.MESSAGE", message,
                "--ez", "android.intent.extra.alarm.SKIP_UI", "true"]) is not None


def open_url(url):
    return (run(["termux-open-url", url]) is not None) or (run(["xdg-open", url]) is not None)


# ---------------------------------------------------------------- the conversation with the brain


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

    def record(self, message, reply):
        """Put an exchange answered by a skill into the brain's context (or correct the last one)."""
        turn = f"user: {tokenizer.normalize(message).lower().strip()}\nmorpheus: "
        if self.turns and self.turns[-1].startswith(turn):
            self.turns[-1] = turn + reply
        else:
            self.turns.append(turn + reply)


# ---------------------------------------------------------------- small parsers


def keywords(text):
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def second_person(text):
    """'my code is 1234' -> 'your code is 1234'."""
    swaps = {"my": "your", "mine": "yours", "i": "you", "me": "you", "myself": "yourself",
             "i'm": "you're", "am": "are", "your": "my", "yours": "mine"}
    return " ".join(swaps.get(w, w) for w in text.split())


def parse_clock(text):
    """'5pm', '5:30 pm', '17:30', 'noon', 'midnight', '7 am' -> (hour, minute) or None."""
    text = text.strip().lower().replace(".", "")
    if text in ("noon", "midday"):
        return 12, 0
    if text == "midnight":
        return 0, 0
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not m:
        return None
    hour, minute, half = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if half == "pm" and hour < 12:
        hour += 12
    if half == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


UNITS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}


def parse_duration(text):
    """'10 minutes', '1 hour and 30 minutes', 'an hour' -> seconds or None."""
    text = text.lower().replace("an hour", "1 hour").replace("a minute", "1 minute")
    total, found = 0, False
    for n, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(second|sec|minute|min|hour|hr)s?\b", text):
        total += float(n) * UNITS[unit]
        found = True
    return int(total) if found else None


WORD_OPS = [(r"\bmultiplied by\b", "*"), (r"\btimes\b", "*"), (r"\bdivided by\b", "/"), (r"\bover\b", "/"),
            (r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\bto the power of\b", "**"), (r"\bsquared\b", "**2"),
            (r"\bcubed\b", "**3"), (r"(?<=[\d)])\s*x\s*(?=[\d(])", "*"), (r"\^", "**"), (r"\bmod\b", "%")]
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
       ast.USub: operator.neg, ast.UAdd: operator.pos}


def parse_math(message):
    """The arithmetic expression in a question, or None. 'what is 12 times 7?' -> '12*7'."""
    text = message.lower().strip().rstrip("?.! ")
    text = re.sub(r"^(what is|what's|whats|calculate|compute|how much is|solve)\s+", "", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*% of\s*", r"\1/100*", text)  # "20% of 80"
    for pattern, op in WORD_OPS:
        text = re.sub(pattern, op, text)
    text = text.replace(",", "").replace("=", "").strip()
    if not re.fullmatch(r"[\d\s.+\-*/%()]+", text) or not re.search(r"\d\s*[-+*/%]", text):
        return None
    return re.sub(r"\s+", "", text)


def safe_eval(expr):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("that power is too big for me")
            return OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError("not arithmetic")
    return ev(ast.parse(expr, mode="eval"))


def fmt_number(x):
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return f"{x:.6g}"
    return str(x)


def pretty(expr):
    return re.sub(r"(\*\*|[-+*/%])", r" \1 ", expr).replace(" * ", " x ").replace("  ", " ").strip()


def first_sentences(text, limit=220):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = ""
    for s in sentences:
        if out and len(out) + len(s) > limit:
            break
        out = (out + " " + s).strip()
    return out[:limit]


# ---------------------------------------------------------------- the assistant


class Assistant:

    def __init__(self, brain, home, temperature=0.0, top_k=None, now=None):
        self.chat = Conversation(brain, temperature=temperature, top_k=top_k)
        self.home = home
        self.now = now or dt.datetime.now  # injectable clock, for tests
        self.skills = [
            (r"^(help|what can you do\??|skills)$", self.skill_help),
            (r"\b(what time is it|what'?s the time|what is the time|tell me the time)\b", self.skill_time),
            (r"\b(what day is (it|today)|what'?s (the |today'?s )?date|what is (the |today'?s )?date|"
             r"what is today)\b", self.skill_date),
            (r"\bbattery\b", self.skill_battery),
            (r"^remember (that )?(?P<fact>.+)$", self.skill_remember),
            (r"^forget (that |about )?(?P<fact>.+)$", self.skill_forget),
            (r"^what do you remember\??$", self.skill_list_notes),
            (r"^(add|put) (?P<item>.+?) (to|on) (my |the )?(to-?do |todo |shopping )?list$", self.skill_todo_add),
            (r"^(remove|delete|cross off|check off) (?P<item>.+?) (from|off) (my |the )?(to-?do |todo |shopping )?list$",
             self.skill_todo_remove),
            (r"^(done|finished|i did) (?P<item>.+)$", self.skill_todo_remove),
            (r"^(what'?s|what is|show|read) (on )?(me )?(my |the )?(to-?do |todo |shopping )?list\??$",
             self.skill_todo_show),
            (r"^clear (my |the )?(to-?do |todo |shopping )?list$", self.skill_todo_clear),
            (r"^set (a |an )?timer for (?P<when>.+)$", self.skill_timer),
            (r"^(set |make )?(an |a )?alarm (for|at) (?P<when>.+)$", self.skill_alarm),
            (r"^remind me (?P<rest>.+)$", self.skill_remind),
            (r"^(search( the web)? for|google|search) (?P<query>.+)$", self.skill_search),
            (r"^(look up|lookup|tell me about|search wikipedia for|who is|who was) (?P<topic>.+?)\??$",
             self.skill_lookup),
        ]

    # ------------------------------------------------------------ entry point

    def respond(self, message):
        text = tokenizer.normalize(message).lower().strip()
        text = re.sub(r"^(hey |hi |ok |okay )?morpheus[,!]?\s+", "", text)  # "hey morpheus, ..."
        if not text:
            return ""
        notices = self.due_reminders()
        reply = None
        for pattern, skill in self.skills:
            m = re.search(pattern, text)
            if m:
                reply = skill(text, m)
                if reply is not None:
                    self.chat.record(text, reply)
                    break
        if reply is None:
            checked = self.check(text)
            if checked:
                reply = self.skill_checked(text, *checked)
        if reply is None:
            reply = self.chat.ask(text)
            if reply == UNKNOWN_ANSWER:
                reply = self.fallback(text) or reply
        return "\n".join(notices + [reply])

    def fallback(self, text):
        """The brain does not know: try the notes, then (if allowed) look it up."""
        note = self.recall(text)
        if note:
            self.chat.record(text, note)
            return note
        m = re.match(r"^(what|who) (is|are|was|were) (a |an |the )?(?P<topic>.+?)\??$", text)
        if m and self.home.settings["allow_internet"]:
            answer = self.skill_lookup(text, m)
            if answer:
                return answer
        return None

    # ------------------------------------------------------------ skills

    def skill_help(self, text, m):
        return ("i can answer what i learned (letters, numbers, math, words, colors, animals, days), "
                "remember notes, keep your list, set reminders, timers and alarms, tell the time, "
                "do any math, and look things up if you allow the internet.")

    def skill_time(self, text, m):
        return "it is " + self.now().strftime("%I:%M %p").lstrip("0").lower() + "."

    def skill_date(self, text, m):
        now = self.now()
        return f"today is {now.strftime('%A, %B')} {now.day}, {now.year}.".lower()

    def skill_battery(self, text, m):
        from .learn import power_status
        charging, percent = power_status()
        if percent is None:
            return "i can not see the battery from here. on android, install termux-api."
        return f"the battery is at {percent}%" + (" and charging." if charging else ".")

    # notes ---------------------------------------------------------

    def notes(self):
        return self.home.read_list("notes.json")

    def skill_remember(self, text, m):
        fact = m.group("fact").strip().rstrip(".")
        notes = [n for n in self.notes() if n["text"] != fact] + [{"text": fact, "when": time.strftime("%Y-%m-%d %H:%M")}]
        self.home.write_list("notes.json", notes)
        return f"ok, i will remember that {second_person(fact)}."

    def recall(self, question):
        asked = keywords(question)
        if not asked:
            return None
        best, best_score = None, 0.0
        for note in self.notes():
            overlap = len(asked & keywords(note["text"]))
            score = overlap / len(asked)
            if overlap and score > best_score:
                best, best_score = note, score
        if best is None or best_score < 0.5:
            return None
        return second_person(best["text"]) + "."

    def skill_forget(self, text, m):
        target = keywords(m.group("fact"))
        notes = self.notes()
        keep = [n for n in notes if not target or len(target & keywords(n["text"])) < max(1, len(target) // 2 + 1)]
        self.home.write_list("notes.json", keep)
        gone = len(notes) - len(keep)
        return f"ok, i forgot {gone} note{'s' if gone != 1 else ''}." if gone else "i did not have a note about that."

    def skill_list_notes(self, text, m):
        notes = self.notes()
        if not notes:
            return "i have no notes yet. say: remember that ..."
        return " ".join(f"{i + 1}) {second_person(n['text'])}." for i, n in enumerate(notes[-10:]))

    # to-do list ----------------------------------------------------

    def skill_todo_add(self, text, m):
        item = m.group("item").strip()
        items = self.home.read_list("todo.json")
        if item not in items:
            items.append(item)
            self.home.write_list("todo.json", items)
        return f"added {item}. you have {len(items)} thing{'s' if len(items) != 1 else ''} on your list."

    def skill_todo_remove(self, text, m):
        item = m.group("item").strip().rstrip(".")
        items = self.home.read_list("todo.json")
        match = next((i for i in items if i == item), None) or next(
            (i for i in items if keywords(item) & keywords(i)), None)
        if match is None:
            return None if text.startswith(("done", "finished", "i did")) else f"{item} is not on your list."
        items.remove(match)
        self.home.write_list("todo.json", items)
        return f"removed {match}. " + (f"{len(items)} left." if items else "your list is empty!")

    def skill_todo_show(self, text, m):
        items = self.home.read_list("todo.json")
        if not items:
            return "your list is empty."
        return "on your list: " + ", ".join(items) + "."

    def skill_todo_clear(self, text, m):
        self.home.write_list("todo.json", [])
        return "your list is empty now."

    # reminders -----------------------------------------------------

    def skill_timer(self, text, m):
        seconds = parse_duration(m.group("when"))
        if not seconds:
            return "for how long? say: set a timer for 10 minutes."
        return self.schedule(time.time() + seconds, "timer", seconds=seconds)

    def skill_alarm(self, text, m):
        clock = parse_clock(m.group("when"))
        if not clock:
            return "at what time? say: set an alarm for 7:30 am."
        return self.schedule(self.next_time(*clock), "alarm", clock=clock)

    def skill_remind(self, text, m):
        rest = m.group("rest")
        patterns = [r"^(to )?(?P<what>.+?) in (?P<dur>.+)$", r"^in (?P<dur>.+?) to (?P<what>.+)$",
                    r"^(to )?(?P<what>.+?) at (?P<at>.+)$", r"^at (?P<at>.+?) to (?P<what>.+)$"]
        for pattern in patterns:
            r = re.match(pattern, rest)
            if not r:
                continue
            what = r.group("what").strip()
            if "dur" in r.groupdict() and r.group("dur"):
                seconds = parse_duration(r.group("dur"))
                if seconds:
                    return self.schedule(time.time() + seconds, what, seconds=seconds)
            elif r.groupdict().get("at"):
                clock = parse_clock(r.group("at"))
                if clock:
                    return self.schedule(self.next_time(*clock), what, clock=clock)
        return "when should i remind you? say: remind me to call mom at 5pm, or in 20 minutes."

    def next_time(self, hour, minute):
        now = self.now()
        when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if when <= now:
            when += dt.timedelta(days=1)
        return when.timestamp()

    def schedule(self, due, what, seconds=None, clock=None):
        reminders = self.home.read_list("reminders.json")
        reminders.append({"due": due, "what": what})
        self.home.write_list("reminders.json", reminders)
        when_text = dt.datetime.fromtimestamp(due).strftime("%I:%M %p").lstrip("0").lower()
        label = "morpheus: " + what
        on_phone = (android_timer(seconds, label) if seconds else android_alarm(clock[0], clock[1], label)
                    ) if on_android() else False
        where = " on your clock app" if on_phone else ""
        if what == "timer":
            return f"timer set for {fmt_duration(seconds)}{where}."
        if what == "alarm":
            return f"alarm set for {when_text}{where}."
        return f"ok, i will remind you to {second_person(what)} at {when_text}{where}."

    def due_reminders(self):
        reminders = self.home.read_list("reminders.json")
        now = self.now().timestamp()
        due = [r for r in reminders if r["due"] <= now]
        if due:
            self.home.write_list("reminders.json", [r for r in reminders if r["due"] > now])
        return [f"(reminder: {second_person(r['what'])})" for r in due if r["what"] not in ("timer", "alarm")]

    # checked answers: the brain answers, a tool checks ----------------

    def skill_checked(self, text, correct, same, learnable):
        attempt = self.chat.ask(text)  # the brain tries first...
        if same(attempt, correct):
            return attempt
        if learnable:  # ...a tool checks it, and mistakes it could learn become lessons
            self.home.teach(text, correct)
        self.chat.record(text, correct)
        return correct

    def check(self, text):
        """The exact answer to a question a tool can verify, or None."""
        m = re.match(r"^which is (?P<w>bigger|larger|greater|smaller|less), (?P<a>-?\d+) or (?P<b>-?\d+)\??$", text)
        if m:
            a, b = int(m.group("a")), int(m.group("b"))
            word = "bigger" if m.group("w") in ("bigger", "larger", "greater") else "smaller"
            value = max(a, b) if word == "bigger" else min(a, b)
            return f"the {word} of {a} and {b} is {value}.", str.__eq__, max(abs(a), abs(b)) <= 100
        m = re.match(r"^(?P<name>[a-z]+) has (?P<a>\d+) (?P<item>[a-z]+) and gets (?P<b>\d+) more\. "
                     r"how many (?P<items>[a-z]+) now\??$", text)
        if m:
            total = int(m.group("a")) + int(m.group("b"))
            item = m.group("items") if total != 1 else m.group("item")
            return f"{m.group('name')} has {total} {item}.", str.__eq__, total <= 40
        expr = parse_math(text)
        if expr:
            try:
                value = safe_eval(expr)
            except ZeroDivisionError:
                return "you can not divide by zero.", str.__eq__, False
            except Exception:
                return None
            numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", expr)]
            learnable = len(numbers) == 2 and max(numbers) <= 100 and float(value).is_integer()
            return f"{pretty(expr)} = {fmt_number(value)}.", same_number, learnable
        return None

    # the outside world -----------------------------------------------

    def skill_search(self, text, m):
        query = m.group("query").strip()
        url = "https://duckduckgo.com/?q=" + urllib.parse.quote_plus(query)
        opened = open_url(url)
        return f"searching for {query}." if opened else f"here is a search for {query}: {url}"

    def skill_lookup(self, text, m):
        topic = re.sub(r"^(a|an|the) ", "", m.group("topic").strip().rstrip("?"))
        if not self.home.settings["allow_internet"]:
            return (f"i have not learned about {topic} yet. if you allow the internet "
                    "(python -m morpheus settings allow_internet on), i can look it up.")
        found = wikipedia_summary(topic)
        if not found:
            return f"i could not find anything about {topic}."
        title, extract, url = found
        answer = first_sentences(extract).lower()
        self.home.remember(f"{title}\n{extract}", url)  # read it properly in the next learning session
        if len(answer) <= 90:
            self.home.teach(text, answer)
        self.home.save_journal()
        return answer


def same_number(attempt, correct):
    """'15 minus 6 is 9.' and '15 - 6 = 9.' agree: both end in the same number."""
    said, want = re.findall(r"-?\d+(?:\.\d+)?", attempt), re.findall(r"-?\d+(?:\.\d+)?", correct)
    return bool(said) and said[-1] == want[-1]


def fmt_duration(seconds):
    parts = []
    for name, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        n, seconds = divmod(seconds, size)
        if n:
            parts.append(f"{int(n)} {name}{'s' if n != 1 else ''}")
    return " and ".join(parts) or "0 seconds"


def wikipedia_summary(topic, lang="simple"):
    """(title, extract, url) from Wikipedia (Simple English by default), or None."""
    from .learn import USER_AGENT
    import urllib.request

    def get(url):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))
    base = f"https://{lang}.wikipedia.org"
    try:
        hits = get(f"{base}/w/api.php?action=opensearch&format=json&limit=1&search=" + urllib.parse.quote(topic))
        if not hits[1]:
            return None
        title = hits[1][0]
        data = get(f"{base}/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")))
        extract = data.get("extract", "")
        if not extract:
            return None
        return data.get("title", title), extract, data.get("content_urls", {}).get("desktop", {}).get("page", base)
    except Exception:
        return None
