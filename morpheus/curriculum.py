"""
The school of Morpheus: a procedurally generated curriculum.

There is no dataset to download. Every lesson is produced on demand by the
small generators below, so the "training data" for all stages is this file
(a few KB of code standing in for an unlimited stream of examples).

Every fact is written once as a (prompt, answer, rest) triple:
  - the lesson Morpheus studies is  prompt + answer + rest
  - the exam question is            prompt  ->  answer (+ "\n" when rest is "",
                                                  i.e. Morpheus must also know to stop)
so lessons and exams can never drift apart. rest=None marks an open-ended fact
(like "a b c" -> "d", where the run may go on) whose stopping point is not graded.

Stages, in the order Morpheus meets them:
  1. letters    a b c, after/before, big and little letters
  2. numbers    counting, number words, counting stars, bigger/smaller
  3. math       + - x / facts, doubles and halves
  4. words      spelling, a is for apple, animals, colors, opposites, plurals
  5. world      days, months, simple world facts, story problems
  6. talk       conversation: Morpheus answers questions about all of the above
"""

import random

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# ----------------------------------------------------------------------------
# helpers


def pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def run(items):
    return " ".join(str(i) for i in items)


ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_word(n):
    if n < 20:
        return ONES[n]
    if n == 100:
        return "one hundred"
    tens, ones = divmod(n, 10)
    return TENS[tens] + ("" if ones == 0 else " " + ONES[ones])


MASS_NOUNS = {"bread", "milk", "rice", "soup", "cheese", "jam", "hair", "cake"}


def article(word):
    return "an" if word[0] in "aeiou" else "a"


def the_thing(word):
    """'a cat', 'an egg', but just 'milk'."""
    return word if word in MASS_NOUNS else f"{article(word)} {word}"


IRREGULAR_PLURALS = {
    "mouse": "mice", "child": "children", "foot": "feet", "man": "men", "woman": "women",
    "tooth": "teeth", "sheep": "sheep", "fish": "fish", "person": "people", "goose": "geese",
    "leaf": "leaves", "wolf": "wolves", "baby": "babies", "puppy": "puppies", "bunny": "bunnies",
    "strawberry": "strawberries", "cherry": "cherries",
}


def plural(word):
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def amount(n, word):
    return f"{n} {word if n == 1 else plural(word)}"


# ----------------------------------------------------------------------------
# the lexicon: everything Morpheus can know about the world, in one place

ANIMALS = {  # animal: (sound or None, legs)
    "cat": ("meow", 4), "dog": ("woof", 4), "cow": ("moo", 4), "pig": ("oink", 4),
    "duck": ("quack", 2), "sheep": ("baa", 4), "horse": ("neigh", 4), "bird": ("tweet", 2),
    "frog": ("ribbit", 4), "lion": ("roar", 4), "bee": ("buzz", 6), "owl": ("hoot", 2),
    "hen": ("cluck", 2), "mouse": ("squeak", 4), "snake": ("hiss", 0), "fish": (None, 0),
    "spider": (None, 8), "ant": (None, 6), "goat": ("maa", 4), "wolf": ("howl", 4),
    "goose": ("honk", 2), "bear": ("growl", 4), "monkey": ("ooh ooh", 2), "elephant": ("toot", 4),
}
CATEGORIES = {
    "animal": list(ANIMALS),
    "color": "red blue green yellow orange purple pink black white brown gray".split(),
    "food": "apple banana bread milk egg cake rice soup cheese grape lemon carrot cookie jam".split(),
    "toy": "ball doll kite drum block robot yoyo teddy xylophone".split(),
    "body part": "eye ear nose mouth hand foot arm leg head hair tooth knee".split(),
    "shape": "circle square triangle star heart".split(),
}
COLOR_FACTS = [  # (subject, verb, color)
    ("the sky", "is", "blue"), ("grass", "is", "green"), ("the sun", "is", "yellow"),
    ("snow", "is", "white"), ("the night", "is", "black"), ("apples", "are", "red"),
    ("bananas", "are", "yellow"), ("leaves", "are", "green"), ("the sea", "is", "blue"),
    ("fire", "is", "red"), ("clouds", "are", "white"),
    ("carrots", "are", "orange"), ("frogs", "are", "green"), ("lemons", "are", "yellow"),
    ("grapes", "are", "purple"), ("chocolate", "is", "brown"), ("strawberries", "are", "red"),
    ("pigs", "are", "pink"), ("mud", "is", "brown"), ("crows", "are", "black"),
    ("elephants", "are", "gray"), ("oranges", "are", "orange"), ("coal", "is", "black"),
    ("tomatoes", "are", "red"), ("the moon", "is", "white"), ("bears", "are", "brown"),
]
OPPOSITES = [
    ("hot", "cold"), ("big", "small"), ("up", "down"), ("day", "night"), ("fast", "slow"),
    ("happy", "sad"), ("yes", "no"), ("open", "closed"), ("in", "out"), ("on", "off"),
    ("wet", "dry"), ("old", "new"), ("full", "empty"), ("light", "dark"), ("tall", "short"),
    ("loud", "quiet"), ("first", "last"), ("good", "bad"), ("left", "right"), ("top", "bottom"),
    ("hard", "soft"), ("early", "late"), ("near", "far"), ("clean", "dirty"), ("black", "white"),
    ("boy", "girl"), ("push", "pull"), ("win", "lose"), ("give", "take"), ("come", "go"),
]
OPPOSITE = {a: b for a, b in OPPOSITES} | {b: a for a, b in OPPOSITES}
ABC_WORDS = dict(zip(LETTERS, (
    "apple ball cat dog egg fish grape hat ice jam key lion moon nose owl pig queen rain sun "
    "tree up van water xylophone yoyo zoo").split()))
RHYMES = [
    "cat hat bat mat rat sat", "dog log frog fog hog", "sun run fun bun", "bee see tree me three",
    "cake lake make bake snake", "ball tall wall fall call", "red bed fed", "star car far jar",
    "moon soon spoon noon", "king ring sing wing", "boat goat coat",
]
DAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
MONTHS = ("january february march april may june july august september october november "
          "december").split()
SEASONS = "spring summer fall winter".split()
WORLD_FACTS = [  # (subject, number, unit): "a week has 7 days"
    ("a week", 7, "day"), ("a year", 12, "month"), ("a day", 24, "hour"), ("an hour", 60, "minute"),
    ("a minute", 60, "second"), ("a hand", 5, "finger"), ("a foot", 5, "toe"), ("a car", 4, "wheel"),
    ("a bike", 2, "wheel"), ("a triangle", 3, "side"), ("a square", 4, "side"), ("a rainbow", 7, "color"),
    ("a person", 2, "eye"), ("a face", 1, "nose"), ("a dozen", 12, "egg"), ("a tricycle", 3, "wheel"),
]
NAMES = "sam mia tom ava max zoe leo ivy ben amy jack lily".split()
ITEMS = "apple ball book cookie car egg hat cup pen shell block kite toy star".split()
ADJECTIVES = "big little happy sleepy fast slow red blue funny tiny".split()
VERBS = "runs jumps sleeps eats plays sings sits hops swims reads".split()
PLACES = "park house garden school farm zoo beach".split()

SPELLING_WORDS = sorted(
    set(ANIMALS) | set(sum(CATEGORIES.values(), [])) | set(ABC_WORDS.values()) | set(DAYS)
    | set(OPPOSITE) | set(ITEMS) | {"morpheus"}
)
SPELLING_WORDS = [w for w in SPELLING_WORDS if " " not in w]
WORD_CATEGORY = {w: cat for cat, words in CATEGORIES.items() for w in words}


def spell(word):
    return " ".join(word)


# ----------------------------------------------------------------------------
# stage 1: letters


def alphabet(rng):
    letters = LETTERS if rng.random() < 0.7 else LETTERS.upper()
    return run(letters), "", ""


def letter_run(rng):
    letters = LETTERS if rng.random() < 0.75 else LETTERS.upper()
    if rng.random() < 0.15:
        letters = letters[::-1]  # backwards
    n = rng.randint(4, 10)
    i = rng.randrange(len(letters) - n + 1)
    seq = letters[i:i + n]
    k = rng.randint(3, n - 1)
    return run(seq[:k]) + " ", seq[k], (" " + run(seq[k + 1:])) if k + 1 < n else None


def after_letter(rng):
    i = rng.randrange(25)
    return f"after {LETTERS[i]} comes ", LETTERS[i + 1], ""


def before_letter(rng):
    i = rng.randrange(1, 26)
    return f"before {LETTERS[i]} comes ", LETTERS[i - 1], ""


def big_little(rng):
    c = pick(rng, LETTERS)
    if rng.random() < 0.5:
        return f"big {c.upper()}, little ", c, ""
    return f"little {c}, big ", c.upper(), ""


def vowels(rng):
    return "the vowels are a e i o u", "", ""


# ----------------------------------------------------------------------------
# stage 2: numbers


def count_run(rng):
    step = 1 if rng.random() < 0.75 else pick(rng, (2, 5, 10))
    start = step * rng.randrange(0 if step > 1 else 0, (60 if step == 1 else 100 // step - 3))
    if step == 1 and rng.random() < 0.3:
        start = 1
    n = rng.randint(4, 12)
    seq = [start + step * j for j in range(n) if start + step * j <= 100]
    if len(seq) < 4:
        seq = [step * j for j in range(1, 5)]
    k = rng.randint(3, len(seq) - 1)
    rest = (" " + run(seq[k + 1:])) if k + 1 < len(seq) else None
    return run(seq[:k]) + " ", str(seq[k]), rest


def after_number(rng):
    n = rng.randrange(100)
    return f"after {n} comes ", str(n + 1), ""


def before_number(rng):
    n = rng.randrange(1, 101)
    return f"before {n} comes ", str(n - 1), ""


def number_to_word(rng):
    n = rng.randrange(21) if rng.random() < 0.5 else rng.randrange(101)
    return f"{n} is ", number_word(n), ""


def word_to_number(rng):
    n = rng.randrange(21) if rng.random() < 0.5 else rng.randrange(101)
    return f"{number_word(n)} is ", str(n), ""


def count_stars(rng):
    n = rng.randint(1, 10)
    return " ".join("*" * n) + " is ", str(n), ""


def bigger_smaller(rng):
    a, b = rng.sample(range(0, 101 if rng.random() < 0.5 else 21), 2)
    if rng.random() < 0.5:
        return f"the bigger of {a} and {b} is ", str(max(a, b)), ""
    return f"the smaller of {a} and {b} is ", str(min(a, b)), ""


def letter_number(rng):
    i = rng.randrange(26)
    if rng.random() < 0.5:
        return f"{LETTERS[i]} is letter ", str(i + 1), ""
    return f"letter {i + 1} is ", LETTERS[i], ""


def odd_even(rng):
    kind, first = pick(rng, (("odd", 1), ("even", 2)))
    return f"{kind} numbers: " + run(range(first, 21, 2)), "", ""


# ----------------------------------------------------------------------------
# stage 3: math


def add(rng):
    a, b = rng.randint(0, 20), rng.randint(0, 20)
    if rng.random() < 0.5:
        return f"{a} + {b} = ", str(a + b), ""
    return f"{a} plus {b} is ", str(a + b), ""


def subtract(rng):
    a = rng.randint(0, 20)
    b = rng.randint(0, a)
    if rng.random() < 0.5:
        return f"{a} - {b} = ", str(a - b), ""
    return f"{a} minus {b} is ", str(a - b), ""


def multiply(rng):
    a, b = rng.randint(0, 10), rng.randint(0, 10)
    if rng.random() < 0.5:
        return f"{a} x {b} = ", str(a * b), ""
    return f"{a} times {b} is ", str(a * b), ""


def divide(rng):
    b, q = rng.randint(1, 10), rng.randint(0, 10)
    return f"{b * q} / {b} = ", str(q), ""


def double_half(rng):
    n = rng.randint(0, 20)
    if rng.random() < 0.5:
        return f"double {n} is ", str(2 * n), ""
    return f"half of {2 * n} is ", str(n), ""


# ----------------------------------------------------------------------------
# stage 4: words


def spelled(rng):
    w = pick(rng, SPELLING_WORDS)
    return f"{w} is spelled ", spell(w), ""


def spells(rng):
    w = pick(rng, SPELLING_WORDS)
    return f"{spell(w)} spells ", w, ""


def starts_ends(rng):
    w = pick(rng, SPELLING_WORDS)
    if rng.random() < 0.6:
        return f"{w} starts with ", w[0], ""
    return f"{w} ends with ", w[-1], ""


def letter_count(rng):
    w = pick(rng, SPELLING_WORDS)
    return f"{w} has ", amount(len(w), "letter"), ""


def is_for(rng):
    c = pick(rng, LETTERS)
    return f"{c} is for ", ABC_WORDS[c], ""


def animal_sound(rng):
    a = pick(rng, [a for a, (s, _) in ANIMALS.items() if s])
    return f"{article(a)} {a} says ", ANIMALS[a][0], ""


def animal_legs(rng):
    a = pick(rng, list(ANIMALS))
    return f"{article(a)} {a} has ", amount(ANIMALS[a][1], "leg"), ""


def color_of(rng):
    subject, verb, color = pick(rng, COLOR_FACTS)
    return f"{subject} {verb} ", color, ""


def category(rng):
    w = pick(rng, list(WORD_CATEGORY))
    cat = WORD_CATEGORY[w]
    subject = w if cat == "color" else the_thing(w)  # "red is a color", "milk is a food"
    return f"{subject} is {article(cat)} ", cat, ""


def opposite(rng):
    w = pick(rng, list(OPPOSITE))
    return f"the opposite of {w} is ", OPPOSITE[w], ""


def one_two(rng):
    w = pick(rng, list(IRREGULAR_PLURALS) + ITEMS + list(ANIMALS))
    return f"one {w}, two ", plural(w), ""


def rhyme(rng):
    words = pick(rng, RHYMES).split()
    rng.shuffle(words)
    return " and ".join(words[:rng.randint(2, len(words))]) + " rhyme", "", ""


def little_sentence(rng):
    a = pick(rng, list(ANIMALS))
    return f"the {pick(rng, ADJECTIVES)} {a} {pick(rng, VERBS)} in the {pick(rng, PLACES)}.", "", ""


# ----------------------------------------------------------------------------
# stage 5: world


def sequence_run(rng):
    seq = DAYS if rng.random() < 0.5 else MONTHS
    n = rng.randint(4, len(seq))
    i = rng.randrange(len(seq) - n + 1)
    part = seq[i:i + n]
    k = rng.randint(3, n - 1)
    rest = (" " + run(part[k + 1:])) if k + 1 < n else None
    return run(part[:k]) + " ", part[k], rest


def after_before_word(rng):
    seq = pick(rng, (DAYS, MONTHS, SEASONS))
    i = rng.randrange(len(seq))
    if rng.random() < 0.5:
        return f"after {seq[i]} comes ", seq[(i + 1) % len(seq)], ""
    return f"before {seq[i]} comes ", seq[i - 1], ""


def today_tomorrow(rng):
    i = rng.randrange(7)
    return f"today is {DAYS[i]}, tomorrow is ", DAYS[(i + 1) % 7], ""


def world_fact(rng):
    subject, n, unit = pick(rng, WORLD_FACTS)
    return f"{subject} has ", amount(n, unit), ""


def story_add(rng):
    name, item = pick(rng, NAMES), pick(rng, ITEMS)
    a, b = rng.randint(1, 10), rng.randint(1, 10)
    story = f"{name} has {amount(a, item)}. {name} gets {b} more. now {name} has "
    return story, amount(a + b, item) + ".", ""


def story_sub(rng):
    name, item = pick(rng, NAMES), pick(rng, ITEMS)
    a = rng.randint(2, 12)
    b = rng.randint(1, a)
    story = f"{name} has {amount(a, item)}. {name} gives away {b}. now {name} has "
    return story, amount(a - b, item) + ".", ""


def story_count(rng):
    animals = rng.sample(list(ANIMALS), rng.randint(2, 5))
    listed = ", ".join(f"{article(a)} {a}" for a in animals[:-1])
    listed += f" and {article(animals[-1])} {animals[-1]}"
    return f"i see {listed}. that is ", amount(len(animals), "animal") + ".", ""


def season_months(rng):
    return "spring, summer, fall and winter are the four seasons", "", ""


# ----------------------------------------------------------------------------
# stage 6: talk. Morpheus answers questions about everything it learned.

UNKNOWN_ANSWER = "i have not learned that yet. will you teach me?"
UNKNOWN_TOPICS = (
    "the capital of france|the weather today|the time|love|a computer|a planet|python|money|"
    "the internet|a rocket|a volcano|the president|gravity|a galaxy|a country|music|the news|"
    "a car engine|a black hole|the stock market|a castle|the best movie|art|a dream"
).split("|")
UNKNOWN_QUESTIONS = (
    "who won the game|how far away is the moon|what should i eat|"
    "why is the sky blue|can you drive a car|how old are you|what is the meaning of life|"
    "can you see me|where is the park|how does a phone work|what will happen tomorrow|who am i"
).split("|")
GREETINGS = "hi|hello|hey|good morning|good evening|hi there|hello morpheus|hey morpheus".split("|")
JOKES = [
    "why did the cat sit on the computer? to keep an eye on the mouse!",
    "what do you call a sleeping bull? a bulldozer!",
    "why is 6 afraid of 7? because 7 8 9!",
    "what has 4 legs but can not walk? a table!",
]


def qa(question, answer, rng):
    if rng.random() < 0.2:
        question = question.rstrip("?")
    return f"user: {question}\nmorpheus: ", answer, ""


def talk_identity(rng):
    q, a = pick(rng, (
        ("who are you?", "i am morpheus. i learned from abc and 123."),
        ("what is your name?", "my name is morpheus."),
        ("what are you?", "i am morpheus, a tiny mind that grew from letters and numbers."),
        ("what can you do?", "i can count, do math, spell words and tell you about letters, "
                             "animals, colors and days."),
        ("help", "ask me about letters, numbers, math, spelling, animals, colors or days."),
        ("how are you?", "i am good, thank you. how can i help?"),
        ("thank you", "you are welcome!"), ("thanks", "you are welcome!"),
        ("bye", "goodbye! sweet dreams."), ("goodbye", "goodbye! sweet dreams."),
        ("good night", "good night! sweet dreams."),
        ("where do you live?", "i live inside your computer."),
        ("who made you?", "i was raised on abc and 123, one lesson at a time."),
        ("tell me a joke", JOKES[0]), ("tell me another joke", JOKES[2]),
        ("tell me a number joke", JOKES[2]), ("tell me an animal joke", JOKES[1]),
        (pick(rng, GREETINGS), "hello! ask me about letters, numbers, words or math."),
    ))
    return qa(q, a, rng)


def talk_letters(rng):
    kind = rng.randrange(5)
    if kind == 0:
        p, a, _ = after_letter(rng)
        return qa(f"what comes after {p.split()[1]}?", p + a + ".", rng)
    if kind == 1:
        p, a, _ = before_letter(rng)
        return qa(f"what comes before {p.split()[1]}?", p + a + ".", rng)
    if kind == 2:
        q = pick(rng, ("say the abc", "sing the alphabet", "what is the alphabet?", "say your abc"))
        return qa(q, run(LETTERS) + ".", rng)
    if kind == 3:
        i = rng.randrange(26)
        return qa(f"what number letter is {LETTERS[i]}?", f"{LETTERS[i]} is letter {i + 1}.", rng)
    c = pick(rng, LETTERS)
    return qa(f"what is {c} for?", f"{c} is for {ABC_WORDS[c]}.", rng)


def talk_numbers(rng):
    kind = rng.randrange(6)
    if kind == 0:
        n = rng.randint(2, 20)
        return qa(f"count to {n}", run(range(1, n + 1)) + ".", rng)
    if kind == 1:
        a = rng.randint(0, 60)
        b = a + rng.randint(3, 12)
        return qa(f"count from {a} to {b}", run(range(a, b + 1)) + ".", rng)
    if kind == 2:
        p, a, _ = after_number(rng)
        return qa(f"what comes after {p.split()[1]}?", p + a + ".", rng)
    if kind == 3:
        n = rng.randrange(101)
        return qa(f"what is {n} in words?", f"{n} is {number_word(n)}.", rng)
    if kind == 4:
        n = rng.randint(1, 10)
        return qa(f"how many stars? {' '.join('*' * n)}", f"i count {amount(n, 'star')}.", rng)
    a, b = rng.sample(range(101), 2)
    word = pick(rng, ("bigger", "smaller"))
    ans = max(a, b) if word == "bigger" else min(a, b)
    return qa(f"which is {word}, {a} or {b}?", f"the {word} of {a} and {b} is {ans}.", rng)


def talk_math(rng):
    prompt, answer, _ = pick(rng, (add, subtract, multiply, divide, double_half))(rng)
    expr = prompt.rstrip("= ").removesuffix(" is")  # "3 + 4", "3 plus 4", "double 6"
    # the answer mirrors the question: symbols in, symbols out; words in, words out
    if "=" in prompt and rng.random() < 0.3:
        return qa(expr.replace(" ", "") + "?", prompt + answer + ".", rng)  # "3+4?"
    return qa(f"what is {expr}?", prompt + answer + ".", rng)


def talk_words(rng):
    kind = rng.randrange(8)
    if kind == 0:
        w = pick(rng, SPELLING_WORDS)
        return qa(pick(rng, (f"how do you spell {w}?", f"spell {w}")), f"{w} is spelled {spell(w)}.", rng)
    if kind == 1:
        a = pick(rng, [a for a, (s, _) in ANIMALS.items() if s])
        q = pick(rng, (f"what does {article(a)} {a} say?", f"what sound does {article(a)} {a} make?"))
        return qa(q, f"{article(a)} {a} says {ANIMALS[a][0]}.", rng)
    if kind == 2:
        subject, verb, color = pick(rng, COLOR_FACTS)
        return qa(f"what color {verb} {subject}?", f"{subject} {verb} {color}.", rng)
    if kind == 3:
        w = pick(rng, list(OPPOSITE))
        return qa(f"what is the opposite of {w}?", f"the opposite of {w} is {OPPOSITE[w]}.", rng)
    if kind == 4:
        a = pick(rng, list(ANIMALS))
        return qa(f"how many legs does {article(a)} {a} have?",
                  f"{article(a)} {a} has {amount(ANIMALS[a][1], 'leg')}.", rng)
    if kind == 5:
        prompt, cat, _ = category(rng)
        subject = prompt.split(" is ")[0]
        return qa(f"what is {subject}?", prompt + cat + ".", rng)
    if kind == 6:
        w = pick(rng, list(IRREGULAR_PLURALS) + ITEMS + list(ANIMALS))
        return qa(f"what is the plural of {w}?", f"one {w}, two {plural(w)}.", rng)
    w = pick(rng, SPELLING_WORDS)
    return qa(f"how many letters are in {w}?", f"{w} has {amount(len(w), 'letter')}.", rng)


def talk_world(rng):
    kind = rng.randrange(5)
    if kind == 0:
        p, a, _ = after_before_word(rng)
        return qa(f"what comes {p.split()[0]} {p.split()[1]}?", p + a + ".", rng)
    if kind == 1:
        subject, n, unit = pick(rng, WORLD_FACTS)
        return qa(f"how many {plural(unit)} does {subject} have?", f"{subject} has {amount(n, unit)}.", rng)
    if kind == 2:
        name, item = pick(rng, NAMES), pick(rng, ITEMS)
        a, b = rng.randint(1, 10), rng.randint(1, 10)
        return qa(f"{name} has {amount(a, item)} and gets {b} more. how many {plural(item)} now?",
                  f"{name} has {amount(a + b, item)}.", rng)
    if kind == 3:
        return qa(pick(rng, ("what are the days of the week?", "say the days of the week")),
                  run(DAYS) + ".", rng)
    i = rng.randrange(7)
    return qa(f"if today is {DAYS[i]}, what is tomorrow?", f"tomorrow is {DAYS[(i + 1) % 7]}.", rng)


def talk_unknown(rng):
    if rng.random() < 0.5:
        q = f"what is {pick(rng, UNKNOWN_TOPICS)}?"
    else:
        q = pick(rng, UNKNOWN_QUESTIONS) + "?"
    return qa(q, UNKNOWN_ANSWER, rng)


def talk_gibberish(rng):
    q = "".join(pick(rng, LETTERS) for _ in range(rng.randint(4, 9)))
    return qa(q, "i do not understand. can you say it another way?", rng)


TALK = (talk_identity, talk_letters, talk_numbers, talk_math, talk_words, talk_world)


def conversation(rng):
    """A few turns in a row, so Morpheus learns to answer the latest question."""
    turns = []
    for _ in range(rng.randint(2, 4)):
        fn = talk_unknown if rng.random() < 0.1 else pick(rng, TALK)
        p, a, r = fn(rng)
        turns.append(p + a + r)
    return "\n".join(turns), "", ""


# ----------------------------------------------------------------------------
# the stages


class Stage:
    """A school year. `facts` is a list of (lesson weight, exam weight, fact function)."""

    def __init__(self, name, title, facts):
        self.name = name
        self.title = title
        self.facts = facts
        self._lesson_w = [lw for lw, _, _ in facts]
        self._exam_facts = [(ew, fn) for _, ew, fn in facts if ew > 0]

    def lesson(self, rng):
        fn = rng.choices(self.facts, weights=self._lesson_w)[0][2]
        prompt, answer, rest = fn(rng)
        return prompt + answer + (rest or "")

    def question(self, rng):
        """An exam question: (prompt, expected answer). The answer ends in a newline
        when the fact is complete, so Morpheus is also graded on knowing when to stop."""
        fn = rng.choices([fn for _, fn in self._exam_facts], weights=[w for w, _ in self._exam_facts])[0]
        prompt, answer, rest = fn(rng)
        return prompt, answer + ("\n" if rest == "" else "")


STAGES = [
    Stage("letters", "the ABCs", [
        (2, 0, alphabet), (4, 2, letter_run), (4, 3, after_letter), (4, 3, before_letter),
        (3, 2, big_little), (0.3, 0, vowels),
    ]),
    Stage("numbers", "1 2 3s", [
        (5, 2, count_run), (3, 2, after_number), (3, 2, before_number), (2, 1, number_to_word),
        (2, 1, word_to_number), (2, 2, count_stars), (2, 2, bigger_smaller), (2, 1, letter_number),
        (0.3, 0, odd_even),
    ]),
    Stage("math", "adding, taking away, times", [
        (4, 3, add), (3, 2, subtract), (3, 2, multiply), (1, 1, divide), (1, 1, double_half),
    ]),
    Stage("words", "spelling and the world of words", [
        (4, 2, spelled), (2, 1, spells), (2, 1, starts_ends), (2, 1, letter_count), (2, 1, is_for),
        (2, 1, animal_sound), (2, 1, animal_legs), (2, 1, color_of), (2, 1, category),
        (2, 1, opposite), (2, 1, one_two), (0.5, 0, rhyme), (1, 0, little_sentence),
    ]),
    Stage("world", "days, facts and little stories", [
        (2, 1, sequence_run), (2, 1, after_before_word), (1, 1, today_tomorrow), (2, 1, world_fact),
        (2, 1, story_add), (2, 1, story_sub), (1, 1, story_count), (0.2, 0, season_months),
    ]),
    Stage("talk", "conversation", [
        (1, 1, talk_identity), (2, 1, talk_letters), (2, 1, talk_numbers), (3, 1, talk_math),
        (4, 2, talk_words), (2, 1, talk_world), (1, 0.5, talk_unknown), (0.5, 0.2, talk_gibberish),
        (2, 0, conversation),
    ]),
]
STAGE_BY_NAME = {s.name: s for s in STAGES}


class Library(Stage):
    """Optional extra stage: your own text files. There is no exam; Morpheus just reads,
    and graduates after a fixed number of steps."""

    def __init__(self, text, name="library"):
        super().__init__(name, "reading your books", [])
        self.text = text

    def lesson(self, rng):
        n = rng.randint(64, 256)
        i = rng.randrange(max(1, len(self.text) - n))
        return self.text[i:i + n]


def exam(stage, n, seed):
    rng = random.Random(seed)
    return [stage.question(rng) for _ in range(n)]
