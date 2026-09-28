# Morpheus on Android

Morpheus runs **on the phone itself**: no cloud, no account, no PyTorch. It needs about
60 MB for Python and numpy plus a **153 KB** brain (1.58 bits per weight). It chats, helps as a personal assistant, and
keeps learning from files you give it, from things you teach it, and from the internet
**only if you allow it**. Learning only happens while the phone is charging.

## One tap: the Morpheus app (easiest)

1. On your phone, open this link:
   **<https://github.com/brahmzj/nanoGPT/raw/claude/clever-heisenberg-qmoont/android/Morpheus.apk>**
   It's a small download (about 330 KB) that includes the 153 KB brain and its curriculum for studying.
   If the link won't download, see the note below.
2. Tap the downloaded file. The first time, Android asks you to allow installs from your
   browser: allow it and go back. Play Protect may say the app is from an unknown developer;
   tap **Install anyway**. It's your own app, built from this repository.
3. Tap the **☾ Morpheus** icon.

If the link won't download, it was probably opened inside another app's built-in browser. Copy it
into Chrome itself, or use
<https://raw.githubusercontent.com/brahmzj/nanoGPT/claude/clever-heisenberg-qmoont/android/Morpheus.apk>.
Chrome's "file might be harmful" prompt is normal for any `.apk`: tap **Download anyway**.
Installing a new version over the old one keeps your notes and everything it learned.

What the app does, entirely on the phone, with no account and nothing uploaded:

- **Chat** with the brain (type, or tap 🎤 to talk). 🔊 makes it answer out loud.
- **Assistant skills**: notes, to-do list, alarms and timers (your Clock app), time and date,
  battery, math that is checked by a calculator, web search.
- **Teach it**: when it says "i have not learned that yet", the question is ready in the text box.
  Type the answer after `=` and send. It answers from memory right away, and studies it into its
  brain at the next learning session.
- **Look things up** on Simple English Wikipedia, but only after you tap 🌐 to allow the internet.
- **Share → Morpheus** from any app: text is read at the next session, and links too once 🌐 is on.

The app's permissions: internet (only used after you tap 🌐), setting alarms, and restarting its
learning schedule after a reboot.

### Self-learning

Morpheus studies **by itself, only while the phone is charging** (about every 6 hours, via
Android's job scheduler), or right now when you tap 🧠. What it learns from:

1. **What you teach it**: `/teach question = answer`, or the one-tap teach.
2. **Its own mistakes**: when the calculator corrects its math, the correction becomes a lesson.
3. **Its own curiosity**: every question it could not answer is remembered. With 🌐 on, a session
   looks those up on Simple English Wikipedia and studies the answers. Ask it "what are you
   curious about?"
4. **What it reads**: your interests ("learn about volcanoes") and where they lead. See
   *Browsing* below.
5. **What you share with it**: Share → Morpheus from any app.

How a session keeps it safe:

- **Its brain stays frozen.** New knowledge goes into small low-rank adapters (LoRA, about 33K
  numbers, a 130 KB file) on top of the 153 KB ternary brain. We tried retraining the ternary
  weights themselves first. That learned new facts, but each flip of a -1/0/+1 weight is a big
  jump, and it quietly broke how it spells new words ("brahm" became "brahc"). With adapters,
  every skill we tested survived.
- **It keeps reviewing its ABCs** from a 12,000-lesson snapshot of its curriculum, matching its
  own pre-session answers (learning without forgetting).
- **It sits an exam before and after**, and **rolls back** any session that costs more than 2
  points, measured against both its last brain and its very first one.

In testing, one 200-step session on the real brain took 84 seconds on a 4-core computer (expect a
few minutes on a phone):

| question                         | before                           | after                           |
|----------------------------------|----------------------------------|---------------------------------|
| what is the capital of france? (taught) | i have not learned that yet.     | paris is the capital of france. |
| what is the capital of spain?    | i have not learned that yet.     | i have not learned that yet.    |
| how do you spell zebra? / brahm? | spelled right                    | spelled right                   |
| what color are apples?           | apples are red.                  | apples are red.                 |
| curriculum exam                  | 97.1%                            | 97.1%                           |

### Browsing: how Morpheus reads and teaches itself

Tap 🌐 to let it use the internet, then tell it what to read:

```
you> learn about volcanoes
morpheus> ok! i will read about volcanoes the next time i study (while you charge me, or tap the brain now).
```

Each session (charging, or 🧠) it reads about 6 Simple English Wikipedia pages: first the
questions it could not answer (its own curiosity), then your interests, then a few pages they
link to (Volcano → Lava → Mount Vesuvius…). Links you share go in too. Then it **absorbs** what it read:

- **Every sentence goes into a searchable memory.** Ask "what do you know about lava?", "tell
  me about the eiffel tower", or anything its brain does not know, and it answers with the
  sentence it read, word for word. Nothing is made up.
- **Simple sentences become questions and answers it studies.** For example "paris is the capital
  and largest city of france." gives "what is the capital of france?" and "what is the largest city
  of france?". It studies **about 4 new ones per session** plus a little review of ones it already
  mastered (spaced repetition). In testing, cramming a whole page into one session made it forget
  its ABCs, and the guard rolled that back.
- **If its brain half-learned a fact and says something its reading contradicts, the reading wins**
  (grounding). "what is a volcano?" is answered with the sentence that *defines* a volcano.

"what are you reading about?" shows its reading list and what it read lately.

**Tested on the real brain**, over three sessions in a row with sample Wikipedia-style pages about
Paris and volcanoes. All three sessions were kept, it mastered 6 facts, and the exam moved 95.7% →
93.9%, within its 2-point budget. Afterwards every answer was right: "paris is the capital and
largest city of france.", "the eiffel tower is 330 metres tall.", "lava is hot melted rock.",
"a volcano is a mountain where lava comes out of the ground.". It still says "i have not learned
that yet" about Spain, and it still spells zebra, brahm and river.

**The guard is strict:** a session is kept only if the exam stays within 2 points of *both* its last
and its first score, and no single skill (letters, numbers, math, words, world, talk, spelling new
words) drops more than 3 questions. When it is below its first score, the next session heals first:
fewer new facts, more review.

Ask "what did you learn?" to hear about its last session. **Long-press 🧠** to send it back to the
brain it shipped with (your notes, list and taught answers are kept).

About signing: the app is signed with a public debug key (`android/app/morpheus-debug.keystore`),
so new versions install over old ones and keep your notes. Because the key is public, only install
`Morpheus.apk` from this repository. To rebuild it yourself: `bash android/app/build.sh` (needs a JDK
and `apt install aapt apksigner zipalign dalvik-exchange android-sdk-platform-23`).

## The full version in Termux (keeps learning)

1. Install **Termux** from F-Droid: <https://f-droid.org/packages/com.termux/>.
   Don't use the Play Store version, which is outdated.
2. Optional but recommended, also from F-Droid:
   - **Termux:API** (<https://f-droid.org/packages/com.termux.api/>) for voice, battery, alarms and scheduled learning
   - **Termux:Widget** (<https://f-droid.org/packages/com.termux.widget/>) for one-tap home-screen shortcuts
3. Open Termux and paste:

   ```sh
   curl -fsSL https://raw.githubusercontent.com/brahmzj/nanoGPT/claude/clever-heisenberg-qmoont/android/install.sh | bash
   ```

   When Android asks whether Termux may access your files, allow it. That's how Morpheus
   reads files you put in `Download/morpheus`.
4. Say hello:

   ```sh
   morpheus chat
   ```

Running the same command again later updates Morpheus. Your brain and notes are kept, since
they live in `~/.morpheus`, separate from the app.

## Use it

| you say / type                                   | what happens                                          |
|--------------------------------------------------|-------------------------------------------------------|
| `morpheus chat`                                  | talk (type `help` for the skills)                     |
| `morpheus chat --voice`                          | talk out loud: it listens and answers aloud (Termux:API) |
| `morpheus chat --ask "what is 7 x 8?"`           | one question, one answer                              |
| "remember that my locker code is 4071"           | saved to your notes; ask "what is my locker code?"    |
| "add milk to my list" / "what's on my list?"     | your to-do list                                       |
| "remind me to stretch in 20 minutes"             | a timer on your phone's Clock app                     |
| "set an alarm for 6:45 am"                       | an alarm on your phone's Clock app                    |
| "what time is it?" / "how much battery?"         | clock and battery                                     |
| "what is 1234 x 5678?"                           | Morpheus answers, a calculator checks it              |
| "tell me about volcanoes"                        | looks it up on Simple English Wikipedia (only if allowed) |
| "search the web for pizza near me"               | opens your browser                                    |
| `/teach what is my dog called? = rex is your dog.` | a lesson for the next learning session              |

Want it even smaller? Two extreme brains ship too: `brains/morpheus-nano-binary.morph` (100 KB,
93.5% on the exams) and `brains/morpheus-loop.morph` (85 KB, 91.4%). To make one your Morpheus,
including for learning, copy it into place:
`cp ~/morpheus-app/brains/morpheus-loop.morph ~/.morpheus/brain.morph`.

The home-screen widget (Termux:Widget) gets three shortcuts: **Morpheus** (chat),
**Morpheus voice**, and **Morpheus learn now**.

## How it keeps learning

```sh
morpheus learn            # one session now (add --force if the phone is not charging)
morpheus status           # brain, library, power, the last sessions
morpheus undo             # go back to the brain before the last session
morpheus settings         # see everything Morpheus is allowed to do
```

- **Automatically:** the installer schedules a session every 6 hours that Android only starts
  while the phone is **charging and on Wi-Fi**. The result appears as a notification and in
  `~/.morpheus/learn.log`.
- **Files:** save `.txt`, `.md` or `.html` files into your phone's `Download/morpheus` folder
  (or `~/.morpheus/inbox`). Each file is read once, then kept LZMA-compressed in
  `~/.morpheus/library`.
- **Teaching:** `/teach question = answer` in chat, or `morpheus teach "question" "answer"`.
  When Morpheus answers a sum wrong, the calculator's correction is saved as a lesson automatically.
- **Internet (off by default):** `morpheus settings allow_internet on`. Then learning sessions
  read a few random Simple English Wikipedia articles, and "tell me about ..." can look things up.
  Turn it off again with `morpheus settings allow_internet off`. You can change the sources:
  `morpheus settings sources '["https://simple.wikipedia.org/api/rest_v1/page/random/summary"]'`.
- **It never gets worse.** Every session keeps reviewing the ABC curriculum, sits an exam
  before and after, and is **rolled back** if it forgot more than 2 points compared with the
  last brain *or* the original one. `morpheus undo` is always there too.

## Energy

- Chatting costs about 1.6 million operations per character, around 150x less than GPT-2 small.
  Skills like notes, lists, alarms and time use almost none.
- Learning runs only while charging. A 200-step session is a few minutes of CPU.
- Nothing runs in the background except the scheduled job, which Android starts only while charging.

## Privacy

Everything stays in `~/.morpheus` on your phone: notes, lists, reminders, the brain and what
it has read. Nothing is uploaded. Internet access is used only for the sources you allow, and
only to download.

## Troubleshooting

- **`pkg` errors:** run `termux-change-repo`, pick a mirror, then run the installer again.
- **Voice/battery/alarms do nothing:** install the Termux:API *app* from F-Droid (the
  `termux-api` package alone isn't enough) and open it once.
- **It can't see `Download/morpheus`:** run `termux-setup-storage` and allow access.
- **Learning never starts on its own:** check `termux-job-scheduler --pending`, and that the phone
  was charging on Wi-Fi. Use `morpheus learn --force` to run a session now.
