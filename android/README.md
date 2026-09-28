# Morpheus on Android

Morpheus runs **on the phone itself**: no cloud, no account, no PyTorch. It needs about
60 MB for Python and numpy plus a **153 KB** brain (1.58 bits per weight). It chats, helps as a personal assistant, and
keeps learning from files you give it, from things you teach it, and from the internet
**only if you allow it**. Learning only happens while the phone is charging.

## One tap: the Morpheus app (easiest)

1. On your phone, open this link:
   **<https://github.com/brahmzj/nanoGPT/raw/claude/clever-heisenberg-qmoont/android/Morpheus.apk>**
   It's a 204 KB download that includes the 153 KB brain.
2. Tap the downloaded file. The first time, Android asks you to allow installs from your
   browser: allow it and go back. Play Protect may say the app is from an unknown developer;
   tap **Install anyway**. It's your own app, built from this repository.
3. Tap the **☾ Morpheus** icon.

What the app does, entirely on the phone, with no account and nothing uploaded:

- **Chat** with the brain (type, or tap 🎤 to talk). 🔊 makes it answer out loud.
- **Assistant skills**: notes, to-do list, alarms and timers (your Clock app), time and date,
  battery, math that is checked by a calculator, web search.
- **Teach it**: when it says "i have not learned that yet", the question is ready in the text box.
  Type the answer after `=` and send. From then on it answers that question.
- **Look things up** on Simple English Wikipedia, but only after you tap 🌐 to allow the internet.
  The app's only permissions are internet (for that) and setting alarms.

The app remembers what you teach it, but its brain doesn't retrain on the phone yet. For
Morpheus to keep *learning*, meaning its brain weights change while it reads your files and
Wikipedia, use the Termux version below. It installs alongside the app and runs the same brain.

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
