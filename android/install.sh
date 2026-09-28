#!/data/data/com.termux/files/usr/bin/bash
# Install Morpheus on Android, inside Termux (get Termux from F-Droid, not the Play Store):
#
#   curl -fsSL https://raw.githubusercontent.com/brahmzj/nanoGPT/claude/clever-heisenberg-qmoont/android/install.sh | bash
#
# What it does: installs python + numpy (no PyTorch needed on the phone), downloads Morpheus
# with its trained brain, adds a `morpheus` command, home-screen shortcuts (Termux:Widget),
# and a learning job that only runs while the phone is charging and on Wi-Fi (Termux:API).
# Run it again at any time to update.
set -e
REPO="${MORPHEUS_REPO:-https://github.com/brahmzj/nanoGPT}"
BRANCH="${MORPHEUS_BRANCH:-claude/clever-heisenberg-qmoont}"
APP="${MORPHEUS_APP:-$HOME/morpheus-app}"
BIN="${PREFIX:-/data/data/com.termux/files/usr}/bin"

say() { printf '\n\033[1;35m[morpheus]\033[0m %s\n' "$*"; }

if [ -z "$TERMUX_VERSION" ] && [ ! -d /data/data/com.termux ]; then
  say "this installer is for Termux on Android. On a computer:  git clone $REPO && python -m morpheus chat"
  exit 1
fi

say "1/5 installing python and numpy (one time, about 60 MB)"
pkg update -y
pkg install -y python python-numpy git
pkg install -y termux-api || true  # voice, battery, notifications, scheduling (with the Termux:API app)

say "2/5 downloading Morpheus"
if [ -d "$APP/.git" ]; then
  git -C "$APP" pull --ff-only
else
  git clone --depth 1 --branch "$BRANCH" "$REPO" "$APP"
fi

say "3/5 adding the 'morpheus' command"
cat > "$BIN/morpheus" <<LAUNCHER
#!/data/data/com.termux/files/usr/bin/sh
PYTHONPATH="$APP" exec python -m morpheus "\$@"
LAUNCHER
chmod +x "$BIN/morpheus"

say "4/5 home-screen shortcuts (add the Termux:Widget widget to your home screen)"
mkdir -p "$HOME/.shortcuts/tasks"
printf '#!/data/data/com.termux/files/usr/bin/sh\nmorpheus chat\n' > "$HOME/.shortcuts/Morpheus"
printf '#!/data/data/com.termux/files/usr/bin/sh\nmorpheus chat --voice\n' > "$HOME/.shortcuts/Morpheus voice"
printf '#!/data/data/com.termux/files/usr/bin/sh\nsh "%s/android/learn-job.sh" --force\n' "$APP" > "$HOME/.shortcuts/tasks/Morpheus learn now"
chmod +x "$HOME/.shortcuts/Morpheus" "$HOME/.shortcuts/Morpheus voice" "$HOME/.shortcuts/tasks/Morpheus learn now"

say "5/5 learning while you sleep: every 6 hours, only when charging and on Wi-Fi"
if command -v termux-job-scheduler >/dev/null 2>&1 && \
   termux-job-scheduler --job-id 4242 --script "$APP/android/learn-job.sh" --period-ms 21600000 \
     --charging true --network unmetered --battery-not-low true --persisted true; then
  say "scheduled. (see or cancel it: termux-job-scheduler --pending / --cancel --job-id 4242)"
else
  say "not scheduled: install the Termux:API app from F-Droid, then run this installer again"
fi

# a Downloads/morpheus folder: save .txt/.md/.html files there and Morpheus will read them
termux-setup-storage >/dev/null 2>&1 || true
mkdir -p "$HOME/storage/downloads/morpheus" 2>/dev/null || true

morpheus status || true
say "done! talk to Morpheus with:  morpheus chat      (or: morpheus chat --voice)"
say "the internet stays OFF until you allow it:  morpheus settings allow_internet on"
