#!/data/data/com.termux/files/usr/bin/sh
# One Morpheus learning session. Started by the Android job scheduler (only while charging and
# on Wi-Fi), or by the "Morpheus learn now" shortcut (which passes --force).
APP="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${MORPHEUS_HOME:-$HOME/.morpheus}/learn.log"
mkdir -p "$(dirname "$LOG")"
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock
PY="$(command -v python || command -v python3)"
RESULT="$(PYTHONPATH="$APP" "$PY" -m morpheus learn --quiet "$@" 2>&1 | tail -n 1)"
echo "$RESULT" >> "$LOG"
command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock
case "$RESULT" in
  *kept*|*"rolled back"*|*Error*)
    command -v termux-notification >/dev/null 2>&1 && \
      termux-notification --id morpheus-learn --title "Morpheus studied" --content "$RESULT" ;;
esac
exit 0
