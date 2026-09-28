#!/usr/bin/env bash
# Build Morpheus.apk: the one-tap Android app (no Termux, no Python on the phone).
#
#   bash android/app/build.sh [brain.morph]        -> android/Morpheus.apk
#
# Needs a JDK and the Android build tools. Either set ANDROID_HOME to an Android SDK, or on
# Debian/Ubuntu:  sudo apt install aapt apksigner zipalign dalvik-exchange android-sdk-platform-23
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BRAIN="${1:-$REPO/brains/morpheus-nano.morph}"
OUT="$HERE/build"
APK="$REPO/android/Morpheus.apk"
export JAVA_TOOL_OPTIONS=""

# --- find the tools: an Android SDK if there is one, else the Debian/Ubuntu packages
SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-/usr/lib/android-sdk}}"
BT="$(ls -d "$SDK"/build-tools/*/ 2>/dev/null | sort -V | tail -1)"
tool() { if [ -n "$BT" ] && [ -x "$BT/$1" ]; then echo "$BT/$1"; else command -v "$2" || command -v "$1"; fi; }
AAPT2="$(tool aapt2 aapt2)"; ZIPALIGN="$(tool zipalign zipalign)"; APKSIGNER="$(tool apksigner apksigner)"
D8="$(tool d8 d8 || true)"; DX="$(tool dx dalvik-exchange || true)"
JAR="$(ls -d "$SDK"/platforms/android-*/android.jar 2>/dev/null | sort -V | tail -1)"
[ -n "$JAR" ] || { echo "no android.jar found (install android-sdk-platform-23 or set ANDROID_HOME)"; exit 1; }
echo "tools: $(basename "$AAPT2"), ${D8:-$DX}, $(basename "$JAR" .jar) from $(dirname "$JAR")"

rm -rf "$OUT"
mkdir -p "$OUT/assets" "$OUT/classes" "$OUT/dex"

# --- 1. the brain and the icon
python3 "$HERE/export_brain.py" "$BRAIN" "$OUT/assets/brain.bin"
python3 "$HERE/make_icon.py" "$OUT/res"

# --- 2. resources + manifest
"$AAPT2" compile --dir "$OUT/res" -o "$OUT/res.zip"
"$AAPT2" link -o "$OUT/base.apk" -I "$JAR" --manifest "$HERE/AndroidManifest.xml" -R "$OUT/res.zip" \
    -A "$OUT/assets" -0 bin --min-sdk-version 24 --target-sdk-version 34

# --- 3. code
javac -source 8 -target 8 -bootclasspath "$JAR" -Xlint:-options -d "$OUT/classes" \
    $(find "$HERE/src" -name '*.java')
if [ -n "$D8" ]; then
    "$D8" --release --min-api 24 --lib "$JAR" --output "$OUT/dex" $(find "$OUT/classes" -name '*.class')
else
    "$DX" --dex --min-sdk-version=24 --output="$OUT/dex/classes.dex" "$OUT/classes"
fi
cp "$OUT/base.apk" "$OUT/unsigned.apk"
(cd "$OUT/dex" && zip -q -j "$OUT/unsigned.apk" classes.dex)

# --- 4. align and sign (with the public debug key in this folder; see android/README.md)
"$ZIPALIGN" -f -p 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"
"$APKSIGNER" sign --ks "$HERE/morpheus-debug.keystore" --ks-pass pass:morpheus --key-pass pass:morpheus \
    --out "$APK" "$OUT/aligned.apk"
"$APKSIGNER" verify "$APK"
rm -f "$APK.idsig"
echo "built $APK ($(du -k "$APK" | cut -f1) KB)"
