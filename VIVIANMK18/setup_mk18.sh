#!/bin/bash
# VIVIAN Mk18 setup — run once on the Pi after copying this folder over.
#
#   bash setup_mk18.sh
#
# Installs the Python dependencies (Anthropic SDK; Kokoro kept for fallback)
# and downloads the Piper TTS binary + voice models. Existing Mk17 deps
# (torch, ultralytics, insightface, ...) are assumed to already be installed;
# requirements.txt has the full list if starting from a clean OS image.

set -e
cd "$(dirname "$0")"

echo "=== Installing Python packages ==="
python3 -m pip install --upgrade anthropic kokoro-onnx soundfile

echo "=== Downloading Piper TTS binary (aarch64) ==="
if [ ! -x piper/piper ]; then
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz"
    curl -L -o /tmp/piper.tar.gz "$PIPER_URL"
    tar xzf /tmp/piper.tar.gz          # extracts a 'piper/' dir with the binary + libs + espeak-ng-data
    rm -f /tmp/piper.tar.gz
fi

echo "=== Downloading Piper voice models ==="
mkdir -p voices
# Assistant voice: en_GB alba (medium). Sentry voice: dnhkng's trained GLaDOS.
ALBA="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx"
GLADOS="https://github.com/dnhkng/GLaDOS/releases/download/0.1/glados.onnx"
GLADOS_CFG="https://raw.githubusercontent.com/dnhkng/GLaDOS/main/models/TTS/glados.json"
[ -f voices/alba.onnx ]        || curl -L -o voices/alba.onnx        "$ALBA"
[ -f voices/alba.onnx.json ]   || curl -L -o voices/alba.onnx.json   "$ALBA.json"
[ -f voices/glados.onnx ]      || curl -L -o voices/glados.onnx      "$GLADOS"
[ -f voices/glados.onnx.json ] || curl -L -o voices/glados.onnx.json "$GLADOS_CFG"

echo "=== Downloading Kokoro model files (fallback engine) ==="
mkdir -p models
BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
[ -f models/kokoro-v1.0.onnx ] || curl -L -o models/kokoro-v1.0.onnx "$BASE/kokoro-v1.0.onnx"
[ -f models/voices-v1.0.bin ]  || curl -L -o models/voices-v1.0.bin  "$BASE/voices-v1.0.bin"

echo "=== Ensuring whisper base.en model is present ==="
WHISPER_MODELS="$HOME/whisper.cpp/models"
if [ -d "$WHISPER_MODELS" ] && [ ! -f "$WHISPER_MODELS/ggml-base.en.bin" ]; then
    echo "Downloading whisper base.en (better accuracy than tiny.en)..."
    ( cd "$HOME/whisper.cpp" && bash ./models/download-ggml-model.sh base.en ) || \
        echo "WARNING: could not download base.en — set whisper.model_path back to tiny.en or download manually."
fi

echo "=== Regenerating sentry voice lines with the new voice ==="
python3 Sentry/generate_audio.py --overwrite || {
    echo "WARNING: generate_audio.py failed — run it manually after fixing the issue."
}

echo ""
echo "Done. Remaining manual steps:"
echo "  1. Add your Anthropic API key to config.yaml (anthropic.api_key)"
echo "  2. Update systemd:  sudo cp vivian.service /etc/systemd/system/ && sudo systemctl daemon-reload"
echo "  3. Restart:         sudo systemctl restart vivian"
