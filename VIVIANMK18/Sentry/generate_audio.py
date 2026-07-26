#!/usr/bin/env python3
"""Pre-generate all static VIVIAN sentry TTS lines as WAV files.

Mk18: renders locally with Piper in the "sentry" voice profile (GLaDOS) — no
network, no API key needed. Run once on the Pi (or whenever faces are added)
to populate Sentry/audio/ so sentry never synthesizes repeated lines at runtime.

Usage:
    cd /home/cjatkinson/VIVIANMK18
    python3 Sentry/generate_audio.py

    # Force re-render everything (overwrite existing files):
    python3 Sentry/generate_audio.py --overwrite
"""

import argparse
import sys
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

SENTRY_DIR  = Path(__file__).parent
VIVIAN_DIR  = SENTRY_DIR.parent
AUDIO_DIR   = SENTRY_DIR / 'audio'
FACE_DB_DIR = SENTRY_DIR / 'face_database'

sys.path.insert(0, str(VIVIAN_DIR))

FACE_IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

# ============================================================================
# Static lines: output filename (no extension) -> spoken text
# ============================================================================

STATIC_LINES = {
    "person_detected":  "Person detected. Running facial recognition.",
    "unknown_person":   "Unknown person detected. Running threat analysis.",
    "person_no_face":   "Person detected. Unable to identify. Running threat analysis.",
    "tamper_detected":  "Potential tampering detected. Treating as unknown threat.",
    "tamper_alert_sent":"Tamper alert sent. Continuing to monitor.",
    "alert_sent":       "Alert sent. Continuing to monitor.",
    "sentry_deactivated":
                        "Sentry mode deactivated.",
    "sentry_error":     "Sentry mode encountered an error and has been deactivated.",
    "server_error":     "I cannot reach the server right now. Please try again in a minute.",
    "vivian_error":     "I encountered an error. Please try again.",

    # Generic deterrents by threat level — instant fallbacks when the
    # AI-written deterrent is unavailable or synthesis would be too slow
    # (TTS competes with YOLO for CPU during sentry mode).
    "deterrent_low":    "Notice. This vehicle is monitored by camera. The owner has been notified of your presence.",
    "deterrent_medium": "You are being recorded. This activity has been reported to the vehicle's owner.",
    "deterrent_high":   "Warning. Your image and biometric data have been logged. All actions are being recorded, and the owner has been alerted.",
}

# ============================================================================
# Helpers
# ============================================================================

def get_person_names() -> list:
    """Return person names from face_database/ — subfolders AND flat images."""
    if not FACE_DB_DIR.exists():
        return []
    names = set()
    for entry in sorted(FACE_DB_DIR.iterdir()):
        if entry.name.startswith('.'):
            continue
        if entry.is_dir():
            names.add(entry.name)
        elif entry.is_file() and entry.suffix.lower() in FACE_IMAGE_EXTS:
            names.add(entry.stem)
    return sorted(names)


SENTRY_PROFILE = "sentry"   # GLaDOS voice for all sentry-context lines


def process_line(tts, filename: str, text: str, overwrite: bool) -> str:
    """Render one WAV. Returns 'ok', 'skip', or 'fail'."""
    out_path = AUDIO_DIR / f"{filename}.wav"
    if out_path.exists() and not overwrite:
        return 'skip'
    print(f"  Rendering    {filename}.wav")
    print(f"               \"{text}\"")
    if tts.synthesize_to_wav(text, out_path, profile=SENTRY_PROFILE):
        print(f"               -> saved")
        return 'ok'
    print(f"               -> FAILED")
    return 'fail'


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pre-generate VIVIAN sentry TTS audio files")
    parser.add_argument('--overwrite', action='store_true',
                        help="Re-render and overwrite existing WAV files")
    args = parser.parse_args()

    from config import Config
    from tts import TextToSpeech

    config = Config(str(VIVIAN_DIR / 'config.yaml'))
    tts = TextToSpeech(config)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {AUDIO_DIR}")
    print()

    counts = {'ok': 0, 'skip': 0, 'fail': 0}

    # --- Static lines ---
    print("=== Static sentry lines ===")
    for filename, text in STATIC_LINES.items():
        result = process_line(tts, filename, text, args.overwrite)
        counts[result] += 1
    print()

    # --- Per-person welcome lines ---
    print("=== Welcome back lines (per known person) ===")
    names = get_person_names()
    if not names:
        print("  No persons found in face_database/ — skipping")
    for name in names:
        filename = f"welcome_back_{name.lower()}"
        text     = f"Identity confirmed. Welcome back, {name}."
        result   = process_line(tts, filename, text, args.overwrite)
        counts[result] += 1
    print()

    print(
        f"Done: {counts['ok']} rendered, "
        f"{counts['skip']} already existed, "
        f"{counts['fail']} failed"
    )
    if counts['fail']:
        sys.exit(1)


if __name__ == "__main__":
    main()
