#!/usr/bin/env python3
"""Record real "Hey VIVIAN" samples for threshold tuning.

A synthetically-trained wake model is only as good as the threshold you pick,
and the only honest way to pick one is with audio from the actual cabin, at the
actual mic, with the engine running. Run this in the car.

  # positives — say the wake phrase once per prompt
  python3 tools/record_wake_samples.py --out wake_samples/positive --count 25

  # negatives — normal driving/talking, anything EXCEPT the phrase
  python3 tools/record_wake_samples.py --out wake_samples/negative \
      --count 10 --duration 30 --no-prompt

Stop VIVIAN first (`sudo systemctl stop vivian`) so the mic is free.
Writes 16kHz mono WAVs, which is what openWakeWord consumes.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

SR = 16000          # openWakeWord's rate; record here to avoid resampling later
MIC_HINT = "vivian_mic"


def find_mic():
    """Prefer the named vivian_mic PCM, as the app does."""
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and MIC_HINT in d["name"]:
            print(f"Using input: {d['name']} (index {i})")
            return i
    print(f"'{MIC_HINT}' not found — falling back to the default input device")
    return None


def record(seconds, device):
    frames = int(seconds * SR)
    audio = sd.rec(frames, samplerate=SR, channels=1, dtype="int16", device=device)
    sd.wait()
    return audio.flatten()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--count", type=int, default=25, help="number of clips")
    ap.add_argument("--duration", type=float, default=3.0, help="seconds per clip")
    ap.add_argument("--no-prompt", action="store_true",
                    help="record back-to-back without waiting for Enter (negatives)")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device is not None else find_mic()

    existing = len(list(out.glob("*.wav")))
    if existing:
        print(f"({existing} clips already in {out} — new ones are appended)")

    kind = "AMBIENT" if args.no_prompt else "WAKE PHRASE"
    print(f"\nRecording {args.count} x {args.duration:.0f}s clips of {kind} into {out}")
    if not args.no_prompt:
        print('Say "Hey VIVIAN" once per clip. Vary it: near/far, quiet/loud,')
        print("mid-sentence, while the blower is on. Realistic beats clean.\n")
    else:
        print("Talk, play music, drive normally — anything EXCEPT the wake phrase.\n")

    peaks = []
    try:
        for n in range(args.count):
            idx = existing + n + 1
            if not args.no_prompt:
                try:
                    input(f"[{n+1}/{args.count}] Enter, then say it: ")
                except EOFError:
                    print("\n(no tty — switching to automatic timing)")
                    args.no_prompt = True
            else:
                print(f"[{n+1}/{args.count}] recording {args.duration:.0f}s...")
                time.sleep(0.3)

            audio = record(args.duration, device)
            peak = float(np.max(np.abs(audio))) / 32768.0
            peaks.append(peak)
            path = out / f"{out.name}_{idx:03d}.wav"
            wavfile.write(path, SR, audio)

            flag = ""
            if peak < 0.02:
                flag = "  <-- almost silent, is the mic live?"
            elif peak > 0.99:
                flag = "  <-- CLIPPING, back off or lower mic gain"
            print(f"      saved {path.name}  peak={peak:.2f}{flag}")
    except KeyboardInterrupt:
        print("\nStopped early — clips recorded so far are kept.")

    if peaks:
        print(f"\n{len(peaks)} clips in {out}")
        print(f"peak level: min={min(peaks):.2f} mean={np.mean(peaks):.2f} max={max(peaks):.2f}")
        if np.mean(peaks) < 0.05:
            print("WARNING: very quiet overall — check the mic before trusting any tuning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
