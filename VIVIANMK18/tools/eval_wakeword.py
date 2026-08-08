#!/usr/bin/env python3
"""Sweep openWakeWord thresholds against real recordings and recommend one.

Guessing a threshold is how you end up either deaf at 60mph or triggering on
road noise. This measures both, on your audio:

  python3 tools/eval_wakeword.py --model models/hey_vivian.onnx \
      --positive wake_samples/positive --negative wake_samples/negative

Positives  = clips containing one "Hey VIVIAN"  -> detection rate (recall)
Negatives  = clips containing none              -> false accepts per hour

Recommends the HIGHEST-recall threshold that still yields zero false accepts,
because in a car a false trigger (she interrupts, mutes your music, records)
is far more annoying than one missed wake.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

FRAME = 1280        # openWakeWord's chunk: 80ms @ 16kHz
SR = 16000


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if sr != SR:
        # linear resample — fine for evaluation
        n = int(round(len(data) * SR / sr))
        data = np.interp(np.linspace(0, len(data), n, endpoint=False),
                         np.arange(len(data)), data.astype(np.float32))
    return data.astype(np.int16)


def max_score(model, audio):
    """Highest score any frame in this clip produced."""
    model.reset() if hasattr(model, "reset") else None
    best = 0.0
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        scores = model.predict(audio[i:i + FRAME])
        if scores:
            best = max(best, max(scores.values()))
    return best


def count_activations(model, audio, threshold):
    """How many separate times this clip crosses the threshold.

    Consecutive frames above threshold are one activation, matching how the
    wake loop behaves (it stops reading and runs the callback on first hit).
    """
    model.reset() if hasattr(model, "reset") else None
    hits, armed = 0, True
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        scores = model.predict(audio[i:i + FRAME])
        s = max(scores.values()) if scores else 0.0
        if s >= threshold and armed:
            hits += 1
            armed = False
        elif s < threshold * 0.7:
            armed = True
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--positive", required=True, help="dir of clips WITH the phrase")
    ap.add_argument("--negative", help="dir of clips WITHOUT the phrase")
    args = ap.parse_args()

    try:
        from openwakeword.model import Model
    except ImportError:
        print("openwakeword is not installed here. pip3 install openwakeword")
        return 2

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        return 2

    model = Model(wakeword_models=[args.model], inference_framework="onnx")

    pos = sorted(Path(args.positive).glob("*.wav"))
    neg = sorted(Path(args.negative).glob("*.wav")) if args.negative else []
    if not pos:
        print(f"No .wav files in {args.positive}")
        return 2
    print(f"Scoring {len(pos)} positive and {len(neg)} negative clips...\n")

    pos_scores = [max_score(model, load_wav(p)) for p in pos]

    neg_secs = 0.0
    neg_audio = []
    for p in neg:
        a = load_wav(p)
        neg_audio.append(a)
        neg_secs += len(a) / SR

    print(f"positive peak scores: min={min(pos_scores):.3f} "
          f"median={np.median(pos_scores):.3f} max={max(pos_scores):.3f}")
    if neg_secs:
        print(f"negative audio: {neg_secs/60:.1f} minutes\n")
    else:
        print("no negatives given — false-accept rate cannot be measured\n")

    print(f"{'thresh':>7}{'detected':>11}{'recall':>9}{'false/hr':>11}")
    print("-" * 38)
    best = None
    for t in [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)]:
        detected = sum(1 for s in pos_scores if s >= t)
        recall = detected / len(pos_scores)
        fa = sum(count_activations(model, a, t) for a in neg_audio) if neg_audio else 0
        fa_hr = (fa / (neg_secs / 3600.0)) if neg_secs else float("nan")
        mark = ""
        if neg_secs and fa == 0 and (best is None or recall > best[1]):
            best = (t, recall)
            mark = "  <- zero false accepts"
        print(f"{t:>7.2f}{detected:>7}/{len(pos_scores):<3}{recall:>8.0%}"
              f"{fa_hr:>11.1f}{mark}")

    print()
    if best and best[1] >= 0.8:
        print(f"RECOMMENDED  wake_word.threshold: {best[0]:.2f}   "
              f"({best[1]:.0%} of your samples detected, no false accepts)")
    elif best:
        print(f"Best zero-false-accept threshold is {best[0]:.2f} but it only "
              f"catches {best[1]:.0%} of your samples.")
        print("That is a weak model, not a bad threshold — retrain with more "
              "positives (ideally recordings of your own voice) before shipping it.")
    else:
        print("Every threshold produced false accepts. Either the negatives "
              "contain the phrase, or the model is too loose — retrain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
