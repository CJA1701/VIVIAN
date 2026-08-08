# Training a "Hey VIVIAN" wake word (openWakeWord)

Picovoice deleted the account behind VIVIAN's Porcupine key and no longer offers
a free tier, so the wake word moved to **openWakeWord** (Apache-2.0, ONNX, no
account, no activation server, nothing phoning home).

This is the one part of VIVIAN that can't be done from the car: training needs a
GPU. Everything else here runs on the laptop or the Pi.

---

## 1. Train the model (Colab, ~1.5-2.5 h)

**Use the community 2026 notebook, not the official one.**

- 2026 edition: <https://github.com/alfiedennen/openwakeword-colab-2026>
- Official (for reference): <https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb>

The official `automatic_model_training.ipynb` has bit-rotted — it fails out of
the box on Python 3.12 (`piper-phonemize`), removed `torchaudio 2.x` APIs, and a
changed YAML schema. The 2026 notebook replaces upstream's `auto_train` with a
self-contained ~250-line PyTorch trainer that does the same curriculum.

⚠️ It is third-party code that downloads several GB of datasets into your Colab
session. Skim cell 14 before running it — it is deliberately short enough to read.

**Settings — change two lines:**

```python
target_phrase = "hey vivian"     # lowercase; how it is pronounced, not spelled
model_name    = "hey_vivian"     # -> hey_vivian.onnx
```

**Runtime:** Colab Pro (L4 + High RAM) ≈ 75-90 min. Free tier (T4) works but
takes ≈2.5 h and may disconnect — if you're on free tier, keep the tab active.

What it does: synthesises positives with Piper TTS across many voices, pulls
negatives from FMA + ACAV100M, augments with noise/reverb, trains with
hard-negative mining, and exports a single `.onnx` with the sigmoid baked in.

**Download `hey_vivian.onnx` when it finishes.**

---

## 2. Install on the Pi

```bash
ssh pi5.local
pip3 install openwakeword
# Pre-download the shared feature models NOW, while there is network.
# openWakeWord fetches melspectrogram/embedding models on first use, and the
# car is frequently offline at boot — a cold start must never depend on this.
python3 -c "import openwakeword.utils as u; u.download_models()"
```

Put the model where the config expects it:

```bash
mkdir -p ~/VIVIANMK18/models
scp hey_vivian.onnx pi5.local:~/VIVIANMK18/models/
```

## 3. Switch VIVIAN over

In `config.yaml`:

```yaml
wake_word:
  engine: "openwakeword"          # porcupine | openwakeword | none
  model_path: "models/hey_vivian.onnx"
  threshold: 0.5                  # tune with step 4 — do not guess
```

Then `sudo systemctl restart vivian` and check:

```bash
cd ~/VIVIANMK18 && python3 preflight.py     # must be 0 FAIL
journalctl -u vivian -b | grep "Wake engine"
# expect: Wake engine 'openwakeword' ready (target SR: 16000Hz, frame length: 1280)
```

Nothing else changes — the callback deadline, device rotation, liveness
watchdog and USB-reset recovery are all engine-agnostic and still apply.

---

## 4. Tune the threshold — do not skip this

A synthetic-trained model is decent on the bench and *mediocre in a moving
'69 Mustang* until the threshold is set against real audio. Too low and road
noise triggers her; too high and she ignores you at speed.

**Record real samples in the car** (engine running, window as you'd normally
have it):

```bash
# on the Pi, with VIVIAN stopped so the mic is free:
sudo systemctl stop vivian
python3 tools/record_wake_samples.py --out wake_samples/positive --count 25
#   say "Hey VIVIAN" once per prompt, varying distance and how you say it

python3 tools/record_wake_samples.py --out wake_samples/negative --count 10 \
    --duration 30 --no-prompt
#   just drive/talk normally — anything EXCEPT the wake phrase
sudo systemctl start vivian
```

**Sweep thresholds against those recordings:**

```bash
python3 tools/eval_wakeword.py \
    --model models/hey_vivian.onnx \
    --positive wake_samples/positive \
    --negative wake_samples/negative
```

It prints detection rate and false-accepts per hour at each threshold and
recommends the highest-recall threshold with zero false accepts. Put that number
in `wake_word.threshold`.

Rules of thumb:
- **False triggers while driving** → raise the threshold 0.05 at a time.
- **She ignores you** → lower it, and if that causes false accepts, the model
  needs more/better positives (record yourself and retrain rather than pushing
  the threshold below ~0.3).

---

## Fallbacks

- `engine: "none"` cleanly disables wake detection. The button, Glass HUD and
  sentry all keep working, and nothing spams the log.
- If the model file or the package is missing, VIVIAN logs one actionable error
  and carries on with the button — she never crash-loops over it.
