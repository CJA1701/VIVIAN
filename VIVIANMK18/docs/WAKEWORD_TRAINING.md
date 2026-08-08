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

**Open it directly in Colab:**

<https://colab.research.google.com/github/alfiedennen/openwakeword-colab-2026/blob/main/train_wakeword.ipynb>

**Runtime → Change runtime type:** GPU (L4 if you have Pro, else T4) **+ High RAM**.

**Edit two lines in Cell 10** — note `TARGET_PHRASE` is a *list* of pronunciation
variants, all of which map to the same single output:

```python
TARGET_PHRASE = ['hey vivian', 'hey vivien']   # two common pronunciations
MODEL_NAME    = 'hey_vivian'                   # -> hey_vivian.onnx
```

Keep the variant list short and close together. Two near-identical
pronunciations improve robustness; throwing in genuinely different words
(`viviana`, `vivi`) blurs the decision boundary and makes it worse.

Then **Runtime → Run all** and leave it.

- **Cell 4** is a fast dependency preflight (~60 s) — if it fails, stop there;
  you have not yet burned the long downloads.
- **~25 GB** of FMA + ACAV100M features download *into the Colab session*, not
  onto your Mac. Your local free space is irrelevant here.
- No HuggingFace token, no Google Drive mount needed.
- **L4 + High RAM ≈ 75-90 min. Free T4 ≈ 2.5 h**, and free Colab disconnects if
  the tab is backgrounded — keep it visible, and stop the Mac sleeping:
  `caffeinate -dims` in a terminal for the duration.

What it does: synthesises positives with Piper TTS across many voices, pulls
negatives from FMA + ACAV100M, augments with noise/reverb, trains with
hard-negative mining, and exports a single `.onnx` with the sigmoid baked in.

`hey_vivian.onnx` **auto-downloads to your browser's download folder** at the end.

### Bench results for the model in `models/hey_vivian.onnx`

Validated against macOS `say` voices (7 voices, deliberately NOT the Piper
voices it was trained on) with 0.5s lead-in and 1.2s trailing silence:

| clip type | n | median score | min |
|---|---|---|---|
| "Hey Vivian" | 7 | 0.971 | 0.518 |
| "Hey Vivien" | 7 | 0.936 | 0.162 |
| "Hey Vivian turn on the radio" | 7 | 0.734 | 0.143 |
| negatives (Hey Google/Siri/Brian, "Vivian" alone, ...) | 56 | 0.000 | max 0.091 |

90% recall at zero false accepts. Specificity is the strong part — every
negative scored ~0, including other "Hey X" phrases.

Two things this tells you:

- **Pad your test clips.** openWakeWord scores a sliding 1.28s window, so a clip
  that ends the instant the phrase does never completes a window and scores near
  zero. That is a measurement artifact, not a bad model — it cost me a wrong
  conclusion the first time.
- **Pause after the wake phrase.** Running straight on ("Hey Vivian turn on the
  radio") scores materially lower than a clean "Hey Vivian". VIVIAN's flow wants
  a pause anyway: say it, wait for the chime, then talk.

This is clean synthetic speech, so treat it as proof the model is not a dud —
NOT as evidence it works in the car. That still needs step 4.

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

---

## Troubleshooting: "she hears nothing" (read this first)

Before suspecting the model or the threshold, confirm **which dongle the mic is
actually on**. This cost a long debugging session once.

There are two identical GeneralPlus dongles (`1b3f:2008`, no serial numbers):

| USB port | ALSA card | what it is |
|---|---|---|
| `1-1.1` | `Device_1` | **microphone** + car-stereo out |
| `1-2`   | `Device`   | internal speakers; **mic jack empty** |

The empty one has AGC enabled, so it winds its gain up hunting for signal and
emits loud, pulsing, hissy noise. It is **louder than the real microphone**, so
"which card has more signal" is exactly the wrong test — it picks the dead one.

Tell them apart by **crest factor**, not level:

```bash
sudo systemctl stop vivian     # frees the mic
arecord -D plughw:1,0 -f S16_LE -r 48000 -c 1 -d 10 /tmp/t.wav   # then tap the mic
```

- real mic: crest ≈ 60x, per-second variation ≈ 60x, <1% energy above 8kHz
- empty input: crest ≈ 1.5x, variation ≈ 2x, ~14% energy above 8kHz

Two more traps:

- **Record at 48000, not 16000.** The hardware only supports 44100/48000; asking
  ALSA's `plug` layer for 16k makes it resample, and the result sounds like
  digital crackling. Every diagnostic capture at 16k is misleading.
- **Just listen to it.** `scp` the wav over and play it. Thirty seconds of
  listening beat an hour of my spectral analysis at identifying "this is not a
  microphone".

`system/85-vivian-audio.rules` pins both dongles to fixed card names by physical
USB port so the mapping cannot flip between boots. Install with:

```bash
sudo cp system/85-vivian-audio.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules      # takes effect at next boot
cat /sys/class/sound/card*/id            # verify: Device, Device_1, Camera
```
