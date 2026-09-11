# VIVIAN

VIVIAN is a voice assistant built into a 1969 Ford Mustang, running on a
Raspberry Pi 5. It listens for a custom wake word, answers spoken questions
using the Claude API, controls Spotify playback, and switches into an
unattended "sentry" mode that watches the car with a camera when it's parked
— using YOLO for person detection, face recognition to tell the owner from a
stranger, and Claude's vision model to assess and react to a threat.

This is `VIVIANMK18`, the current build. Prior iterations (`MK1`–`MK17`) are
not published here — this repo tracks the current, working system rather
than every prototype along the way.

## What it does

- **Wake word → conversation.** A locally-trained [openWakeWord](https://github.com/dscripka/openWakeWord)
  model listens continuously; on trigger it records, transcribes with
  `whisper.cpp`, and sends the request to Claude (`claude-haiku-4-5`), which
  can answer directly or call tools (music control, GPS/weather lookups,
  system diagnostics, arming sentry mode).
- **Local text-to-speech.** Two [Piper](https://github.com/rhasspy/piper)
  voice profiles — a warm assistant voice and a distinct "sentry" voice for
  security alerts — synthesized entirely on-device, streamed sentence-by-
  sentence so the reply starts speaking before the whole response has
  generated.
- **Sentry mode.** When armed, a YOLOv8 + ByteTrack pipeline detects people,
  InsightFace identifies known vs. unknown faces, and an unknown/lingering
  presence triggers a Claude vision call that assesses threat level and
  generates a specific spoken deterrent — grounded in what the camera
  actually shows, not a generic canned line.
- **Hardware I/O.** A composite CRT display shows assistant/sentry state, a
  small touchscreen shows Spotify playback and queue, a rotary switch and
  physical button provide non-voice control, and a companion Google
  Glass app (`VivianGlass/`) mirrors state over a local WebSocket.
- **Built for an unreliable environment.** The Pi loses power every time the
  car is turned off — every boot is a cold boot, often with no network yet.
  A lot of the engineering here is about surviving that: watchdogs for a
  wedged USB mic, a hard deadline around the wake-word callback so a stuck
  API call can't leave the assistant permanently deaf, atomic writes for
  state files that could otherwise get corrupted mid-write on power loss,
  and a `preflight.py` script that verifies the whole stack (audio devices,
  model files, credentials, disk space) before trusting a fresh deploy.

## Structure

```
VIVIANMK18/            Main application (Python)
  main.py                orchestrator / wake-word loop
  assistant.py            Claude tool-use loop
  wake_word.py             wake-word engine (openWakeWord, pluggable)
  tts.py                   local Piper/Kokoro TTS
  Sentry/                  camera-based sentry system
  docs/                    setup notes (wake-word training, etc.)
  tools/                   wake-word sample recording + threshold tuning
  preflight.py             pre-deploy hardware/config sanity check
VivianGlass/            Companion Google Glass app (Java)
1.9inch_Touch_LCD_Pi/    Vendor driver for the touchscreen (see below)
```

## Running it yourself

This is tightly coupled to specific hardware (Pi 5, a particular USB audio
setup, a car-mounted camera), so it isn't a drop-in install — but the shape
is:

1. `cd VIVIANMK18 && ./setup_mk18.sh` — installs dependencies, downloads the
   Piper voice models.
2. Copy `config.example.yaml` → `config.yaml` and `Sentry/sentry_config.example.yaml`
   → `Sentry/sentry_config.yaml`, filling in your own Anthropic API key,
   Spotify app credentials, and (optionally) OpenWeather/Google/Discord keys.
   **Never commit the filled-in files** — they're gitignored for a reason.
3. `python3 preflight.py` — checks the wake-word model, audio devices, and
   config before you trust it with anything.
4. `python3 main.py`, or install `vivian.service` to run it under systemd.

To train your own wake word rather than reusing `models/hey_vivian.onnx`,
see `docs/WAKEWORD_TRAINING.md`.

## Third-party code

`1.9inch_Touch_LCD_Pi/` is Waveshare's vendor sample driver for their 1.9"
touch LCD, included here (with local modifications in `VIVIANMK18/spotify_display.py`
and `crt_display.py`) because it's otherwise only distributed via their wiki.
All credit for that driver belongs to Waveshare.

## License

MIT — see `LICENSE`. This does not extend to the vendored Waveshare driver
code noted above, which retains whatever terms Waveshare distributes it under.
