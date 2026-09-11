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
  small touchscreen shows Spotify playback and queue, a rotary switch selects
  what the CRT is showing, and a companion Google Glass app (`VivianGlass/`)
  mirrors state over a local WebSocket. Wake control started out as a
  physical button; it's since been removed in favor of voice-only wake word
  detection (see `docs/WAKEWORD_TRAINING.md`).
- **Built for an unreliable environment.** The Pi loses power every time the
  car is turned off — every boot is a cold boot, often with no network yet.
  A lot of the engineering here is about surviving that: watchdogs for a
  wedged USB mic, a hard deadline around the wake-word callback so a stuck
  API call can't leave the assistant permanently deaf, atomic writes for
  state files that could otherwise get corrupted mid-write on power loss,
  and a `preflight.py` script that verifies the whole stack (audio devices,
  model files, credentials, disk space) before trusting a fresh deploy.

## Hardware

![VIVIAN unit installed in the Mustang](Photos/VIVIAN%20Outside%20Shot.png)

**Enclosure.** Custom-designed and 3D-printed in four pieces (top, bottom,
CRT hood, and cover plates) — STLs are in `Enclosure Design/`. The front
panel carries the mode switch, the CRT power/select rotary, and a
high-density connector so the whole harness disconnects as one unit rather
than a handful of loose wires.

**Display.** The 4.5" black-and-white CRT is a real tube, pulled out of a
vintage portable RV television. The TV's original control board only
accepted an RF-modulated signal (its built-in tuner), so it was reverse
engineered to intercept the signal after RF demodulation and feed it
composite video directly from the Pi 5's TP7 test pad instead — no RF
modulator needed. Rendered at 320×240 NTSC via a pygame framebuffer
(`VIVIANMK18/crt_display.py`).

![Bench-testing the composite video path during the CRT integration](Photos/CRT%20Reverse%20Engineering.jpeg)

**Power.** A 12V automotive electrical system is not a clean power source —
cranking the starter can sag the battery to ~10V, which browned out the
original cheap buck converters and rebooted the Pi mid-drive. That's now two
Pololu step-down regulators: one feeds the Pi and every onboard system, the
other independently feeds a Netgear LM1200 cellular modem, so the car keeps
a network connection (and VIVIAN keeps its Claude API access) through the
same voltage dips that used to reboot everything.

**Other peripherals:**
- Waveshare 1.9" touchscreen (ST7789V2 + CST816) for Spotify control
- A VK-162 USB GPS receiver
- A USB camera at the base of the windshield, facing rearward, for sentry
  mode
- Two USB audio dongles — one carrying the mic input and feeding the car
  stereo, the other driving the internal speakers — pinned to fixed device
  names by physical USB port (see `docs/WAKEWORD_TRAINING.md` for why: two
  electrically identical dongles otherwise enumerate in a different order
  every boot)

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

## Third-party code

`1.9inch_Touch_LCD_Pi/` is Waveshare's vendor sample driver for their 1.9"
touch LCD, included here (with local modifications in `VIVIANMK18/spotify_display.py`
and `crt_display.py`) because it's otherwise only distributed via their wiki.
All credit for that driver belongs to Waveshare.

## License

MIT — see `LICENSE`. This does not extend to the vendored Waveshare driver
code noted above, which retains whatever terms Waveshare distributes it under.
