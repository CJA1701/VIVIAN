#!/usr/bin/env python3
"""Pre-deploy sanity check — run this ON THE PI after syncing, before driving.

Checks the things that only the real hardware can answer, and that would
otherwise fail silently or at 60mph:

  * every module imports (catches a syntax/name error before the car does)
  * spotipy is new enough for the explicit timeout/retry overrides
  * sudoers actually permits the privileged calls init_audio.sh makes
  * Piper binary + both voice models exist and are runnable
  * config has the keys the code reads, and secrets are NOT world-readable
  * ALSA capture/playback devices are present
  * whisper binary + model exist
  * disk headroom for snapshots/recordings

Exit code 0 = safe to drive. 1 = at least one FAIL.
Nothing here touches the running service; it is read-only.
"""

import importlib
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "Sentry"))

FAILS, WARNS = [], []


def ok(msg):
    print(f"  \033[32mPASS\033[0m  {msg}")


def fail(msg):
    print(f"  \033[31mFAIL\033[0m  {msg}")
    FAILS.append(msg)


def warn(msg):
    print(f"  \033[33mWARN\033[0m  {msg}")
    WARNS.append(msg)


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------- imports
section("Module imports")
# hardware-bound modules are imported lazily by main.py; import them anyway —
# on the Pi they should all be available.
for mod in ["config", "memory", "audio_mute", "system_info", "hardware",
            "tts", "wake_word", "audio", "spotify_control", "assistant",
            "sentry_controller", "gps_controller", "crt_display",
            "spotify_display", "glass_server", "dashboard_server"]:
    try:
        importlib.import_module(mod)
        ok(f"import {mod}")
    except Exception as e:
        fail(f"import {mod}: {type(e).__name__}: {e}")

try:
    sys.path.insert(0, str(HERE / "Sentry"))
    importlib.import_module("SentryMk3")
    ok("import SentryMk3")
except Exception as e:
    fail(f"import SentryMk3: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- spotipy
section("spotipy version (explicit timeout/retry overrides)")
try:
    import spotipy
    import inspect
    params = inspect.signature(spotipy.Spotify.__init__).parameters
    ver = getattr(spotipy, "__version__", "unknown")
    if "requests_timeout" in params and "retries" in params:
        ok(f"spotipy {ver} accepts requests_timeout + retries")
    elif "requests_timeout" in params:
        warn(f"spotipy {ver} has no `retries` kwarg — timeout still applied, "
             f"but retry count falls back to the library default. "
             f"`pip install -U spotipy` to get bounded retries.")
    else:
        fail(f"spotipy {ver} accepts NEITHER override — Spotify calls on the "
             f"wake-word path are unbounded. Upgrade spotipy.")
except Exception as e:
    fail(f"spotipy check failed: {e}")


# ---------------------------------------------------------------- sudoers
section("sudoers (init_audio.sh runs unprivileged and needs these)")
for desc, cmd in [
    ("usbreset (mic dongle recovery)", ["sudo", "-n", "/usr/bin/usbreset", "--help"]),
    ("sh -c for spidev unbind (LCD backlight GPIO 12)", ["sudo", "-n", "/bin/sh", "-c", "true"]),
]:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=10)
        # usbreset --help may exit non-zero; what matters is that sudo did not
        # refuse with a password prompt / "not allowed" message.
        errtxt = (r.stderr or b"").decode(errors="replace").lower()
        if "password is required" in errtxt or "not allowed" in errtxt or "may not run" in errtxt:
            fail(f"sudo -n denied: {desc} -> {errtxt.strip()[:80]}")
        else:
            ok(f"sudo -n permitted: {desc}")
    except FileNotFoundError:
        fail(f"missing binary for {desc}: {cmd[2] if len(cmd) > 2 else cmd}")
    except Exception as e:
        warn(f"could not test {desc}: {e}")


# ---------------------------------------------------------------- piper/tts
section("Piper TTS assets")
try:
    from config import Config
    cfg = Config("config.yaml")
    tts_cfg = cfg.tts
    piper_bin = HERE / tts_cfg.get("piper_bin", "piper/piper")
    if piper_bin.exists() and os.access(piper_bin, os.X_OK):
        ok(f"piper binary executable: {piper_bin.name}")
    else:
        fail(f"piper binary missing or not executable: {piper_bin} "
             f"(run setup_mk18.sh)")
    profiles = tts_cfg.get("profiles") or {}
    if not profiles:
        warn("no tts.profiles configured — falling back to single-voice config")
    for name, prof in profiles.items():
        model = HERE / prof.get("model", "")
        cfgj = Path(str(model) + ".json")
        if model.exists() and cfgj.exists():
            ok(f"voice '{name}': {model.name} (+ .json)")
        else:
            missing = "model" if not model.exists() else "model .json sidecar"
            fail(f"voice '{name}': missing {missing} -> {model} (run setup_mk18.sh)")
except Exception as e:
    fail(f"TTS asset check failed: {e}")


# ---------------------------------------------------------------- config
section("Config + secret file permissions")
try:
    from config import Config
    cfg = Config("config.yaml")
    checks = [
        ("anthropic.api_key", (cfg.anthropic or {}).get("api_key")),
        ("porcupine.access_key", (cfg.porcupine or {}).get("access_key")),
        ("spotify.client_id", (cfg.spotify or {}).get("client_id")),
        ("spotify.refresh_token", (cfg.spotify or {}).get("refresh_token")),
    ]
    for key, val in checks:
        if val:
            ok(f"{key} present")
        else:
            fail(f"{key} is EMPTY in config.yaml")
    glass = cfg.glass or {}
    if glass.get("enabled") and not glass.get("control_token"):
        warn("glass.control_token is empty — anyone on the car's WiFi can arm/"
             "disarm sentry and control playback over the WebSocket")
except Exception as e:
    fail(f"config load failed: {e}")

for secret in ["config.yaml", "Sentry/sentry_config.yaml"]:
    p = HERE / secret
    if not p.exists():
        fail(f"{secret} not found on this machine")
        continue
    mode = stat.S_IMODE(p.stat().st_mode)
    if mode & 0o077:
        warn(f"{secret} is mode {oct(mode)} (group/world readable) — "
             f"holds live credentials; run: chmod 600 {secret}")
    else:
        ok(f"{secret} permissions {oct(mode)}")


# ---------------------------------------------------------------- audio devs
section("ALSA devices")
try:
    r = subprocess.run(["aplay", "-L"], capture_output=True, timeout=10)
    names = (r.stdout or b"").decode(errors="replace")
    for pcm in ["stereo_direct", "internal_plug"]:
        (ok if pcm in names else fail)(
            f"playback PCM '{pcm}' {'found' if pcm in names else 'MISSING (check /etc/asound.conf)'}")
except Exception as e:
    warn(f"aplay -L failed: {e}")
try:
    r = subprocess.run(["arecord", "-L"], capture_output=True, timeout=10)
    names = (r.stdout or b"").decode(errors="replace")
    (ok if "vivian_mic" in names else fail)(
        f"capture PCM 'vivian_mic' {'found' if 'vivian_mic' in names else 'MISSING'}")
except Exception as e:
    warn(f"arecord -L failed: {e}")


# ---------------------------------------------------------------- whisper
section("Whisper STT")
try:
    from config import Config
    cfg = Config("config.yaml")
    w = cfg.whisper or {}
    for label, key in [("binary", "binary_path"), ("model", "model_path")]:
        val = w.get(key)
        if not val:
            fail(f"whisper.{key} not set in config")
            continue
        p = Path(val)
        if not p.is_absolute():
            p = HERE / p
        (ok if p.exists() else fail)(
            f"whisper {label}: {p} {'' if p.exists() else 'MISSING'}")
except Exception as e:
    warn(f"whisper check skipped: {e}")


# ---------------------------------------------------------------- disk
section("Disk headroom")
try:
    total, used, free = shutil.disk_usage(str(HERE))
    free_gb = free / 1024 ** 3
    pct = 100.0 * used / total
    if free_gb < 1.0:
        fail(f"only {free_gb:.2f} GB free ({pct:.0f}% used) — snapshots, "
             f"recordings and logs will start failing silently")
    elif free_gb < 3.0:
        warn(f"{free_gb:.2f} GB free ({pct:.0f}% used) — thin for sentry "
             f"recordings; consider lowering snapshots.max_files")
    else:
        ok(f"{free_gb:.1f} GB free ({pct:.0f}% used)")
except Exception as e:
    warn(f"disk check failed: {e}")


# ---------------------------------------------------------------- summary
print("\n" + "=" * 62)
if FAILS:
    print(f"{len(FAILS)} FAIL, {len(WARNS)} WARN — do NOT rely on this build yet:")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print(f"0 FAIL, {len(WARNS)} WARN — safe to start the service."
      if WARNS else "ALL CHECKS PASSED — safe to start the service.")
for w in WARNS:
    print(f"  warn: {w}")
sys.exit(0)
