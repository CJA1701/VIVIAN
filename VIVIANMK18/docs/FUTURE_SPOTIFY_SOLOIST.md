# Planned: migrate playback from raspotify to Spotify Soloist (~Sep 2027)

Decision 2026-09-06: adopt Soloist, but not before the Pi moves off Bookworm.
Evaluated against the real arm64 binary, not just the docs.

## Why

Soloist's local WebSocket API removes the network from every playback control:
`play {uri}` (track/album/playlist), `pause`, `skip_next`, `skip_prev`, `seek`,
`set_volume`, `add_to_queue`, `activate`, plus pushed `track_changed` /
`playback_changed` / `position_sync` events with full metadata. That replaces
every 1 Hz `current_playback` Web API poll (spotify_control writer thread,
spotify_display, glass_server). Only `sp.search` and `sp.artist_top_tracks`
keep needing the Web API.

## Blockers to clear first (all verified 2026-09-06)

1. **glibc >= 2.38.** Pi is Bookworm (glibc 2.36). Requires the Trixie upgrade
   -> rebuild whisper.cpp, torch, onnxruntime, InsightFace, Piper, openWakeWord,
   PyAudio. Do Soloist as part of that upgrade, not before.
2. **PipeWire/PulseAudio only** (`libpipewire-0.3`, `libpulse`; no libasound;
   only `--pipewire-device`). The whole VIVIAN audio stack is raw ALSA:
   asound.conf dmix/softvol chains, the `Stereo` softvol mute (audio_mute.py),
   `aplay -D stereo_direct` for TTS, PyAudio/sounddevice mic capture. Plan the
   PipeWire re-plumb explicitly; the two identical USB dongles make this
   non-trivial (see docs/WAKEWORD_TRAINING.md troubleshooting).
   Upside: Soloist `set_volume 0` over the local socket could replace the
   amixer mute entirely.
3. **90-day hard build expiry** (`exit code 10`, "client expired, please update
   to a newer version"). No auto-update. Needs a systemd timer that
   re-downloads `soloist_release_arm64.tar.gz` well inside the window AND a
   fallback so an offline car does not lose music when a build lapses.
   Re-check whether Spotify has relaxed this before committing.

Also confirm at the time: it has matured past its Aug-2026 launch (33 stars,
7 issues then), and re-read the WebSocket reference for API changes.

## Migration sketch

1. Install binary to /usr/local/bin; API key from developer dashboard
   (/dashboard/soloist, Premium required) -> env file, never argv.
2. `soloist --pair --device-name "VIVIAN"` once, phone on the same LAN
   (VIVIAN-AP works). Session persists in `~/.local/share/soloist`.
3. Run as a systemd unit with `--ws 127.0.0.1:9090` (NO auth on the socket —
   loopback only, never 0.0.0.0).
4. spotify_control.py: route pause/resume/skip/previous/play_playlist/
   play_track(uri)/get_current_playback through the WebSocket; keep spotipy for
   search + artist_top_tracks only. Subscribe to events instead of polling.
5. main.py `_mute_on_detect`: consider `set_volume 0` via the socket instead of
   amixer (instant, local, no card-mapping dependency).
6. Retire raspotify; keep it installed but disabled for one release as
   rollback.

Docs: https://developer.spotify.com/documentation/soloist
