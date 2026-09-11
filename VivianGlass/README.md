# VivianGlass — Google Glass XE-C HUD for VIVIAN

Heads-up display for a 1969 Ford Mustang running the VIVIAN AI assistant system. Connects to the Pi 5 over the car's WiFi AP and displays real-time driving data on Google Glass Explorer Edition.

## Architecture

```
Glass XE-C  <--WiFi (192.168.4.1)--> Pi 5 (VIVIAN)
                                      |
                                   glass_server.py
                                      +-- :9100  WebSocket (state broadcast every 0.5s)
                                      +-- :9101  HTTP (pre-rendered OSM map tile JPEG)
```

The Pi reads the same `/tmp/vivian_*.json` state files that the CRT display consumes and broadcasts consolidated JSON to the Glass over WebSocket. The Glass renders a Canvas-based HUD on a Live Card at 4 FPS. No changes to existing VIVIAN data producers were needed.

## HUD Layout (640x360, white on black)

```
+------------------------------------------------------------------+
| [GPS] [SENTRY]                         72F Clear           12:45 |
|------------------------------------------------------------------|
|                                                                    |
|         45              +-------------------+                      |
|        MPH              |   OSM Map Tile    |  Hotel California    |
|      LIMIT 55           |   (200x180)       |  Eagles              |
|                         |        .          |  > 1:23 / 6:31      |
|                         +-------------------+                      |
+------------------------------------------------------------------+
```

- **Speed** (left) — large speedometer with posted speed limit from OSM below
- **Map** (center) — live OSM tile centered on GPS position, zooms with speed
- **Music** (right) — current track, artist, interpolated progress bar
- **Status bar** (top) — GPS fix, satellite count, sentry status, weather, time

Black pixels are transparent on the Glass prism, so only white content is visible — ideal for a driving HUD overlay.

## Features

- **Live OSM map** — Pi fetches 3x3 tile grid, crops/converts to white-on-black, serves as JPEG. Glass fetches every 2s. Zoom scales with speed (zoom 17 parked to zoom 12 at highway speed).
- **Speed limit** — Overpass API query for nearest road's `maxspeed` tag. Cached for 10s or 50m movement. Displayed as "LIMIT XX" under speedometer.
- **Music control** — touchpad menu (tap Live Card) for play/pause, next, previous. Progress bar interpolated client-side between server updates for smooth ticking.
- **Weather** — OpenWeatherMap, cached 120s, non-blocking background fetch.
- **Wake lock** — Glass stays at full brightness with touchpad responsive while HUD is active.
- **Auto-reconnect** — WebSocket reconnects with exponential backoff (1s to 15s) if connection drops.
- **Voice trigger** — "OK Glass, control my car" launches the HUD.
- **Touch launcher** — appears in Glass app list as "VIVIAN" with the V+I logo.

## Project Structure

### Glass App (this directory)

```
app/src/main/java/com/vivian/glass/
  VivianLaunchActivity.java    — Thin launcher (starts service, finishes)
  VivianGlassService.java      — Live Card service, wake lock, START_STICKY
  VivianLiveCardRenderer.java  — Canvas HUD renderer at 4 FPS
  VivianState.java             — Thread-safe state POJO (volatile fields)
  WebSocketManager.java        — java-websocket client, auto-reconnect
  MenuActivity.java            — Touchpad menu (music controls, stop HUD)

app/src/main/res/
  xml/voice_trigger.xml        — "CONTROL_MY_CAR" predefined voice command
  drawable/ic_vivian.png       — 50x50 white V+I logo (app icon)
  drawable/vivian_logo.png     — 150x150 white V+I logo
  values/strings.xml

app/libs/
  gdk.jar                      — Glass Development Kit (compile-only stub)
```

### Pi Side (in VIVIANMK15/)

```
glass_server.py    — NEW: WebSocket + HTTP server
                     Reads /tmp/ state files, broadcasts JSON
                     Renders OSM map tiles, serves as JPEG on :9101
                     Queries Overpass API for speed limits
                     Handles music commands from Glass

config.py          — Added: glass property (optional, like gps/sentry)
config.yaml        — Added: glass section (enabled, host, port)
main.py            — Added: GlassServer init/start/cleanup (~10 lines)
```

## Pi Config (config.yaml)

```yaml
glass:
  enabled: true
  host: "0.0.0.0"
  port: 9100
```

## WebSocket Protocol

**Pi -> Glass** (every 0.5s, skipped if unchanged):
```json
{
  "type": "state",
  "gps": {"speed_mph": 45.2, "track": 182, "has_fix": true, "sats_used": 8, "lat": 34.1167, "lon": -84.3976, "alt": 302},
  "music": {"is_playing": true, "track": "Hotel California", "artist": "Eagles", "progress_ms": 45000, "duration_ms": 391000},
  "assistant": {"state": "idle", "transcript": ""},
  "weather": {"temp_f": 72, "condition": "Clear"},
  "speed_limit": 55,
  "sentry_active": false
}
```

**Glass -> Pi** (commands):
```json
{"type": "cmd", "action": "music_toggle|music_next|music_prev"}
```

## Build

Requires Android SDK with API 19 platform. GDK jar is included in `app/libs/`.

```bash
# Build
./gradlew assembleDebug

# Install to Glass (USB)
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Force-start (first launch)
adb shell am startservice com.vivian.glass/.VivianGlassService
```

## Dependencies

**Pi:** `websockets`, `Pillow`, `requests` (all already available on the VIVIAN Pi)

**Glass:** `org.java-websocket:Java-WebSocket:1.5.3` (bundled in APK)

## Hardware

- **Google Glass Explorer Edition XE-C** running XE24 firmware, debug mode enabled
- **Pi 5** hosting WiFi AP at 192.168.4.1, running VIVIAN with `glass.enabled: true`
- Glass connects to Pi AP, receives state over WebSocket, fetches map over HTTP
