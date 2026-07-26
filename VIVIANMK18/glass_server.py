"""VIVIAN Glass HUD WebSocket Server

Reads /tmp/ state files (same ones CRT display consumes) and broadcasts
consolidated JSON to connected Google Glass clients over WebSocket.
Serves pre-rendered OSM map tiles as JPEG over HTTP.
Accepts music control commands from Glass.

Usage:
    Integrated via main.py when glass.enabled is true in config.yaml.
    Can also be run standalone for testing:
        python glass_server.py
"""

import asyncio
import io
import json
import logging
import math
import os
import threading
import time
from typing import Dict, Optional, Set

import websockets
import requests
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# Shared data files (same paths used by crt_display.py)
GPS_SHARED_FILE = "/tmp/vivian_gps.json"
SPOTIFY_SHARED_FILE = "/tmp/vivian_spotify.json"
ASSISTANT_SHARED_FILE = "/tmp/vivian_assistant.json"
SENTRY_FLAG_PATH = "/tmp/vivian_sentry_enabled"

WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_CACHE_TTL = 120  # seconds

MAP_WIDTH = 200
MAP_HEIGHT = 180
MAP_REFRESH_INTERVAL = 5  # seconds between map render attempts

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
SPEED_LIMIT_CACHE_TTL = 10  # seconds between queries
SPEED_LIMIT_MOVE_THRESHOLD = 50  # meters before re-querying


def _read_json(path: str, max_age: float = 5.0) -> Optional[dict]:
    """Read a JSON file, returning None if missing or stale."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        if time.time() - data.get('timestamp', 0) > max_age:
            return None
        return data
    except Exception:
        return None


def _read_gps() -> dict:
    """Read GPS state from shared file. Rounds values to reduce delta noise."""
    data = _read_json(GPS_SHARED_FILE, max_age=5.0)
    if data is None:
        return {"speed_mph": None, "track": None, "has_fix": False,
                "sats_used": 0, "lat": None, "lon": None, "alt": None}
    speed = data.get("speed_mph")
    track = data.get("track")
    lat = data.get("lat")
    lon = data.get("lon")
    alt = data.get("alt")
    return {
        "speed_mph": round(speed, 1) if speed is not None else None,
        "track": round(track, 0) if track is not None else None,
        "has_fix": data.get("has_fix", False),
        "sats_used": data.get("sats_used", 0),
        "lat": round(lat, 5) if lat is not None else None,
        "lon": round(lon, 5) if lon is not None else None,
        "alt": round(alt, 0) if alt is not None else None,
    }


def _read_spotify() -> dict:
    """Read Spotify playback state from shared file."""
    data = _read_json(SPOTIFY_SHARED_FILE, max_age=30.0)
    if data is None:
        return {"is_playing": False}
    return {
        "is_playing": data.get("is_playing", False),
        "track": data.get("track", ""),
        "artist": data.get("artist", ""),
        "progress_ms": data.get("progress_ms", 0),
        "duration_ms": data.get("duration_ms", 0),
    }


def _read_assistant() -> dict:
    """Read assistant state from shared file."""
    try:
        if not os.path.exists(ASSISTANT_SHARED_FILE):
            return {"state": "idle", "transcript": ""}
        mtime = os.path.getmtime(ASSISTANT_SHARED_FILE)
        if time.time() - mtime > 60:
            return {"state": "idle", "transcript": ""}
        with open(ASSISTANT_SHARED_FILE, 'r') as f:
            data = json.load(f)
        return {
            "state": data.get("state", "idle"),
            "transcript": data.get("transcript", ""),
        }
    except Exception:
        return {"state": "idle", "transcript": ""}


def _haversine(p1, p2) -> float:
    """Distance between two (lat, lon) points in meters."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(a))


def _zoom_for_speed(speed_mph) -> int:
    """Pick OSM zoom level based on GPS speed (same logic as crt_display.py)."""
    if speed_mph is None or speed_mph < 5:
        return 17
    elif speed_mph < 15:
        return 16
    elif speed_mph < 35:
        return 15
    elif speed_mph < 55:
        return 14
    elif speed_mph < 70:
        return 13
    else:
        return 12


def _lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to slippy map tile coordinates + pixel offset."""
    n = 2 ** zoom
    x_float = (lon + 180) / 360 * n
    lat_rad = math.radians(lat)
    y_float = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n
    tile_x = int(x_float)
    tile_y = int(y_float)
    pixel_x = int((x_float - tile_x) * 256)
    pixel_y = int((y_float - tile_y) * 256)
    return tile_x, tile_y, pixel_x, pixel_y


class GlassServer:
    """WebSocket server that broadcasts VIVIAN state to Google Glass clients."""

    def __init__(self, glass_config: dict, spotify, weather_config: dict, sentry_controller=None):
        self._host = glass_config.get('host', '0.0.0.0')
        self._port = glass_config.get('port', 9100)
        self._map_port = glass_config.get('map_port', 9101)
        self._interval = glass_config.get('broadcast_interval', 0.5)
        self._spotify = spotify
        self._weather_config = weather_config
        self._sentry_controller = sentry_controller
        # Optional shared secret for commands (glass.control_token in
        # config.yaml). When set, the Glass app must include it in each
        # cmd message; without it anyone on the car AP can send commands.
        self._control_token = glass_config.get('control_token', '')

        self._clients: Set = set()
        self._last_state_json: str = ""
        self._weather_cache = (None, None, 0.0)
        self._speed_limit_cache = {
            'limit': None, 'lat': None, 'lon': None, 'time': 0
        }
        self._map_jpeg: bytes = b''
        self._map_cache = {
            'lat': None, 'lon': None, 'zoom': None, 'time': 0
        }
        self._tile_session = requests.Session()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def _fetch_weather(self) -> tuple:
        """Return cached weather. Triggers background refresh if stale."""
        temp_f, condition, last_fetch = self._weather_cache
        if time.time() - last_fetch >= WEATHER_CACHE_TTL:
            self._weather_cache = (temp_f, condition, time.time())
            threading.Thread(
                target=self._fetch_weather_bg, daemon=True
            ).start()
        return temp_f, condition

    def _fetch_weather_bg(self):
        """Background weather fetch (non-blocking)."""
        try:
            city = self._weather_config.get('city', 'Woodstock')
            state = self._weather_config.get('state', 'GA')
            country = self._weather_config.get('country', 'US')
            r = requests.get(
                WEATHER_API_URL,
                params={
                    "q": f"{city},{state},{country}",
                    "appid": self._weather_config.get('api_key', ''),
                    "units": "imperial",
                },
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
            temp_f = round(data["main"]["temp"])
            condition = data["weather"][0]["main"]
            self._weather_cache = (temp_f, condition, time.time())
        except Exception as e:
            logger.debug(f"Weather fetch failed: {e}")

    # ------------------------------------------------------------------
    # Speed Limit (OSM Overpass)
    # ------------------------------------------------------------------

    def _get_speed_limit(self, lat, lon) -> Optional[int]:
        """Return cached speed limit. Triggers background refresh if stale."""
        cache = self._speed_limit_cache
        if cache['lat'] is not None:
            dist = _haversine((lat, lon), (cache['lat'], cache['lon']))
            age = time.time() - cache['time']
            if dist < SPEED_LIMIT_MOVE_THRESHOLD and age < SPEED_LIMIT_CACHE_TTL:
                return cache['limit']

        # Mark as fetched to prevent duplicate requests
        self._speed_limit_cache['time'] = time.time()
        self._speed_limit_cache['lat'] = lat
        self._speed_limit_cache['lon'] = lon
        threading.Thread(
            target=self._fetch_speed_limit_bg, args=(lat, lon), daemon=True
        ).start()
        return cache['limit']

    def _fetch_speed_limit_bg(self, lat, lon):
        """Background Overpass API query for speed limit."""
        query = (
            f'[out:json][timeout:5];'
            f'way(around:20,{lat},{lon})'
            f'["highway"~"^(motorway|trunk|primary|secondary|tertiary|'
            f'residential|unclassified|motorway_link|trunk_link|'
            f'primary_link|secondary_link)$"]'
            f'["maxspeed"];out tags;'
        )
        try:
            r = requests.post(
                OVERPASS_API_URL,
                data={'data': query},
                timeout=5,
            )
            r.raise_for_status()
            elements = r.json().get('elements', [])
            if elements:
                raw = elements[0].get('tags', {}).get('maxspeed', '')
                # Parse "45 mph", "45", "30 mph", etc.
                limit = self._parse_speed_limit(raw)
                self._speed_limit_cache.update(
                    limit=limit, lat=lat, lon=lon, time=time.time()
                )
            else:
                self._speed_limit_cache.update(
                    limit=None, lat=lat, lon=lon, time=time.time()
                )
        except Exception as e:
            logger.debug(f"Speed limit fetch failed: {e}")

    @staticmethod
    def _parse_speed_limit(raw: str) -> Optional[int]:
        """Parse OSM maxspeed value to integer mph."""
        if not raw:
            return None
        raw = raw.strip().lower()
        # Handle "45 mph", "45", "30 mph"
        parts = raw.split()
        try:
            value = int(parts[0])
            # If no unit or "mph", it's already mph (US default)
            if len(parts) > 1 and parts[1] == 'km/h':
                value = int(value * 0.621371)
            return value
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # OSM Map Rendering
    # ------------------------------------------------------------------

    def _render_map(self, lat, lon, speed_mph) -> bytes:
        """Render OSM map tile centered on position. Returns JPEG bytes."""
        zoom = _zoom_for_speed(speed_mph)

        # Check cache -- skip if position hasn't moved significantly
        cache = self._map_cache
        if cache['lat'] is not None and cache['zoom'] == zoom:
            dist = _haversine((lat, lon), (cache['lat'], cache['lon']))
            max_dist = 50 * (2 ** (17 - zoom))
            if dist < max_dist and time.time() - cache['time'] < 30:
                return self._map_jpeg  # still valid

        tile_x, tile_y, px_off, py_off = _lat_lon_to_tile(lat, lon, zoom)
        tile_size = 256
        canvas = Image.new('RGB', (tile_size * 3, tile_size * 3), (255, 255, 255))
        headers = {'User-Agent': 'VIVIAN/1.0'}

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                tx, ty = tile_x + dx, tile_y + dy
                url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
                try:
                    r = self._tile_session.get(url, headers=headers, timeout=5)
                    r.raise_for_status()
                    tile_img = Image.open(io.BytesIO(r.content)).convert('RGB')
                    canvas.paste(tile_img, ((dx + 1) * tile_size, (dy + 1) * tile_size))
                except Exception:
                    pass

        # Center pixel on the 3x3 canvas
        cx = tile_size + px_off
        cy = tile_size + py_off

        # Crop around position
        left = cx - MAP_WIDTH // 2
        top = cy - MAP_HEIGHT // 2
        cropped = canvas.crop((left, top, left + MAP_WIDTH, top + MAP_HEIGHT))

        # Convert to white-on-black for Glass prism
        gray = cropped.convert('L')
        enhanced = gray.point(lambda p: 255 if p < 230 else 0)
        rgb = enhanced.convert('RGB')

        # Draw location dot
        draw = ImageDraw.Draw(rgb)
        cx_dot, cy_dot = MAP_WIDTH // 2, MAP_HEIGHT // 2
        draw.ellipse(
            [cx_dot - 5, cy_dot - 5, cx_dot + 5, cy_dot + 5],
            fill=(255, 255, 255), outline=(0, 0, 0)
        )

        buf = io.BytesIO()
        rgb.save(buf, 'JPEG', quality=75)
        jpeg_bytes = buf.getvalue()

        self._map_cache = {'lat': lat, 'lon': lon, 'zoom': zoom, 'time': time.time()}
        return jpeg_bytes

    async def _map_render_loop(self):
        """Periodically render the map tile in a background thread."""
        while not self._stop_event.is_set():
            gps = _read_gps()
            if gps.get('has_fix') and gps.get('lat') and gps.get('lon'):
                try:
                    loop = asyncio.get_event_loop()
                    jpeg = await loop.run_in_executor(
                        None, self._render_map,
                        gps['lat'], gps['lon'], gps.get('speed_mph', 0)
                    )
                    if jpeg:
                        self._map_jpeg = jpeg
                except Exception as e:
                    logger.debug(f"Map render failed: {e}")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=MAP_REFRESH_INTERVAL
                )
                break
            except asyncio.TimeoutError:
                pass

    async def _map_http_handler(self, reader, writer):
        """Minimal HTTP handler that serves the current map JPEG."""
        try:
            await asyncio.wait_for(reader.read(1024), timeout=2)
        except Exception:
            writer.close()
            return

        if self._map_jpeg:
            header = (
                b"HTTP/1.0 200 OK\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(self._map_jpeg)).encode() + b"\r\n"
                b"Cache-Control: no-cache\r\n"
                b"\r\n"
            )
            writer.write(header + self._map_jpeg)
        else:
            writer.write(b"HTTP/1.0 204 No Content\r\n\r\n")

        try:
            await writer.drain()
        except Exception:
            pass
        writer.close()

    # ------------------------------------------------------------------
    # State broadcast
    # ------------------------------------------------------------------

    def _build_state(self) -> dict:
        """Read all /tmp/ state files and build consolidated state dict."""
        temp_f, condition = self._fetch_weather()
        gps = _read_gps()

        weather = None
        if temp_f is not None:
            weather = {"temp_f": temp_f, "condition": condition}

        speed_limit = None
        if gps.get('has_fix') and gps.get('lat') and gps.get('lon'):
            speed_limit = self._get_speed_limit(gps['lat'], gps['lon'])

        return {
            "type": "state",
            "gps": gps,
            "music": _read_spotify(),
            "assistant": _read_assistant(),
            "weather": weather,
            "speed_limit": speed_limit,
            "sentry_active": os.path.exists(SENTRY_FLAG_PATH),
        }

    def _toggle_sentry(self):
        """Toggle sentry mode on/off via Glass command."""
        if os.path.exists(SENTRY_FLAG_PATH):
            # Sentry is active — deactivate it
            if self._sentry_controller is not None:
                self._sentry_controller.deactivate()
                logger.info("Glass: sentry deactivate requested")
            else:
                logger.warning("Glass: sentry deactivate requested but no controller")
        else:
            # Sentry is inactive — write request file for main loop to pick up
            try:
                with open('/tmp/vivian_sentry_request', 'w') as f:
                    f.write('glass\n')
                logger.info("Glass: sentry activate request written")
            except Exception as e:
                logger.error(f"Glass: failed to write sentry request: {e}")

    async def _handle_command(self, msg: dict):
        """Dispatch a command from Glass to the appropriate controller."""
        action = msg.get("action", "")

        if self._control_token and msg.get("token") != self._control_token:
            logger.warning(f"Glass command '{action}' rejected: bad or missing token")
            return

        loop = asyncio.get_event_loop()

        try:
            if action == "music_toggle":
                playback = await loop.run_in_executor(
                    None, self._spotify.get_current_playback
                )
                if playback and playback.get("is_playing"):
                    await loop.run_in_executor(None, self._spotify.pause)
                else:
                    await loop.run_in_executor(None, self._spotify.resume)
            elif action == "music_next":
                await loop.run_in_executor(None, self._spotify.skip)
            elif action == "music_prev":
                await loop.run_in_executor(None, self._spotify.previous)
            elif action == "sentry_toggle":
                await loop.run_in_executor(None, self._toggle_sentry)
            else:
                logger.warning(f"Unknown Glass command: {action}")
        except Exception as e:
            logger.error(f"Glass command '{action}' failed: {e}")

    async def _handler(self, websocket):
        """Handle a single Glass client WebSocket connection."""
        self._clients.add(websocket)
        remote = websocket.remote_address
        logger.info(f"Glass client connected: {remote}")

        try:
            state = self._build_state()
            await websocket.send(json.dumps(state))

            async for message in websocket:
                try:
                    msg = json.loads(message)
                    if msg.get("type") == "cmd":
                        await self._handle_command(msg)
                except json.JSONDecodeError:
                    logger.debug(f"Invalid JSON from Glass: {message}")
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info(f"Glass client disconnected: {remote}")

    async def _send_state(self, state_json):
        """Send state to every client, dropping any that hangs or errors.
        A Glass that leaves WiFi range without a TCP FIN fills its write buffer,
        and a bare `await ws.send()` then waits on drain forever — that blocked
        the single broadcast task and killed the HUD for every other client too.
        Bounded per-client so one dead radio can't take everyone down."""
        disconnected = set()
        for ws in self._clients.copy():
            try:
                await asyncio.wait_for(ws.send(state_json), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(f"Glass client {ws.remote_address} send stalled -- dropping")
                disconnected.add(ws)
            except Exception:
                # ConnectionClosed and any transport error: drop this client only
                disconnected.add(ws)
        self._clients -= disconnected
        for ws in disconnected:
            # Fire-and-forget close (bounded by websockets' close_timeout) so the
            # half-written socket and its handler task get cleaned up.
            try:
                asyncio.ensure_future(ws.close())
            except Exception:
                pass

    async def _broadcast_loop(self):
        """Periodically broadcast state to all connected clients."""
        while not self._stop_event.is_set():
            if self._clients:
                state = self._build_state()
                state_json = json.dumps(state)
                self._last_state_json = state_json
                await self._send_state(state_json)

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._interval
                )
                break
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _serve(self):
        """Start WebSocket server, HTTP map server, and broadcast loop."""
        self._stop_event = asyncio.Event()

        # HTTP server for map tiles
        http_server = await asyncio.start_server(
            self._map_http_handler, self._host, self._map_port
        )
        logger.info(f"Glass map server listening on {self._host}:{self._map_port}")

        # Start map render loop
        map_task = asyncio.ensure_future(self._map_render_loop())

        # Keepalives on: a Glass that drives out of WiFi range never sends a TCP
        # FIN, so without pings the server keeps a dead client in _clients forever.
        async with websockets.serve(
            self._handler, self._host, self._port,
            ping_interval=20, ping_timeout=20,
        ):
            logger.info(
                f"Glass HUD server listening on {self._host}:{self._port}"
            )
            await self._broadcast_loop()

        map_task.cancel()
        http_server.close()
        logger.info("Glass HUD server stopped")

    def _thread_main(self):
        """Entry point for the server daemon thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            logger.error(f"Glass server thread error: {e}")
        finally:
            self._loop.close()

    def push_now(self):
        """Immediately broadcast current state to all Glass clients (thread-safe)."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._immediate_broadcast())
            )

    async def _immediate_broadcast(self):
        """Send state to all clients right now, bypassing the interval timer."""
        if not self._clients:
            return
        state = self._build_state()
        state_json = json.dumps(state)
        self._last_state_json = state_json
        await self._send_state(state_json)

    def start(self):
        """Start the Glass HUD server in a daemon thread."""
        self._thread = threading.Thread(
            target=self._thread_main, name="glass-server", daemon=True
        )
        self._thread.start()
        logger.info("Glass HUD server thread started")

    def stop(self):
        """Stop the Glass HUD server."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop_event.set)
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Glass HUD server shut down")


# Standalone test mode
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class _MockSpotify:
        def get_current_playback(self):
            return {"is_playing": False}
        def pause(self):
            print("[mock] pause")
        def resume(self):
            print("[mock] resume")
        def skip(self):
            print("[mock] skip")
        def previous(self):
            print("[mock] previous")

    # Weather config from config.yaml — no secrets in code
    try:
        import yaml
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")) as f:
            weather_cfg = (yaml.safe_load(f) or {}).get("weather", {})
    except Exception:
        weather_cfg = {"api_key": "", "city": "Woodstock", "state": "GA", "country": "US"}
    glass_cfg = {"host": "0.0.0.0", "port": 9100, "map_port": 9101}

    server = GlassServer(glass_cfg, _MockSpotify(), weather_cfg)
    server.start()

    try:
        print("Glass server running -- WS :9100, Map :9101")
        print("Test with: wscat -c ws://localhost:9100")
        print("Map: curl http://localhost:9101/map.jpg -o map.jpg")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
