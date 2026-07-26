#!/usr/bin/env python3
"""
VIVIAN CRT Display Module v1.0
Unified display for 4.5" B&W CRT via composite video (Pi 5 TP7)
Renders to framebuffer at 320x240 NTSC using pygame.
Replaces the 128x64 OLED module with a single-page layout.
"""

import argparse
import io
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytz
import requests

# Set SDL video driver before importing pygame
# Pi 5 uses KMS/DRM; fallback to fbcon or dummy for headless testing
if not os.environ.get('SDL_VIDEODRIVER'):
    os.environ['SDL_VIDEODRIVER'] = 'kmsdrm'
if not os.environ.get('SDL_DRM_DEVICE'):
    os.environ['SDL_DRM_DEVICE'] = '/dev/dri/card1'

import pygame

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

# Shared data files written by other controllers
GPS_SHARED_FILE = "/tmp/vivian_gps.json"
SPOTIFY_SHARED_FILE = "/tmp/vivian_spotify.json"
ASSISTANT_SHARED_FILE = "/tmp/vivian_assistant.json"
DISPLAY_MODE_FILE = "/tmp/vivian_display_mode"


def _load_weather_config():
    """Read the weather API key/city from config.yaml (no secrets in code)."""
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(cfg_path) as f:
            weather = (yaml.safe_load(f) or {}).get("weather", {})
        key = weather.get("api_key", "")
        city = ",".join(filter(None, [
            weather.get("city", "Woodstock"),
            weather.get("state", "GA"),
            weather.get("country", "US"),
        ]))
        return key, city
    except Exception:
        return "", "Woodstock,GA,US"


_WEATHER_API_KEY, _WEATHER_CITY = _load_weather_config()


class Config:
    WIDTH = 320
    HEIGHT = 240

    SVG_PATH = "/home/cjatkinson/logo.svg"
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    SENTRY_FLAG_PATH = "/tmp/vivian_sentry_enabled"

    WEATHER_API_KEY = _WEATHER_API_KEY
    CITY = _WEATHER_CITY
    UNITS = "imperial"

    BOOT_DURATION = 10
    DISPLAY_REFRESH = 0.25
    WEATHER_REFRESH = 120
    SENTRY_FLASH_ON = 2.0
    SENTRY_FLASH_OFF = 1.0

    LOGO_SIZE = 18
    LOGO_SIZE_BOOT = 80
    LOGO_SIZE_MAIN = 60
    TIMEZONE = "America/New_York"

    PAGE_CYCLE_INTERVAL = 4
    PAGES = ["main", "diagnostics", "weather", "navigation"]
    WEATHER_DETAIL_REFRESH = 120
    OSM_TILE_REFRESH = 30


# Colors (B&W only for CRT)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def read_gps_data() -> Dict:
    """Read GPS data from shared file written by GPSController"""
    default = {
        'lat': None, 'lon': None, 'alt': None,
        'speed_mps': None, 'speed_mph': None,
        'track': None,
        'mode': 0, 'sats_used': 0, 'sats_visible': 0,
        'has_fix': False, 'timestamp': 0
    }
    try:
        if os.path.exists(GPS_SHARED_FILE):
            with open(GPS_SHARED_FILE, 'r') as f:
                data = json.load(f)
                if time.time() - data.get('timestamp', 0) > 5:
                    data['has_fix'] = False
                return data
    except Exception:
        pass
    return default


def read_spotify_data() -> Dict:
    """Read Spotify playback data from shared file written by SpotifyController."""
    try:
        if os.path.exists(SPOTIFY_SHARED_FILE):
            with open(SPOTIFY_SHARED_FILE, 'r') as f:
                data = json.load(f)
                if time.time() - data.get('timestamp', 0) > 30:
                    return {}
                return data
    except Exception:
        pass
    return {}


class CRTDisplay:
    """CRT display via pygame framebuffer. Drop-in replacement for OLEDDisplay."""

    def __init__(self):
        self.config = Config()
        self.screen = None
        self.logo = None
        self.logo_boot = None
        self.logo_main = None
        self.fonts = {}
        self._session = requests.Session()
        self._weather_cache = (None, None, 0)
        self._weather_detail_cache = (None, 0)
        self._osm_tile_cache = (None, None, None, 0, None)  # (surface, lat, lon, timestamp, zoom)
        # Raw RGB bytes published by the fetcher, converted to a Surface by the
        # render thread on next read (see _get_osm_tile).
        self._osm_pending = None
        self._radar_pending = None
        self._radar_cache = (None, 0)  # (surface, timestamp)
        # Every HTTP request in this module runs on the background fetcher
        # thread (see _network_worker). The render loop only reads these
        # caches: a degraded cellular hotspot used to freeze the CRT for ~45s
        # (9 sequential tile fetches x 5s timeout) with no LISTENING overlay,
        # no TTS waveform and no pygame event draining. These two fields tell
        # the fetcher what the render loop last asked for.
        self._osm_wanted = None    # (lat, lon, speed_mph)
        self._radar_wanted = None  # (lat, lon)
        self._net_lock = threading.Lock()  # guards every cache above (2 threads now)
        self._net_thread = None
        self._net_thread_lock = threading.Lock()  # start-once guard
        self.trip_start_time = None  # Set when driving starts
        self.trip_distance = 0.0
        self.last_position = None
        self._last_moving_time = None  # Last time speed was >= 5 mph with good GPS
        self._trip_active = False
        self._min_trip_sats = 4  # Ignore speed readings below this sat count
        self.running = False
        self._current_page = 0
        self._page_switch_time = time.time()

    def init_display(self) -> 'CRTDisplay':
        """Initialize pygame and open fullscreen display on framebuffer."""
        pygame.init()
        pygame.mouse.set_visible(False)

        # Try fullscreen first, fall back to windowed for dev/testing
        try:
            self.hw_screen = pygame.display.set_mode(
                (0, 0), pygame.FULLSCREEN
            )
        except pygame.error:
            self.hw_screen = pygame.display.set_mode(
                (self.config.WIDTH, self.config.HEIGHT)
            )

        self.hw_size = self.hw_screen.get_size()
        # Render to a virtual 320x240 surface, then scale to hardware
        self.screen = pygame.Surface((self.config.WIDTH, self.config.HEIGHT))

        pygame.display.set_caption("VIVIAN CRT")
        self._load_fonts()
        return self

    def _load_fonts(self):
        """Load fonts scaled for 320x240 CRT resolution."""
        try:
            self.fonts = {
                'tiny': pygame.font.Font(self.config.FONT_PATH, 12),
                'small': pygame.font.Font(self.config.FONT_PATH, 14),
                'medium': pygame.font.Font(self.config.FONT_PATH, 18),
                'medium_large': pygame.font.Font(self.config.FONT_PATH, 21),
                'large': pygame.font.Font(self.config.FONT_BOLD_PATH, 24),
                'xlarge': pygame.font.Font(self.config.FONT_BOLD_PATH, 36),
                'huge': pygame.font.Font(self.config.FONT_BOLD_PATH, 52),
            }
        except Exception:
            default = pygame.font.Font(None, 16)
            self.fonts = {k: default for k in ['tiny', 'small', 'medium', 'large', 'xlarge', 'huge']}

    def _load_logo(self, size) -> Optional[pygame.Surface]:
        """Load VIVIAN SVG logo as a pygame surface."""
        if not CAIROSVG_AVAILABLE or not os.path.exists(self.config.SVG_PATH):
            return None
        try:
            png_bytes = cairosvg.svg2png(
                url=self.config.SVG_PATH, output_width=size, output_height=size
            )
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(png_bytes))
            pil_img = pil_img.convert("RGBA")

            # Convert to B&W: white logo on transparent
            pixels = pil_img.load()
            for y in range(pil_img.height):
                for x in range(pil_img.width):
                    r, g, b, a = pixels[x, y]
                    if a > 128:
                        gray = (r + g + b) // 3
                        if gray < 128:
                            pixels[x, y] = (255, 255, 255, 255)
                        else:
                            pixels[x, y] = (0, 0, 0, 0)
                    else:
                        pixels[x, y] = (0, 0, 0, 0)

            # Convert PIL to pygame surface
            raw = pil_img.tobytes()
            surf = pygame.image.fromstring(raw, pil_img.size, "RGBA").convert_alpha()
            return surf
        except Exception:
            return None

    def _get_logo(self) -> Optional[pygame.Surface]:
        if self.logo is None:
            self.logo = self._load_logo(self.config.LOGO_SIZE)
        return self.logo

    def _get_logo_boot(self) -> Optional[pygame.Surface]:
        if self.logo_boot is None:
            self.logo_boot = self._load_logo(self.config.LOGO_SIZE_BOOT)
        return self.logo_boot

    def _get_logo_main(self) -> Optional[pygame.Surface]:
        if self.logo_main is None:
            self.logo_main = self._load_logo(self.config.LOGO_SIZE_MAIN)
        return self.logo_main

    # ------------------------------------------------------------------
    # Data sources
    # ------------------------------------------------------------------

    def _get_gps(self) -> Dict:
        """Get GPS data from shared file and update trip distance/timer."""
        data = read_gps_data()
        now = time.time()

        if data['has_fix'] and data['lat'] and data['lon']:
            current = (data['lat'], data['lon'])

            # Only trust speed when we have enough satellites
            speed = data.get('speed_mph') or 0
            good_signal = data.get('sats_used', 0) >= self._min_trip_sats

            if good_signal and speed >= 5:
                self._last_moving_time = now
                if not self._trip_active:
                    # Start a new trip
                    self._trip_active = True
                    self.trip_start_time = now
                    self.trip_distance = 0.0
                    self.last_position = current
            elif good_signal and self._trip_active and self._last_moving_time:
                # Check if stationary for 5+ minutes
                if now - self._last_moving_time > 300:
                    self._trip_active = False
                    self.trip_start_time = None
                    self.trip_distance = 0.0
                    self.last_position = None

            # Accumulate distance only during active trip with good signal
            if self._trip_active and good_signal and self.last_position:
                dist = self._haversine(self.last_position, current)
                if dist < 1000:
                    self.trip_distance += dist
            if good_signal:
                self.last_position = current
        return data

    def _haversine(self, p1, p2) -> float:
        """Distance between two points in meters."""
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371000 * 2 * math.asin(math.sqrt(a))

    def _start_network_thread(self):
        """Start the background fetcher once (idempotent, thread-safe)."""
        with self._net_thread_lock:
            if self._net_thread is not None and self._net_thread.is_alive():
                return
            self._net_thread = threading.Thread(
                target=self._network_worker, name="crt-net", daemon=True
            )
            self._net_thread.start()

    def _network_worker(self):
        """Every network call in this module lives here, off the render loop.
        Each fetcher checks its own TTL, so this just ticks once a second."""
        while self.running:
            for fetch in (self._fetch_weather, self._fetch_weather_detail,
                          self._fetch_osm_tile, self._fetch_radar_tile):
                try:
                    fetch()
                except Exception:
                    # Must never die: if this thread exits, weather and maps
                    # silently stop updating for the rest of the drive.
                    pass
            time.sleep(1.0)

    def _get_weather(self) -> Tuple[Optional[int], Optional[str]]:
        """Read the cached summary weather. No network here — render path only."""
        with self._net_lock:
            temp, desc, _last_fetch = self._weather_cache
        return temp, desc

    def _fetch_weather(self):
        """Background: refresh the summary weather cache (runs off the render loop)."""
        with self._net_lock:
            temp, desc, last_fetch = self._weather_cache
        if time.time() - last_fetch < self.config.WEATHER_REFRESH:
            return
        try:
            r = self._session.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": self.config.CITY,
                    "appid": self.config.WEATHER_API_KEY,
                    "units": self.config.UNITS,
                },
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
            temp = round(data["main"]["temp"])
            desc = data["weather"][0]["main"]
        except Exception:
            pass
        with self._net_lock:
            self._weather_cache = (temp, desc, time.time())

    def _get_weather_detail(self) -> Optional[Dict]:
        """Read cached detailed weather for the weather page (no network here)."""
        with self._net_lock:
            data, _last_fetch = self._weather_detail_cache
        return data

    def _fetch_weather_detail(self):
        """Background: refresh the detailed weather cache."""
        with self._net_lock:
            data, last_fetch = self._weather_detail_cache
        if time.time() - last_fetch < self.config.WEATHER_DETAIL_REFRESH:
            return
        try:
            r = self._session.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": self.config.CITY,
                    "appid": self.config.WEATHER_API_KEY,
                    "units": self.config.UNITS,
                },
                timeout=5,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            pass
        with self._net_lock:
            self._weather_detail_cache = (data, time.time())

    def _get_system_info(self) -> Dict:
        """Get Pi system diagnostics."""
        info = {}
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                info['cpu_temp'] = int(f.read().strip()) / 1000
        except Exception:
            info['cpu_temp'] = None
        try:
            with open('/proc/uptime') as f:
                info['uptime_secs'] = int(float(f.read().split()[0]))
        except Exception:
            info['uptime_secs'] = 0
        try:
            with open('/proc/meminfo') as f:
                lines = f.readlines()
                meminfo = {}
                for line in lines:
                    parts = line.split(':')
                    if len(parts) == 2:
                        meminfo[parts[0].strip()] = int(parts[1].strip().split()[0])
                total = meminfo.get('MemTotal', 1)
                avail = meminfo.get('MemAvailable', 0)
                info['mem_total_mb'] = total // 1024
                info['mem_used_mb'] = (total - avail) // 1024
                info['mem_pct'] = int(((total - avail) / total) * 100)
        except Exception:
            info['mem_total_mb'] = 0
            info['mem_used_mb'] = 0
            info['mem_pct'] = 0
        try:
            st = os.statvfs('/')
            total = st.f_blocks * st.f_frsize
            free = st.f_bavail * st.f_frsize
            info['disk_total_gb'] = total / (1024 ** 3)
            info['disk_used_gb'] = (total - free) / (1024 ** 3)
            info['disk_pct'] = int(((total - free) / total) * 100)
        except Exception:
            info['disk_total_gb'] = 0
            info['disk_used_gb'] = 0
            info['disk_pct'] = 0
        try:
            with open('/proc/loadavg') as f:
                info['load_avg'] = f.read().split()[0]
        except Exception:
            info['load_avg'] = '--'
        try:
            result = subprocess.run(
                ['hostname', '-I'], capture_output=True, text=True, timeout=2
            )
            ips = result.stdout.strip().split()
            info['ip'] = ips[0] if ips else '--'
        except Exception:
            info['ip'] = '--'
        return info

    def _lat_lon_to_tile(self, lat, lon, zoom):
        """Convert lat/lon to slippy map tile coordinates + pixel offset within tile."""
        n = 2 ** zoom
        x_float = (lon + 180) / 360 * n
        lat_rad = math.radians(lat)
        y_float = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n
        tile_x = int(x_float)
        tile_y = int(y_float)
        pixel_x = int((x_float - tile_x) * 256)
        pixel_y = int((y_float - tile_y) * 256)
        return tile_x, tile_y, pixel_x, pixel_y

    def _zoom_for_speed(self, speed_mph) -> int:
        """Pick OSM zoom level based on GPS speed."""
        if speed_mph is None or speed_mph < 5:
            return 17   # ~250ft view - walking/parked
        elif speed_mph < 15:
            return 16   # ~500ft - neighborhood
        elif speed_mph < 35:
            return 15   # ~0.5mi - city streets
        elif speed_mph < 55:
            return 14   # ~1mi - arterials
        elif speed_mph < 70:
            return 13   # ~2mi - highway
        else:
            return 12   # ~5mi - interstate

    def _get_osm_tile(self, lat, lon, speed_mph=0) -> Optional[Tuple[pygame.Surface, int, int]]:
        """Return the cached map tile and tell the fetcher where we are.
        Never fetches: the 3x3 tile grid is 9 requests and blocking the render
        loop on it froze the CRT for ~45s on a bad hotspot. A stale tile is
        drawn until the fetcher publishes a fresh one (None = 'Loading map...')."""
        with self._net_lock:
            self._osm_wanted = (lat, lon, speed_mph)
            pending = self._osm_pending
            self._osm_pending = None

        if pending is not None:
            # Build the pygame Surface HERE, on the render thread. The fetcher
            # publishes raw RGB bytes instead of a Surface: SDL only guarantees
            # display/event calls are main-thread-only, but constructing
            # Surfaces off-thread is untested on this hardware and this
            # conversion is ~1ms — not worth the risk in a car.
            raw, size, cx, cy, p_lat, p_lon, p_ts, p_zoom = pending
            try:
                surf = pygame.image.fromstring(raw, size, 'RGB')
                with self._net_lock:
                    self._osm_tile_cache = ((surf, cx, cy), p_lat, p_lon, p_ts, p_zoom)
            except Exception:
                pass  # keep whatever tile is already cached

        with self._net_lock:
            return self._osm_tile_cache[0]

    def _fetch_osm_tile(self):
        """Background: fetch OSM tiles, stitch, convert to white-on-black.
        Publishes (surface, center_x, center_y) into the tile cache."""
        with self._net_lock:
            wanted = self._osm_wanted
            cached_surf, cached_lat, cached_lon, cached_time, cached_zoom = self._osm_tile_cache
        if not wanted:
            return  # navigation page hasn't been rendered yet — don't waste data
        lat, lon, speed_mph = wanted
        zoom = self._zoom_for_speed(speed_mph)
        if cached_surf and time.time() - cached_time < self.config.OSM_TILE_REFRESH:
            if cached_lat is not None and cached_lon is not None and cached_zoom == zoom:
                dist = self._haversine((lat, lon), (cached_lat, cached_lon))
                # Scale cache distance threshold with zoom (higher zoom = smaller area)
                max_dist = 50 * (2 ** (17 - zoom))
                if dist < max_dist:
                    return
        try:
            from PIL import Image as PILImage
            tile_x, tile_y, px_off, py_off = self._lat_lon_to_tile(lat, lon, zoom)

            # Fetch 3x3 grid of tiles centered on current position
            tile_size = 256
            canvas = PILImage.new('RGB', (tile_size * 3, tile_size * 3), (255, 255, 255))
            headers = {'User-Agent': 'VIVIAN/1.0'}
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    tx = tile_x + dx
                    ty = tile_y + dy
                    url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
                    try:
                        r = self._session.get(url, headers=headers, timeout=5)
                        r.raise_for_status()
                        tile_img = PILImage.open(io.BytesIO(r.content)).convert('RGB')
                        canvas.paste(tile_img, ((dx + 1) * tile_size, (dy + 1) * tile_size))
                    except Exception:
                        pass

            # Center pixel of our position on the canvas
            center_canvas_x = tile_size + px_off
            center_canvas_y = tile_size + py_off

            # Crop around center
            crop_w, crop_h = 300, 170
            left = center_canvas_x - crop_w // 2
            top = center_canvas_y - crop_h // 2
            cropped = canvas.crop((left, top, left + crop_w, top + crop_h))

            # Convert to grayscale — roads/text are darker than background (~238)
            # Roads/labels become bright white, land becomes dim gray to avoid CRT overdrive
            gray = cropped.convert('L')
            enhanced = gray.point(lambda p: 255 if p < 230 else 60)
            rgb = enhanced.convert('RGB')

            # Publish raw bytes, not a Surface — the render thread converts.
            with self._net_lock:
                self._osm_pending = (rgb.tobytes(), rgb.size,
                                     crop_w // 2, crop_h // 2,
                                     lat, lon, time.time(), zoom)
        except Exception:
            pass  # keep whatever tile is already cached

    def _get_greeting(self, dt) -> str:
        h = dt.hour
        if 5 <= h <= 11:
            return "Good Morning"
        elif 12 <= h <= 17:
            return "Good Afternoon"
        return "Good Evening"

    def _cardinal(self, degrees) -> str:
        if degrees is None:
            return "--"
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        return dirs[int((degrees + 22.5) / 45) % 8]

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _flip(self):
        """Scale virtual surface to hardware screen and flip (with 10% overscan margin)."""
        margin = 0.25
        target_w = int(self.hw_size[0] * (1 - margin))
        target_h = int(self.hw_size[1] * (1 - margin))
        offset_x = (self.hw_size[0] - target_w) // 2
        offset_y = (self.hw_size[1] - target_h) // 2
        scaled = pygame.transform.smoothscale(self.screen, (target_w, target_h))
        self.hw_screen.blit(scaled, (offset_x, offset_y))
        pygame.display.flip()

    def _draw_text(self, x, y, text, font_name='small', color=WHITE):
        """Draw text at (x, y)."""
        surf = self.fonts[font_name].render(text, True, color)
        self.screen.blit(surf, (x, y))

    def _draw_text_right(self, x, y, text, font_name='small', color=WHITE):
        """Draw text right-aligned to x."""
        surf = self.fonts[font_name].render(text, True, color)
        self.screen.blit(surf, (x - surf.get_width(), y))

    def _draw_text_centered(self, y, text, font_name='small', color=WHITE):
        """Draw text horizontally centered."""
        surf = self.fonts[font_name].render(text, True, color)
        self.screen.blit(surf, ((self.config.WIDTH - surf.get_width()) // 2, y))

    def _text_width(self, text, font_name='small') -> int:
        surf = self.fonts[font_name].render(text, True, WHITE)
        return surf.get_width()

    def _fit_width(self, text, font_name='small', max_w=None) -> str:
        """Truncate text with an ellipsis so it fits within max_w pixels
        (default: screen width minus a small margin), never clipping."""
        if max_w is None:
            max_w = self.config.WIDTH - 8
        if self._text_width(text, font_name) <= max_w:
            return text
        while text and self._text_width(text + "…", font_name) > max_w:
            text = text[:-1]
        return text + "…"

    def _draw_compass(self, cx, cy, radius, heading):
        """Draw compass circle with heading arrow."""
        # Outer circle
        pygame.draw.circle(self.screen, WHITE, (cx, cy), radius, 1)

        # Cardinal tick marks
        for label, angle_deg in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
            angle = math.radians(angle_deg - 90)
            x1 = cx + int((radius - 3) * math.cos(angle))
            y1 = cy + int((radius - 3) * math.sin(angle))
            x2 = cx + int((radius + 3) * math.cos(angle))
            y2 = cy + int((radius + 3) * math.sin(angle))
            pygame.draw.line(self.screen, WHITE, (x1, y1), (x2, y2), 1)

        # N label
        n_surf = self.fonts['tiny'].render("N", True, WHITE)
        self.screen.blit(n_surf, (cx - n_surf.get_width() // 2, cy - radius - 14))

        # Heading arrow
        if heading is not None:
            angle = math.radians(heading - 90)
            tip_len = radius - 4
            tip_x = cx + int(tip_len * math.cos(angle))
            tip_y = cy + int(tip_len * math.sin(angle))

            tail_len = 5
            tail_x = cx - int(tail_len * math.cos(angle))
            tail_y = cy - int(tail_len * math.sin(angle))

            wing_len = 8
            wing_angle = 2.5
            wing1_x = cx + int(wing_len * math.cos(angle + wing_angle))
            wing1_y = cy + int(wing_len * math.sin(angle + wing_angle))
            wing2_x = cx + int(wing_len * math.cos(angle - wing_angle))
            wing2_y = cy + int(wing_len * math.sin(angle - wing_angle))

            pygame.draw.polygon(
                self.screen, WHITE,
                [(tip_x, tip_y), (wing1_x, wing1_y), (tail_x, tail_y), (wing2_x, wing2_y)]
            )

        # Center dot
        pygame.draw.circle(self.screen, WHITE, (cx, cy), 2)

    def _draw_weather_icon(self, cx, cy, condition, size=30):
        """Draw a weather condition icon using primitives."""
        r = size // 2
        cond = condition.lower() if condition else ""

        if "clear" in cond or "sun" in cond:
            # Sun: circle with rays
            pygame.draw.circle(self.screen, WHITE, (cx, cy), r // 2, 2)
            for angle_deg in range(0, 360, 45):
                a = math.radians(angle_deg)
                x1 = cx + int((r // 2 + 3) * math.cos(a))
                y1 = cy + int((r // 2 + 3) * math.sin(a))
                x2 = cx + int((r - 2) * math.cos(a))
                y2 = cy + int((r - 2) * math.sin(a))
                pygame.draw.line(self.screen, WHITE, (x1, y1), (x2, y2), 1)

        elif "cloud" in cond or "overcast" in cond:
            # Cloud: overlapping arcs
            pygame.draw.ellipse(self.screen, WHITE, (cx - r, cy - r // 3, r * 2, r), 1)
            pygame.draw.ellipse(self.screen, WHITE, (cx - r // 2, cy - r // 2 - 4, r, r // 2 + 4), 1)

        elif "rain" in cond or "drizzle" in cond:
            # Cloud + rain drops
            pygame.draw.ellipse(self.screen, WHITE, (cx - r, cy - r // 2, r * 2, r // 2 + 4), 1)
            for dx in [-r // 2, 0, r // 2]:
                x1 = cx + dx
                y1 = cy + 6
                pygame.draw.line(self.screen, WHITE, (x1, y1), (x1 - 2, y1 + 6), 1)

        elif "thunder" in cond or "storm" in cond:
            # Cloud + lightning bolt
            pygame.draw.ellipse(self.screen, WHITE, (cx - r, cy - r // 2, r * 2, r // 2 + 4), 1)
            bolt = [(cx, cy + 4), (cx - 4, cy + 12), (cx + 2, cy + 12),
                    (cx - 2, cy + r)]
            pygame.draw.lines(self.screen, WHITE, False, bolt, 2)

        elif "snow" in cond:
            # Snowflakes: asterisks
            for dx, dy in [(-r // 3, -2), (r // 3, 4), (0, -6)]:
                sx, sy = cx + dx, cy + dy
                for angle_deg in [0, 60, 120]:
                    a = math.radians(angle_deg)
                    x1 = sx + int(4 * math.cos(a))
                    y1 = sy + int(4 * math.sin(a))
                    x2 = sx - int(4 * math.cos(a))
                    y2 = sy - int(4 * math.sin(a))
                    pygame.draw.line(self.screen, WHITE, (x1, y1), (x2, y2), 1)

        elif "mist" in cond or "fog" in cond or "haze" in cond:
            # Horizontal wavy lines
            for i in range(4):
                y_line = cy - r // 2 + i * (r // 3)
                pygame.draw.line(self.screen, WHITE, (cx - r, y_line), (cx + r, y_line), 1)

        else:
            # Unknown: question mark
            self._draw_text(cx - 5, cy - 8, "?", 'medium')

    def _get_radar_tile(self, lat, lon) -> Optional[pygame.Surface]:
        """Return the cached radar image and tell the fetcher where we are.
        Same reason as _get_osm_tile: 10 requests must not run on the render loop."""
        with self._net_lock:
            self._radar_wanted = (lat, lon)
            pending = self._radar_pending
            self._radar_pending = None

        if pending is not None:
            # Surface built on the render thread — same reason as _get_osm_tile.
            raw, size, ts = pending
            try:
                with self._net_lock:
                    self._radar_cache = (pygame.image.fromstring(raw, size, 'RGB'), ts)
            except Exception:
                pass  # keep whatever radar image is already cached

        with self._net_lock:
            return self._radar_cache[0]

    def _fetch_radar_tile(self):
        """Background: fetch weather radar from RainViewer API."""
        with self._net_lock:
            wanted = self._radar_wanted
            cached = self._radar_cache
        if not wanted:
            return  # radar page hasn't been rendered yet
        lat, lon = wanted
        if cached[0] and time.time() - cached[1] < 120:
            return
        try:
            # Get latest radar timestamp
            r = self._session.get("https://api.rainviewer.com/public/weather-maps.json", timeout=5)
            r.raise_for_status()
            rv_data = r.json()
            past = rv_data.get('radar', {}).get('past', [])
            if not past:
                return
            latest = past[-1]['path']

            # Calculate tile coords for zoom level 7
            zoom = 7
            n = 2 ** zoom
            tile_x = int((lon + 180) / 360 * n)
            lat_rad = math.radians(lat)
            tile_y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)

            # Fetch 3x3 grid of tiles and stitch for better coverage
            tile_size = 256
            canvas_size = tile_size * 3
            from PIL import Image as PILImage
            canvas = PILImage.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    tx = tile_x + dx
                    ty = tile_y + dy
                    tile_url = f"https://tilecache.rainviewer.com{latest}/256/{zoom}/{tx}/{ty}/2/1_1.png"
                    try:
                        tr = self._session.get(tile_url, timeout=5)
                        tr.raise_for_status()
                        tile_img = PILImage.open(io.BytesIO(tr.content)).convert('RGBA')
                        px_x = (dx + 1) * tile_size
                        px_y = (dy + 1) * tile_size
                        canvas.paste(tile_img, (px_x, px_y))
                    except Exception:
                        pass

            # Crop center portion for display
            crop_w, crop_h = 280, 140
            left = (canvas_size - crop_w) // 2
            top = (canvas_size - crop_h) // 2
            cropped = canvas.crop((left, top, left + crop_w, top + crop_h))

            # Convert to high-contrast B&W
            rgb = cropped.convert('RGB')
            pixels = rgb.load()
            for y in range(rgb.height):
                for x in range(rgb.width):
                    r_val, g, b = pixels[x, y]
                    # Radar returns colored precipitation; any non-black pixel = rain
                    if r_val + g + b > 40:
                        pixels[x, y] = (255, 255, 255)
                    else:
                        pixels[x, y] = (0, 0, 0)

            # Publish raw bytes, not a Surface — the render thread converts.
            with self._net_lock:
                self._radar_pending = (rgb.tobytes(), rgb.size, time.time())
        except Exception:
            pass  # keep whatever radar image is already cached

    def _draw_signal_bars(self, x, y, sats):
        """Draw GPS signal strength bars."""
        bars = min(4, sats // 2) if sats else 0
        for i in range(4):
            h = 4 + i * 3
            bx = x + i * 6
            by = y + (14 - h)
            rect = pygame.Rect(bx, by, 4, h)
            if i < bars:
                pygame.draw.rect(self.screen, WHITE, rect)
            else:
                pygame.draw.rect(self.screen, WHITE, rect, 1)

    # ------------------------------------------------------------------
    # Layout rendering
    # ------------------------------------------------------------------

    def _render_main(self):
        """Render the unified main display — all data on one page."""
        W, H = self.config.WIDTH, self.config.HEIGHT
        tz = pytz.timezone(self.config.TIMEZONE)
        now = datetime.now(tz)
        gps = self._get_gps()

        # === HEADER BAR ===
        # "VIVIAN" label
        self._draw_text(4, 2, "VIVIAN", 'small')

        # Time — right side of header
        time_str = now.strftime("%I:%M %p").lstrip("0")
        self._draw_text_right(W - 4, 2, time_str, 'small')

        # Weather — next to time or below
        temp, desc = self._get_weather()
        if temp is not None:
            weather_str = f"{temp}F {desc[:8]}" if desc else f"{temp}F"
        else:
            weather_str = ""
        if weather_str:
            weather_w = self._text_width(weather_str, 'tiny')
            self._draw_text(W - weather_w - 4, 18, weather_str, 'tiny')

        # Header divider line
        pygame.draw.line(self.screen, WHITE, (0, 34), (W, 34), 1)

        # === SPEED BOX (center-left) ===
        speed_box_x = 10
        speed_box_y = 42
        speed_box_w = 120
        speed_box_h = 90

        # Speed box border
        pygame.draw.rect(
            self.screen, WHITE,
            pygame.Rect(speed_box_x, speed_box_y, speed_box_w, speed_box_h),
            1
        )

        # Speed value
        if gps['has_fix'] and gps['speed_mph'] is not None:
            speed_val = str(int(gps['speed_mph']))
        else:
            speed_val = "--"

        speed_surf = self.fonts['huge'].render(speed_val, True, WHITE)
        sx = speed_box_x + (speed_box_w - speed_surf.get_width()) // 2
        sy = speed_box_y + 5
        self.screen.blit(speed_surf, (sx, sy))

        # "MPH" label
        mph_surf = self.fonts['medium'].render("MPH", True, WHITE)
        mx = speed_box_x + (speed_box_w - mph_surf.get_width()) // 2
        self.screen.blit(mph_surf, (mx, speed_box_y + speed_box_h - 28))

        # === TRIP STATS (right column) ===
        trip_x = 145
        trip_y = 42

        self._draw_text(trip_x, trip_y, "TRIP", 'small')
        pygame.draw.line(self.screen, WHITE, (trip_x, trip_y + 16), (W - 4, trip_y + 16), 1)

        # Trip time
        if self.trip_start_time is not None:
            secs = int(time.time() - self.trip_start_time)
            mins = secs // 60
            hrs = mins // 60
            if hrs > 0:
                time_val = f"{hrs}h {mins % 60}m"
            else:
                time_val = f"{mins}m"
        else:
            secs = 0
            time_val = "--"
        self._draw_text(trip_x, trip_y + 20, f"Time: {time_val}", 'tiny')

        # Distance
        miles = self.trip_distance / 1609.34
        self._draw_text(trip_x, trip_y + 36, f"Dist: {miles:.1f} mi", 'tiny')

        # Average speed
        if secs > 60 and self.trip_distance > 100:
            avg = (self.trip_distance / 1609.34) / (secs / 3600)
            self._draw_text(trip_x, trip_y + 52, f"Avg:  {avg:.0f} mph", 'tiny')
        else:
            self._draw_text(trip_x, trip_y + 52, "Avg:  --", 'tiny')

        # Altitude
        if gps['alt'] is not None:
            alt_ft = int(gps['alt'] * 3.28084)
            self._draw_text(trip_x, trip_y + 68, f"Alt:  {alt_ft} ft", 'tiny')
        else:
            self._draw_text(trip_x, trip_y + 68, "Alt:  --", 'tiny')

        # === BOTTOM SECTION ===
        bottom_y = 140

        # Nav info (bottom-left, below speed box)
        nav_y = bottom_y + 2
        if gps['has_fix'] and gps['lat'] is not None:
            lat_str = f"{abs(gps['lat']):.4f} {'N' if gps['lat'] >= 0 else 'S'}"
            lon_str = f"{abs(gps['lon']):.4f} {'W' if gps['lon'] < 0 else 'E'}"
            self._draw_text(4, nav_y, lat_str, 'tiny')
            self._draw_text(4, nav_y + 14, lon_str, 'tiny')
        else:
            self._draw_text(4, nav_y, "No GPS Fix", 'tiny')

        # Fix status + sats
        if gps['has_fix']:
            fix_names = ["No Fix", "No Fix", "2D Fix", "3D Fix"]
            fix = fix_names[min(gps['mode'], 3)]
            self._draw_text(4, nav_y + 30, f"{fix}  {gps['sats_used']}/{gps['sats_visible']} sats", 'tiny')
            self._draw_signal_bars(110, nav_y + 28, gps['sats_used'])
        else:
            self._draw_text(4, nav_y + 30, "No GPS Data", 'tiny')

        # Logo (bottom-right, where compass was)
        main_logo = self._get_logo_main()
        if main_logo:
            logo_x = W - 80
            logo_y = nav_y
            self.screen.blit(main_logo, (logo_x, logo_y))

        # === NOW PLAYING BAR (very bottom) ===
        bar_y = H - 34
        pygame.draw.line(self.screen, WHITE, (0, bar_y), (W, bar_y), 1)

        spotify = read_spotify_data()
        if spotify.get('is_playing') and spotify.get('track'):
            track = spotify['track']
            artist = spotify.get('artist', '')

            # Truncate long names to fit
            label = f"{track} - {artist}"
            max_w = W - 8
            while self._text_width(label, 'tiny') > max_w and len(label) > 20:
                label = label[:-4] + "..."
            self._draw_text(4, bar_y + 3, label, 'tiny')

            # Progress bar
            prog_ms = spotify.get('progress_ms', 0)
            dur_ms = spotify.get('duration_ms', 1)
            pct = min(prog_ms / dur_ms, 1.0) if dur_ms > 0 else 0

            bar_x, bar_w, bar_h = 4, W - 90, 6
            bar_top = bar_y + 18
            pygame.draw.rect(self.screen, WHITE, pygame.Rect(bar_x, bar_top, bar_w, bar_h), 1)
            fill_w = int((bar_w - 2) * pct)
            if fill_w > 0:
                pygame.draw.rect(self.screen, WHITE, pygame.Rect(bar_x + 1, bar_top + 1, fill_w, bar_h - 2))

            # Time label to the right of progress bar
            prog_s = prog_ms // 1000
            dur_s = dur_ms // 1000
            time_str = f"{prog_s // 60}:{prog_s % 60:02d}/{dur_s // 60}:{dur_s % 60:02d}"
            self._draw_text(bar_x + bar_w + 4, bar_top - 2, time_str, 'tiny')
        else:
            self._draw_text(4, bar_y + 10, "No track playing", 'tiny')

    SENTRY_STATUS_PATH = "/tmp/vivian_sentry_status.json"

    def _read_sentry_status(self):
        """Return (header, detail) for the last sentry event, or None.

        header is a short status line (threat level + time); detail is the
        longer human description (shown on a second line, empty for plain
        events). Threat level is on a 1-10 scale.
        """
        try:
            if not os.path.exists(self.SENTRY_STATUS_PATH):
                return None
            with open(self.SENTRY_STATUS_PATH) as f:
                data = json.load(f)
            event = data.get("event", "")
            level = data.get("level")
            desc = data.get("description", "")
            when = data.get("time_str", "")
            if level is not None:
                header = f"THREAT {level}/10   {when}".strip()
                detail = desc
            else:
                header = f"{when}   {event}".strip()
                detail = ""
            return (header, detail)
        except Exception:
            return None

    def _render_sentry(self, flash_on):
        """Render sentry mode: camera feed background with flashing overlay text.

        Distinguishes three states so a stale/missing frame never masquerades
        as a live feed: fresh frame -> ACTIVE, no frame yet -> STARTING,
        frame older than 3s -> NO FEED (writer died or camera lost).
        """
        W, H = self.config.WIDTH, self.config.HEIGHT
        SENTRY_FRAME = "/tmp/vivian_sentry_frame.jpg"
        FRAME_FRESH_S = 3.0

        frame_exists = os.path.exists(SENTRY_FRAME)
        frame_fresh = False
        try:
            if frame_exists:
                frame_fresh = (time.time() - os.path.getmtime(SENTRY_FRAME)) < FRAME_FRESH_S
            if frame_fresh:
                import cv2
                import numpy as np
                img = cv2.imread(SENTRY_FRAME, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if img.shape[:2] != (H, W):
                        img = cv2.resize(img, (W, H))
                    img = (img * 0.35).astype(np.uint8)
                    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    frame_surf = pygame.image.fromstring(rgb.tobytes(), (W, H), 'RGB')
                    self.screen.blit(frame_surf, (0, 0))
                else:
                    frame_fresh = False
        except Exception:
            frame_fresh = False  # Black background fallback

        # Flashing overlay text (dim gray so feed stays visible)
        if flash_on:
            DIM = (140, 140, 140)
            if frame_fresh:
                status = "ACTIVE"
            elif frame_exists:
                status = "NO FEED"
            else:
                status = "STARTING..."
            self._draw_text_centered(90, "SENTRY MODE", 'large', color=DIM)
            self._draw_text_centered(120, status, 'large', color=DIM)

        # Last threat event (always shown, not flashed) along the bottom.
        # Threats get two lines: an amber "THREAT n/10  time" header and the
        # description beneath it; plain events get a single grey line.
        last = self._read_sentry_status()
        if last:
            header, detail = last
            H = self.config.HEIGHT
            if detail:
                self._draw_text_centered(
                    H - 30, self._fit_width(header, 'small'), 'small', color=(230, 180, 90))
                self._draw_text_centered(
                    H - 15, self._fit_width(detail, 'small'), 'small', color=(175, 175, 175))
            else:
                self._draw_text_centered(
                    H - 16, self._fit_width(header, 'small'), 'small', color=(175, 175, 175))

    def _render_boot(self):
        """Render boot splash screen."""
        H = self.config.HEIGHT
        logo = self._get_logo_boot()
        if logo:
            lx = (self.config.WIDTH - logo.get_width()) // 2
            ly = (H - logo.get_height()) // 2 - 15
            self.screen.blit(logo, (lx, ly))
            self._draw_text_centered(ly + logo.get_height() + 5, "VIVIAN", 'large')
        else:
            self._draw_text_centered(H // 2 - 15, "VIVIAN", 'large')
        self._draw_text_centered(H - 35, "Starting...", 'medium')

    def _read_assistant_state(self) -> Optional[Dict]:
        """Read assistant state from shared file. Returns None if idle."""
        try:
            if not os.path.exists(ASSISTANT_SHARED_FILE):
                return None
            mtime = os.path.getmtime(ASSISTANT_SHARED_FILE)
            if time.time() - mtime > 60:
                return None  # Stale — assistant probably crashed
            with open(ASSISTANT_SHARED_FILE, 'r') as f:
                data = json.load(f)
            if data.get('state') == 'idle':
                return None
            return data
        except Exception:
            return None

    def _render_assistant_overlay(self, state_data: Dict):
        """Render full-screen assistant state overlay (listening, transcribed, thinking, speaking)."""
        W, H = self.config.WIDTH, self.config.HEIGHT
        state = state_data.get('state', 'listening')
        transcript = state_data.get('transcript', '')

        if state == 'listening':
            # Logo centered with LISTENING above and below
            logo = self._get_logo_boot()
            if logo:
                lx = (W - logo.get_width()) // 2
                ly = (H - logo.get_height()) // 2 - 5
                self._draw_text_centered(ly - 25, "LISTENING", 'large')
                self.screen.blit(logo, (lx, ly))
                self._draw_text_centered(ly + logo.get_height() + 8, "LISTENING", 'large')
            else:
                self._draw_text_centered(H // 2 - 20, "LISTENING", 'xlarge')

        elif state == 'transcribed':
            # Show what was heard
            self._draw_text_centered(20, "HEARD:", 'medium')
            self._draw_wrapped_text(transcript, 50, 'large')

        elif state == 'thinking':
            # Show transcript + thinking indicator
            self._draw_text_centered(15, "HEARD:", 'small')
            self._draw_wrapped_text(transcript, 38, 'medium')
            self._draw_text_centered(H - 40, "THINKING...", 'large')

        elif state == 'speaking':
            # Transcript at top
            self._draw_text_centered(10, "HEARD:", 'small')
            self._draw_wrapped_text(transcript, 28, 'medium')
            # KITT-style voice waveform in middle
            self._render_kitt_waveform()
            # Label below waveform
            self._draw_text_centered(H - 30, "VIVIAN", 'large')

        elif state == 'error':
            # Error message centered on screen
            self._draw_text_centered(H // 2 - 30, "ERROR", 'xlarge')
            self._draw_wrapped_text(transcript, H // 2 + 10, 'medium')

    def _read_tts_amplitude(self) -> float:
        """Read current TTS audio amplitude from shared file."""
        try:
            amp_file = "/tmp/vivian_tts_amplitude.json"
            if not os.path.exists(amp_file):
                return 0.0
            with open(amp_file, 'r') as f:
                data = json.load(f)
            # Stale if older than 200ms
            if time.time() - data.get('t', 0) > 0.2:
                return 0.0
            return data.get('amplitude', 0.0)
        except Exception:
            return 0.0

    def _render_kitt_waveform(self):
        """Render KITT-style voice waveform driven by real TTS audio amplitude."""
        W, H = self.config.WIDTH, self.config.HEIGHT
        t = time.time()
        amp = self._read_tts_amplitude()

        # Waveform parameters
        num_bars = 32
        bar_width = 4
        bar_gap = 2
        total_w = num_bars * (bar_width + bar_gap) - bar_gap
        start_x = (W - total_w) // 2
        center_y = H // 2 + 10
        max_bar_h = 70

        if amp < 0.01:
            # Silent: just show a flat center line
            pygame.draw.line(self.screen, (80, 80, 80),
                             (start_x - 10, center_y),
                             (start_x + total_w + 10, center_y), 1)
            return

        for i in range(num_bars):
            # Distance from center (0.0 = center, 1.0 = edge)
            center_dist = abs(i - (num_bars - 1) / 2) / ((num_bars - 1) / 2)

            # Base envelope: taller in center, tapering to edges
            envelope = 1.0 - center_dist ** 1.5

            # Animated wave pattern modulated by real amplitude
            wave1 = math.sin(t * 8.0 + i * 0.5) * 0.35
            wave2 = math.sin(t * 13.0 + i * 0.9) * 0.25
            wave3 = math.sin(t * 18.0 + i * 1.3) * 0.15

            # Sweeping bright point (KITT scanner)
            sweep_pos = (math.sin(t * 3.5) + 1) / 2
            sweep_i = sweep_pos * (num_bars - 1)
            sweep_boost = max(0, 1.0 - abs(i - sweep_i) / 4) * 0.3

            # Combine waves, then scale by real amplitude (boosted)
            wave_mix = 0.4 + wave1 + wave2 + wave3 + sweep_boost
            bar_amp = envelope * wave_mix * (amp * 1.5 + 0.1)
            bar_amp = max(0.0, min(1.0, bar_amp))

            bar_h = int(bar_amp * max_bar_h)
            if bar_h < 1:
                continue

            x = start_x + i * (bar_width + bar_gap)

            # Brightness scales with amplitude
            brightness = int(120 + 135 * amp * (1.0 - center_dist * 0.4))
            brightness = min(255, brightness)
            color = (brightness, brightness, brightness)

            rect = pygame.Rect(x, center_y - bar_h, bar_width, bar_h * 2)
            pygame.draw.rect(self.screen, color, rect)

        # Center line
        pygame.draw.line(self.screen, (50, 50, 50),
                         (start_x - 10, center_y),
                         (start_x + total_w + 10, center_y), 1)

    def _draw_wrapped_text(self, text: str, y_start: int, size: str = 'small'):
        """Draw word-wrapped text centered on screen."""
        W = self.config.WIDTH
        font = self.fonts[size]
        words = text.split()
        lines = []
        current_line = ""
        max_w = W - 20  # 10px margin each side

        for word in words:
            test = f"{current_line} {word}".strip()
            if font.size(test)[0] <= max_w:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        line_h = font.get_linesize() + 2
        for i, line in enumerate(lines[:6]):  # Max 6 lines
            tw = font.size(line)[0]
            x = (W - tw) // 2
            self.screen.blit(font.render(line, True, WHITE), (x, y_start + i * line_h))

    def _render_page_header(self, title):
        """Render a standard page header with title and page dots."""
        W = self.config.WIDTH
        tz = pytz.timezone(self.config.TIMEZONE)
        now = datetime.now(tz)

        self._draw_text(4, 2, title, 'small')
        time_str = now.strftime("%I:%M %p").lstrip("0")
        self._draw_text_right(W - 4, 2, time_str, 'small')

        pygame.draw.line(self.screen, WHITE, (0, 22), (W, 22), 1)

    def _render_diagnostics(self):
        """Render system diagnostics page."""
        W, H = self.config.WIDTH, self.config.HEIGHT
        self._render_page_header("DIAGNOSTICS")
        info = self._get_system_info()

        y = 42
        gap = 18

        # CPU Temperature
        if info['cpu_temp'] is not None:
            temp_str = f"{info['cpu_temp']:.1f} C"
            self._draw_text(10, y, f"CPU Temp:  {temp_str}", 'small')
            # Temperature bar
            bar_x, bar_w, bar_h = 200, 100, 10
            pygame.draw.rect(self.screen, WHITE, pygame.Rect(bar_x, y + 3, bar_w, bar_h), 1)
            fill = min(int((info['cpu_temp'] / 85) * (bar_w - 2)), bar_w - 2)
            if fill > 0:
                pygame.draw.rect(self.screen, WHITE, pygame.Rect(bar_x + 1, y + 4, fill, bar_h - 2))
        y += gap

        # Memory
        self._draw_text(10, y, f"Memory:    {info['mem_used_mb']}MB / {info['mem_total_mb']}MB  ({info['mem_pct']}%)", 'small')
        y += gap

        # Disk
        self._draw_text(10, y, f"Disk:      {info['disk_used_gb']:.1f}GB / {info['disk_total_gb']:.1f}GB  ({info['disk_pct']}%)", 'small')
        y += gap

        # Load Average
        self._draw_text(10, y, f"Load:      {info['load_avg']}", 'small')
        y += gap

        # Uptime
        up = info['uptime_secs']
        days = up // 86400
        hrs = (up % 86400) // 3600
        mins = (up % 3600) // 60
        if days > 0:
            up_str = f"{days}d {hrs}h {mins}m"
        elif hrs > 0:
            up_str = f"{hrs}h {mins}m"
        else:
            up_str = f"{mins}m"
        self._draw_text(10, y, f"Uptime:    {up_str}", 'small')
        y += gap

        # IP Address
        self._draw_text(10, y, f"IP:        {info['ip']}", 'small')
        y += gap

        # VIVIAN service status
        sentry = "ARMED" if os.path.exists(self.config.SENTRY_FLAG_PATH) else "OFF"
        self._draw_text(10, y, f"Sentry:    {sentry}", 'small')

    def _render_weather_page(self):
        """Render detailed weather page."""
        W, H = self.config.WIDTH, self.config.HEIGHT
        self._render_page_header("WEATHER")
        data = self._get_weather_detail()

        if not data:
            self._draw_text_centered(100, "No weather data", 'medium')
            return

        try:
            main = data['weather'][0]
            temp = data['main']
            wind = data.get('wind', {})
            clouds = data.get('clouds', {})
            rain = data.get('rain', {})
            snow = data.get('snow', {})
            sys_data = data.get('sys', {})

            condition = main.get('main', 'Unknown')
            desc = main.get('description', '').title()

            # Condition + temp
            cur_temp = round(temp.get('temp', 0))
            self._draw_text(4, 42, f"{cur_temp} F", 'large')
            self._draw_text(4, 68, desc, 'small')

            # Feels like + hi/lo on the right
            feels = round(temp.get('feels_like', 0))
            hi = round(temp.get('temp_max', 0))
            lo = round(temp.get('temp_min', 0))
            self._draw_text_right(W - 4, 42, f"Feels {feels} F", 'small')
            self._draw_text_right(W - 4, 58, f"Hi {hi}  Lo {lo}", 'tiny')

            # Divider
            y = 88
            pygame.draw.line(self.screen, WHITE, (4, y), (W - 4, y), 1)
            y += 6

            gap = 16

            # Humidity + Pressure
            humidity = temp.get('humidity', 0)
            pressure = temp.get('pressure', 0)
            dew_point = None
            if humidity and cur_temp:
                # Magnus formula approximation
                a, b = 17.27, 237.7
                t_c = (cur_temp - 32) * 5 / 9
                alpha = (a * t_c) / (b + t_c) + math.log(humidity / 100)
                dew_c = (b * alpha) / (a - alpha)
                dew_point = round(dew_c * 9 / 5 + 32)
            hum_str = f"Humidity: {humidity}%"
            if dew_point is not None:
                hum_str += f"   Dew: {dew_point} F"
            self._draw_text(4, y, hum_str, 'tiny')
            y += gap

            # Pressure
            self._draw_text(4, y, f"Pressure: {pressure} hPa", 'tiny')
            y += gap

            # Wind
            speed = round(wind.get('speed', 0))
            deg = wind.get('deg', 0)
            direction = self._cardinal(deg)
            gust = wind.get('gust')
            wind_str = f"Wind: {speed} mph {direction}"
            if gust:
                wind_str += f"  (gust {round(gust)})"
            self._draw_text(4, y, wind_str, 'tiny')
            y += gap

            # Precipitation
            rain_1h = rain.get('1h', 0)
            rain_3h = rain.get('3h', 0)
            snow_1h = snow.get('1h', 0)
            if rain_1h or rain_3h or snow_1h:
                precip_str = "Precip:"
                if rain_1h:
                    precip_str += f" {rain_1h}mm/1h"
                if rain_3h:
                    precip_str += f" {rain_3h}mm/3h"
                if snow_1h:
                    precip_str += f" Snow {snow_1h}mm/1h"
                self._draw_text(4, y, precip_str, 'tiny')
            else:
                self._draw_text(4, y, "Precip: None", 'tiny')
            y += gap

            # Clouds + Visibility
            cloud_pct = clouds.get('all', 0)
            vis = data.get('visibility', 0)
            vis_mi = vis / 1609.34
            self._draw_text(4, y, f"Clouds: {cloud_pct}%   Vis: {vis_mi:.1f} mi", 'tiny')
            y += gap

            # Sunrise/Sunset
            tz = pytz.timezone(self.config.TIMEZONE)
            sunrise = sys_data.get('sunrise')
            sunset = sys_data.get('sunset')
            if sunrise and sunset:
                sr = datetime.fromtimestamp(sunrise, tz).strftime("%I:%M %p").lstrip("0")
                ss = datetime.fromtimestamp(sunset, tz).strftime("%I:%M %p").lstrip("0")
                self._draw_text(4, y, f"Rise: {sr}   Set: {ss}", 'tiny')

        except Exception:
            self._draw_text_centered(120, "Error parsing weather", 'small')

    def _render_radar(self):
        """Render weather radar page."""
        W, H = self.config.WIDTH, self.config.HEIGHT
        self._render_page_header("RADAR")
        gps = self._get_gps()

        if not gps['has_fix'] or gps['lat'] is None:
            # Use default location (Woodstock, GA)
            lat, lon = 34.1015, -84.5194
        else:
            lat, lon = gps['lat'], gps['lon']

        radar = self._get_radar_tile(lat, lon)
        if radar:
            tile_x = (W - radar.get_width()) // 2
            tile_y = 40
            self.screen.blit(radar, (tile_x, tile_y))

            # Crosshair at center
            cx = tile_x + radar.get_width() // 2
            cy = tile_y + radar.get_height() // 2
            pygame.draw.line(self.screen, WHITE, (cx - 6, cy), (cx + 6, cy), 1)
            pygame.draw.line(self.screen, WHITE, (cx, cy - 6), (cx, cy + 6), 1)

            # Legend
            self._draw_text(4, H - 36, "White = precipitation", 'tiny')
        else:
            self._draw_text_centered(100, "Loading radar...", 'small')

        self._draw_text_centered(H - 18, f"{lat:.3f}, {lon:.3f}", 'tiny')

    def _render_navigation(self):
        """Render navigation page with OSM map tile."""
        W, H = self.config.WIDTH, self.config.HEIGHT
        self._render_page_header("NAVIGATION")
        gps = self._get_gps()

        if not gps['has_fix'] or gps['lat'] is None:
            self._draw_text_centered(100, "No GPS Fix", 'medium')
            self._draw_text_centered(130, "Waiting for signal...", 'small')
            return

        # Coordinates
        lat_str = f"{abs(gps['lat']):.5f} {'N' if gps['lat'] >= 0 else 'S'}"
        lon_str = f"{abs(gps['lon']):.5f} {'W' if gps['lon'] < 0 else 'E'}"
        self._draw_text(4, 38, f"{lat_str}  {lon_str}", 'tiny')

        # Speed + heading
        speed = int(gps['speed_mph']) if gps['speed_mph'] is not None else 0
        heading = gps.get('track')
        # `if heading` treated a heading of exactly 0 as missing — due north
        # rendered "-- --" instead of "0 N".
        hdg_str = f"{int(heading)} {self._cardinal(heading)}" if heading is not None else "-- --"
        self._draw_text_right(W - 4, 38, f"{speed} mph  {hdg_str}", 'tiny')

        # Map tile — zoom scales with speed
        result = self._get_osm_tile(gps['lat'], gps['lon'], speed_mph=gps['speed_mph'])
        if result:
            tile_surf, loc_x, loc_y = result
            tile_x = (W - tile_surf.get_width()) // 2
            tile_y = 62
            self.screen.blit(tile_surf, (tile_x, tile_y))

            # Location dot at center
            cx = tile_x + loc_x
            cy = tile_y + loc_y
            pygame.draw.circle(self.screen, BLACK, (cx, cy), 5)
            pygame.draw.circle(self.screen, WHITE, (cx, cy), 5, 1)
            pygame.draw.circle(self.screen, WHITE, (cx, cy), 2)
        else:
            self._draw_text_centered(120, "Loading map...", 'small')

    # ------------------------------------------------------------------
    # Public interface (matches OLEDDisplay)
    # ------------------------------------------------------------------

    def clear(self):
        """Clear the display to black."""
        if self.screen:
            self.screen.fill(BLACK)
            self._flip()

    def show_boot(self):
        """Show boot splash screen for BOOT_DURATION seconds."""
        self.screen.fill(BLACK)
        self._render_boot()
        self._flip()
        time.sleep(self.config.BOOT_DURATION)
        self.clear()

    def _read_display_mode(self) -> int:
        """Read display mode from shared file (written by main.py from rotary switch)."""
        try:
            if os.path.exists(DISPLAY_MODE_FILE):
                with open(DISPLAY_MODE_FILE, 'r') as f:
                    return int(f.read().strip())
        except Exception:
            pass
        return 0  # Default to auto-cycle

    def _render_page_by_name(self, page: str):
        """Render a specific page by name."""
        if page == "main":
            self._render_main()
        elif page == "diagnostics":
            self._render_diagnostics()
        elif page == "weather":
            self._render_weather_page()
        elif page == "navigation":
            self._render_navigation()

    def _render_current_page(self):
        """Render the current page based on page index."""
        pages = self.config.PAGES
        page = pages[self._current_page % len(pages)]
        self._render_page_by_name(page)

    def run_daemon(self):
        """Main daemon loop — watches sentry flag, cycles through display pages."""
        sentry_flash = True
        sentry_toggle = time.time()
        self._page_switch_time = time.time()
        self.running = True
        self._start_network_thread()  # weather/tile HTTP must not run in this loop

        try:
            while self.running:
                # Drain pygame events to prevent OS from thinking we're frozen
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        return

                self.screen.fill(BLACK)
                assistant_state = None

                if os.path.exists(self.config.SENTRY_FLAG_PATH):
                    # Sentry mode — flashing display
                    elapsed = time.time() - sentry_toggle
                    toggle_time = (
                        self.config.SENTRY_FLASH_ON if sentry_flash
                        else self.config.SENTRY_FLASH_OFF
                    )
                    if elapsed >= toggle_time:
                        sentry_flash = not sentry_flash
                        sentry_toggle = time.time()
                    self._render_sentry(sentry_flash)
                else:
                    # Check if assistant is active — override page cycling
                    assistant_state = self._read_assistant_state()
                    if assistant_state:
                        self._render_assistant_overlay(assistant_state)
                    else:
                        # Check rotary switch for locked page or auto-cycle
                        display_mode = self._read_display_mode()
                        if display_mode == 4:
                            # Position 5: auto-cycle
                            if time.time() - self._page_switch_time >= self.config.PAGE_CYCLE_INTERVAL:
                                self._current_page = (self._current_page + 1) % len(self.config.PAGES)
                                self._page_switch_time = time.time()
                            self._render_current_page()
                        else:
                            # Locked to specific page
                            mode_page = {0: "main", 1: "navigation", 2: "weather", 3: "diagnostics"}
                            page = mode_page.get(display_mode, "main")
                            self._render_page_by_name(page)

                self._flip()
                # Faster refresh for sentry camera feed and waveform animation
                if os.path.exists(self.config.SENTRY_FLAG_PATH):
                    time.sleep(0.1)  # ~10fps for camera feed
                elif assistant_state and assistant_state.get('state') == 'speaking':
                    time.sleep(0.04)  # ~25fps for smooth waveform
                else:
                    time.sleep(self.config.DISPLAY_REFRESH)

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.clear()

    def stop(self):
        """Stop the daemon loop (thread-safe)."""
        self.running = False

    def test(self):
        """Display a test pattern for 3 seconds."""
        self.screen.fill(BLACK)
        pygame.draw.rect(
            self.screen, WHITE,
            pygame.Rect(0, 0, self.config.WIDTH - 1, self.config.HEIGHT - 1),
            1
        )
        self._draw_text_centered(100, "CRT OK", 'large')
        self._flip()
        time.sleep(3)
        self.clear()

    def cleanup(self):
        """Shut down pygame."""
        self.clear()
        pygame.quit()


# ------------------------------------------------------------------
# Standalone CLI (same interface as oled.py)
# ------------------------------------------------------------------

def set_sentry_mode(enabled):
    try:
        if enabled:
            with open(Config.SENTRY_FLAG_PATH, "w") as f:
                f.write("on\n")
            print("Sentry mode enabled")
        else:
            if os.path.exists(Config.SENTRY_FLAG_PATH):
                os.remove(Config.SENTRY_FLAG_PATH)
            print("Sentry mode disabled")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="VIVIAN CRT Display v1.0")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--boot", action="store_true", help="Show boot splash")
    group.add_argument("--main", action="store_true", help="Run main display")
    group.add_argument("--sentry", action="store_true", help="Run sentry flash display")
    group.add_argument("--daemon", action="store_true", help="Run as daemon (watches sentry flag)")
    group.add_argument("--boot-daemon", action="store_true", help="Show boot splash then run daemon")
    group.add_argument("--sentry-on", action="store_true", help="Set sentry flag")
    group.add_argument("--sentry-off", action="store_true", help="Clear sentry flag")
    group.add_argument("--clear", action="store_true", help="Clear display")
    group.add_argument("--test", action="store_true", help="Display test pattern")
    args = parser.parse_args()

    if args.sentry_on:
        set_sentry_mode(True)
        return
    if args.sentry_off:
        set_sentry_mode(False)
        return

    crt = CRTDisplay().init_display()

    try:
        if args.boot:
            crt.show_boot()
        elif args.sentry:
            sentry_flash = True
            crt.running = True
            while crt.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        crt.running = False
                crt.screen.fill(BLACK)
                crt._render_sentry(sentry_flash)
                crt._flip()  # was self._flip() — NameError killed --sentry on frame 1
                time.sleep(crt.config.SENTRY_FLASH_ON if sentry_flash else crt.config.SENTRY_FLASH_OFF)
                sentry_flash = not sentry_flash
        elif args.boot_daemon:
            crt.show_boot()
            crt.run_daemon()
        elif args.daemon:
            crt.run_daemon()
        elif args.clear:
            crt.clear()
        elif args.test:
            crt.test()
        else:
            crt.run_daemon()
    except KeyboardInterrupt:
        pass
    finally:
        crt.cleanup()


if __name__ == "__main__":
    main()
