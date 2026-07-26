import logging
import math
import os
import queue
import threading
import time
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import pygame

logger = logging.getLogger(__name__)

# ── Pin definitions (avoids GPIO 17, 18, 27, 22 used by rotary switch) ──────
RST_PIN = 24
DC_PIN = 25
BL_PIN = 12
TP_RST_PIN = 16
TP_IRQ_PIN = 4
CST816_ADDR = 0x15

# ── Colors ───────────────────────────────────────────────────────────────────
SPOTIFY_GREEN = (29, 185, 84)
BG = (17, 17, 17)
SURFACE_CLR = (26, 26, 26)
WHITE = (255, 255, 255)
GRAY = (170, 170, 170)
DIM = (85, 85, 85)
DIVIDER = (30, 30, 30)
BAR_BG = (42, 42, 42)
DOT_INACTIVE = (51, 51, 51)
QUEUE_DOT_OFF = (42, 42, 42)
SUB_TEXT = (102, 102, 102)
SCROLL_THUMB = (68, 68, 68)

THUMB_COLORS = [
    (29, 185, 84), (233, 30, 140), (245, 155, 35),
    (79, 156, 249), (185, 88, 247), (226, 90, 75),
]

# ── Screens ──────────────────────────────────────────────────────────────────
SCREEN_NOW_PLAYING = 0
SCREEN_QUEUE = 1
SCREEN_PLAYLISTS = 2
SCREEN_RECENT = 3
NUM_SCREENS = 4

# ── CST816 gestures ─────────────────────────────────────────────────────────
GESTURE_NONE = 0x00
GESTURE_SWIPE_UP = 0x01
GESTURE_SWIPE_DOWN = 0x02
GESTURE_SWIPE_LEFT = 0x03
GESTURE_SWIPE_RIGHT = 0x04
GESTURE_CLICK = 0x05

# ── Fonts ────────────────────────────────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Panel is 170×320; landscape mode swaps to 320×170 for our coordinate system
PANEL_W = 170
PANEL_H = 320
WIDTH = 320
HEIGHT = 170
X_OFFSET = 35  # ST7789 240px buffer, 170px visible → 35px offset


class SpotifyDisplay:
    """Touch-driven Spotify UI for Waveshare 1.9" LCD (ST7789V2 + CST816)."""

    def __init__(self, config, spotify_controller):
        self.config = config
        self.spotify = spotify_controller
        self._stop_event = threading.Event()

        # Screen state
        self.current_screen = SCREEN_NOW_PLAYING
        self.queue_scroll = 0
        self.playlist_scroll = 0
        self.recent_scroll = 0

        # Cached Spotify data
        self._playback = None
        self._queue_tracks: List[dict] = []
        self._playlists: List[dict] = []
        self._playlists_ts = 0.0
        self._recent_albums: List[dict] = []
        self._recent_ts = 0.0

        # Album art
        self._art_surface: Optional[pygame.Surface] = None
        self._art_url: Optional[str] = None
        self._art_lock = threading.Lock()

        # Thumbnail caches (url → pygame.Surface)
        self._thumb_cache: dict = {}
        self._thumb_pending: set = set()

        # Touch state
        self._touch_active = False
        self._touch_raw_x = 0
        self._touch_raw_y = 0
        self._ripple_x = 0
        self._ripple_y = 0
        self._ripple_time = 0.0
        self._recent_scope_missing = False

        # Hardware handles (set in init_display)
        self.spi = None
        self.bus = None
        self._dc = None
        self._rst = None
        self._tp_rst = None
        self._tp_irq = None

    # ── Public lifecycle ─────────────────────────────────────────────────────

    def init_display(self):
        import spidev
        import smbus2
        from gpiozero import DigitalOutputDevice

        self._dc = DigitalOutputDevice(DC_PIN, active_high=True, initial_value=True)
        self._rst = DigitalOutputDevice(RST_PIN, active_high=True, initial_value=True)

        self.spi = spidev.SpiDev(0, 0)
        self.spi.max_speed_hz = 40_000_000
        self.spi.mode = 0

        self.bus = smbus2.SMBus(1)

        if not pygame.font.get_init():
            pygame.font.init()

        self._f_title = pygame.font.Font(FONT_BOLD_PATH, 12)
        self._f_artist = pygame.font.Font(FONT_PATH, 9)
        self._f_time = pygame.font.Font(FONT_PATH, 7)
        self._f_hdr = pygame.font.Font(FONT_BOLD_PATH, 10)
        self._f_hdr_dim = pygame.font.Font(FONT_BOLD_PATH, 9)
        self._f_track = pygame.font.Font(FONT_PATH, 10)
        self._f_track_bold = pygame.font.Font(FONT_BOLD_PATH, 10)
        self._f_sub = pygame.font.Font(FONT_PATH, 8)
        self._f_name = pygame.font.Font(FONT_BOLD_PATH, 9)
        self._f_count = pygame.font.Font(FONT_PATH, 7)

        self._init_st7789()
        self._init_backlight()
        self._init_touch()
        logger.info("Spotify display initialized")

    def _poll_loop(self):
        """Background thread: refresh Spotify state without blocking the UI.

        The Spotify API calls (playback/queue/playlists) can take 1-3s over
        cellular; running them on the render thread froze the display and
        dropped touches. This runs them off-thread on a 2s cadence.
        """
        while not self._stop_event.is_set():
            try:
                self._poll_spotify()
            except Exception as e:
                logger.debug(f"Spotify poll error: {e}")
            self._stop_event.wait(2.0)

    def run_daemon(self):
        frame_interval = 1.0 / 15

        # Spotify polling runs on its own thread so a slow API call never
        # freezes rendering or eats touch input.
        threading.Thread(target=self._poll_loop, daemon=True).start()

        while not self._stop_event.is_set():
            # Per-iteration guard: a transient draw error must not kill the
            # whole display until restart.
            try:
                t0 = time.monotonic()

                self._poll_touch()

                surface = pygame.Surface((WIDTH, HEIGHT))
                surface.fill(BG)

                if self.current_screen == SCREEN_NOW_PLAYING:
                    self._draw_now_playing(surface)
                elif self.current_screen == SCREEN_QUEUE:
                    self._draw_queue(surface)
                elif self.current_screen == SCREEN_PLAYLISTS:
                    self._draw_playlists(surface)
                elif self.current_screen == SCREEN_RECENT:
                    self._draw_recent(surface)

                self._draw_nav_dots(surface, self.current_screen)
                self._draw_ripple(surface)
                self._flush_frame(surface)

                dt = time.monotonic() - t0
                if dt < frame_interval:
                    self._stop_event.wait(frame_interval - dt)
            except Exception as e:
                logger.error(f"Spotify display frame error: {e}", exc_info=True)
                self._stop_event.wait(0.5)

    def stop(self):
        self._stop_event.set()
        if self.spi:
            try:
                self.spi.close()
            except Exception:
                pass
        if self.bus:
            try:
                self.bus.close()
            except Exception:
                pass

    # ── ST7789V2 (Waveshare init sequence) ───────────────────────────────────

    BL_FLAG = "/tmp/vivian_bl_on"

    def _init_backlight(self):
        import os
        if os.path.exists(self.BL_FLAG):
            logger.info("Backlight already on (flag exists), skipping")
            return
        from gpiozero import PWMOutputDevice
        bl = PWMOutputDevice(BL_PIN, frequency=1000)
        bl.value = 0.8
        time.sleep(0.5)
        bl.close()
        with open(self.BL_FLAG, "w") as f:
            f.write("1")
        logger.info("Backlight triggered and pin released")

    def _init_st7789(self):
        self._rst.on()
        time.sleep(0.01)
        self._rst.off()
        time.sleep(0.01)
        self._rst.on()
        time.sleep(0.01)

        self._command(0x11)       # Sleep out
        time.sleep(0.12)

        self._command(0x36)       # MADCTL
        self._data_byte(0x08)

        self._command(0x3A)       # Pixel format: 16-bit RGB565
        self._data_byte(0x05)

        self._command(0xF0)       # Command Set Control
        self._data_byte(0xC3)
        self._command(0xF0)
        self._data_byte(0x96)

        self._command(0xB4)
        self._data_byte(0x01)
        self._command(0xB7)
        self._data_byte(0xC6)

        self._command(0xC0)
        self._data_byte(0x80)
        self._data_byte(0x45)

        self._command(0xC1)
        self._data_byte(0x13)

        self._command(0xC2)
        self._data_byte(0xA7)

        self._command(0xC5)
        self._data_byte(0x0A)

        self._command(0xF0)
        self._data_byte(0x3C)
        self._command(0xF0)
        self._data_byte(0x69)

        self._command(0x21)       # Inversion on
        self._command(0x11)       # Sleep out (again)
        time.sleep(0.1)
        self._command(0x29)       # Display on

    def _command(self, cmd):
        self._dc.off()
        self.spi.writebytes([cmd])

    def _data_byte(self, val):
        self._dc.on()
        self.spi.writebytes([val])

    def _set_window(self, x0, y0, x1, y1):
        """Set pixel window in landscape orientation with 35px X offset."""
        # Landscape: swap axes and apply offset to the short axis
        # CASET (column) = y in landscape, RASET (row) = x in landscape
        x0o = x0 + X_OFFSET
        x1o = x1 + X_OFFSET
        self._command(0x2B)  # CASET → landscape X (long axis)
        self._data_byte(y0 >> 8)
        self._data_byte(y0 & 0xFF)
        self._data_byte(y1 >> 8)
        self._data_byte(y1 & 0xFF)
        self._command(0x2A)  # RASET → landscape Y (short axis, with offset)
        self._data_byte(x0o >> 8)
        self._data_byte(x0o & 0xFF)
        self._data_byte(x1o >> 8)
        self._data_byte(x1o & 0xFF)
        self._command(0x2C)  # RAMWR

    def _write_pixels(self, data):
        self._dc.on()
        for i in range(0, len(data), 4096):
            self.spi.writebytes2(data[i:i + 4096])

    def _flush_frame(self, surface):
        from PIL import Image as PILImage
        # pygame surface (320×170) → PIL Image → portrait transform
        # surfarray is (width, height, 3), transpose to (height, width, 3)
        arr = pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        img = PILImage.fromarray(arr.astype(np.uint8), 'RGB')
        # Same rotation as working test programs
        port = img.transpose(PILImage.ROTATE_90).transpose(PILImage.FLIP_LEFT_RIGHT)
        # RGB888 → RGB565 big-endian (ST7789 byte order)
        parr = np.array(port, dtype=np.uint16)
        r = (parr[:, :, 0] >> 3).astype(np.uint16)
        g = (parr[:, :, 1] >> 2).astype(np.uint16)
        b = (parr[:, :, 2] >> 3).astype(np.uint16)
        rgb565 = (r << 11) | (g << 5) | b
        # Pack as big-endian bytes: high byte = R5+G3, low byte = G3+B5
        data = np.empty((parr.shape[0], parr.shape[1], 2), dtype=np.uint8)
        data[:, :, 0] = (rgb565 >> 8).astype(np.uint8)
        data[:, :, 1] = (rgb565 & 0xFF).astype(np.uint8)
        data = data.tobytes()

        x0 = X_OFFSET
        x1 = X_OFFSET + PANEL_W - 1
        self._command(0x2A)
        self._data_byte(x0 >> 8)
        self._data_byte(x0 & 0xFF)
        self._data_byte(x1 >> 8)
        self._data_byte(x1 & 0xFF)
        self._command(0x2B)
        self._data_byte(0)
        self._data_byte(0)
        self._data_byte((PANEL_H - 1) >> 8)
        self._data_byte((PANEL_H - 1) & 0xFF)
        self._command(0x2C)

        try:
            self._write_pixels(data)
        except Exception:
            time.sleep(0.1)
            try:
                self._write_pixels(data)
            except Exception:
                pass

    # ── CST816 touch ─────────────────────────────────────────────────────────

    def _init_touch(self):
        from gpiozero import DigitalOutputDevice
        self._tp_rst = DigitalOutputDevice(TP_RST_PIN, active_high=True, initial_value=False)
        for attempt in range(5):
            self._tp_rst.off()
            time.sleep(0.05)
            self._tp_rst.on()
            time.sleep(0.5)
            try:
                self.bus.read_byte(CST816_ADDR)
                logger.info("CST816 touch controller detected on I2C")
                return
            except Exception as e:
                logger.warning(f"CST816 init attempt {attempt + 1}/5 failed: {e}")
                time.sleep(1.0)
        logger.error("CST816 touch controller not detected after 5 attempts")

    def _poll_touch(self):
        """Poll CST816 for touch events (called from render loop).

        Process on finger-lift so the CST816 has time to determine
        whether it was a tap or swipe gesture.
        """
        try:
            finger_count = self.bus.read_i2c_block_data(CST816_ADDR, 0x02, 1)[0]

            if finger_count > 0:
                if not self._touch_active:
                    self._touch_active = True
                # Continuously update coordinates while finger is down
                buf = self.bus.read_i2c_block_data(CST816_ADDR, 0x03, 4)
                self._touch_raw_x = ((buf[0] & 0x0F) << 8) | buf[1]
                self._touch_raw_y = ((buf[2] & 0x0F) << 8) | buf[3]
                return

            if not self._touch_active:
                return

            # Finger just lifted — read gesture and process
            self._touch_active = False
            buf = self.bus.read_i2c_block_data(CST816_ADDR, 0x01, 1)
            gesture = buf[0]
            screen_x = int(-0.0739 * self._touch_raw_x + -0.9312 * self._touch_raw_y + 315.9725)
            screen_y = int(1.0562 * self._touch_raw_x + -0.0510 * self._touch_raw_y + 14.8656)
            screen_x = max(0, min(WIDTH - 1, screen_x))
            screen_y = max(0, min(HEIGHT - 1, screen_y))
            self._ripple_x = screen_x
            self._ripple_y = screen_y
            self._ripple_time = time.monotonic()
            self._process_touch(gesture, screen_x, screen_y, 1)
        except Exception:
            pass

    # ── Touch processing ─────────────────────────────────────────────────────

    def _process_touch(self, gesture, x, y, finger_count):
        # CST816 reports gestures in portrait — rotate 90° for landscape:
        # portrait up=left, down=right, left=down, right=up
        if gesture == GESTURE_SWIPE_UP:
            self.current_screen = (self.current_screen - 1) % NUM_SCREENS
        elif gesture == GESTURE_SWIPE_DOWN:
            self.current_screen = (self.current_screen + 1) % NUM_SCREENS
        elif gesture == GESTURE_SWIPE_LEFT:
            self._scroll(1)
        elif gesture == GESTURE_SWIPE_RIGHT:
            self._scroll(-1)
        elif gesture in (GESTURE_NONE, GESTURE_CLICK) and finger_count == 1:
            self._handle_tap(x, y)

    def _scroll(self, direction):
        step = 2
        if self.current_screen == SCREEN_QUEUE:
            mx = max(0, len(self._queue_tracks) - 2)
            self.queue_scroll = max(0, min(self.queue_scroll + direction * step, mx))
        elif self.current_screen == SCREEN_PLAYLISTS:
            mx = max(0, len(self._playlists) - 2)
            self.playlist_scroll = max(0, min(self.playlist_scroll + direction * step, mx))
        elif self.current_screen == SCREEN_RECENT:
            mx = max(0, len(self._recent_albums) - 2)
            self.recent_scroll = max(0, min(self.recent_scroll + direction * step, mx))

    def _handle_tap(self, x, y):
        s = self.current_screen
        if s == SCREEN_NOW_PLAYING:
            self._tap_now_playing(x, y)
        elif s == SCREEN_QUEUE:
            self._tap_queue(x, y)
        elif s == SCREEN_PLAYLISTS:
            self._tap_list(x, y, self._playlists, self.playlist_scroll, 26, 66,
                           self._on_playlist_tap)
        elif s == SCREEN_RECENT:
            self._tap_list(x, y, self._recent_albums, self.recent_scroll, 26, 66,
                           self._on_recent_tap)

    def _tap_now_playing(self, x, y):
        # Shuffle button (bottom-right area)
        if x > 255 and y > 120:
            threading.Thread(target=self.spotify.toggle_shuffle, daemon=True).start()
            return
        if x < 118 or y < 34 or y > 114:
            return
        btn_w = 62
        gap = 4
        rx = 118
        rel = x - rx
        idx = int(rel / (btn_w + gap))
        if idx > 2:
            idx = 2

        if idx == 0:
            threading.Thread(target=self.spotify.previous, daemon=True).start()
        elif idx == 1:
            if self._playback and self._playback.get('is_playing'):
                threading.Thread(target=self.spotify.pause, daemon=True).start()
            else:
                threading.Thread(target=self.spotify.resume, daemon=True).start()
        elif idx == 2:
            threading.Thread(target=self.spotify.skip, daemon=True).start()

    def _tap_queue(self, x, y):
        # >= 158, not > 158: only two rows are drawn (y 26-157), and y == 158
        # computes row 2 — an undrawn entry, so an exact-158 tap played a track
        # the user could not see.
        if y < 26 or y >= 158:
            return
        row = (y - 26) // 66
        idx = self.queue_scroll + row
        if idx >= len(self._queue_tracks):
            return
        track = self._queue_tracks[idx]
        threading.Thread(target=self._play_queue_track, args=(track,),
                         daemon=True).start()

    def _tap_list(self, x, y, items, scroll, header_h, row_h, callback, vis=2):
        # Bound the tap to the rows actually drawn (vis rows, same as
        # _draw_playlists/_draw_recent). Touch y is clamped to HEIGHT-1, so
        # without an upper bound a tap on the bottom nav-dot strip resolved to
        # row 2 and started an off-screen playlist/album mid-drive.
        if y < header_h or y >= header_h + vis * row_h:
            return
        row = (y - header_h) // row_h + scroll
        if row < len(items):
            callback(items[row])

    def _on_playlist_tap(self, pl):
        uri = pl.get('uri', '')
        if uri:
            threading.Thread(target=self.spotify.play_playlist, args=(uri,),
                             daemon=True).start()

    def _on_recent_tap(self, album):
        uri = album.get('uri', '')
        if uri:
            threading.Thread(target=self.spotify.play_album_at_position,
                             args=(uri, 0), daemon=True).start()

    def _play_queue_track(self, track):
        """Skip forward through the queue to reach the tapped track."""
        target_uri = track.get('uri', '')
        if not target_uri:
            return
        try:
            sp = self.spotify.get_client()
            device_id = self.spotify.get_device_id()
            if not device_id:
                return
            # Skip until we reach the target track (max 20 to avoid runaway)
            for _ in range(20):
                sp.next_track(device_id=device_id)
                time.sleep(0.25)
                current = sp.current_playback()
                if current and current.get('item', {}).get('uri') == target_uri:
                    break
            self.spotify.save_state(True)
        except Exception as e:
            logger.error(f"Queue track skip failed: {e}")

    # ── Spotify polling ──────────────────────────────────────────────────────

    def _poll_spotify(self):
        try:
            sp = self.spotify.get_client()

            # Current playback
            raw = sp.current_playback()
            if raw and raw.get('item'):
                item = raw['item']
                images = item.get('album', {}).get('images', [])
                art_url = images[0]['url'] if images else None
                self._playback = {
                    'is_playing': raw.get('is_playing', False),
                    'track_name': item.get('name', 'Unknown'),
                    'artists': ', '.join(a['name'] for a in item.get('artists', [])),
                    'album': item.get('album', {}).get('name', ''),
                    'album_uri': item.get('album', {}).get('uri', ''),
                    'duration_ms': item.get('duration_ms', 0),
                    'progress_ms': raw.get('progress_ms', 0),
                    'shuffle': raw.get('shuffle_state', False),
                    'art_url': art_url,
                }
                if art_url and art_url != self._art_url:
                    threading.Thread(target=self._fetch_album_art,
                                     args=(art_url,), daemon=True).start()
            else:
                self._playback = None

            # Queue
            try:
                q = sp.queue()
                tracks = []
                cp = q.get('currently_playing')
                if cp:
                    tracks.append(self._parse_queue_track(cp, True))
                for t in q.get('queue', []):
                    tracks.append(self._parse_queue_track(t, False))
                self._queue_tracks = tracks
            except Exception:
                pass

            now = time.time()

            # Playlists (refresh every 60s)
            if now - self._playlists_ts > 60:
                try:
                    res = sp.current_user_playlists(limit=50)
                    self._playlists = res.get('items', [])
                    self._playlists_ts = now
                    self._prefetch_thumbnails(
                        [img[0]['url'] for p in self._playlists
                         if (img := p.get('images')) and img])
                except Exception:
                    pass

            # Recently played albums (refresh every 60s, skip if scope missing)
            if now - self._recent_ts > 60 and not self._recent_scope_missing:
                try:
                    res = sp.current_user_recently_played(limit=50)
                    seen = set()
                    albums = []
                    for entry in res.get('items', []):
                        trk = entry.get('track', {})
                        alb = trk.get('album', {})
                        uri = alb.get('uri', '')
                        if uri and uri not in seen:
                            seen.add(uri)
                            imgs = alb.get('images', [])
                            albums.append({
                                'name': alb.get('name', ''),
                                'artist': ', '.join(
                                    a['name'] for a in alb.get('artists',
                                                               trk.get('artists', []))),
                                'uri': uri,
                                'image_url': imgs[-1]['url'] if imgs else None,
                            })
                    self._recent_albums = albums
                    self._recent_ts = now
                    self._prefetch_thumbnails(
                        [a['image_url'] for a in albums if a['image_url']])
                except Exception as e:
                    if '403' in str(e):
                        self._recent_scope_missing = True
                        logger.warning("Recently played scope missing — disabling")

        except Exception as e:
            logger.error(f"Spotify poll error: {e}")

    @staticmethod
    def _parse_queue_track(t, is_current):
        return {
            'name': t.get('name', ''),
            'artist': ', '.join(a['name'] for a in t.get('artists', [])),
            'duration_ms': t.get('duration_ms', 0),
            'uri': t.get('uri', ''),
            'album_uri': t.get('album', {}).get('uri', ''),
            'is_current': is_current,
        }

    # ── Image fetching ───────────────────────────────────────────────────────

    def _fetch_album_art(self, url):
        try:
            import requests
            from PIL import Image
            resp = requests.get(url, timeout=2)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert('RGB')
            img = img.resize((110, HEIGHT), Image.LANCZOS)
            surf = pygame.image.fromstring(img.tobytes(), (110, HEIGHT), 'RGB')
            with self._art_lock:
                self._art_surface = surf
                self._art_url = url
        except Exception as e:
            logger.debug(f"Album art fetch failed: {e}")

    def _prefetch_thumbnails(self, urls):
        for url in urls:
            if url not in self._thumb_cache and url not in self._thumb_pending:
                self._thumb_pending.add(url)
                threading.Thread(target=self._fetch_thumb, args=(url,),
                                 daemon=True).start()

    def _fetch_thumb(self, url):
        try:
            import requests
            from PIL import Image
            resp = requests.get(url, timeout=2)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert('RGB')
            img = img.resize((28, 28), Image.LANCZOS)
            self._thumb_cache[url] = pygame.image.fromstring(
                img.tobytes(), (28, 28), 'RGB')
        except Exception:
            pass
        finally:
            self._thumb_pending.discard(url)

    # ── Drawing helpers ──────────────────────────────────────────────────────

    def _trunc(self, font, text, max_w):
        if font.size(text)[0] <= max_w:
            return text
        while len(text) > 0 and font.size(text + '…')[0] > max_w:
            text = text[:-1]
        return text + '…'

    @staticmethod
    def _fmt_time(ms):
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"

    def _draw_nav_dots(self, surface, active):
        dot_r = 2
        gap = 5
        total_w = NUM_SCREENS * 5 + (NUM_SCREENS - 1) * gap
        sx = (WIDTH - total_w) // 2
        cy = 161
        for i in range(NUM_SCREENS):
            color = SPOTIFY_GREEN if i == active else DOT_INACTIVE
            cx = sx + i * (5 + gap) + dot_r
            pygame.draw.circle(surface, color, (cx, cy), dot_r)

    def _draw_ripple(self, surface):
        elapsed = time.monotonic() - self._ripple_time
        if elapsed > 0.4:
            return
        # Expanding circle that fades out
        progress = elapsed / 0.4
        radius = int(10 + 15 * progress)
        alpha = int(180 * (1.0 - progress))
        ripple = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(ripple, (29, 185, 84, alpha), (radius, radius), radius, 2)
        surface.blit(ripple, (self._ripple_x - radius, self._ripple_y - radius))

    def _draw_scroll_indicator(self, surface, offset, total, visible, y0, y1):
        track_h = y1 - y0
        ix = WIDTH - 3
        pygame.draw.rect(surface, DIVIDER, (ix, y0, 3, track_h))
        thumb_h = max(8, int(track_h * visible / total))
        mx = total - visible
        ty = y0 + (int((track_h - thumb_h) * offset / mx) if mx > 0 else 0)
        pygame.draw.rect(surface, SCROLL_THUMB, (ix, ty, 3, thumb_h))

    # ── Screen 1: Now Playing ────────────────────────────────────────────────

    def _draw_now_playing(self, surface):
        pb = self._playback

        # Left panel — album art
        with self._art_lock:
            art = self._art_surface if pb else None
        if art:
            surface.blit(art, (0, 0))
        else:
            self._draw_vinyl_placeholder(surface)

        # Right panel
        rx, rw = 118, 194

        if pb:
            title = pb['track_name']
            sub = f"{pb['artists']} · {pb['album']}"
            active = True
        else:
            title, sub, active = "No playback", "Open Spotify on your phone", False

        # Title
        ts = self._f_title.render(self._trunc(self._f_title, title, rw),
                                  True, WHITE if active else DIM)
        surface.blit(ts, (rx, 4))

        # Artist · Album
        asf = self._f_artist.render(self._trunc(self._f_artist, sub, rw),
                                    True, GRAY if active else DIM)
        surface.blit(asf, (rx, 20))

        # Transport buttons (3 equal, gap 4, 80px tall)
        btn_gap = 4
        btn_w = (rw - 2 * btn_gap) // 3
        btn_y, btn_h = 34, 80
        is_playing = pb and pb.get('is_playing')

        for i in range(3):
            bx = rx + i * (btn_w + btn_gap)
            if i == 1 and active:
                bg = SPOTIFY_GREEN
            else:
                bg = SURFACE_CLR
            pygame.draw.rect(surface, bg, (bx, btn_y, btn_w, btn_h),
                             border_radius=8)
            cx = bx + btn_w // 2
            cy = btn_y + btn_h // 2

            if i == 0:
                ic = (191, 191, 191) if active else DIM
                pygame.draw.rect(surface, ic, (cx - 10, cy - 8, 3, 16))
                pygame.draw.polygon(surface, ic,
                                    [(cx + 8, cy - 8), (cx + 8, cy + 8),
                                     (cx - 5, cy)])
            elif i == 1:
                ic = (0, 0, 0) if active else DIM
                if is_playing:
                    pygame.draw.rect(surface, ic, (cx - 7, cy - 8, 5, 16))
                    pygame.draw.rect(surface, ic, (cx + 2, cy - 8, 5, 16))
                else:
                    pygame.draw.polygon(surface, ic,
                                        [(cx - 5, cy - 8), (cx - 5, cy + 8),
                                         (cx + 8, cy)])
            else:
                ic = (191, 191, 191) if active else DIM
                pygame.draw.polygon(surface, ic,
                                    [(cx - 8, cy - 8), (cx - 8, cy + 8),
                                     (cx + 5, cy)])
                pygame.draw.rect(surface, ic, (cx + 7, cy - 8, 3, 16))

        # Progress bar (no timestamps)
        py = btn_y + btn_h + 6
        pygame.draw.rect(surface, BAR_BG, (rx, py, rw, 3), border_radius=2)
        if pb and pb['duration_ms'] > 0:
            fw = int(rw * pb['progress_ms'] / pb['duration_ms'])
            if fw > 0:
                pygame.draw.rect(surface, SPOTIFY_GREEN, (rx, py, fw, 3),
                                 border_radius=2)

        # Shuffle button (bottom-right corner)
        shuf_on = pb and pb.get('shuffle', False)
        shuf_bg = (29, 60, 40) if shuf_on else SURFACE_CLR
        shuf_color = SPOTIFY_GREEN if shuf_on else DIM
        shuf_rect = (rx + rw - 55, py + 6, 55, 36)
        pygame.draw.rect(surface, shuf_bg, shuf_rect, border_radius=6)
        sx = shuf_rect[0] + shuf_rect[2] // 2 - 12
        sy = shuf_rect[1] + shuf_rect[3] // 2 - 6
        pygame.draw.line(surface, shuf_color, (sx, sy + 12), (sx + 24, sy), 2)
        pygame.draw.line(surface, shuf_color, (sx, sy), (sx + 24, sy + 12), 2)
        pygame.draw.polygon(surface, shuf_color,
                            [(sx + 24, sy), (sx + 19, sy - 3), (sx + 19, sy + 3)])
        pygame.draw.polygon(surface, shuf_color,
                            [(sx + 24, sy + 12), (sx + 19, sy + 9), (sx + 19, sy + 15)])

    def _draw_vinyl_placeholder(self, surface):
        for y in range(HEIGHT):
            t = y / HEIGHT
            r = int(26 * (1 - t) + 10 * t)
            g = int(58 * (1 - t) + 21 * t)
            b = int(39 * (1 - t) + 16 * t)
            pygame.draw.line(surface, (r, g, b), (0, y), (109, y))
        cx, cy = 55, 85
        pygame.draw.circle(surface, (13, 13, 13), (cx, cy), 27)
        pygame.draw.circle(surface, SPOTIFY_GREEN, (cx, cy), 27, 7)
        pygame.draw.circle(surface, (15, 32, 24), (cx, cy), 5)

    # ── Screen 2: Queue ──────────────────────────────────────────────────────

    def _draw_queue(self, surface):
        hdr_h = 26
        pygame.draw.line(surface, (34, 34, 34), (0, hdr_h - 1), (WIDTH, hdr_h - 1))

        for yo in (6, 12, 18):
            pygame.draw.rect(surface, (128, 128, 128), (10, yo, 14, 2))

        hs = self._f_hdr.render("Up next", True, WHITE)
        surface.blit(hs, (30, 6))

        if not self._queue_tracks:
            m = self._f_artist.render("No tracks in queue", True, DIM)
            surface.blit(m, ((WIDTH - m.get_width()) // 2, 80))
            return

        vis = 2
        row_h = 66
        total = len(self._queue_tracks)

        for i in range(vis):
            ti = self.queue_scroll + i
            if ti >= total:
                break
            t = self._queue_tracks[ti]
            cur = t['is_current']
            ry = hdr_h + i * row_h

            if cur:
                hl = pygame.Surface((WIDTH, row_h), pygame.SRCALPHA)
                hl.fill((29, 185, 84, 25))
                surface.blit(hl, (0, ry))

            dc = SPOTIFY_GREEN if cur else QUEUE_DOT_OFF
            pygame.draw.circle(surface, dc, (14, ry + row_h // 2), 5)

            text_x = 28
            mw = WIDTH - 68
            nc = SPOTIFY_GREEN if cur else WHITE
            nf = self._f_title if cur else self._f_title
            ns = nf.render(self._trunc(nf, t['name'], mw), True, nc)
            surface.blit(ns, (text_x, ry + 12))

            af = self._f_track.render(self._trunc(self._f_track, t['artist'], mw - 50),
                                      True, SUB_TEXT)
            surface.blit(af, (text_x, ry + 34))

            ds = self._f_track.render(self._fmt_time(t['duration_ms']), True, DIM)
            surface.blit(ds, (300 - ds.get_width(), ry + row_h // 2 -
                              ds.get_height() // 2))

            pygame.draw.line(surface, DIVIDER, (0, ry + row_h - 1),
                             (WIDTH, ry + row_h - 1))

        if total > vis:
            self._draw_scroll_indicator(surface, self.queue_scroll, total, vis,
                                        hdr_h, hdr_h + vis * row_h)

    # ── Screen 3: Playlists (2-col grid) ─────────────────────────────────────

    def _draw_playlists(self, surface):
        hdr_h = 26
        row_h = 66
        vis = 2
        pygame.draw.line(surface, (34, 34, 34), (0, hdr_h - 1), (WIDTH, hdr_h - 1))
        hs = self._f_hdr_dim.render("YOUR PLAYLISTS", True, DIM)
        surface.blit(hs, (10, 6))

        if not self._playlists:
            m = self._f_artist.render("Could not load playlists", True, DIM)
            surface.blit(m, ((WIDTH - m.get_width()) // 2, 80))
            return

        total = len(self._playlists)
        for i in range(vis):
            idx = self.playlist_scroll + i
            if idx >= total:
                break
            pl = self._playlists[idx]
            y = hdr_h + i * row_h

            tx, ty = 10, y + (row_h - 40) // 2
            imgs = pl.get('images', [])
            url = imgs[0]['url'] if imgs else None
            thumb = self._thumb_cache.get(url) if url else None
            if thumb:
                surface.blit(thumb, (tx, ty))
            else:
                c = THUMB_COLORS[idx % len(THUMB_COLORS)]
                pygame.draw.rect(surface, c, (tx, ty, 40, 40), border_radius=5)

            text_x = 58
            mw = WIDTH - 68
            ns = self._f_title.render(self._trunc(self._f_title, pl.get('name', ''), mw),
                                      True, WHITE)
            surface.blit(ns, (text_x, y + 12))
            cnt = pl.get('tracks', {}).get('total', 0)
            cs = self._f_track.render(f"{cnt} songs", True, SUB_TEXT)
            surface.blit(cs, (text_x, y + 34))

            pygame.draw.line(surface, DIVIDER, (0, y + row_h - 1),
                             (WIDTH, y + row_h - 1))

        if total > vis:
            self._draw_scroll_indicator(surface, self.playlist_scroll, total, vis,
                                        hdr_h, hdr_h + vis * row_h)

    # ── Screen 4: Recently Played ────────────────────────────────────────────

    def _draw_recent(self, surface):
        hdr_h = 26
        row_h = 66
        vis = 2
        pygame.draw.line(surface, (34, 34, 34), (0, hdr_h - 1), (WIDTH, hdr_h - 1))
        hs = self._f_hdr_dim.render("RECENTLY PLAYED", True, DIM)
        surface.blit(hs, (10, 6))

        if not self._recent_albums:
            m = self._f_artist.render("No recent albums", True, DIM)
            surface.blit(m, ((WIDTH - m.get_width()) // 2, 80))
            return

        total = len(self._recent_albums)
        for i in range(vis):
            idx = self.recent_scroll + i
            if idx >= total:
                break
            album = self._recent_albums[idx]
            y = hdr_h + i * row_h

            tx, ty = 10, y + (row_h - 40) // 2
            url = album.get('image_url')
            thumb = self._thumb_cache.get(url) if url else None
            if thumb:
                surface.blit(thumb, (tx, ty))
            else:
                c = THUMB_COLORS[idx % len(THUMB_COLORS)]
                pygame.draw.rect(surface, c, (tx, ty, 40, 40), border_radius=5)

            text_x = 58
            mw = WIDTH - 68
            ns = self._f_title.render(self._trunc(self._f_title, album.get('name', ''), mw),
                                      True, WHITE)
            surface.blit(ns, (text_x, y + 12))
            ar = self._f_track.render(
                self._trunc(self._f_track, album.get('artist', ''), mw), True, SUB_TEXT)
            surface.blit(ar, (text_x, y + 34))

            pygame.draw.line(surface, DIVIDER, (0, y + row_h - 1),
                             (WIDTH, y + row_h - 1))

        if total > vis:
            self._draw_scroll_indicator(surface, self.recent_scroll, total, vis,
                                        hdr_h, hdr_h + vis * row_h)
