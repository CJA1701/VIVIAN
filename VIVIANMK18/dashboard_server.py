"""VIVIAN Remote Dashboard Server

Password-protected web dashboard serving live vehicle state, GPS location,
system diagnostics, and music info over SSE. Designed to sit behind a
Cloudflare Tunnel for secure remote access from any browser.

Usage:
    Integrated via main.py when dashboard.enabled is true in config.yaml.
    Can also be run standalone for testing:
        python dashboard_server.py
"""

import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from functools import wraps
from typing import Any, Dict, Optional

import bcrypt
import psutil
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for

logger = logging.getLogger(__name__)

# Shared data files (same paths used by crt_display.py, glass_server.py)
GPS_SHARED_FILE = "/tmp/vivian_gps.json"
SPOTIFY_SHARED_FILE = "/tmp/vivian_spotify.json"
ASSISTANT_SHARED_FILE = "/tmp/vivian_assistant.json"
SENTRY_FLAG_PATH = "/tmp/vivian_sentry_enabled"
SENTRY_REQUEST_PATH = "/tmp/vivian_sentry_request"
SENTRY_FRAME_PATH = "/tmp/vivian_sentry_frame.jpg"
SENTRY_STATUS_PATH = "/tmp/vivian_sentry_status.json"

# Silent camera preview (when sentry is not active)
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_ROTATE_180 = True
CAMERA_IDLE_RELEASE = 5.0  # release camera after N seconds of no requests


class _SilentCameraPreview:
    """On-demand camera capture for silent preview from the dashboard.

    One dedicated thread owns the V4L2 device and publishes the newest JPEG
    bytes; HTTP requests only read that attribute and never touch the device.
    get_frame() used to open and read() the camera while holding a lock, and
    neither call has a timeout — one stalled USB camera wedged the lock, so
    dashboard.html's 200ms polling parked ~5 Flask threads per second inside
    VIVIAN until thread creation failed process-wide.

    Opens the V4L2 device only while clients are actively requesting frames.
    Auto-releases after CAMERA_IDLE_RELEASE seconds of inactivity, and yields
    cleanly to sentry mode when it activates (sentry owns the camera).
    """

    def __init__(self):
        self._cap = None            # only the capture thread touches this
        self._latest_jpeg = None    # published frame — the only thing routes read
        self._last_request = 0.0
        self._cv2_missing = False
        self._thread = None
        # Start-once guard: the old check-then-set on a bare bool let two
        # concurrent first requests both call start() -> RuntimeError
        # ("threads can only be started once").
        self._thread_lock = threading.Lock()
        self._stop = threading.Event()

    def _ensure_thread(self):
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._capture_loop, daemon=True,
                                            name="dashboard-cam")
            self._thread.start()

    def _release_cap(self):
        """Close the device. Capture thread only (or release() once it is gone)."""
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _capture_loop(self):
        """Sole owner of the camera. Publishes JPEG bytes to _latest_jpeg."""
        # Defer cv2 import so dashboard can run without it installed
        try:
            import cv2
        except ImportError:
            self._cv2_missing = True  # so get_frame() stops respawning this thread
            return

        try:
            while not self._stop.is_set():
                try:
                    idle = time.time() - self._last_request
                    if idle > CAMERA_IDLE_RELEASE or os.path.exists(SENTRY_FLAG_PATH):
                        # Nobody is watching, or sentry owns the camera now.
                        self._release_cap()
                        self._latest_jpeg = None
                        if idle > CAMERA_IDLE_RELEASE:
                            return  # the next get_frame() restarts us
                        time.sleep(0.5)
                        continue

                    if self._cap is None:
                        cap = cv2.VideoCapture(CAMERA_INDEX)
                        if not cap.isOpened():
                            try:
                                cap.release()
                            except Exception:
                                pass
                            time.sleep(1.0)  # don't hammer a missing device
                            continue
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self._cap = cap

                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        self._release_cap()  # reopen on the next pass
                        self._latest_jpeg = None
                        time.sleep(0.5)
                        continue
                    if CAMERA_ROTATE_180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        self._latest_jpeg = buf.tobytes()  # atomic rebind; no lock needed
                    time.sleep(0.1)  # ~10fps so every 200ms poll gets a fresh frame
                except Exception:
                    # Never die silently: the preview would stay dark until restart.
                    self._release_cap()
                    time.sleep(1.0)
        finally:
            self._release_cap()

    def release(self):
        """Stop the capture thread and free the device (bounded wait)."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)  # bounded: a wedged V4L2 read must not hang shutdown
        if thread is None or not thread.is_alive():
            self._release_cap()
        self._latest_jpeg = None

    def get_frame(self) -> Optional[bytes]:
        """Return the most recently published JPEG, or None if unavailable.
        Pure memory read — never holds a lock across a camera call, so a stalled
        USB camera can only make this return None (route answers 503)."""
        # If sentry is active, do not touch the camera — sentry owns it.
        # The route layer reads SENTRY_FRAME_PATH instead.
        if os.path.exists(SENTRY_FLAG_PATH) or self._cv2_missing:
            return None

        self._last_request = time.time()
        self._ensure_thread()
        return self._latest_jpeg

WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_CACHE_TTL = 120  # seconds


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
    """Read GPS state from shared file."""
    data = _read_json(GPS_SHARED_FILE, max_age=5.0)
    if data is None:
        return {"speed_mph": None, "track": None, "has_fix": False,
                "sats_used": 0, "lat": None, "lon": None, "alt": None}
    return {
        "speed_mph": round(data["speed_mph"], 1) if data.get("speed_mph") is not None else None,
        "track": round(data["track"], 0) if data.get("track") is not None else None,
        "has_fix": data.get("has_fix", False),
        "sats_used": data.get("sats_used", 0),
        "lat": round(data["lat"], 5) if data.get("lat") is not None else None,
        "lon": round(data["lon"], 5) if data.get("lon") is not None else None,
        "alt": round(data["alt"], 0) if data.get("alt") is not None else None,
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


def _read_sentry_status() -> Optional[dict]:
    """Read the last sentry event/threat, or None if not present."""
    try:
        if not os.path.exists(SENTRY_STATUS_PATH):
            return None
        with open(SENTRY_STATUS_PATH) as f:
            data = json.load(f)
        return {
            "event": data.get("event", ""),
            "level": data.get("level"),
            "description": data.get("description", ""),
            "time_str": data.get("time_str", ""),
        }
    except Exception:
        return None


def _get_diagnostics() -> dict:
    """Gather system diagnostics via psutil."""
    try:
        # CPU temperature
        cpu_temp = None
        temps = psutil.sensors_temperatures()
        if temps:
            for name in ('cpu_thermal', 'cpu-thermal', 'coretemp'):
                if name in temps and temps[name]:
                    cpu_temp = temps[name][0].current
                    break
        if cpu_temp is None:
            # Fallback for Pi
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = int(f.read().strip()) / 1000.0
            except Exception:
                pass

        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        boot = psutil.boot_time()
        load = os.getloadavg()

        return {
            "cpu_temp_c": round(cpu_temp, 1) if cpu_temp is not None else None,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": round(mem.used / (1024 * 1024)),
            "ram_total_mb": round(mem.total / (1024 * 1024)),
            "ram_percent": mem.percent,
            "disk_used_gb": round(disk.used / (1024 ** 3), 1),
            "disk_total_gb": round(disk.total / (1024 ** 3), 1),
            "disk_percent": disk.percent,
            "uptime_seconds": int(time.time() - boot),
            "load_1m": round(load[0], 2),
        }
    except Exception as e:
        logger.error(f"Failed to get diagnostics: {e}")
        return {}


class DashboardServer:
    """Flask-based remote dashboard with SSE state streaming."""

    def __init__(self, dashboard_config: dict, weather_config: dict = None,
                 sentry_controller=None):
        self._config = dashboard_config
        self._weather_config = weather_config or {}
        self._sentry_controller = sentry_controller
        self._silent_cam = _SilentCameraPreview()
        self._port = dashboard_config.get('port', 9200)
        self._username = dashboard_config.get('username', '')
        self._password_hash = dashboard_config.get('password_hash', '').encode()
        self._secret_key = dashboard_config.get('secret_key', '')
        if not self._secret_key:
            # Random per-boot key beats a well-known fallback string:
            # sessions reset on restart, but they can't be forged.
            import secrets as _secrets
            self._secret_key = _secrets.token_hex(32)
            logger.warning("dashboard.secret_key not set — using a random "
                           "per-boot key (sessions won't survive restarts)")
        self._thread = None
        self._shutdown = threading.Event()

        # Weather cache
        self._weather_cache = None
        self._weather_ts = 0

        # Login rate limiting: {ip: [timestamps]}
        self._login_attempts: Dict[str, list] = {}

        # Build Flask app
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        self._app = Flask(__name__, template_folder=template_dir)
        self._app.secret_key = self._secret_key
        self._app.config['SESSION_COOKIE_HTTPONLY'] = True
        self._app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
        # Enable when access is exclusively over HTTPS (Cloudflare Tunnel).
        # Leave false if you also log in over plain http:// on the local AP.
        if dashboard_config.get('secure_cookie', False):
            self._app.config['SESSION_COOKIE_SECURE'] = True
        self._register_routes()

    def _register_routes(self):
        app = self._app

        def login_required(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                if not session.get('authenticated'):
                    return redirect(url_for('login_page'))
                return f(*args, **kwargs)
            return decorated

        @app.route('/')
        def login_page():
            if session.get('authenticated'):
                return redirect(url_for('dashboard'))
            return render_template('login.html', error=None)

        @app.route('/login', methods=['POST'])
        def login():
            # Behind Cloudflare Tunnel every request has the tunnel's local
            # address — use the real client IP header when present so the
            # rate limit is per-attacker, not one shared global bucket.
            ip = request.headers.get('CF-Connecting-IP', request.remote_addr)
            now = time.time()

            # Rate limiting: 5 attempts per minute per IP
            attempts = self._login_attempts.get(ip, [])
            attempts = [t for t in attempts if now - t < 60]
            if len(attempts) >= 5:
                return render_template('login.html', error="Too many attempts. Try again in a minute.")

            username = request.form.get('username', '')
            password = request.form.get('password', '')
            if (username == self._username and
                    self._password_hash and
                    bcrypt.checkpw(password.encode(), self._password_hash)):
                session['authenticated'] = True
                self._login_attempts.pop(ip, None)
                return redirect(url_for('dashboard'))

            attempts.append(now)
            self._login_attempts[ip] = attempts
            return render_template('login.html', error="Invalid password.")

        @app.route('/logout')
        def logout():
            session.clear()
            return redirect(url_for('login_page'))

        @app.route('/dashboard')
        @login_required
        def dashboard():
            return render_template('dashboard.html')

        @app.route('/api/sentry/toggle', methods=['POST'])
        @login_required
        def sentry_toggle():
            if os.path.exists(SENTRY_FLAG_PATH):
                # Active — deactivate via controller
                deactivated = False
                if self._sentry_controller is not None:
                    try:
                        deactivated = self._sentry_controller.deactivate()
                    except Exception as e:
                        return jsonify({"ok": False, "error": str(e)}), 500
                if not deactivated:
                    # Flag exists but no sentry is running — stale leftover
                    # from a crash. Clear it so the UI isn't stuck "active".
                    logger.warning("Sentry flag was stale (no live run) — clearing")
                    try:
                        os.remove(SENTRY_FLAG_PATH)
                    except OSError:
                        pass
                return jsonify({"ok": True, "active": False})
            else:
                # Inactive — request activation via flag file
                try:
                    with open(SENTRY_REQUEST_PATH, 'w') as f:
                        f.write('dashboard\n')
                except Exception as e:
                    return jsonify({"ok": False, "error": str(e)}), 500
                return jsonify({"ok": True, "active": True, "pending": True})

        @app.route('/sentry/frame.jpg')
        @login_required
        def sentry_frame():
            # Sentry active → use the frame file it writes (no double-open)
            if os.path.exists(SENTRY_FLAG_PATH) and os.path.exists(SENTRY_FRAME_PATH):
                try:
                    with open(SENTRY_FRAME_PATH, 'rb') as f:
                        data = f.read()
                    return Response(data, mimetype='image/jpeg',
                                    headers={'Cache-Control': 'no-store'})
                except Exception:
                    return Response(status=500)

            # Sentry off → silent on-demand capture
            data = self._silent_cam.get_frame()
            if data is None:
                return Response(status=503)
            return Response(data, mimetype='image/jpeg',
                            headers={'Cache-Control': 'no-store'})

        @app.route('/api/logs')
        @login_required
        def get_logs():
            kind = request.args.get('kind', 'vivian')
            lines = int(request.args.get('lines', 200))
            lines = max(10, min(lines, 1000))
            try:
                if kind == 'system':
                    out = subprocess.run(
                        ['journalctl', '-u', 'vivian', '-n', str(lines), '--no-pager', '-o', 'short-iso'],
                        capture_output=True, text=True, timeout=5
                    )
                    text = out.stdout or out.stderr or "(no output)"
                else:
                    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vivian.log')
                    if not os.path.exists(log_path):
                        log_path = 'vivian.log'
                    if os.path.exists(log_path):
                        with open(log_path, 'r', errors='replace') as f:
                            text = ''.join(deque(f, maxlen=lines))
                    else:
                        text = "(vivian.log not found)"
            except Exception as e:
                text = f"Error reading logs: {e}"
            return jsonify({"text": text})

        @app.route('/api/state')
        @login_required
        def state_stream():
            def generate():
                while not self._shutdown.is_set():
                    state = self._build_state()
                    yield f"data: {json.dumps(state)}\n\n"
                    time.sleep(1)
            return Response(generate(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache',
                                     'X-Accel-Buffering': 'no'})

    def _build_state(self) -> dict:
        return {
            "gps": _read_gps(),
            "spotify": _read_spotify(),
            "assistant": _read_assistant(),
            "sentry_active": os.path.exists(SENTRY_FLAG_PATH),
            "sentry_status": _read_sentry_status(),
            "weather": self._get_weather(),
            "diagnostics": _get_diagnostics(),
        }

    def _get_weather(self) -> Optional[dict]:
        """Fetch weather with caching."""
        now = time.time()
        if self._weather_cache and now - self._weather_ts < WEATHER_CACHE_TTL:
            return self._weather_cache

        api_key = self._weather_config.get('api_key')
        city = self._weather_config.get('city')
        if not api_key or not city:
            return None

        try:
            import requests
            state = self._weather_config.get('state', '')
            country = self._weather_config.get('country', 'US')
            q = f"{city},{state},{country}" if state else city
            resp = requests.get(WEATHER_API_URL, params={
                'q': q, 'appid': api_key, 'units': 'imperial'
            }, timeout=5)
            data = resp.json()
            self._weather_cache = {
                "temp_f": round(data['main']['temp']),
                "condition": data['weather'][0]['main'],
            }
            self._weather_ts = now
            return self._weather_cache
        except Exception as e:
            logger.debug(f"Weather fetch failed: {e}")
            return self._weather_cache

    def start(self):
        """Start the dashboard server in a daemon thread."""
        self._thread = threading.Thread(target=self._thread_main, daemon=True,
                                        name="dashboard-server")
        self._thread.start()
        logger.info(f"Dashboard server started on port {self._port}")

    def stop(self):
        """Signal shutdown and wait for thread."""
        self._shutdown.set()
        try:
            self._silent_cam.release()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Dashboard server stopped")

    def _thread_main(self):
        """Thread entry point — runs Flask."""
        # Suppress Flask/Werkzeug request logging
        wlog = logging.getLogger('werkzeug')
        wlog.setLevel(logging.ERROR)

        self._app.run(
            host='0.0.0.0',
            port=self._port,
            threaded=True,
            use_reloader=False,
        )


# Standalone testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    # Generate a test password hash if needed
    if len(sys.argv) > 1 and sys.argv[1] == '--hash':
        pw = input("Enter password to hash: ")
        hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
        print(f"Hash: {hashed}")
        sys.exit(0)

    # Standalone test credentials come from the environment — never
    # hardcode a password here, this file is tracked in git.
    test_user = os.environ.get('DASHBOARD_TEST_USER', 'test')
    test_pass = os.environ.get('DASHBOARD_TEST_PASSWORD')
    if not test_pass:
        sys.exit("Set DASHBOARD_TEST_PASSWORD (and optionally DASHBOARD_TEST_USER) "
                 "to run the dashboard standalone.")

    config = {
        'enabled': True,
        'port': 9200,
        'username': test_user,
        'password_hash': bcrypt.hashpw(test_pass.encode(), bcrypt.gensalt()).decode(),
        'secret_key': '',  # random per-boot key
    }
    weather = {
        'api_key': '',
        'city': 'Woodstock',
        'state': 'GA',
        'country': 'US',
    }

    server = DashboardServer(config, weather)
    print(f"Dashboard running at http://localhost:9200 (user: {test_user})")
    server._thread_main()
