import os
import sys
import signal
import time
import logging
import threading
import tempfile
import subprocess
from pathlib import Path

import pytz

logger = logging.getLogger(__name__)

# CRT display sentry flag file path
SENTRY_FLAG_PATH = "/tmp/vivian_sentry_enabled"

# Shared camera frame for CRT display during sentry mode
SENTRY_FRAME_PATH = "/tmp/vivian_sentry_frame.jpg"

# Last threat assessment, for the CRT / dashboard / Glass to display
SENTRY_STATUS_PATH = "/tmp/vivian_sentry_status.json"

# Pre-cached WAV audio directory (populated by Sentry/generate_audio.py)
AUDIO_DIR = Path(__file__).parent / 'Sentry' / 'audio'

# Image extensions recognized in the face database (flat-file layout)
FACE_IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

# Give up on a blind camera after this long and hand control back to VIVIAN.
# Generous enough to ride out a USB re-enumeration or several recovery cycles,
# but bounded — the sentry loop runs inside the wake callback, so retrying
# forever left VIVIAN permanently deaf with no way out but a power cycle.
CAMERA_BLIND_FAILSAFE_S = 180.0


class SentryController:
    """Bridge between VIVIAN main system and Vehicle Sentry.

    Wraps VehicleSentry with TTS announcements at each pipeline step
    and returns control to main.py when a known person is detected.
    """

    def __init__(self, config, tts, hardware):
        self.config = config
        self.tts = tts
        self.hardware = hardware

        # Resolve sentry config path relative to this file's directory
        sentry_cfg = self.config.sentry
        cfg_path = sentry_cfg.get('config_path', 'Sentry/sentry_config.yaml')
        self.sentry_config_path = Path(__file__).parent / cfg_path

        # TTS output device for sentry announcements
        self.tts_output = sentry_cfg.get('tts_output', 'internal_plug')

        # Active run pointer (set only while sentry is running)
        self._sentry = None
        self._original_tts_device = None

        # Cached integrated sentry — YOLO + InsightFace stay loaded across
        # activations (~20-30s model load otherwise paid on EVERY activation,
        # shown as a black "SENTRY MODE" screen on the CRT). Rebuilt only
        # when sentry_config.yaml changes.
        self._integrated = None
        self._integrated_cfg_mtime = None

        logger.info("SentryController initialized")

    def _get_integrated(self):
        """Return the cached _IntegratedSentry, building it on first use or
        when sentry_config.yaml has changed since it was built."""
        try:
            cfg_mtime = os.path.getmtime(self.sentry_config_path)
        except OSError:
            cfg_mtime = None

        if self._integrated is not None and cfg_mtime == self._integrated_cfg_mtime:
            self._integrated.reset_for_run()
            logger.info("Reusing cached sentry (models already loaded)")
            return self._integrated

        # Import here to avoid loading heavy ML libs at VIVIAN startup
        sys.path.insert(0, str(Path(__file__).parent / 'Sentry'))
        from SentryMk3 import SentryConfig

        sentry_config = SentryConfig.from_yaml(self.sentry_config_path)

        # Override: known person should stop the sentry loop
        sentry_config.known_person_stops = True

        # Resolve paths relative to Sentry directory
        sentry_dir = self.sentry_config_path.parent
        if not Path(sentry_config.face_database_path).is_absolute():
            sentry_config.face_database_path = str(sentry_dir / sentry_config.face_database_path)
        if not Path(sentry_config.snapshot_dir).is_absolute():
            sentry_config.snapshot_dir = str(sentry_dir / sentry_config.snapshot_dir)
        if not Path(sentry_config.log_dir).is_absolute():
            sentry_config.log_dir = str(sentry_dir / sentry_config.log_dir)
        if not Path(sentry_config.yolo_model).is_absolute():
            sentry_config.yolo_model = str(sentry_dir / sentry_config.yolo_model)
        if not Path(sentry_config.recording_dir).is_absolute():
            sentry_config.recording_dir = str(sentry_dir / sentry_config.recording_dir)

        logger.info("Building integrated sentry (loading detection models)...")
        self._integrated = _IntegratedSentry(
            sentry_config, self.tts, self.tts_output, self.config.timezone
        )
        self._integrated_cfg_mtime = cfg_mtime
        return self._integrated

    def activate(self) -> dict:
        """Run sentry mode (blocking).

        Returns:
            dict with keys:
                'reason': 'known_person' | 'manual_stop' | 'error'
                'known_names': list of recognized names (empty if none)
        """
        result = {
            'reason': 'error',
            'known_names': [],
        }

        # Save original signal handlers before sentry overwrites them
        orig_sigint = signal.getsignal(signal.SIGINT)
        orig_sigterm = signal.getsignal(signal.SIGTERM)

        try:
            # 1. Build or reuse the integrated sentry (heavy on first use)
            sentry = self._get_integrated()
            self._sentry = sentry
            logger.info("Integrated sentry ready, starting detection loop")

            # 2. Flag on only once we're about to run — the CRT shows
            #    "STARTING" until camera frames flow (see crt_display).
            self._write_sentry_flag(True)
            self._notify_discord("🛡️ Sentry armed — monitoring the vehicle.")

            # 3. Run sentry main loop (blocking - returns when known person or stopped)
            sentry.run()

            # 4. Determine result
            if sentry.detected_known:
                result['reason'] = 'known_person'
                result['known_names'] = list(sentry.detected_known)
            elif getattr(sentry, 'camera_blind_failure', False):
                # Camera never recovered — not a manual stop, and the user needs
                # to hear that the car stopped being watched.
                result['reason'] = 'error'
            else:
                result['reason'] = 'manual_stop'

        except Exception as e:
            logger.error(f"Sentry activation error: {e}", exc_info=True)
            result['reason'] = 'error'

        finally:
            # 5. Restore original signal handlers so Ctrl+C works for VIVIAN again
            try:
                signal.signal(signal.SIGINT, orig_sigint)
                signal.signal(signal.SIGTERM, orig_sigterm)
            except Exception:
                pass

            # 6. Remove OLED flag + last-threat status; keep the built sentry
            #    cached for next time
            self._write_sentry_flag(False)
            self._notify_discord(f"✅ Sentry disarmed ({result['reason']}).")
            for path in (SENTRY_STATUS_PATH, SENTRY_FRAME_PATH):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
            self._sentry = None

        return result

    def _notify_discord(self, text: str) -> None:
        """Post a short arm/disarm notice to the sentry Discord webhook (if set).
        Fire-and-forget on a background thread so it never blocks activation."""
        webhook = None
        try:
            if self._integrated is not None:
                webhook = self._integrated._sentry.config.discord_webhook_url
        except Exception:
            webhook = None
        if not webhook:
            return

        def _post():
            try:
                import requests
                requests.post(webhook, json={"content": text}, timeout=10)
            except Exception as e:
                logger.debug(f"Discord arm/disarm notify failed: {e}")
        threading.Thread(target=_post, daemon=True).start()

    def deactivate(self) -> bool:
        """Stop sentry mode externally.

        Returns True if a live sentry run was signalled to stop, False if
        no sentry is running (lets callers detect a stale flag file).
        """
        if self._sentry is not None:
            self._sentry.running = False
            return True
        return False

    def _write_sentry_flag(self, enabled: bool):
        """Write or remove the OLED sentry flag file."""
        try:
            if enabled:
                with open(SENTRY_FLAG_PATH, 'w') as f:
                    f.write('on\n')
                logger.info("OLED sentry flag written")
            else:
                if os.path.exists(SENTRY_FLAG_PATH):
                    os.remove(SENTRY_FLAG_PATH)
                    logger.info("OLED sentry flag removed")
        except Exception as e:
            logger.error(f"Sentry flag file error: {e}")


class _IntegratedSentry:
    """VehicleSentry wrapper that adds TTS announcements and known-person exit.

    Uses file-based TTS (synthesize to WAV → play with aplay) instead of
    VIVIAN's chunked-pipeline speak(), so each step completes fully even
    while the camera and YOLO are consuming CPU/USB bandwidth.
    """

    # Sentry speaks in the GLaDOS voice profile (the assistant uses "default").
    TTS_PROFILE = "sentry"

    def __init__(self, config, tts_engine, tts_device, timezone="US/Eastern"):
        from SentryMk3 import VehicleSentry
        # Set when the camera-blind failsafe ends the run, so activate() can
        # report reason='error' rather than a misleading 'manual_stop'.
        self.camera_blind_failure = False
        self._tts_engine = tts_engine
        self._tts_device = tts_device
        # Display timezone for CRT/dashboard timestamps (config, not the Pi's
        # system clock — which may be set to a different region).
        try:
            self._tz = pytz.timezone(timezone)
        except Exception:
            self._tz = pytz.timezone("US/Eastern")
        self.detected_known = None
        self._announced_person = False
        self._announced_recognition = False
        self._tts_lock = threading.Lock()

        # Create the actual sentry
        self._sentry = VehicleSentry(config)

        # Monkey-patch the handler methods to add TTS
        self._sentry.handle_known_person = self._handle_known_person
        self._sentry.handle_unknown_person = self._handle_unknown_person
        self._sentry.handle_person_no_face = self._handle_person_no_face
        self._sentry.handle_tamper_alert = self._handle_tamper_alert

        # Patch process_frame to add TTS at confirmation stage
        self._original_process_frame = self._sentry.process_frame
        self._sentry.process_frame = self._process_frame_with_tts

        # Patch run() to write camera frames for CRT display
        self._original_run = self._sentry.run
        self._sentry.run = self._run_with_frame_sharing
        self._frame_write_interval = 0.08  # Write frame every 80ms (~12fps)
        self._last_frame_write = 0
        self._latest_frame = None
        self._frame_id = 0
        self._frame_lock = threading.Lock()

        # Background: generate any missing welcome WAVs for known persons
        threading.Thread(target=self._ensure_welcome_wavs, daemon=True).start()

    @property
    def running(self):
        return self._sentry.running

    @running.setter
    def running(self, value):
        self._sentry.running = value

    def reset_for_run(self):
        """Clear per-run state so a cached instance (models stay loaded)
        starts a fresh sentry session."""
        self.detected_known = None
        self._announced_person = False
        self._announced_recognition = False
        with self._frame_lock:
            self._latest_frame = None
            self._frame_id = 0

        s = self._sentry
        s.running = False
        s.paused_until = None
        s.unknown_paused_until = None
        s.frame_count = 0
        s._confirmed_no_face_frames = 0
        s._last_tamper_alert = None
        s.person_detector.reset_confirmation()
        s.person_detector.tracked_persons.clear()
        s.person_detector.alerted_tracks.clear()
        if s.face_recognizer:
            s.face_recognizer.reset_voting()
        s.motion_detector.reset()
        s.tamper_detector.reset()

    def run(self):
        self._run_with_frame_sharing()

    def _camera_reader_loop(self):
        """Background thread: continuously reads camera frames at camera rate.

        Keeps _latest_frame always fresh, independent of YOLO processing speed.
        Both the frame writer and the YOLO loop pull from _latest_frame.
        """
        sentry = self._sentry
        while sentry.running:
            try:
                ret, frame = sentry.watchdog.read_frame()
                if ret and frame is not None and frame.size > 0:
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._frame_id += 1
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

    def _frame_writer_loop(self):
        """Background thread: encodes and writes latest frame to disk for CRT."""
        import cv2
        sentry = self._sentry
        last_written_id = -1
        while sentry.running:
            with self._frame_lock:
                frame = self._latest_frame
                frame_id = self._frame_id
            if frame is not None and frame_id != last_written_id:
                last_written_id = frame_id
                try:
                    display_frame = frame
                    if sentry.config.rotate_180:
                        display_frame = cv2.rotate(frame, cv2.ROTATE_180)
                    display_frame = cv2.resize(display_frame, (320, 240))
                    tmp_path = SENTRY_FRAME_PATH.replace(".jpg", "_tmp.jpg")
                    ok = cv2.imwrite(tmp_path, display_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    if ok:
                        os.replace(tmp_path, SENTRY_FRAME_PATH)
                except Exception:
                    pass
            time.sleep(0.05)

    def _run_with_frame_sharing(self):
        """Wrap the sentry run loop, writing camera frames for CRT display."""
        import cv2

        sentry = self._sentry
        sentry._setup_signal_handlers()
        sentry.logger.info("Starting Vehicle Sentry System Mk3 (with AI Threat Analysis)...")

        if sentry.watchdog.initialize_camera() is None:
            sentry.logger.error("Failed to initialize camera - exiting")
            return

        # Minimize frame buffer so read_frame() returns fresh frames
        try:
            sentry.watchdog.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Camera warm-up
        sentry.logger.info("Camera warming up (2 seconds)...")
        warmup_frames = 0
        warmup_start = time.time()
        while time.time() - warmup_start < 2.0:
            ret, _ = sentry.watchdog.read_frame()
            if ret:
                warmup_frames += 1
            time.sleep(0.05)
        sentry.logger.info(f"Warm-up complete ({warmup_frames} frames)")

        sentry.running = True
        sentry.motion_detector.start_time = time.time()
        sentry.logger.info("Sentry system active. Monitoring for persons...")
        sentry.logger.info(f"Motion warmup: bypassing motion filter for {sentry.config.motion_warmup_seconds}s")

        # Start background threads for camera reading and CRT frame writing
        threading.Thread(target=self._camera_reader_loop, daemon=True).start()
        threading.Thread(target=self._frame_writer_loop, daemon=True).start()

        # Camera-disconnect watchdog state: if the reader thread stops
        # producing fresh frames, alert + recover instead of going blind.
        last_seen_frame_id = -1
        last_fresh_frame = time.time()
        # When the camera first went blind, or None while frames are flowing.
        blind_since = None
        # Throttle recorder feeding to the recording fps — the camera runs
        # ~30fps and unthrottled feeding overflows the writer queue and
        # produces slow-motion clips.
        rec_interval = 1.0 / max(sentry.config.recording_fps, 1.0)
        last_rec_feed = 0.0

        try:
            while sentry.running:
                # Get latest frame from reader thread instead of blocking on camera
                with self._frame_lock:
                    frame = self._latest_frame
                    frame_id = self._frame_id

                # Only process FRESH frames — re-processing the same frame
                # burns CPU (motion diff at 720p) for zero information.
                if frame is None or frame_id == last_seen_frame_id:
                    # Enforce the recording duration limit here too: this branch
                    # skips feed_frame(), so without it a recording that started
                    # just before the camera stalled would latch on forever and
                    # block every later recording for the rest of the run.
                    if sentry.video_recorder is not None:
                        sentry.video_recorder.poll()
                    if time.time() - last_fresh_frame > sentry.config.watchdog_timeout:
                        # No new frames — camera unplugged or wedged. Fire the
                        # disconnect tamper alert, then attempt recovery.
                        sentry.logger.warning("Camera unhealthy - Loss of camera feed detected")
                        if not sentry._tamper_cooldown_active():
                            sentry.handle_tamper_alert('disconnect')
                        if sentry.watchdog.attempt_recovery():
                            try:
                                sentry.watchdog.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            except Exception:
                                pass
                            sentry.motion_detector.reset()
                            sentry.tamper_detector.reset()
                            sentry.logger.info("Camera recovered after feed loss")
                        else:
                            sentry.logger.error("Camera recovery failed - waiting before retry")
                            time.sleep(sentry.config.watchdog_retry_delay)
                        last_fresh_frame = time.time()

                        # Failsafe: if the camera never comes back, stop instead
                        # of retrying forever. This loop runs inside VIVIAN's
                        # wake callback, so an endless retry meant she never
                        # returned to wake-word listening — the worst outcome.
                        # camera_blind_failure makes activate() report
                        # reason='error', which main.py speaks and recovers from.
                        if blind_since is None:
                            blind_since = time.time()
                        elif time.time() - blind_since > CAMERA_BLIND_FAILSAFE_S:
                            sentry.logger.error(
                                f"No camera frames for "
                                f"{CAMERA_BLIND_FAILSAFE_S:.0f}s despite recovery "
                                f"attempts — stopping sentry so VIVIAN can resume"
                            )
                            self.camera_blind_failure = True
                            sentry.running = False
                            break
                    else:
                        time.sleep(0.01)
                    continue

                last_seen_frame_id = frame_id
                last_fresh_frame = time.time()
                blind_since = None  # frames flowing again

                # Tamper: obstruction check
                if sentry.frame_count % sentry.config.tamper_obstruction_check_interval == 0:
                    if sentry.tamper_detector.check_frame(frame):
                        if not sentry._tamper_cooldown_active():
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            tamper_dir = sentry.snapshot_manager.unknown_dir
                            snapshot_path = tamper_dir / f"tamper_obstruction_{timestamp}.jpg"
                            cv2.imwrite(str(snapshot_path), frame)
                            sentry.logger.info(f"Tamper snapshot saved: {snapshot_path}")
                            sentry.handle_tamper_alert('obstruction', snapshot_path)
                        sentry.tamper_detector.reset()

                # Feed frame to video recorder (throttled to recording fps)
                now = time.time()
                if (sentry.video_recorder is not None
                        and now - last_rec_feed >= rec_interval):
                    last_rec_feed = now
                    rec_frame = frame
                    if sentry.config.rotate_180:
                        rec_frame = cv2.rotate(frame, cv2.ROTATE_180)
                    sentry.video_recorder.feed_frame(rec_frame)

                sentry.process_frame(frame)
                sentry.periodic_maintenance()

        except Exception as e:
            sentry.logger.error(f"Error in main loop: {e}")
            import traceback
            sentry.logger.error(traceback.format_exc())
        finally:
            sentry.running = False
            if sentry.video_recorder is not None:
                sentry.video_recorder.release()
            sentry.watchdog.release()
            # Clean up shared frame file
            try:
                if os.path.exists(SENTRY_FRAME_PATH):
                    os.remove(SENTRY_FRAME_PATH)
            except Exception:
                pass
            sentry.logger.info("Sentry system stopped")

    def _file_speak(self, text):
        """File-based TTS: synthesize locally to a WAV file → aplay.

        More reliable than the chunked speak() pipeline under heavy CPU/USB
        load from YOLO + camera. Each step completes fully before the next.
        """
        wav_path = None
        try:
            logger.info(f"Sentry TTS: {text}")

            wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
            os.close(wav_fd)

            if not self._tts_engine.synthesize_to_wav(
                text, wav_path, profile=self.TTS_PROFILE
            ):
                logger.error("Sentry TTS synthesis failed")
                return

            subprocess.run(
                ["aplay", "-D", self._tts_device, wav_path],
                timeout=30,
                stderr=subprocess.PIPE
            )

            logger.info("Sentry TTS playback complete")

        except Exception as e:
            logger.error(f"Sentry TTS error: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _speak_async(self, text):
        """Non-blocking TTS in a background thread."""
        def _speak():
            with self._tts_lock:
                self._file_speak(text)
        t = threading.Thread(target=_speak, daemon=True)
        t.start()

    def _speak_blocking(self, text):
        """Blocking TTS - waits for speech to finish."""
        with self._tts_lock:
            self._file_speak(text)

    # ------------------------------------------------------------------
    # Pre-cached WAV playback helpers
    # ------------------------------------------------------------------

    def _play_wav_blocking(self, wav_path: Path) -> None:
        """Block until a pre-cached WAV file finishes playing."""
        try:
            subprocess.run(
                ["aplay", "-D", self._tts_device, str(wav_path)],
                timeout=30,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            logger.error(f"WAV playback error ({wav_path.name}): {e}")

    def _play_wav_async(self, wav_path: Path) -> None:
        """Play a pre-cached WAV in a background thread (serialised via lock)."""
        def _play():
            with self._tts_lock:
                self._play_wav_blocking(wav_path)
        threading.Thread(target=_play, daemon=True).start()

    def _speak_or_play_async(self, text: str, wav_key: str) -> None:
        """Use pre-cached WAV if available, otherwise fall back to live TTS (async)."""
        wav_path = AUDIO_DIR / f"{wav_key}.wav"
        if wav_path.exists():
            self._play_wav_async(wav_path)
        else:
            self._speak_async(text)

    def _speak_or_play_blocking(self, text: str, wav_key: str) -> None:
        """Use pre-cached WAV if available, otherwise fall back to live TTS (blocking)."""
        wav_path = AUDIO_DIR / f"{wav_key}.wav"
        if wav_path.exists():
            with self._tts_lock:
                self._play_wav_blocking(wav_path)
        else:
            self._speak_blocking(text)

    # ------------------------------------------------------------------
    # Auto-generate welcome WAVs for new/unknown faces
    # ------------------------------------------------------------------

    def _generate_wav_file(self, text: str, out_path: Path) -> bool:
        """Synthesize TTS audio locally and save as WAV. Returns True on success."""
        return self._tts_engine.synthesize_to_wav(
            text, out_path, profile=self.TTS_PROFILE
        )

    def _ensure_welcome_wavs(self) -> None:
        """Scan face_database/ and generate any missing 'Welcome back' WAVs.

        Runs in a background daemon thread at sentry startup so it never
        blocks detection. If a WAV already exists it is skipped.
        Supports both database layouts: face_database/PersonName/*.jpg
        (subfolders) and face_database/PersonName.jpg (flat files).
        """
        face_db = Path(self._sentry.config.face_database_path)
        if not face_db.exists():
            return

        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        names = set()
        for entry in sorted(face_db.iterdir()):
            if entry.name.startswith('.'):
                continue
            if entry.is_dir():
                names.add(entry.name)
            elif entry.is_file() and entry.suffix.lower() in FACE_IMAGE_EXTS:
                names.add(entry.stem)

        for name in sorted(names):
            wav_path = AUDIO_DIR / f"welcome_back_{name.lower()}.wav"
            if wav_path.exists():
                logger.info(f"Welcome WAV already exists for {name}")
                continue
            logger.info(f"Generating welcome WAV for new person: {name}")
            text = f"Identity confirmed. Welcome back, {name}."
            if self._generate_wav_file(text, wav_path):
                logger.info(f"Welcome WAV saved: {wav_path.name}")
            else:
                logger.warning(f"Failed to generate welcome WAV for {name} — will use live TTS")

    def _process_frame_with_tts(self, frame):
        """Wrapper around process_frame that announces detection milestones."""
        # Track confirmation state before processing
        was_confirmed = (
            len(self._sentry.person_detector.confirmation_buffer) >= self._sentry.config.confirmation_frames
            and sum(self._sentry.person_detector.confirmation_buffer) >= self._sentry.config.confirmation_min_hits
        )

        # Run the original process_frame
        self._original_process_frame(frame)

        # Check if confirmation just happened
        is_confirmed = (
            len(self._sentry.person_detector.confirmation_buffer) >= self._sentry.config.confirmation_frames
            and sum(self._sentry.person_detector.confirmation_buffer) >= self._sentry.config.confirmation_min_hits
        )

        if is_confirmed and not was_confirmed and not self._announced_person:
            self._announced_person = True
            self._speak_or_play_async("Person detected. Running facial recognition.", "person_detected")

        # Check if face voting is active (recognition in progress)
        if (self._sentry.face_recognizer and
                self._sentry.face_recognizer.track_voting and
                not self._announced_recognition and
                self._announced_person):
            for tid, votes in self._sentry.face_recognizer.track_voting.items():
                if len(votes) >= 2:
                    self._announced_recognition = True
                    break

    def _handle_known_person(self, names, confidence, frame):
        """Known person detected - announce (blocking) and stop sentry."""
        names_str = ', '.join(names)
        logger.info(f"Sentry: Known person(s) identified: {names_str}")

        # Use pre-cached WAV for a single identity; fall back to live TTS for multiple
        if len(names) == 1:
            wav_path = AUDIO_DIR / f"welcome_back_{names[0].lower()}.wav"
            if wav_path.exists():
                with self._tts_lock:
                    self._play_wav_blocking(wav_path)
            else:
                self._speak_blocking(f"Identity confirmed. Welcome back, {names[0]}.")
        else:
            self._speak_blocking(f"Identity confirmed. Welcome back, {names_str}.")

        # Save snapshot via the real sentry's snapshot manager
        person_count = len(self._sentry.person_detector.tracked_persons)
        for name in names:
            self._sentry.snapshot_manager.save_snapshot(
                frame, person_count, is_known=True, person_name=name
            )

        # Pause via cooldown (parent logic)
        from datetime import datetime, timedelta
        self._sentry.paused_until = datetime.now() + timedelta(
            seconds=self._sentry.config.known_person_cooldown
        )

        # Record detected names and stop
        self._write_status(f"known: {names_str}")
        self.detected_known = names
        self._sentry.running = False

    def _write_status(self, event: str, threat=None) -> None:
        """Write the latest sentry event/threat to a shared file for the CRT,
        dashboard, and Glass to display."""
        import json
        from datetime import datetime
        now = datetime.now(self._tz)
        data = {"event": event, "timestamp": time.time(),
                "time_str": now.strftime("%I:%M %p").lstrip("0")}
        if threat is not None:
            data["level"] = getattr(threat, "threat_level", None)
            data["description"] = getattr(threat, "description", "")
        try:
            tmp = SENTRY_STATUS_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, SENTRY_STATUS_PATH)
        except Exception as e:
            logger.debug(f"Sentry status write failed: {e}")

    def _play_deterrent(self, threat) -> None:
        """Speak the AI-written deterrent; if analysis failed or synthesis
        would be too slow, fall back to a pre-rendered generic deterrent
        matched to the threat level (instant — no CPU contention with YOLO)."""
        if threat and not threat.error and threat.deterrent_message:
            self._speak_async(threat.deterrent_message)
            return
        if threat and not threat.error:
            level = threat.threat_level
            key = ("deterrent_low" if level <= 3
                   else "deterrent_medium" if level <= 6
                   else "deterrent_high")
            wav = AUDIO_DIR / f"{key}.wav"
            if wav.exists():
                self._play_wav_async(wav)
                return
        self._speak_or_play_async("Alert sent. Continuing to monitor.", "alert_sent")

    def _handle_unknown_person(self, frame, track_ids):
        """Unknown person detected - announce, run threat analysis, send alert."""
        self._speak_or_play_async("Unknown person detected. Running threat analysis.", "unknown_person")
        logger.info(f"Sentry: Unknown person(s) detected, tracks: {track_ids}")

        # Start video recording FIRST — threat analysis + email + Discord
        # take seconds over cellular, and those are the seconds that matter.
        if self._sentry.video_recorder is not None:
            self._sentry.video_recorder.start_recording('unknown')

        # Use the real sentry's handler for snapshot + threat analysis + email
        person_count = len(track_ids)
        snapshot_path = self._sentry.snapshot_manager.save_snapshot(
            frame, person_count, is_known=False
        )
        threat = self._sentry.threat_analyzer.analyze_threat(snapshot_path)
        self._write_status("unknown_person", threat)
        # Email + Discord fire in the background so detection isn't blocked
        self._sentry._send_alerts_async(snapshot_path, person_count, threat)

        # Set unknown cooldown
        from datetime import datetime, timedelta
        self._sentry.unknown_paused_until = datetime.now() + timedelta(
            seconds=self._sentry.config.unknown_cooldown_seconds
        )

        # Speak deterrent aloud near the vehicle
        self._play_deterrent(threat)

        # Reset announcement flags for next detection cycle
        self._announced_person = False
        self._announced_recognition = False

    def _handle_person_no_face(self, frame, track_ids):
        """Person detected but face not identifiable."""
        self._speak_or_play_async("Person detected. Unable to identify. Running threat analysis.", "person_no_face")
        logger.info(f"Sentry: Person with no identifiable face, tracks: {track_ids}")

        # Start video recording FIRST (see _handle_unknown_person)
        if self._sentry.video_recorder is not None:
            self._sentry.video_recorder.start_recording('no_face')

        # Use the real sentry's handler for snapshot + threat analysis + email
        person_count = len(track_ids)
        snapshot_path = self._sentry.snapshot_manager.save_snapshot(
            frame, person_count, is_known=False
        )
        threat = self._sentry.threat_analyzer.analyze_threat(snapshot_path)
        self._write_status("person_no_face", threat)
        # Email + Discord fire in the background so detection isn't blocked
        self._sentry._send_alerts_async(snapshot_path, person_count, threat)

        # Set unknown cooldown
        from datetime import datetime, timedelta
        self._sentry.unknown_paused_until = datetime.now() + timedelta(
            seconds=self._sentry.config.unknown_cooldown_seconds
        )

        # An unidentifiable person gets the same deterrent as an unknown one
        self._play_deterrent(threat)

        # Reset announcement flags for next detection cycle
        self._announced_person = False
        self._announced_recognition = False

    def _handle_tamper_alert(self, tamper_type, snapshot_path=None):
        """Camera tamper detected - announce, send level-10 alert."""
        # Start recording FIRST while the camera still produces frames
        # (obstruction = covered lens, disconnect = no frames to record)
        if tamper_type == 'obstruction' and self._sentry.video_recorder is not None:
            self._sentry.video_recorder.start_recording('tamper')

        self._speak_or_play_blocking("Potential tampering detected. Treating as unknown threat.", "tamper_detected")
        logger.warning(f"Sentry: Tamper alert - {tamper_type}")

        from SentryMk3 import ThreatAssessment
        from datetime import datetime

        self._sentry._last_tamper_alert = datetime.now()

        threat = ThreatAssessment(
            threat_level=10,
            description=f"Camera {tamper_type} - potential tampering",
            reasoning=(
                f"Camera {tamper_type} detected while sentry was active. "
                "Treating as unknown threat."
            ),
        )
        self._write_status(f"tamper: {tamper_type}", threat)

        if snapshot_path is not None:
            self._sentry._send_alerts_async(snapshot_path, 0, threat)
        else:
            threading.Thread(
                target=self._sentry._send_tamper_alert_no_image,
                args=(tamper_type, threat), daemon=True
            ).start()

        self._speak_or_play_async("Tamper alert sent. Continuing to monitor.", "tamper_alert_sent")
