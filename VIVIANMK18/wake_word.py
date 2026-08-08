import pvporcupine
import pyaudio
import numpy as np
import os
import subprocess
import sys
import threading
import time
import logging

logger = logging.getLogger(__name__)

# --- Audio-stall watchdog tuning ---------------------------------------------
# A car-start brownout (since the regulator change no longer reboots the Pi)
# can leave a USB audio dongle wedged. There are TWO variants:
#   1. It still returns buffers, but only silence/frozen data. The in-loop
#      check below catches that fast and rotates devices.
#   2. The in-flight stream.read() blocks in the kernel (do_sys_poll) and
#      NEVER returns — observed after a 2s under-voltage dip. The in-loop
#      check can't see this (it runs *after* read() returns, which never
#      happens), so a separate liveness thread force-exits the process for
#      a clean systemd restart (proven to re-open the device fine).
STALL_TIMEOUT_S = 18.0   # variant 1: dead/frozen audio this long => wedged stream
MAX_RECOVERIES = 4       # consecutive failed reopens => sys.exit for clean systemd restart
# variant 2: no healthy frame for this long while armed => the read is wedged
# on a blocked stream.read(). Slightly longer than STALL_TIMEOUT_S so the
# lighter in-loop device-rotation recovery gets first crack at variant 1.
LIVENESS_TIMEOUT_S = 25.0
LIVENESS_POLL_S = 4.0    # how often the liveness thread checks

# Tiered recovery for the blocked-read wedge. A USB reset of the mic dongle
# forces the kernel to error the stuck read() so the wake loop's existing
# exception handler reopens the stream IN-PROCESS — VIVIAN keeps running
# (no ~2-min cold restart). os._exit is kept only as a last resort if the
# resets don't bring audio back.
MIC_USB_VID = "1b3f"     # GeneralPlus USB Audio dongle(s)
MIC_USB_PID = "2008"
MAX_USB_RESETS = 2       # consecutive resets that fail to restore audio => hard-exit
RESET_GRACE_S = 16.0     # wait this long after a reset for reopen+frames before re-acting
DEFAULT_SENSITIVITY = 0.65  # Porcupine wake-word sensitivity (was implicit 0.5)

# --- Callback watchdog -------------------------------------------------------
# Both audio watchdogs are DISARMED while the wake callback runs (record ->
# whisper -> Claude -> TTS), because that window legitimately does no mic
# reads. That left the callback itself completely uncovered: if it ever hangs,
# the wake loop never restarts, no watchdog fires, and VIVIAN goes silently
# deaf until the car is power-cycled — she answers once and never listens
# again. This deadline covers that window. A normal interaction is well under
# 60s; sentry mode blocks for hours by design, so it explicitly suspends this.
CALLBACK_TIMEOUT_S = 300.0

# --- Reopen-failure escalation ----------------------------------------------
# The catch-all handler used to retry forever with no escalation and no device
# rotation, so a persistently unopenable mic became an infinite quiet spin
# (same deaf-forever symptom, nothing in the log but repeated errors).
MAX_OPEN_FAILURES = 8    # consecutive failures before exiting for a clean restart

# --- Porcupine licence/activation failures -----------------------------------
# These are NOT audio faults and a restart cannot fix them: Picovoice's
# activation server refused the AccessKey. They must never feed the escalation
# ladder above, or VIVIAN restart-loops every ~20s and TTS, the displays and
# sentry never stabilise. Instead: log something actionable, back off, and keep
# retrying slowly so she self-heals the moment the key is valid again — with
# the rest of the system running normally in the meantime.
LICENCE_RETRY_S = 60.0

# How often to log the peak wake score when wake_word.debug_scores is on.
DEBUG_SCORE_INTERVAL_S = 2.0
try:
    _LICENCE_ERRORS = (
        pvporcupine.PorcupineActivationError,
        pvporcupine.PorcupineActivationLimitError,
        pvporcupine.PorcupineActivationRefusedError,
        pvporcupine.PorcupineActivationThrottledError,
        pvporcupine.PorcupineKeyError,
    )
except AttributeError:      # older pvporcupine without the typed exceptions
    _LICENCE_ERRORS = ()


class _AudioStall(Exception):
    """Raised internally when the input stream delivers dead/frozen audio."""


class WakeEngineUnavailable(Exception):
    """The configured wake engine cannot run at all (missing model/package).

    Distinct from a transient failure: retrying is pointless, so the loop
    stops trying and leaves button/Glass/sentry working.
    """


# --- Wake engines -------------------------------------------------------------
# Everything else in this file — the callback deadline, device rotation, the
# liveness watchdog, USB reset — is engine-agnostic. An engine only has to say
# what audio it wants and answer "was that the wake word?", so swapping the
# detector never touches the reliability machinery.

class _WakeEngine:
    sample_rate = 16000
    frame_length = 512
    name = "base"

    def process(self, frame) -> bool:
        raise NotImplementedError

    def delete(self):
        pass


class _PorcupineEngine(_WakeEngine):
    """Picovoice Porcupine. Requires a valid AccessKey activated against
    Picovoice's servers — see _LICENCE_ERRORS for the refusal path."""

    name = "porcupine"

    def __init__(self, access_key, model_path, sensitivity):
        self._h = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[model_path],
            sensitivities=[sensitivity],
        )
        self.sample_rate = self._h.sample_rate
        self.frame_length = self._h.frame_length

    def process(self, frame) -> bool:
        return self._h.process(frame.tolist()) >= 0

    def delete(self):
        try:
            self._h.delete()
        except Exception as e:
            logger.warning(f"Error deleting Porcupine instance: {e}")


class _OpenWakeWordEngine(_WakeEngine):
    """openWakeWord (Apache-2.0, ONNX). No account, no activation server —
    the reason we can leave Picovoice behind.

    Wants 1280-sample (80ms) int16 frames at 16kHz, and returns a per-model
    score rather than an index, so the threshold lives here.
    """

    name = "openwakeword"
    sample_rate = 16000
    frame_length = 1280

    def __init__(self, model_path, threshold, debug_scores=False):
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise WakeEngineUnavailable(
                f"openwakeword is not installed ({e}). "
                f"Install it on the Pi: pip3 install openwakeword"
            )
        # When on, log the peak score every DEBUG_SCORE_INTERVAL_S. This is the
        # only way to see what the live mic actually produces without stopping
        # the service to record — a silent wake word is otherwise
        # indistinguishable from "model never fires" and "audio never arrives".
        self._debug = debug_scores
        self._peak = 0.0
        self._peak_at = time.monotonic()
        self._lvl = 0.0
        self._nframes = 0
        if not os.path.exists(model_path):
            raise WakeEngineUnavailable(
                f"openWakeWord model not found: {model_path} — train one "
                f"(see docs/WAKEWORD_TRAINING.md) or point wake_word.model_path at it"
            )
        self._threshold = threshold
        self._model = Model(wakeword_models=[model_path],
                            inference_framework="onnx")
        self._key = None

    def process(self, frame) -> bool:
        scores = self._model.predict(frame)
        if self._key is None:
            # The score dict is keyed by model name; resolve it once so a
            # renamed model file doesn't silently never trigger.
            self._key = next(iter(scores), None)
        if self._key is None:
            return False
        score = scores[self._key]

        if self._debug:
            self._peak = max(self._peak, score)
            # Track the audio level of the SAME frames the model sees. A near
            # silent level means audio never arrives (routing/device problem);
            # a healthy level with a zero score means the model is the problem.
            lvl = float(np.abs(frame).max()) / 32768.0
            self._lvl = max(getattr(self, "_lvl", 0.0), lvl)
            now = time.monotonic()
            if now - self._peak_at >= DEBUG_SCORE_INTERVAL_S:
                logger.info(
                    f"wake score peak={self._peak:.3f} "
                    f"(threshold {self._threshold:.2f}) | audio peak={self._lvl:.4f} "
                    f"| frames={self._nframes}"
                )
                self._peak = 0.0
                self._lvl = 0.0
                self._nframes = 0
                self._peak_at = now
            self._nframes = getattr(self, "_nframes", 0) + 1

        return score >= self._threshold

    def delete(self):
        self._model = None


class WakeWordDetector:
    """Detect wake word using Porcupine or button press"""

    def __init__(self, config, hardware_controller):
        self.config = config
        self.hardware = hardware_controller
        self.access_key = config.porcupine['access_key']
        self.model_path = config.porcupine['model_path']
        # Sensitivity is optional in config; default raised to 0.65 for the
        # noisy car cabin (Porcupine's library default is 0.5).
        try:
            self.sensitivity = float(config.porcupine.get('sensitivity', DEFAULT_SENSITIVITY))
        except (AttributeError, TypeError, ValueError):
            self.sensitivity = DEFAULT_SENSITIVITY
        self.running = False
        # Rotates across identical USB dongles when one wedges (enumeration swap)
        self._device_attempt = 0

        # --- Liveness watchdog state (variant-2 blocked-read backstop) -------
        # _last_good_frame uses time.monotonic() (NOT wall clock) so an NTP
        # step — which happens every boot here, the Pi has no RTC battery —
        # can't corrupt the elapsed-time math.
        self._last_good_frame = time.monotonic()
        # Armed only while we're actively reading audio. Cleared during the
        # (potentially long) wake callback / GPT+TTS and during recovery, so
        # the watchdog never force-exits mid-response.
        self._wd_armed = threading.Event()
        self._wd_thread = None

        # --- Callback watchdog state -----------------------------------------
        # Monotonic timestamp of when the current wake callback started, or
        # None when no callback is running. Single assignments of an
        # immutable, so no lock is needed for the watchdog thread to read it.
        self._cb_started = None
        # Set while a legitimately long-running callback (sentry mode) owns
        # the process, so the deadline doesn't kill it.
        self._cb_exempt = threading.Event()

        # Engine selection. Porcupine is no longer usable on this build (the
        # Picovoice account was deleted and there is no free tier), so the
        # engine is configurable and "none" is a first-class option that keeps
        # the button/Glass/sentry paths working without log noise.
        wcfg = getattr(config, 'wake_word', None) or {}
        self.engine_name = str(wcfg.get('engine', 'porcupine')).lower()
        self.oww_model_path = wcfg.get('model_path', 'models/hey_vivian.onnx')
        try:
            self.oww_threshold = float(wcfg.get('threshold', 0.5))
        except (TypeError, ValueError):
            self.oww_threshold = 0.5
        self.oww_debug = bool(wcfg.get('debug_scores', False))
        # Only ever open the named vivian_mic PCM. Default ON: the udev rule
        # pins each dongle to a fixed ALSA card by physical USB port, so
        # vivian_mic is always the real microphone and falling back to another
        # input device can only ever make things worse (see _get_input_device).
        self.strict_device = bool(wcfg.get('strict_device', True))

        logger.info(
            f"Wake word detector initialized "
            f"(engine={self.engine_name}, sensitivity={self.sensitivity})"
        )

    def _make_engine(self):
        """Build the configured wake engine.

        Raises WakeEngineUnavailable when the engine can never work as
        configured (missing package/model) — the loop treats that as fatal for
        wake detection but harmless to the rest of VIVIAN.
        """
        if self.engine_name in ('none', 'off', 'disabled'):
            raise WakeEngineUnavailable("wake_word.engine is set to 'none'")

        if self.engine_name == 'openwakeword':
            path = self.oww_model_path
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            return _OpenWakeWordEngine(path, self.oww_threshold,
                                       debug_scores=self.oww_debug)

        if self.engine_name == 'porcupine':
            return _PorcupineEngine(self.access_key, self.model_path, self.sensitivity)

        raise WakeEngineUnavailable(f"unknown wake_word.engine '{self.engine_name}'")

    def _usb_reset_mics(self):
        """
        USB-reset every GeneralPlus mic dongle (by VID:PID, sysfs — no
        fragile card->node mapping). The port reset makes the kernel error
        the stuck read() so the wake loop's normal exception/reopen path
        runs in-process. The healthy spare just re-enumerates in ~1-2s.
        Returns the number of dongles reset.
        """
        base = "/sys/bus/usb/devices"
        nodes = []
        try:
            for dev in os.listdir(base):
                dpath = os.path.join(base, dev)
                try:
                    with open(os.path.join(dpath, "idVendor")) as f:
                        vid = f.read().strip()
                    with open(os.path.join(dpath, "idProduct")) as f:
                        pid = f.read().strip()
                except (OSError, ValueError):
                    continue  # interfaces/hubs without these attrs
                if vid != MIC_USB_VID or pid != MIC_USB_PID:
                    continue
                try:
                    with open(os.path.join(dpath, "busnum")) as f:
                        bus = int(f.read().strip())
                    with open(os.path.join(dpath, "devnum")) as f:
                        num = int(f.read().strip())
                except (OSError, ValueError):
                    continue
                nodes.append(f"/dev/bus/usb/{bus:03d}/{num:03d}")
        except OSError as e:
            logger.error(f"USB reset: cannot scan {base}: {e}")
            return 0

        if not nodes:
            logger.error(
                f"USB reset: no {MIC_USB_VID}:{MIC_USB_PID} dongle found"
            )
            return 0

        for node in nodes:
            try:
                subprocess.run(
                    ["sudo", "-n", "/usr/bin/usbreset", node],
                    timeout=10, check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                logger.warning(f"USB reset issued: {node}")
            except Exception as e:
                logger.error(f"USB reset failed for {node}: {e}")
        return len(nodes)

    def _liveness_watchdog(self):
        """
        Background daemon backstop for the blocked-read wedge (variant 2),
        which the in-loop watchdog physically can't catch. Tiered so the
        common case recovers WITHOUT a ~2-min process restart:

          tier 2 — no healthy frame for LIVENESS_TIMEOUT_S while armed:
                   USB-reset the mic dongles. That errors the stuck read();
                   the wake loop's existing handler reopens the stream and
                   the rest of VIVIAN keeps running untouched.
          tier 3 — still no audio MAX_USB_RESETS resets later: os._exit(1)
                   (sys.exit can't help — wake thread is stuck in native
                   code) so systemd does a full clean restart as last resort.

        Healthy audio or a disarmed state clears the escalation counter, so
        each fresh wedge starts the ladder over.
        """
        resets = 0
        last_action = 0.0
        while self.running:
            time.sleep(LIVENESS_POLL_S)
            if not self.running:
                return

            # tier 1 — the callback window (uncovered by both audio watchdogs,
            # since it does no mic reads at all). A wedged callback means the
            # wake loop never restarts, so only a hard exit recovers it.
            started = self._cb_started
            if started is not None and not self._cb_exempt.is_set():
                cb_elapsed = time.monotonic() - started
                if cb_elapsed > CALLBACK_TIMEOUT_S:
                    logger.error(
                        f"Callback watchdog: wake callback stuck for "
                        f"{cb_elapsed:.0f}s (limit {CALLBACK_TIMEOUT_S:.0f}s) "
                        f"— hard-exiting for clean systemd restart"
                    )
                    os._exit(1)

            if not self._wd_armed.is_set():
                resets = 0          # reopening / in callback — not a wedge
                continue
            stalled = time.monotonic() - self._last_good_frame
            if stalled <= LIVENESS_TIMEOUT_S:
                resets = 0          # audio flowing again — ladder reset
                continue
            # Wedged while armed. Give the last reset time to take effect
            # (USB re-enumerate + stream reopen) before acting again.
            if last_action and time.monotonic() - last_action < RESET_GRACE_S:
                continue
            if resets >= MAX_USB_RESETS:
                logger.error(
                    f"Liveness: audio still dead after {resets} USB resets "
                    f"({stalled:.0f}s) — hard-exiting for clean systemd restart"
                )
                os._exit(1)
            resets += 1
            last_action = time.monotonic()
            logger.error(
                f"Liveness: read wedged {stalled:.0f}s — USB-resetting mic "
                f"dongle(s) (attempt {resets}/{MAX_USB_RESETS}, in-process "
                f"recovery, no full restart)"
            )
            self._usb_reset_mics()
    
    def _resample_int16_mono(self, pcm_int16: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """
        Linear resample mono int16 -> int16
        Used to convert device sample rate to Porcupine's expected rate
        """
        if from_sr == to_sr or len(pcm_int16) == 0:
            return pcm_int16
        
        x = pcm_int16.astype(np.float32)
        ratio = to_sr / float(from_sr)
        n_out = int(round(len(x) * ratio))
        xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float32)
        xo = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
        y = np.interp(xo, xp, x)
        return np.clip(y, -32768, 32767).astype(np.int16)
    
    def listen(self, on_wake_callback, poll_callback=None):
        """
        Listen for wake word or button press.
        Calls on_wake_callback when either is detected.
        If poll_callback is set, calls it each audio frame — if it returns True,
        triggers the same cleanup/restart cycle as a wake event.
        """
        FIXED_DEVICE_SR = 48000
        self.running = True
        recoveries = 0  # consecutive stall recoveries with no good audio in between
        open_failures = 0  # consecutive setup/read errors with no good audio in between
        licence_failures = 0  # consecutive Porcupine activation refusals

        logger.info("Starting wake word detection loop")

        # Start the blocked-read liveness backstop once.
        if self._wd_thread is None or not self._wd_thread.is_alive():
            self._wd_thread = threading.Thread(
                target=self._liveness_watchdog, daemon=True,
                name="wake-liveness-watchdog")
            self._wd_thread.start()

        while self.running:
            porcupine = None
            pa = None
            stream = None
            # Disarmed until a stream is open and reading — covers reopen,
            # recovery sleeps, and the long post-detection callback window.
            self._wd_armed.clear()

            try:
                # Create Porcupine instance
                porcupine = self._make_engine()

                target_sr = porcupine.sample_rate
                target_frame = porcupine.frame_length

                logger.info(
                    f"Wake engine '{porcupine.name}' ready "
                    f"(target SR: {target_sr}Hz, frame length: {target_frame})"
                )
                
                # Initialize PyAudio
                pa = pyaudio.PyAudio()
                
                # Select input device
                in_dev = self._get_input_device(pa)
                
                # Determine device sample rate
                device_sr = self._get_device_sample_rate(pa, in_dev, FIXED_DEVICE_SR)
                
                # Calculate buffer size
                approx_dev_frames = max(64, int(round(target_frame * device_sr / float(target_sr))))
                
                # Open audio stream
                stream = pa.open(
                    rate=device_sr,
                    channels=1,
                    format=pyaudio.paInt16,
                    input=True,
                    input_device_index=in_dev,
                    frames_per_buffer=approx_dev_frames,
                )
                
                # Set hardware to wake mode
                self.hardware.set_wake_mode()
                logger.info(f"Listening for wake word (device: {device_sr}Hz, target: {target_sr}Hz)")

                # Arm the liveness backstop now that we're about to read.
                self._last_good_frame = time.monotonic()
                self._wd_armed.set()

                # Process audio
                resampled_buf = np.empty(0, dtype=np.int16)
                last_live_ts = time.monotonic()  # last time we saw real mic audio
                prev_raw = None             # previous raw buffer (frozen-stream detection)

                while self.running:
                    # Read audio data
                    data = stream.read(approx_dev_frames, exception_on_overflow=False)
                    pcm_dev = np.frombuffer(data, dtype=np.int16)

                    # --- Audio-stall watchdog -------------------------------
                    # A wedged USB capture (post-brownout) returns exact-zero
                    # buffers or the same frozen buffer repeatedly. A quiet
                    # cabin still has nonzero, varying mic self-noise, so this
                    # will not false-trip when you're simply not talking.
                    is_zero = not pcm_dev.any()
                    is_frozen = prev_raw is not None and data == prev_raw
                    prev_raw = data
                    if is_zero or is_frozen:
                        if time.monotonic() - last_live_ts > STALL_TIMEOUT_S:
                            reason = "all-zero" if is_zero else "frozen"
                            logger.error(
                                f"Audio stream wedged ({reason} for "
                                f">{STALL_TIMEOUT_S:.0f}s) — recovering"
                            )
                            raise _AudioStall(reason)
                    else:
                        now_m = time.monotonic()
                        last_live_ts = now_m
                        # Healthy frame — feed the liveness backstop too.
                        self._last_good_frame = now_m
                        recoveries = 0  # healthy audio — clear recovery streak
                        open_failures = 0
                        licence_failures = 0

                    # Resample to Porcupine's expected rate
                    pcm_target = self._resample_int16_mono(pcm_dev, from_sr=device_sr, to_sr=target_sr)
                    if pcm_target.size:
                        resampled_buf = np.concatenate((resampled_buf, pcm_target))
                    
                    # Process complete frames
                    while resampled_buf.size >= target_frame:
                        frame = resampled_buf[:target_frame]
                        resampled_buf = resampled_buf[target_frame:]
                        
                        # Check for wake word
                        wake_detected = porcupine.process(frame)
                        
                        # Check for button press
                        button_pressed = self.hardware.is_button_pressed()

                        # Check poll callback (e.g. Glass sentry request)
                        poll_result = poll_callback() if poll_callback else None

                        if wake_detected or button_pressed or poll_result:
                            if poll_result:
                                trigger_type = "Poll Callback"
                            elif wake_detected:
                                trigger_type = "Wake Word"
                            else:
                                trigger_type = "Button Press"
                            logger.info(f"[{trigger_type} Detected]")

                            # Disarm: the callback (GPT + TTS) can run far
                            # longer than LIVENESS_TIMEOUT_S with no reads.
                            self._wd_armed.clear()

                            # Cleanup audio resources before callback
                            self._cleanup_audio(stream, pa, porcupine)
                            stream = None
                            pa = None
                            porcupine = None

                            # Small delay for stability
                            time.sleep(0.5)

                            # Trigger appropriate callback. Under the callback
                            # watchdog: both audio watchdogs are disarmed here,
                            # so this deadline is the only thing that can
                            # recover a hang in record/whisper/Claude/TTS.
                            self._cb_started = time.monotonic()
                            try:
                                if poll_result and callable(poll_result):
                                    poll_result()  # Poll returns the action to run
                                elif not poll_result:
                                    on_wake_callback()
                            finally:
                                self._cb_started = None

                            # Restart wake word listener
                            logger.info("Restarting wake word listener")
                            time.sleep(0.5)
                            raise StopIteration
            
            except WakeEngineUnavailable as e:
                # Nothing to retry: the package or model is absent, or the
                # engine is switched off. Say so once and stop the loop —
                # button, Glass and sentry all keep working. (Porcupine's
                # licence refusal is different: that one CAN self-heal, so it
                # keeps retrying below.)
                self._cleanup_audio(stream, pa, porcupine)
                logger.error(
                    f"Wake word DISABLED — {e}. Button, Glass and sentry are "
                    f"unaffected; no further attempts will be made."
                )
                self.running = False
                return

            except StopIteration:
                # Clean restart of wake word detection
                continue

            except _AudioStall:
                # Stream is wedged. Tear down, rotate to the next candidate
                # input device (covers the two-identical-dongle enumeration
                # swap), and reopen. If several reopens in a row still find
                # dead audio, exit so systemd (Restart=on-failure) does a
                # full clean restart — the only thing that reliably clears a
                # brownout-zombied USB device.
                self._cleanup_audio(stream, pa, porcupine)
                stream = pa = porcupine = None
                recoveries += 1
                self._device_attempt += 1
                if recoveries >= MAX_RECOVERIES:
                    logger.error(
                        f"Audio still dead after {recoveries} reopen attempts "
                        f"— exiting for clean systemd restart"
                    )
                    sys.exit(1)
                logger.warning(
                    f"Reopening audio (attempt {recoveries}/{MAX_RECOVERIES}, "
                    f"device rotation {self._device_attempt})"
                )
                time.sleep(1.0)
                continue

            except KeyboardInterrupt:
                logger.info("Wake word detection interrupted by user")
                self.running = False
                self._cleanup_audio(stream, pa, porcupine)
                break
            
            except Exception as e:
                # Usually a transient reopen hiccup right after an interaction
                # (device still releasing), which the retry below clears. But a
                # PERSISTENT failure here used to spin forever — no rotation,
                # no escalation, no restart — so VIVIAN stayed deaf while the
                # process looked healthy. Rotate the input device and escalate.
                self._cleanup_audio(stream, pa, porcupine)
                stream = pa = porcupine = None

                # Licence/activation refusal: restarting and rotating devices
                # are both pointless. Back off and keep the rest of VIVIAN alive.
                if _LICENCE_ERRORS and isinstance(e, _LICENCE_ERRORS):
                    licence_failures += 1
                    if licence_failures == 1 or licence_failures % 20 == 0:
                        logger.error(
                            f"Porcupine ACTIVATION REFUSED ({type(e).__name__}) "
                            f"— this is an AccessKey/licence problem, NOT audio. "
                            f"Check console.picovoice.ai (device limit / key "
                            f"validity). Wake word is offline; button, Glass and "
                            f"sentry still work. Retrying every "
                            f"{LICENCE_RETRY_S:.0f}s (attempt {licence_failures})."
                        )
                    time.sleep(LICENCE_RETRY_S)
                    continue

                open_failures += 1
                self._device_attempt += 1
                if open_failures >= MAX_OPEN_FAILURES:
                    logger.error(
                        f"Wake word audio failed {open_failures} times in a row "
                        f"(last: {e}) — exiting for clean systemd restart"
                    )
                    sys.exit(1)
                logger.error(
                    f"Wake word detection error: {e} "
                    f"(failure {open_failures}/{MAX_OPEN_FAILURES}, "
                    f"device rotation {self._device_attempt})"
                )
                time.sleep(1.0)
                continue
    
    def _get_input_device(self, pa: pyaudio.PyAudio) -> int:
        """
        Pick an input device, rotating across candidates on stall recovery.

        Candidate order: the named `vivian_mic` PCM first, then every other
        input-capable device (the second identical USB dongle, raw hw, etc.).
        `self._device_attempt` advances on each audio stall so a wedged
        dongle gets bypassed.

        With strict_device set (the default now that udev pins the cards by
        physical USB port), ONLY vivian_mic is ever used. Rotation existed to
        survive the two identical dongles swapping enumeration order, which the
        udev pin solves properly — and rotating was actively dangerous: the
        spare dongle has nothing plugged into its mic jack, and its AGC winds
        the gain up until the floating input produces loud, plausible-looking
        noise. No watchdog can tell that apart from a live mic, so a rotation
        onto it left VIVIAN permanently deaf while looking perfectly healthy.
        That is exactly the failure that cost a long debugging session.
        """
        named = []   # vivian_mic by name (preferred)
        others = []  # any other input-capable device
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) <= 0:
                continue
            if 'vivian_mic' in info.get('name', ''):
                named.append((i, info.get('name')))
            else:
                others.append((i, info.get('name')))

        if self.strict_device:
            if not named:
                raise RuntimeError(
                    "strict_device is set but no input device named 'vivian_mic' "
                    "exists — check /etc/asound.conf and that the mic dongle is "
                    "plugged in. Refusing to fall back to an arbitrary device."
                )
            candidates = named
        else:
            candidates = named + others

        if not candidates:
            raise RuntimeError("No input device found with input channels")

        idx, name = candidates[self._device_attempt % len(candidates)]
        logger.info(
            f"Using input device: {name} (index {idx}) "
            f"[candidate {self._device_attempt % len(candidates)+1}/{len(candidates)}]"
        )
        return idx
    
    def _get_device_sample_rate(self, pa: pyaudio.PyAudio, device_index: int, preferred_rate: int) -> int:
        """
        Get the sample rate for the device
        Try preferred rate first, fall back to device default if not supported
        """
        try:
            # Try preferred rate
            pa.is_format_supported(
                rate=preferred_rate,
                input_device=device_index,
                input_channels=1,
                input_format=pyaudio.paInt16,
            )
            logger.info(f"Device supports preferred sample rate: {preferred_rate}Hz")
            return preferred_rate
        except ValueError:
            # Fall back to device default
            device_info = pa.get_device_info_by_index(device_index)
            default_rate = int(device_info.get("defaultSampleRate", 48000))
            logger.warning(f"Device doesn't support {preferred_rate}Hz, using {default_rate}Hz")
            return default_rate
    
    def _cleanup_audio(self, stream, pa, porcupine):
        """Safely cleanup audio resources"""
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                logger.debug("Audio stream closed")
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
        
        if pa is not None:
            try:
                pa.terminate()
                logger.debug("PyAudio terminated")
            except Exception as e:
                logger.warning(f"Error terminating PyAudio: {e}")
        
        if porcupine is not None:
            try:
                porcupine.delete()
                logger.debug("Porcupine instance deleted")
            except Exception as e:
                logger.warning(f"Error deleting Porcupine: {e}")
    
    def suspend_callback_watchdog(self):
        """Exempt the running callback from CALLBACK_TIMEOUT_S.

        For work that legitimately blocks far longer than an interaction —
        sentry mode waits for a known person to return, potentially for hours.
        """
        self._cb_exempt.set()
        logger.debug("Callback watchdog suspended")

    def resume_callback_watchdog(self):
        """Re-apply the callback deadline, restarting the clock so the time
        already spent in the exempt section doesn't trip it immediately."""
        if self._cb_started is not None:
            self._cb_started = time.monotonic()
        self._cb_exempt.clear()
        logger.debug("Callback watchdog resumed")

    def stop(self):
        """Stop the wake word detection loop"""
        logger.info("Stopping wake word detection")
        self.running = False
