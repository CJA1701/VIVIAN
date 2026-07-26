import json
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

# Import all modules
from config import Config
from hardware import HardwareController
from memory import MemoryManager
from system_info import SystemInfo
from audio import AudioRecorder, AudioTranscriber
from audio_mute import AudioMuteController
from tts import TextToSpeech
from spotify_control import SpotifyController
from assistant import Assistant
from wake_word import WakeWordDetector
from sentry_controller import SentryController

# Setup logging — rotating file handler so vivian.log can't grow unbounded
# (10MB x 5 backups; the sentry logs already rotate the same way)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'vivian.log', maxBytes=10 * 1024 * 1024, backupCount=5
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class VIVIAN:
    """Main VIVIAN application orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        logger.info("=" * 60)
        logger.info("Initializing VIVIAN...")
        logger.info("=" * 60)
        
        try:
            # Clear stale sentry state from a previous crashed run — the
            # watchdog's os._exit / systemd restarts skip cleanup, and a
            # leftover flag locks the CRT (and Glass/dashboard) into
            # "sentry active" with no way to clear it from the UI.
            self._clear_stale_sentry_files()

            # Load configuration
            logger.info("Loading configuration...")
            self.config = Config(config_path)

            # Initialize hardware first (LEDs and button)
            logger.info("Initializing hardware controller...")
            self.hardware = HardwareController(self.config)
            
            # Initialize memory manager
            logger.info("Initializing memory manager...")
            self.memory = MemoryManager(self.config)
            
            # Initialize system info
            logger.info("Initializing system info...")
            self.system_info = SystemInfo(self.config)
            
            # Initialize audio components
            logger.info("Initializing audio recorder...")
            self.audio_recorder = AudioRecorder(self.config)

            logger.info("Initializing audio transcriber...")
            self.audio_transcriber = AudioTranscriber(self.config)

            # Initialize audio mute controller for stereo
            logger.info("Initializing audio mute controller...")
            stereo_control = self.config.audio.get('stereo_control', 'Stereo')
            # Card is auto-detected based on USB port
            self.audio_mute = AudioMuteController(control_name=stereo_control)

            # Initialize TTS
            logger.info("Initializing text-to-speech...")
            self.tts = TextToSpeech(self.config)
            
            # Initialize Spotify controller
            logger.info("Initializing Spotify controller...")
            self.spotify = SpotifyController(self.config)
            
            # Initialize sentry controller (optional - only if config present)
            self.sentry_controller = None
            if self.config.sentry:
                logger.info("Initializing sentry controller...")
                self.sentry_controller = SentryController(self.config, self.tts, self.hardware)

            # Initialize Glass HUD server (optional)
            self.glass_server = None
            if self.config.glass.get('enabled', False):
                logger.info("Initializing Glass HUD server...")
                from glass_server import GlassServer
                self.glass_server = GlassServer(
                    self.config.glass, self.spotify, self.config.weather,
                    sentry_controller=self.sentry_controller
                )

            # Initialize Dashboard server (optional)
            self.dashboard_server = None
            if self.config.dashboard.get('enabled', False):
                logger.info("Initializing Dashboard server...")
                from dashboard_server import DashboardServer
                self.dashboard_server = DashboardServer(
                    self.config.dashboard, self.config.weather,
                    sentry_controller=self.sentry_controller
                )

            # Initialize assistant (GPT brain)
            logger.info("Initializing assistant...")
            self.assistant = Assistant(
                self.config,
                self.system_info,
                self.memory,
                self.spotify,
                self.tts,
                sentry_controller=self.sentry_controller
            )

            # Initialize wake word detector
            logger.info("Initializing wake word detector...")
            self.wake_detector = WakeWordDetector(self.config, self.hardware)

            # Initialize CRT display (optional - gracefully handles missing hardware)
            self.display = None
            self.display_thread = None
            self._init_display()

            # Initialize Spotify touch display (optional)
            self.spotify_display = None
            self.spotify_display_thread = None
            self._init_spotify_display()

            logger.info("=" * 60)
            logger.info("VIVIAN initialized successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to initialize VIVIAN: {e}")
            raise
    
    @staticmethod
    def _clear_stale_sentry_files():
        """Remove sentry state files left behind by a crashed/killed run."""
        for path in ("/tmp/vivian_sentry_enabled",
                     "/tmp/vivian_sentry_request",
                     "/tmp/vivian_sentry_frame.jpg",
                     "/tmp/vivian_sentry_status.json"):
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.warning(f"Removed stale sentry file from previous run: {path}")
            except OSError:
                pass

    def _init_display(self):
        """Prepare CRT display object (no pygame init yet — that happens on the display thread)."""
        try:
            from crt_display import CRTDisplay
            self.display = CRTDisplay()
            logger.info("CRT display initialized")
        except Exception as e:
            logger.warning(f"CRT display not available: {e}")
            self.display = None

    def _init_spotify_display(self):
        """Prepare Spotify touch display (Waveshare 1.9" LCD)."""
        try:
            from spotify_display import SpotifyDisplay
            self.spotify_display = SpotifyDisplay(self.config, self.spotify)
            logger.info("Spotify display initialized")
        except Exception as e:
            logger.warning(f"Spotify display not available: {e}")
            self.spotify_display = None

    def _spotify_display_thread_main(self):
        """Run Spotify display on its own thread."""
        try:
            self.spotify_display.init_display()
            self.spotify_display.run_daemon()
        except Exception as e:
            logger.error(f"Spotify display thread error: {e}")

    def _start_spotify_display_daemon(self):
        """Start Spotify touch display in a background thread."""
        if self.spotify_display is None:
            return
        self.spotify_display_thread = threading.Thread(
            target=self._spotify_display_thread_main, daemon=True)
        self.spotify_display_thread.start()
        logger.info("Spotify display daemon started")

    def _display_thread_main(self):
        """Run all pygame operations on this single thread (EGL requires it)."""
        try:
            self.display.init_display()
            self.display.show_boot()
            self.display.run_daemon()
        except Exception as e:
            logger.error(f"CRT display thread error: {e}")

    DISPLAY_MODE_FILE = "/tmp/vivian_display_mode"

    def _display_switch_poller(self):
        """Background thread: poll rotary switch and write mode to shared file."""
        last_mode = -1
        while True:
            try:
                mode = self.hardware.get_display_mode()
                if mode != last_mode:
                    last_mode = mode
                    tmp = self.DISPLAY_MODE_FILE + ".tmp"
                    with open(tmp, 'w') as f:
                        f.write(str(mode))
                    os.replace(tmp, self.DISPLAY_MODE_FILE)
            except Exception:
                pass
            import time
            time.sleep(0.1)

    def _start_display_daemon(self):
        """Start CRT display daemon in a background thread."""
        if self.display is None:
            return

        # Start rotary switch poller
        threading.Thread(target=self._display_switch_poller, daemon=True).start()

        self.display_thread = threading.Thread(target=self._display_thread_main, daemon=True)
        self.display_thread.start()
        logger.info("CRT display daemon started in background thread")

    ASSISTANT_SHARED_FILE = "/tmp/vivian_assistant.json"

    def _set_assistant_state(self, state: str, transcript: str = ""):
        """Write assistant state to shared file for CRT display overlay."""
        data = {"state": state, "transcript": transcript}
        try:
            tmp = self.ASSISTANT_SHARED_FILE + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, self.ASSISTANT_SHARED_FILE)
        except Exception as e:
            logger.debug(f"Could not write assistant state: {e}")

    def _stop_display_daemon(self):
        """Stop CRT display daemon and clear display."""
        if self.display is not None:
            try:
                self.display.stop()
                if self.display_thread is not None:
                    self.display_thread.join(timeout=2.0)
                self.display.clear()
                logger.info("CRT display daemon stopped")
            except Exception as e:
                logger.warning(f"Error stopping CRT display: {e}")

    def _check_sentry_request(self):
        """Check if sentry mode was requested via Glass (file-based signal).
        Returns self.run_sentry_mode if requested, None otherwise.
        Throttled to check at most once per second.
        """
        now = time.time()
        if now - getattr(self, '_last_sentry_check', 0) < 1.0:
            return None
        self._last_sentry_check = now

        request_path = '/tmp/vivian_sentry_request'
        if os.path.exists(request_path):
            try:
                os.remove(request_path)
            except OSError:
                pass
            logger.info("Sentry mode requested via Glass")
            return self.run_sentry_mode
        return None

    def _resume_music_async(self):
        """Resume Spotify in the background, after any in-flight pause.

        soft_pause() now runs off the interaction path so it can't delay the
        mic opening, which means on a slow uplink it may still be in flight
        when we want to resume. Joining it first keeps pause->resume ordered;
        without this a slow pause could land *after* the resume and leave the
        music stopped for the rest of the drive.
        """
        pause_thread = getattr(self, '_pause_thread', None)

        def _wait_then_resume():
            if pause_thread is not None and pause_thread.is_alive():
                pause_thread.join(timeout=15)
            self.spotify.soft_resume()

        threading.Thread(target=_wait_then_resume, daemon=True).start()

    def run_assistant_interaction(self):
        """
        Run a single assistant interaction cycle:
        1. Record audio
        2. Transcribe speech to text
        3. Process with assistant
        4. Respond with TTS
        """
        try:
            logger.info("-" * 60)
            logger.info("Starting assistant interaction")
            logger.info("-" * 60)

            # Mute stereo output instantly, then soft-pause Spotify in background
            # (saves song position without waiting for Spotify API).
            # The thread is load-bearing: soft_pause() makes two Spotify Web API
            # round-trips, and in a weak-signal zone spotipy's retries can burn
            # tens of seconds. Run inline it delayed opening the mic, so the
            # front of the user's sentence was lost — or the whole utterance,
            # ending in "No speech detected".
            logger.info("Muting stereo for interaction...")
            self.audio_mute.mute()
            self._pause_thread = threading.Thread(
                target=self.spotify.soft_pause, daemon=True)
            self._pause_thread.start()

            # Set listening mode — CRT shows LISTENING only after music is paused
            self.hardware.set_listen_mode()
            self._set_assistant_state('listening')

            # Record audio
            logger.info("Recording user input...")
            audio_path = self.audio_recorder.record()

            # Transcribe
            logger.info("Transcribing audio...")
            user_text = self.audio_transcriber.transcribe(audio_path)

            if not user_text or user_text.strip() == "":
                logger.warning("No speech detected in recording")

                # Unmute stereo and resume music in background
                logger.info("No speech detected, unmuting stereo")
                self._set_assistant_state('idle')
                self.audio_mute.unmute()
                self._resume_music_async()

                self.hardware.set_wake_mode()
                return

            # Show transcribed text on CRT
            self._set_assistant_state('transcribed', user_text)

            # Set speaking mode
            self.hardware.set_speak_mode()
            self._set_assistant_state('thinking', user_text)

            # Process with assistant (GPT + command handling)
            logger.info("Processing with assistant...")
            self._set_assistant_state('speaking', user_text)
            self.assistant.process_request(user_text)

            # Check if assistant requested sentry mode
            if self.assistant.sentry_requested:
                self.assistant.sentry_requested = False
                # Unmute stereo before entering sentry (sentry uses internal speakers)
                self.audio_mute.unmute()
                self.hardware.set_wake_mode()
                self.run_sentry_mode()
                return

            # Unmute stereo and resume music in background
            logger.info("Unmuting stereo after interaction")
            self.audio_mute.unmute()
            self._resume_music_async()

            logger.info("-" * 60)
            logger.info("Interaction complete")
            logger.info("-" * 60)
            
        except KeyboardInterrupt:
            logger.info("Interaction interrupted by user")
            raise
        
        except Exception as e:
            logger.error(f"Interaction error: {e}", exc_info=True)

            error_msg = str(e)
            is_rate_limit = '429' in error_msg

            if is_rate_limit:
                # Show on CRT only — TTS will also be rate-limited
                self._set_assistant_state('error', "Rate limit exceeded. Please try again later.")
                logger.warning("Rate limit hit — displaying error on CRT, skipping TTS")
                time.sleep(4)  # Hold the error on screen
            else:
                # Show on CRT and attempt TTS
                self._set_assistant_state('error', "An error occurred. Please try again.")
                try:
                    self.hardware.set_speak_mode()
                    self.tts.speak("I encountered an error. Please try again.")
                except Exception as tts_error:
                    logger.error(f"Could not speak error message: {tts_error}")
                    time.sleep(4)

        finally:
            # Always unmute stereo, resume music if appropriate, and return to wake mode
            self._set_assistant_state('idle')
            if self.audio_mute.ensure_unmuted():
                self._resume_music_async()
            self.hardware.set_wake_mode()
    
    def run_sentry_mode(self):
        """Enter sentry mode, blocking until known person returns or manual stop."""
        if self.sentry_controller is None:
            logger.warning("Sentry mode requested but no sentry controller configured")
            self.tts.speak("Sentry mode is not configured.")
            return

        logger.info("=" * 60)
        logger.info("Entering sentry mode")
        logger.info("=" * 60)

        # Save music state and pause before sentry
        pre_sentry_music = self.spotify.load_state()
        if pre_sentry_music:
            logger.info("Pausing music for sentry mode")
            self.spotify.soft_pause()

        # Turn off all LEDs during sentry mode (VIVIAN is not listening)
        self.hardware.all_off()

        # Sentry blocks until a known person returns — potentially hours — which
        # is far past the wake-callback deadline. Exempt it, since this runs
        # inside the wake callback (via both the spoken and Glass entry paths).
        self.wake_detector.suspend_callback_watchdog()
        try:
            result = self.sentry_controller.activate()
        finally:
            self.wake_detector.resume_callback_watchdog()

        logger.info(f"Sentry mode ended: {result}")

        if result['reason'] == 'known_person':
            names = result.get('known_names') or []
            who = f" ({', '.join(names)} recognized)" if names else ""
            self.tts.speak("Sentry mode deactivated.")
            mem_note = (f"[Sentry mode is now OFF and no longer active — it stood down "
                        f"when a known person returned{who}.]")
        elif result['reason'] == 'manual_stop':
            self.tts.speak("Sentry mode deactivated.")
            mem_note = "[Sentry mode is now OFF and no longer active — it was manually stopped.]"
        else:
            self.tts.speak("Sentry mode encountered an error and has been deactivated.")
            mem_note = "[Sentry mode is now OFF and no longer active — it stopped due to an error.]"

        # Record the deactivation in conversation memory so the assistant knows
        # sentry is no longer running. Without this the model only ever sees the
        # "[Sentry: Activated]" turn and treats a later "activate sentry" request
        # as redundant (it assumes sentry is still engaged from the first time).
        try:
            self.assistant.memory.add_interaction("(sentry mode ended)", mem_note)
        except Exception as e:
            logger.debug(f"Could not record sentry deactivation in memory: {e}")

        # Restore music if it was playing before sentry
        if pre_sentry_music:
            logger.info("Resuming music after sentry mode")
            self._resume_music_async()

        # Restore wake LED
        self.hardware.set_wake_mode()

        logger.info("=" * 60)
        logger.info("Returning to wake word listening")
        logger.info("=" * 60)

    def start(self):
        """Start the main VIVIAN loop"""
        logger.info("")
        logger.info("=" * 60)
        logger.info("VIVIAN is now active and listening...")
        logger.info("Say 'Hey VIVIAN' or press the button to interact")
        logger.info("Press Ctrl+C to exit")
        logger.info("=" * 60)
        logger.info("")

        # Start CRT display daemon (boot screen + background display loop)
        self._start_display_daemon()

        # Start Spotify touch display daemon
        self._start_spotify_display_daemon()

        # Start Glass HUD server if enabled
        if self.glass_server:
            self.glass_server.start()

        # Start Dashboard server if enabled
        if self.dashboard_server:
            self.dashboard_server.start()

        try:
            # Start wake word detection loop
            # This will call run_assistant_interaction() when wake word/button detected
            # poll_callback checks for Glass sentry requests each audio frame
            self.wake_detector.listen(
                on_wake_callback=self.run_assistant_interaction,
                poll_callback=self._check_sentry_request
            )
            
        except KeyboardInterrupt:
            logger.info("")
            logger.info("=" * 60)
            logger.info("Shutdown requested by user")
            logger.info("=" * 60)
        
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources before exit"""
        logger.info("Cleaning up resources...")

        try:
            # Stop Glass HUD server
            if self.glass_server:
                self.glass_server.stop()

            # Stop Dashboard server
            if self.dashboard_server:
                self.dashboard_server.stop()

            # Stop CRT display daemon
            self._stop_display_daemon()

            # Stop Spotify touch display
            if self.spotify_display:
                try:
                    self.spotify_display.stop()
                    if self.spotify_display_thread:
                        self.spotify_display_thread.join(timeout=2.0)
                    logger.info("Spotify display stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Spotify display: {e}")

            # Ensure stereo is unmuted before exit
            self.audio_mute.ensure_unmuted()

            # Stop wake word detector
            self.wake_detector.stop()

            # Stop GPS if running
            if hasattr(self.system_info, 'gps') and self.system_info.gps:
                self.system_info.gps.stop()

            # Turn off all LEDs
            self.hardware.cleanup()

            logger.info("Cleanup complete")
            logger.info("=" * 60)
            logger.info("VIVIAN shutdown complete. Goodbye!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Entry point for VIVIAN"""
    try:
        # Create and start VIVIAN
        vivian = VIVIAN()
        vivian.start()
        
    except FileNotFoundError as e:
        logger.error(f"Configuration or required file not found: {e}")
        logger.error("Please ensure config.yaml exists and all paths are correct")
        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
