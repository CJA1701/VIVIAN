"""
Audio mute controller for VIVIAN
Mutes/unmutes the car stereo output while keeping Spotify playing
"""
import subprocess
import logging
import os

logger = logging.getLogger(__name__)


class AudioMuteController:
    """Control stereo muting via ALSA software volume"""

    def __init__(self, control_name="Stereo", card=None):
        """
        Initialize mute controller

        Args:
            control_name: ALSA control name (from asound.conf)
            card: Sound card number/name for the stereo (auto-detected if None)
        """
        self.control_name = control_name

        # Auto-detect stereo card if not specified
        if card is None:
            card = self._detect_stereo_card()

        self.card = str(card)
        self.is_muted = False
        logger.info(f"Audio mute controller initialized (control: {control_name}, card: {self.card})")

    def _detect_stereo_card(self):
        """Detect which card is the stereo based on USB port"""
        try:
            # Try to read from detection script output
            if os.path.exists('/tmp/vivian_stereo_card'):
                with open('/tmp/vivian_stereo_card', 'r') as f:
                    card = f.read().strip()
                    if card:
                        logger.info(f"Detected stereo card from cache: {card}")
                        return card

            # Fall back to detecting from /proc/asound/cards
            with open('/proc/asound/cards', 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    # Look for USB port 3-1 (stereo)
                    if 'usb-xhci-hcd.1-1' in line:
                        # Previous line has card number
                        if i > 0:
                            card_num = lines[i-1].strip().split()[0]
                            logger.info(f"Detected stereo card from USB port: {card_num}")
                            return card_num

            # Default fallback
            logger.warning("Could not detect stereo card, using card 0")
            return "0"

        except Exception as e:
            logger.error(f"Error detecting stereo card: {e}")
            return "0"

    def mute(self):
        """Mute the car stereo output (Spotify keeps playing)"""
        if self.is_muted:
            logger.debug("Stereo already muted")
            return True

        try:
            # Set volume to 0% using amixer
            cmd = ["amixer", "-c", self.card, "sset", self.control_name, "0%"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)

            if result.returncode == 0:
                self.is_muted = True
                logger.info("Stereo muted")
                return True
            else:
                logger.warning(f"Failed to mute stereo: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Mute command timed out")
            return False
        except Exception as e:
            logger.error(f"Error muting stereo: {e}")
            return False

    def unmute(self):
        """Unmute the car stereo output"""
        if not self.is_muted:
            logger.debug("Stereo already unmuted")
            return True

        try:
            # Set volume to 100% using amixer
            cmd = ["amixer", "-c", self.card, "sset", self.control_name, "100%"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)

            if result.returncode == 0:
                self.is_muted = False
                logger.info("Stereo unmuted")
                return True
            else:
                logger.warning(f"Failed to unmute stereo: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Unmute command timed out")
            return False
        except Exception as e:
            logger.error(f"Error unmuting stereo: {e}")
            return False

    def ensure_unmuted(self):
        """Ensure stereo is unmuted (for cleanup/error recovery)"""
        if self.is_muted:
            return self.unmute()
        return True
