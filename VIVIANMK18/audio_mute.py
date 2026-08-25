"""
Audio mute controller for VIVIAN
Mutes/unmutes the car stereo output while keeping Spotify playing
"""
import subprocess
import logging
import os
import re

# asound.conf declares the 'Stereo' softvol on this card, and udev pins the
# name to a fixed USB port. Names are stable; card numbers are not.
STEREO_CARD_NAME = "Device_1"

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
        """Resolve which ALSA card carries the softvol control named by
        `control_name` (the car-stereo output).

        Returns a card NAME, not a number. asound.conf declares the control on
        `card Device_1`, and udev pins that name to a physical USB port, so the
        name is the stable identifier — card numbers are enumeration order.

        This used to scan /proc/asound/cards for the literal string
        'usb-xhci-hcd.1-1'. The real string is 'usb-xhci-hcd.0-1.1', so it never
        matched and silently fell back to card 0 — which has its own unrelated
        softvol control of the same name. amixer therefore SUCCEEDED, mute()
        logged "Stereo muted", and the music kept playing: the wrong control was
        being turned down. Music only stopped when the Spotify API pause landed
        a second or two later, which is exactly the bug this fixes.
        """
        # 1. Cache written by vivian-audio-detect.service, when it works.
        try:
            if os.path.exists('/tmp/vivian_stereo_card'):
                with open('/tmp/vivian_stereo_card') as f:
                    card = f.read().strip()
                if card and self._control_exists(card):
                    logger.info(f"Stereo card from cache: {card}")
                    return card
        except OSError:
            pass

        # 2. The expected name, verified by actually probing for the control.
        if self._control_exists(STEREO_CARD_NAME):
            logger.info(f"Stereo card: {STEREO_CARD_NAME}")
            return STEREO_CARD_NAME

        # 3. Last resort: ask every card which one really has the control,
        #    rather than assuming and muting something inaudible.
        try:
            with open('/proc/asound/cards') as f:
                for line in f:
                    m = re.match(r"\s*(\d+)\s+\[(\S+)\s*\]", line)
                    if m and self._control_exists(m.group(2)):
                        logger.warning(
                            f"{STEREO_CARD_NAME} unavailable; using card "
                            f"'{m.group(2)}' which does have a '{self.control_name}' control"
                        )
                        return m.group(2)
        except OSError as e:
            logger.error(f"Could not read /proc/asound/cards: {e}")

        logger.error(
            f"No card exposes a '{self.control_name}' control — muting will not "
            f"work, so music will bleed into recordings. Check /etc/asound.conf."
        )
        return STEREO_CARD_NAME

    def _control_exists(self, card) -> bool:
        """True if `control_name` exists on this card. Probing beats guessing:
        the old code's assumption is precisely what broke muting."""
        try:
            r = subprocess.run(["amixer", "-c", str(card), "sget", self.control_name],
                               capture_output=True, timeout=3)
            return r.returncode == 0
        except Exception:
            return False

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
