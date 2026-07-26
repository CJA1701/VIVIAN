from gpiozero import Button
import logging

logger = logging.getLogger(__name__)

# Rotary switch position -> mode value returned by get_display_mode().
# crt_display.py maps these values to actual pages (mode_page dict in
# run_daemon); the resulting page for each position is noted here:
DISPLAY_MODE_CYCLE = 0        # Position 1: no pin grounded — shows MAIN page
DISPLAY_MODE_NAV = 1          # Position 2: GPIO 17 LOW — navigation
DISPLAY_MODE_WEATHER = 2      # Position 3: GPIO 18 LOW — weather
DISPLAY_MODE_DIAGNOSTICS = 3  # Position 4: GPIO 27 LOW — diagnostics
DISPLAY_MODE_UNUSED = 4       # Position 5: GPIO 22 LOW — auto-cycles pages


class HardwareController:
    """Manage button and rotary display switch"""

    def __init__(self, config):
        self.config = config

        # Initialize button
        self.button = Button(
            config.hardware['button_pin'],
            pull_up=True,
            bounce_time=0.05
        )

        # Initialize rotary switch inputs (active LOW with internal pull-ups)
        # RS1010: common pin → GND, position pins → GPIO
        switch_pins = config.hardware.get('display_switch_pins', {})
        self._switch_buttons = {}
        self._switch_order = ['nav', 'weather', 'diagnostics', 'unused']
        for name in self._switch_order:
            pin = switch_pins.get(name)
            if pin is not None:
                self._switch_buttons[name] = Button(pin, pull_up=True, bounce_time=0.02)

        logger.info(f"Hardware initialized (display switch: {len(self._switch_buttons)} positions)")

    def get_display_mode(self) -> int:
        """Read rotary switch position. Returns DISPLAY_MODE_* constant.

        Position 1 = all pins HIGH (pull-up, no connection) → CYCLE
        Position 2-5 = one pin pulled LOW → specific page
        """
        for i, name in enumerate(self._switch_order):
            btn = self._switch_buttons.get(name)
            if btn is not None and btn.is_pressed:  # is_pressed = pin LOW (grounded)
                return i + 1  # 1=nav, 2=weather, 3=diagnostics, 4=unused
        return DISPLAY_MODE_CYCLE

    # Legacy LED methods — now no-ops since LEDs replaced by CRT status
    def set_wake_mode(self):
        pass

    def set_listen_mode(self):
        pass

    def set_speak_mode(self):
        pass

    def all_off(self):
        pass

    def is_button_pressed(self) -> bool:
        """Check if button is pressed"""
        return self.button.is_pressed

    def cleanup(self):
        """Cleanup GPIO resources"""
        logger.info("Hardware cleaned up")
