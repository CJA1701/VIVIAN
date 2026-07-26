#!/usr/bin/env python3
"""LCD test using Waveshare's proven init sequence, adapted for VIVIAN pin mapping."""

import time
import spidev
import numpy as np
from gpiozero import DigitalOutputDevice, PWMOutputDevice

# VIVIAN pin mapping (avoids GPIO 17, 18, 27, 22 used by rotary switch)
RST_PIN = 24
DC_PIN = 25
BL_PIN = 12

WIDTH = 170
HEIGHT = 320


class LCD:
    def __init__(self):
        self.rst = DigitalOutputDevice(RST_PIN, active_high=True, initial_value=True)
        self.dc = DigitalOutputDevice(DC_PIN, active_high=True, initial_value=True)
        self.bl = PWMOutputDevice(BL_PIN, frequency=1000)
        self.bl.value = 0

        self.spi = spidev.SpiDev(0, 0)
        self.spi.max_speed_hz = 40_000_000
        self.spi.mode = 0

        self._init_display()

    def _command(self, cmd):
        self.dc.off()
        self.spi.writebytes([cmd])

    def _data(self, val):
        self.dc.on()
        self.spi.writebytes([val])

    def _reset(self):
        self.rst.on()
        time.sleep(0.01)
        self.rst.off()
        time.sleep(0.01)
        self.rst.on()
        time.sleep(0.01)

    def _init_display(self):
        """Waveshare's exact init sequence for ST7789V2 1.9" panel."""
        self._reset()

        self._command(0x11)       # Sleep out
        time.sleep(0.12)

        self._command(0x36)       # MADCTL
        self._data(0x08)

        self._command(0x3A)       # Pixel format
        self._data(0x05)          # 16-bit RGB565

        self._command(0xF0)       # Command Set Control
        self._data(0xC3)
        self._command(0xF0)
        self._data(0x96)

        self._command(0xB4)
        self._data(0x01)
        self._command(0xB7)
        self._data(0xC6)

        self._command(0xC0)
        self._data(0x80)
        self._data(0x45)

        self._command(0xC1)
        self._data(0x13)

        self._command(0xC2)
        self._data(0xA7)

        self._command(0xC5)
        self._data(0x0A)

        self._command(0xF0)
        self._data(0x3C)
        self._command(0xF0)
        self._data(0x69)

        self._command(0x21)       # Inversion on
        self._command(0x11)       # Sleep out (again)
        time.sleep(0.1)
        self._command(0x29)       # Display on

        print("Display initialized")

    def set_window(self, x0, y0, x1, y1):
        """Set pixel write window (portrait mode, with 35px X offset)."""
        x0 += 35
        x1 += 35
        self._command(0x2A)
        self._data(x0 >> 8)
        self._data(x0 & 0xFF)
        self._data(x1 >> 8)
        self._data(x1 & 0xFF)
        self._command(0x2B)
        self._data(y0 >> 8)
        self._data(y0 & 0xFF)
        self._data(y1 >> 8)
        self._data(y1 & 0xFF)
        self._command(0x2C)

    def fill(self, r, g, b):
        """Fill screen with a solid color."""
        r5 = (r >> 3) & 0x1F
        g6 = (g >> 2) & 0x3F
        b5 = (b >> 3) & 0x1F
        pixel = (r5 << 11) | (g6 << 5) | b5
        hi = (pixel >> 8) & 0xFF
        lo = pixel & 0xFF

        self.set_window(0, 0, WIDTH - 1, HEIGHT - 1)
        buf = bytes([hi, lo]) * (WIDTH * HEIGHT)

        self.dc.on()
        for i in range(0, len(buf), 4096):
            self.spi.writebytes2(buf[i:i + 4096])

    def backlight(self, percent):
        self.bl.value = percent / 100

    def close(self):
        self.bl.value = 0
        self.spi.close()


def main():
    lcd = LCD()
    lcd.backlight(80)
    print("Backlight ON")

    colors = [
        ("RED",           255, 0,   0),
        ("GREEN",         0,   255, 0),
        ("BLUE",          0,   0,   255),
        ("SPOTIFY GREEN", 29,  185, 84),
        ("WHITE",         255, 255, 255),
        ("BLACK",         0,   0,   0),
    ]

    print("Cycling colors — 2 seconds each. Ctrl+C to stop.\n")
    try:
        for name, r, g, b in colors:
            print(f"  {name}")
            lcd.fill(r, g, b)
            time.sleep(2)

        print("\nDone! Press Ctrl+C to exit.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
        lcd.close()
        print("Done.")


if __name__ == "__main__":
    main()
