#!/usr/bin/env python3
"""Touch calibration — one target at a time, computes correct transform."""

import time
import spidev
import smbus2
import numpy as np
from gpiozero import DigitalOutputDevice, PWMOutputDevice
from PIL import Image, ImageDraw

RST_PIN = 24
DC_PIN = 25
BL_PIN = 12
TP_RST_PIN = 16
CST816_ADDR = 0x15

PANEL_W = 170
PANEL_H = 320
X_OFFSET = 35
SW = 320
SH = 170


class Display:
    def __init__(self):
        self.dc = DigitalOutputDevice(DC_PIN, active_high=True, initial_value=True)
        self.rst = DigitalOutputDevice(RST_PIN, active_high=True, initial_value=True)
        self.bl = PWMOutputDevice(BL_PIN, frequency=1000)
        self.bl.value = 0
        self.spi = spidev.SpiDev(0, 0)
        self.spi.max_speed_hz = 40_000_000
        self.spi.mode = 0
        self._init_hw()
        self.bl.value = 0.8

    def _cmd(self, c):
        self.dc.off(); self.spi.writebytes([c])

    def _dat(self, v):
        self.dc.on(); self.spi.writebytes([v])

    def _init_hw(self):
        self.rst.on(); time.sleep(0.01)
        self.rst.off(); time.sleep(0.01)
        self.rst.on(); time.sleep(0.01)
        self._cmd(0x11); time.sleep(0.12)
        self._cmd(0x36); self._dat(0x08)
        self._cmd(0x3A); self._dat(0x05)
        self._cmd(0xF0); self._dat(0xC3)
        self._cmd(0xF0); self._dat(0x96)
        self._cmd(0xB4); self._dat(0x01)
        self._cmd(0xB7); self._dat(0xC6)
        self._cmd(0xC0); self._dat(0x80); self._dat(0x45)
        self._cmd(0xC1); self._dat(0x13)
        self._cmd(0xC2); self._dat(0xA7)
        self._cmd(0xC5); self._dat(0x0A)
        self._cmd(0xF0); self._dat(0x3C)
        self._cmd(0xF0); self._dat(0x69)
        self._cmd(0x21); self._cmd(0x11); time.sleep(0.1)
        self._cmd(0x29)

    def show(self, img):
        port = img.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(port, dtype=np.uint16)
        r = (arr[:, :, 0] >> 3).astype(np.uint16)
        g = (arr[:, :, 1] >> 2).astype(np.uint16)
        b = (arr[:, :, 2] >> 3).astype(np.uint16)
        data = ((r << 11) | (g << 5) | b).tobytes()
        x0, x1 = X_OFFSET, X_OFFSET + PANEL_W - 1
        self._cmd(0x2A)
        self._dat(x0 >> 8); self._dat(x0 & 0xFF)
        self._dat(x1 >> 8); self._dat(x1 & 0xFF)
        self._cmd(0x2B)
        self._dat(0); self._dat(0)
        self._dat((PANEL_H - 1) >> 8); self._dat((PANEL_H - 1) & 0xFF)
        self._cmd(0x2C)
        self.dc.on()
        for i in range(0, len(data), 4096):
            self.spi.writebytes2(data[i:i + 4096])

    def close(self):
        self.bl.value = 0; self.spi.close()


class Touch:
    def __init__(self):
        self.bus = smbus2.SMBus(1)
        self.tp_rst = DigitalOutputDevice(TP_RST_PIN, active_high=True, initial_value=True)
        self.tp_rst.off(); time.sleep(0.01)
        self.tp_rst.on(); time.sleep(0.1)

    def wait_tap(self):
        """Block until a single tap, return (raw_x, raw_y)."""
        # Wait for finger down
        while True:
            try:
                buf = self.bus.read_i2c_block_data(CST816_ADDR, 0x01, 6)
                if buf[1] > 0:  # finger count
                    raw_x = ((buf[2] & 0x0F) << 8) | buf[3]
                    raw_y = ((buf[4] & 0x0F) << 8) | buf[5]
                    # Wait for finger up
                    time.sleep(0.3)
                    return raw_x, raw_y
            except Exception:
                pass
            time.sleep(0.02)


def draw_target(tx, ty, label):
    img = Image.new("RGB", (SW, SH), (17, 17, 17))
    draw = ImageDraw.Draw(img)
    draw.line((tx - 12, ty, tx + 12, ty), fill=(255, 255, 0), width=1)
    draw.line((tx, ty - 12, tx, ty + 12), fill=(255, 255, 0), width=1)
    draw.rectangle((tx - 2, ty - 2, tx + 2, ty + 2), fill=(255, 255, 0))
    draw.text((tx + 14, ty - 6), label, fill=(170, 170, 170))
    return img


def main():
    print("Initializing...")
    disp = Display()
    touch = Touch()

    margin = 30
    targets = [
        (margin,      margin,      "1: Top-Left"),
        (SW - margin, margin,      "2: Top-Right"),
        (SW - margin, SH - margin, "3: Bottom-Right"),
        (margin,      SH - margin, "4: Bottom-Left"),
        (SW // 2,     SH // 2,     "5: Center"),
    ]

    readings = []

    print("\nCalibration: tap each crosshair as it appears.\n")

    for i, (tx, ty, label) in enumerate(targets):
        img = draw_target(tx, ty, label)
        disp.show(img)
        print(f"  Tap {label} ...")
        raw_x, raw_y = touch.wait_tap()
        print(f"    screen=({tx},{ty})  raw=({raw_x},{raw_y})")
        readings.append((tx, ty, raw_x, raw_y))

    # Compute affine transform: screen = A * raw + B
    # Using least squares on the 5 points
    screen_pts = np.array([(r[0], r[1]) for r in readings], dtype=np.float64)
    raw_pts = np.array([(r[2], r[3]) for r in readings], dtype=np.float64)

    # Solve for screen_x = a*raw_x + b*raw_y + c
    #         screen_y = d*raw_x + e*raw_y + f
    A = np.column_stack([raw_pts, np.ones(len(raw_pts))])
    coeff_x, _, _, _ = np.linalg.lstsq(A, screen_pts[:, 0], rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(A, screen_pts[:, 1], rcond=None)

    ax, bx, cx = coeff_x
    ay, by, cy = coeff_y

    print(f"\n  Transform coefficients:")
    print(f"    screen_x = {ax:.4f} * raw_x + {bx:.4f} * raw_y + {cx:.4f}")
    print(f"    screen_y = {ay:.4f} * raw_x + {by:.4f} * raw_y + {cy:.4f}")

    # Verify with calibration points
    print(f"\n  Verification:")
    for tx, ty, rx, ry in readings:
        calc_x = ax * rx + bx * ry + cx
        calc_y = ay * rx + by * ry + cy
        err = ((calc_x - tx)**2 + (calc_y - ty)**2) ** 0.5
        print(f"    expected=({tx},{ty})  got=({calc_x:.0f},{calc_y:.0f})  err={err:.1f}px")

    # Save results to file
    with open("/home/cjatkinson/touch_calibration.txt", "w") as f:
        f.write("Touch Calibration Results\n")
        f.write("========================\n\n")
        f.write("Raw readings:\n")
        for tx, ty, rx, ry in readings:
            f.write(f"  screen=({tx},{ty})  raw=({rx},{ry})\n")
        f.write(f"\nTransform coefficients:\n")
        f.write(f"  screen_x = {ax:.4f} * raw_x + {bx:.4f} * raw_y + {cx:.4f}\n")
        f.write(f"  screen_y = {ay:.4f} * raw_x + {by:.4f} * raw_y + {cy:.4f}\n")
        f.write(f"\nVerification:\n")
        for tx, ty, rx, ry in readings:
            calc_x = ax * rx + bx * ry + cx
            calc_y = ay * rx + by * ry + cy
            err = ((calc_x - tx)**2 + (calc_y - ty)**2) ** 0.5
            f.write(f"  expected=({tx},{ty})  got=({calc_x:.0f},{calc_y:.0f})  err={err:.1f}px\n")
    print("  Saved to /home/cjatkinson/touch_calibration.txt")

    # Live test with computed transform
    print(f"\n  Now tap anywhere — red dots should appear where you tap.")
    print(f"  Ctrl+C to exit.\n")

    dots = []
    last = 0
    try:
        while True:
            try:
                buf = touch.bus.read_i2c_block_data(CST816_ADDR, 0x01, 6)
                if buf[1] == 0 and buf[0] == 0:
                    time.sleep(0.02); continue
                now = time.monotonic()
                if now - last < 0.4:
                    time.sleep(0.02); continue
                last = now
                rx = ((buf[2] & 0x0F) << 8) | buf[3]
                ry = ((buf[4] & 0x0F) << 8) | buf[5]
                sx = int(ax * rx + bx * ry + cx)
                sy = int(ay * rx + by * ry + cy)
                sx = max(0, min(SW - 1, sx))
                sy = max(0, min(SH - 1, sy))
                dots.append((sx, sy))
                print(f"  raw=({rx},{ry}) -> screen=({sx},{sy})")

                img = Image.new("RGB", (SW, SH), (17, 17, 17))
                draw = ImageDraw.Draw(img)
                for dx, dy in dots[-20:]:
                    draw.ellipse((dx - 4, dy - 4, dx + 4, dy + 4), fill=(255, 0, 0))
                disp.show(img)
            except Exception:
                pass
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nDone.")
        disp.close()


if __name__ == "__main__":
    main()
