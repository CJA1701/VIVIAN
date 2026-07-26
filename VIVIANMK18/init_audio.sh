#!/bin/bash
# Initialize ALSA mixer volumes before VIVIAN starts.
# Ensures consistent levels across reboots.
# Uses card names (not numbers) for stability with USB hub enumeration.

# Wait for USB audio devices to enumerate
sleep 3

# --- Proactive USB mic reset -------------------------------------------------
# The car cuts Pi power at shutoff, so every start is a COLD boot, and the
# GeneralPlus USB mic dongle reliably wedges its capture stream on that power
# transient. A wedged stream returns data the wake-word watchdog can't tell
# apart from live mic (not zero, not frozen, not a blocked read), so it never
# recovers and VIVIAN stops hearing "Hey Vivian" until a manual restart.
# Resetting the dongle(s) here — before main.py opens the mic — forces clean
# re-enumeration every boot. Same VID:PID + usbreset as wake_word.py's reactive
# reset (1b3f:2008).
for d in /sys/bus/usb/devices/*; do
    [ -f "$d/idVendor" ] || continue
    [ "$(cat "$d/idVendor" 2>/dev/null)" = "1b3f" ] || continue
    [ "$(cat "$d/idProduct" 2>/dev/null)" = "2008" ] || continue
    bus=$(cat "$d/busnum" 2>/dev/null)
    dev=$(cat "$d/devnum" 2>/dev/null)
    [ -n "$bus" ] && [ -n "$dev" ] || continue
    node=$(printf "/dev/bus/usb/%03d/%03d" "$bus" "$dev")
    echo "init_audio: USB-resetting mic dongle $node"
    sudo -n /usr/bin/usbreset "$node" 2>/dev/null || echo "init_audio: usbreset failed for $node"
done
# Let the dongle(s) re-enumerate before mixer setup / VIVIAN opening the mic
sleep 3

# Unbind SPI5 (spidev10.0) which steals GPIO 12 from the Spotify LCD backlight.
# Needs root: this is an ExecStartPre without a '+' prefix, so it runs as the
# service user and a plain redirect into sysfs is EACCES — which was silently
# swallowed, leaving GPIO 12 claimed and the LCD backlight off every boot.
# Same sudo -n (NOPASSWD) pattern as the usbreset above; log failures instead
# of hiding them.
if [ -e /sys/bus/spi/drivers/spidev/spi10.0 ]; then
    if sudo -n /bin/sh -c 'echo spi10.0 > /sys/bus/spi/drivers/spidev/unbind'; then
        echo "init_audio: unbound spidev spi10.0 (freed GPIO 12)"
    else
        echo "init_audio: spidev unbind of spi10.0 FAILED — LCD backlight GPIO 12 stays claimed"
    fi
else
    echo "init_audio: spidev spi10.0 not bound — nothing to unbind"
fi

# Clear backlight flag so it gets re-triggered on fresh boot
rm -f /tmp/vivian_bl_on

# Internal speakers (card "Device") — 75%
amixer -c Device sset Speaker 75% on || true

# Stereo DAC (card "Device_1") — Speaker to 100%
amixer -c Device_1 sset Speaker 100% on || true

# Stereo softvol control to 100% (unmuted)
amixer -c Device_1 sset Stereo 100% || true

# Always exit success so VIVIAN starts regardless
exit 0
