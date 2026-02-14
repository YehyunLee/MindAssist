#!/usr/bin/env python3
"""
Forget and re-pair the MindWave Mobile Bluetooth device on macOS.
Requires: blueutil (brew install blueutil)
"""

import os
import subprocess
import sys
import time
import re
import json

DEVICE_NAME = "MindWave Mobile"
PIN = "0000"
KNOWN_MAC = None  # will be set dynamically from paired devices


def run(cmd):
    """Run a shell command and return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def check_blueutil():
    """Ensure blueutil is installed."""
    _, _, rc = run(["which", "blueutil"])
    if rc != 0:
        raise RuntimeError(
            "blueutil not found. Install it with: brew install blueutil"
        )


def find_device_mac():
    """Find the MAC address of MindWave Mobile from paired or nearby devices."""
    # Check paired devices first (use JSON for reliable parsing)
    out, _, _ = run(["blueutil", "--paired", "--format", "json"])
    try:
        devices = json.loads(out) if out else []
        for dev in devices:
            if DEVICE_NAME in dev.get("name", ""):
                return dev["address"]
    except json.JSONDecodeError:
        pass

    # Also check system profiler for any known device
    out, _, _ = run(["system_profiler", "SPBluetoothDataType"])
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if DEVICE_NAME in line:
            # Look for address in nearby lines
            for j in range(i, min(i + 10, len(lines))):
                match = re.search(r"Address:\s*([0-9a-fA-F:-]+)", lines[j])
                if match:
                    return match.group(1)
    return None


def main():
    check_blueutil()

    # Make sure Bluetooth is on
    run(["blueutil", "--power", "1"])
    print("Bluetooth is ON.")

    # Find device
    print(f"Looking for '{DEVICE_NAME}'...")
    mac = find_device_mac()

    if mac:
        print(f"Found device: {mac}")
        # Unpair / forget the device
        print(f"Forgetting '{DEVICE_NAME}' ({mac})...")
        out, err, rc = run(["blueutil", "--unpair", mac])
        if rc == 0:
            print("Device forgotten successfully.")
        else:
            print(f"Unpair returned code {rc}: {err}")
    else:
        print(f"'{DEVICE_NAME}' not currently paired. Will scan for it.")

    # Wait a moment before re-pairing
    print("Waiting 3 seconds before scanning & re-pairing...")
    time.sleep(3)

    # If we don't have a MAC yet, do an inquiry scan
    if not mac:
        print("Scanning for nearby Bluetooth devices (30s)...")
        out, _, _ = run(["blueutil", "--inquiry", "30", "--format", "json"])
        print(f"Scan results: {out}")
        try:
            devices = json.loads(out) if out else []
            for dev in devices:
                if DEVICE_NAME in dev.get("name", ""):
                    mac = dev["address"]
                    print(f"Found '{DEVICE_NAME}' at {mac}")
                    break
        except json.JSONDecodeError:
            pass

    if not mac:
        # Last resort: check recent/nearby from system profiler
        out, _, _ = run(["system_profiler", "SPBluetoothDataType"])
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if DEVICE_NAME in line:
                for j in range(i, min(i + 10, len(lines))):
                    match = re.search(r"Address:\s*([0-9a-fA-F:-]+)", lines[j])
                    if match:
                        mac = match.group(1)
                        break
                if mac:
                    break

    if not mac:
        raise RuntimeError(
            f"Could not find '{DEVICE_NAME}'. Make sure it's turned on and in range."
        )

    # Pair the device with retries
    MAX_PAIR_ATTEMPTS = 5
    paired = False
    for attempt in range(1, MAX_PAIR_ATTEMPTS + 1):
        print(f"Pairing attempt {attempt}/{MAX_PAIR_ATTEMPTS} with '{DEVICE_NAME}' ({mac})...")
        out, err, rc = run(["blueutil", "--pair", mac, PIN])
        if rc == 0:
            print("Paired successfully!")
            paired = True
            break

        # Fallback: pipe PIN via stdin
        result = subprocess.run(
            ["blueutil", "--pair", mac],
            input=PIN + "\n", capture_output=True, text=True
        )
        if result.returncode == 0:
            print("Paired successfully!")
            paired = True
            break

        print(f"  Attempt {attempt} failed: {err or result.stderr.strip()}")
        if attempt < MAX_PAIR_ATTEMPTS:
            print("  Cycling Bluetooth and retrying...")
            run(["blueutil", "--power", "0"])
            time.sleep(2)
            run(["blueutil", "--power", "1"])
            time.sleep(3)

    if not paired:
        raise RuntimeError(
            f"Pairing failed after {MAX_PAIR_ATTEMPTS} attempts. "
            f"Make sure '{DEVICE_NAME}' is on and in range."
        )

    # Connect (may need two rounds on macOS to fully establish serial port)
    for c in range(1, 4):
        print(f"Connecting (attempt {c}/3)...")
        time.sleep(2)
        out, err, rc = run(["blueutil", "--connect", mac])
        if rc == 0:
            print(f"Connected to '{DEVICE_NAME}'!")
        else:
            print(f"Connect returned code {rc}: {err}")

        # Check if the serial port appeared
        time.sleep(2)
        if os.path.exists("/dev/tty.MindWaveMobile"):
            print("Serial port /dev/tty.MindWaveMobile is ready!")
            break
        else:
            print("  Serial port not yet available, retrying connect...")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
