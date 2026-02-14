#!/usr/bin/env python3
"""
Forget and re-pair the MindWave Mobile Bluetooth device on macOS.
Requires: blueutil (brew install blueutil)
"""

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

    # Pair the device (syntax: --pair MAC PIN)
    print(f"Pairing with '{DEVICE_NAME}' ({mac}) using PIN {PIN}...")
    out, err, rc = run(["blueutil", "--pair", mac, PIN])
    if rc == 0:
        print("Paired successfully!")
    else:
        # Fallback: pipe PIN via stdin for interactive prompt
        print(f"Direct pair failed ({err}), trying with stdin PIN...")
        result = subprocess.run(
            ["blueutil", "--pair", mac],
            input=PIN + "\n", capture_output=True, text=True
        )
        if result.returncode == 0:
            print("Paired successfully!")
        else:
            raise RuntimeError(
                f"Pairing failed (code {result.returncode}): {result.stderr.strip()}"
            )

    # Connect
    print("Connecting...")
    time.sleep(1)
    out, err, rc = run(["blueutil", "--connect", mac])
    if rc == 0:
        print(f"Connected to '{DEVICE_NAME}'!")
    else:
        print(f"Connect returned code {rc}: {err}")
        print("You may need to connect manually or run your EEG script.")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
