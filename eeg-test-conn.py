import serial
import time
import glob
import signal
import sys
import threading

PORT_PATTERN = '/dev/tty.*Mind*'
BAUD = 57600

ser = None
running = True

def cleanup(*args):
    global ser, running
    running = False
    if ser and ser.is_open:
        ser.close()
        print("\nSerial port closed.")
    sys.exit(0)

def input_listener():
    """Background thread: type 'q' + Enter to quit gracefully."""
    global running
    while running:
        try:
            user_input = input()
            if user_input.strip().lower() == 'q':
                print("\nQuitting...")
                cleanup()
        except EOFError:
            break

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

listener = threading.Thread(target=input_listener, daemon=True)
listener.start()

def find_port():
    ports = glob.glob(PORT_PATTERN)
    return ports[0] if ports else None

while True:
    port = find_port()
    if not port:
        print("MindWave not found. Pair it in Bluetooth settings, then it will auto-detect...")
        while not port:
            time.sleep(2)
            port = find_port()
        print(f"Found: {port}")

    try:
        ser = serial.Serial(port, BAUD, timeout=3)
        print(f"Connected to {port}. Waiting for data... (ear clip on?)")
        print("Press 'q' + Enter to quit gracefully.")
        empty_count = 0
        while running:
            raw = ser.read()
            if raw:
                print(raw.hex(), end=' ', flush=True)
                empty_count = 0
            else:
                empty_count += 1
                if empty_count >= 5:
                    print(f"\nNo data for {empty_count * 3}s. Reconnecting...")
                    ser.close()
                    break
    except serial.SerialException as e:
        print(f"\nConnection lost: {e}")
        print("Will retry in 3s...")
        time.sleep(3)