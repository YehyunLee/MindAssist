import serial
import time
import glob

PORT_PATTERN = '/dev/tty.*Mind*'
BAUD = 57600

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
        empty_count = 0
        while True:
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