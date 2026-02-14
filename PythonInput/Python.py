import serial
import time

# ---------- CONFIG ----------
PORT = "COM3"          # Windows example. Change to your Arduino port.
# PORT = "/dev/ttyACM0"  # Linux example
# PORT = "/dev/tty.usbmodem14101"  # macOS example

BAUD = 9600
TIMEOUT = 1
# ----------------------------

def main():
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)

    # Opening the serial port often resets the Arduino; wait for it to boot.
    time.sleep(2)

    try:
        while True:
            # Replace this with your "Python output"
            x = input("Enter an integer to send (or 'q' to quit): ").strip()
            if x.lower() in ("q", "quit", "exit"):
                break

            # Validate and send as "number + newline"
            n = int(x)
            msg = f"{n}\n".encode("ascii")
            ser.write(msg)

            # Optional: read Arduino response (if Arduino prints something back)
            reply = ser.readline().decode("utf-8", errors="ignore").strip()
            if reply:
                print("Arduino:", reply)

    finally:
        ser.close()

if __name__ == "__main__": 
    main()