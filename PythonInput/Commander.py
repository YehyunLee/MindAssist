import signal
import sys
import time
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from EEG.eeg_stream import UDPSubscriber

# ---------- CONFIG ----------
PORT = "/dev/cu.usbserial-10"  # change to your Arduino serial port
BAUD = 9600
TIMEOUT = 1.0
FOCUS_THRESHOLD = 60
RELAX_THRESHOLD = 60
LOOP_SLEEP = 0.05
# ----------------------------


class ArmFSM:
    """Mind-state to step-command finite state machine."""

    def __init__(self):
        self.phase = "HOME"
        self.step = 0
        self.last_sent = None

    def update(self, eeg):
        state = str(eeg.get("state", "IDLE"))
        attn = float(eeg.get("attention", 0))
        med = float(eeg.get("meditation", 0))
        blink = int(eeg.get("blink", 0))

        if self.phase == "HOME":
            if state == "FOCUS" and attn >= FOCUS_THRESHOLD:
                self.phase = "REACH"
                self.step = 1
        elif self.phase == "REACH":
            if state == "BLINK" or blink > 0:
                self.phase = "GRAB"
                self.step = 2
            elif state == "RELAX" and med >= RELAX_THRESHOLD:
                self.phase = "HOME"
                self.step = 0
        elif self.phase == "GRAB":
            if state == "RELAX" and med >= RELAX_THRESHOLD:
                self.phase = "RETURN"
                self.step = 3
        elif self.phase == "RETURN":
            if state in ("IDLE", "RELAX"):
                self.phase = "HOME"
                self.step = 0

        return self.step, self.phase


def send_step(ser, step):
    msg = f"{step}\n".encode("ascii")
    ser.write(msg)


def main():
    running = True

    def stop_handler(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    sub = UDPSubscriber()
    fsm = ArmFSM()
    last_eeg_ts = 0.0

    time.sleep(2)
    print("Commander online. Waiting for EEG stream from eeg_visualizer.py ...")

    try:
        while running:
            eeg = sub.recv()
            if eeg is None:
                time.sleep(LOOP_SLEEP)
                continue

            last_eeg_ts = float(eeg.get("ts", 0.0))
            step, phase = fsm.update(eeg)
            if step != fsm.last_sent:
                send_step(ser, step)
                fsm.last_sent = step
                print(
                    "cmd=%d phase=%s state=%s attn=%.1f med=%.1f blink=%d"
                    % (
                        step,
                        phase,
                        eeg.get("state", "IDLE"),
                        float(eeg.get("attention", 0)),
                        float(eeg.get("meditation", 0)),
                        int(eeg.get("blink", 0)),
                    )
                )

            reply = ser.readline().decode("utf-8", errors="ignore").strip()
            if reply:
                print("Arduino:", reply)

            if time.time() - last_eeg_ts > 5:
                if fsm.last_sent != 0:
                    send_step(ser, 0)
                    fsm.last_sent = 0
                    print("EEG stream stale -> sent HOME (0)")

    finally:
        sub.close()
        ser.close()
        print("Commander stopped.")


if __name__ == "__main__":
    main()
