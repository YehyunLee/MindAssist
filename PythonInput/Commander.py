import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from EEG.eeg_stream import UDPSubscriber as EEGSubscriber
from ObjectDetection.object_stream import UDPSubscriber as ObjectSubscriber

# ---------- SERIAL CONFIG ----------
PORT = os.environ.get("MINIARM_PORT", "/dev/cu.usbmodem101")
BAUD = int(os.environ.get("MINIARM_BAUD", "9600"))
TIMEOUT = float(os.environ.get("MINIARM_TIMEOUT", "0.05"))
LOOP_SLEEP = float(os.environ.get("COMMANDER_LOOP_SLEEP", "0.05"))
FOCUS_SEARCH_THRESHOLD = float(os.environ.get("FOCUS_SEARCH_THRESHOLD", "40"))
EEG_STALE_S = float(os.environ.get("EEG_STALE_S", "3.0"))

# ---------- SEARCH SWEEP ----------
ROTATION_MIN = int(os.environ.get("ROTATION_MIN", "30"))
ROTATION_MAX = int(os.environ.get("ROTATION_MAX", "150"))
ROTATION_STEP = int(os.environ.get("ROTATION_STEP", "1"))
SWEEP_INTERVAL_S = float(os.environ.get("SWEEP_INTERVAL_S", "0.3"))

# Search pose: [gripper, upper, middle, lower, rotation, aux]
# index 0: gripper (0=open, 90=closed)
# index 1: upper joint
# index 2: middle joint
# index 3: lower joint
# index 4: rotation
# index 5: aux (unused)
SEARCH_GRIPPER = int(os.environ.get("SEARCH_GRIPPER", "0"))
SEARCH_UPPER = int(os.environ.get("SEARCH_UPPER", "180"))
SEARCH_MIDDLE = int(os.environ.get("SEARCH_MIDDLE", "20"))
SEARCH_LOWER = int(os.environ.get("SEARCH_LOWER", "50"))
SEARCH_AUX = int(os.environ.get("SEARCH_AUX", "0"))

# ---------- OBJECT DETECTION ----------
OBJECT_STALE_S = float(os.environ.get("OBJECT_STALE_S", "1.5"))
LOCK_DETECTIONS_REQUIRED = int(os.environ.get("LOCK_DETECTIONS_REQUIRED", "3"))

START_OBJECT_DETECTION = os.environ.get("START_OBJECT_DETECTION", "1") == "1"
OBJECT_SHOW_WINDOW = os.environ.get("OBJECT_SHOW_WINDOW", "1")
OBJECT_DETECTION_SCRIPT = os.environ.get("OBJECT_DETECTION_SCRIPT", "object_detection.py")
OBJECT_DETECTION_FALLBACK = os.environ.get("OBJECT_DETECTION_FALLBACK", "object_detection.py")
OBJECT_RESTART_INTERVAL_S = float(os.environ.get("OBJECT_RESTART_INTERVAL_S", "3.0"))
START_MODAL_SERVER = os.environ.get("START_MODAL_SERVER", "1") == "1"
MODAL_SERVER_SCRIPT = os.environ.get("MODAL_SERVER_SCRIPT", "modal_server.py")
MODAL_RESTART_INTERVAL_S = float(os.environ.get("MODAL_RESTART_INTERVAL_S", "4.0"))

# ---------- APPROACH / PICK / LIFT / SERVE ----------
APPROACH_CENTER_TOL = float(os.environ.get("APPROACH_CENTER_TOL", "0.15"))
APPROACH_TIMEOUT_S = float(os.environ.get("APPROACH_TIMEOUT_S", "3.0"))
MAX_OBJECT_MISS = int(os.environ.get("MAX_OBJECT_MISS", "40"))
PICK_LOWER_S = float(os.environ.get("PICK_LOWER_S", "2.0"))
PICK_GRAB_S = float(os.environ.get("PICK_GRAB_S", "1.5"))
LIFT_HOLD_S = float(os.environ.get("LIFT_HOLD_S", "2.0"))
SERVE_HOLD_S = float(os.environ.get("SERVE_HOLD_S", "6.0"))
LOG_INTERVAL_S = float(os.environ.get("LOG_INTERVAL_S", "1.0"))
# ----------------------------


# ---- Serial helpers (protocol: A{angle}$ .. F{angle}$) ----

def send_servo(ser, index, angle):
    """Send angle to a single servo. index 0-5 maps to letters A-F."""
    letter = chr(ord("A") + index)
    ser.write(f"{letter}{int(angle)}$".encode("ascii"))


def send_pose(ser, gripper, upper, middle, lower, rotation, aux):
    """Send full 6-servo pose."""
    for i, a in enumerate([gripper, upper, middle, lower, rotation, aux]):
        send_servo(ser, i, int(a))


# ---- Subprocess management ----

def maybe_start_object_detection():
    if not START_OBJECT_DETECTION:
        return None
    script = ROOT / "ObjectDetection" / OBJECT_DETECTION_SCRIPT
    if not script.exists():
        fallback = ROOT / "ObjectDetection" / OBJECT_DETECTION_FALLBACK
        print(f"[Commander] {script.name} not found, falling back to {fallback.name}")
        script = fallback
    if not script.exists():
        print(f"[Commander] {script} missing")
        return None
    env = os.environ.copy()
    env["OBJECT_SHOW_WINDOW"] = OBJECT_SHOW_WINDOW
    proc = subprocess.Popen([sys.executable, script.name], cwd=str(script.parent), env=env)
    print(f"[Commander] Started object detection (pid={proc.pid})")
    return proc


def maybe_start_modal_server():
    if not START_MODAL_SERVER:
        return None
    script = ROOT / "ObjectDetection" / MODAL_SERVER_SCRIPT
    if not script.exists():
        print(f"[Commander] modal server script missing: {script}")
        return None
    proc = subprocess.Popen(
        [sys.executable, script.name], cwd=str(script.parent), env=os.environ.copy()
    )
    print(f"[Commander] Started modal server (pid={proc.pid})")
    return proc


def stop_process(proc):
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---- Workflow FSM ----

class WorkflowFSM:
    """Immediate workflow: SEARCH -> APPROACH -> PICK -> LIFT -> SERVE -> SEARCH."""

    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    PICK = "PICK"
    LIFT = "LIFT"
    SERVE = "SERVE"

    def __init__(self):
        self.state = self.SEARCH
        self.state_started = time.time()
        self.rotation_angle = (ROTATION_MIN + ROTATION_MAX) // 2
        self.rotation_dir = 1
        self.last_sweep = 0.0
        self.obj_seen_streak = 0
        self.obj_miss_streak = 0
        self.pick_ready_streak = 0
        self.last_log = 0.0

    def set_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            self.state_started = time.time()
            self.obj_seen_streak = 0
            self.obj_miss_streak = 0
            self.pick_ready_streak = 0
            print(f"[FSM] -> {self.state}")

    def _is_fresh(self, obj):
        if not obj:
            return False
        return (time.time() - float(obj.get("ts", 0.0))) <= OBJECT_STALE_S

    def update(self, ser, obj, focus_active):
        now = time.time()
        obj_fresh = self._is_fresh(obj)
        obj_found = bool(obj and obj.get("found", False) and obj_fresh)

        # Periodic status log
        if now - self.last_log >= LOG_INTERVAL_S:
            score = float(obj.get("score", 0)) if obj else 0
            cx = float(obj.get("cx", 0.5)) if obj else 0.5
            area = float(obj.get("area", 0)) if obj else 0
            print(
                f"[STATUS] state={self.state} rot={self.rotation_angle} "
                f"focus={'Y' if focus_active else 'N'} obj={'Y' if obj_found else 'N'} score={score:.2f} "
                f"cx={cx:.2f} area={area:.3f}"
            )
            self.last_log = now

        # ---- SEARCH: sweep rotation left/right until YOLO locks on ----
        if self.state == self.SEARCH:
            if not focus_active:
                return

            if obj_found:
                self.obj_seen_streak += 1
                print(
                    f"[SEARCH] Detection #{self.obj_seen_streak}: "
                    f"label={obj.get('label')} "
                    f"score={float(obj.get('score', 0)):.2f} "
                    f"cx={float(obj.get('cx', 0.5)):.2f} "
                    f"area={float(obj.get('area', 0)):.3f}"
                )
                if self.obj_seen_streak >= LOCK_DETECTIONS_REQUIRED:
                    print("[SEARCH] >>> Object LOCKED! Transitioning to APPROACH")
                    self.set_state(self.APPROACH)
                    return
            else:
                if self.obj_seen_streak > 0:
                    print(f"[SEARCH] Lost detection (streak was {self.obj_seen_streak})")
                self.obj_seen_streak = 0

            if now - self.last_sweep >= SWEEP_INTERVAL_S:
                self.rotation_angle += self.rotation_dir * ROTATION_STEP
                if self.rotation_angle >= ROTATION_MAX:
                    self.rotation_angle = ROTATION_MAX
                    self.rotation_dir = -1
                elif self.rotation_angle <= ROTATION_MIN:
                    self.rotation_angle = ROTATION_MIN
                    self.rotation_dir = 1
                send_servo(ser, 4, self.rotation_angle)
                self.last_sweep = now
            return

        # ---- APPROACH: nudge rotation toward object, then proceed to PICK ----
        if self.state == self.APPROACH:
            elapsed = now - self.state_started

            if obj_found:
                self.obj_miss_streak = 0
                cx = float(obj.get("cx", 0.5))
                # Nudge rotation to roughly center the object
                if cx < (0.5 - APPROACH_CENTER_TOL):
                    self.rotation_angle = min(self.rotation_angle + 1, ROTATION_MAX)
                    send_servo(ser, 4, self.rotation_angle)
                elif cx > (0.5 + APPROACH_CENTER_TOL):
                    self.rotation_angle = max(self.rotation_angle - 1, ROTATION_MIN)
                    send_servo(ser, 4, self.rotation_angle)
            else:
                self.obj_miss_streak += 1
                if self.obj_miss_streak >= MAX_OBJECT_MISS:
                    print("[APPROACH] Lost object too long -> back to SEARCH")
                    self.set_state(self.SEARCH)
                    return

            # After timeout, move to PICK regardless
            if elapsed >= APPROACH_TIMEOUT_S:
                print(f"[APPROACH] Timeout ({APPROACH_TIMEOUT_S}s) -> lowering arm to PICK")
                # Lower arm: open gripper, extend forward/down
                send_pose(ser, 0, 120, 0, 0, self.rotation_angle, 0)
                self.set_state(self.PICK)
            return

        # ---- PICK: arm is lowering, then close gripper ----
        if self.state == self.PICK:
            elapsed = now - self.state_started
            if elapsed < PICK_LOWER_S:
                # Wait for arm to reach lowered position
                return
            if elapsed < PICK_LOWER_S + PICK_GRAB_S:
                # Close gripper once
                if self.pick_ready_streak == 0:
                    send_servo(ser, 0, 90)
                    print("[PICK] Gripper closing...")
                    self.pick_ready_streak = 1
                return
            # Gripper closed, now lift straight up
            print("[PICK] Grabbed -> LIFT")
            send_pose(ser, 90, 90, 90, 90, self.rotation_angle, 0)
            self.set_state(self.LIFT)
            return

        # ---- LIFT: raise arm with object ----
        if self.state == self.LIFT:
            if now - self.state_started >= LIFT_HOLD_S:
                print("[LIFT] Raised -> SERVE (straight up)")
                send_pose(ser, 90, 90, 90, 90, 90, 0)
                self.set_state(self.SERVE)
            return

        # ---- SERVE: hold, then release and restart ----
        if self.state == self.SERVE:
            if now - self.state_started >= SERVE_HOLD_S:
                send_servo(ser, 0, 0)  # open gripper
                print("[SERVE] Released -> back to SEARCH")
                time.sleep(1.0)
                send_pose(
                    ser, SEARCH_GRIPPER, SEARCH_UPPER, SEARCH_MIDDLE,
                    SEARCH_LOWER, (ROTATION_MIN + ROTATION_MAX) // 2, SEARCH_AUX,
                )
                self.rotation_angle = (ROTATION_MIN + ROTATION_MAX) // 2
                self.set_state(self.SEARCH)
            return


# ---- Main loop ----

def main():
    running = True

    def stop_handler(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    eeg_sub = EEGSubscriber()
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    obj_sub = ObjectSubscriber()
    fsm = WorkflowFSM()

    modal_proc = maybe_start_modal_server()
    if modal_proc is not None:
        time.sleep(1.0)
    detector_proc = maybe_start_object_detection()
    last_restart_try = time.time()
    last_modal_restart_try = time.time()
    latest_eeg = None
    last_eeg_ts = 0.0
    latest_obj = None

    time.sleep(2)
    print("[Commander] Online. Search will run only while EEG focus is high.")

    # Send initial search pose
    send_pose(
        ser, SEARCH_GRIPPER, SEARCH_UPPER, SEARCH_MIDDLE,
        SEARCH_LOWER, fsm.rotation_angle, SEARCH_AUX,
    )

    try:
        while running:
            # Keep latest EEG packet
            eeg = eeg_sub.recv()
            if eeg:
                latest_eeg = eeg
                last_eeg_ts = float(eeg.get("ts", 0.0))

            # Drain object queue, keep latest
            while True:
                obj = obj_sub.recv()
                if not obj:
                    break
                latest_obj = obj

            eeg_fresh = (time.time() - last_eeg_ts) <= EEG_STALE_S if last_eeg_ts else False
            attention = float(latest_eeg.get("attention", 0.0)) if latest_eeg else 0.0
            focus_active = eeg_fresh and attention >= FOCUS_SEARCH_THRESHOLD

            fsm.update(ser, latest_obj, focus_active)

            # Read Arduino replies
            if ser.in_waiting:
                reply = ser.readline().decode("utf-8", errors="ignore").strip()
                if reply:
                    print("Arduino:", reply)

            # Auto-restart crashed subprocesses
            if detector_proc is not None and detector_proc.poll() is not None:
                print(f"[Commander] object_detection exited (code={detector_proc.returncode})")
                detector_proc = None
            if START_OBJECT_DETECTION and detector_proc is None:
                if time.time() - last_restart_try >= OBJECT_RESTART_INTERVAL_S:
                    detector_proc = maybe_start_object_detection()
                    last_restart_try = time.time()

            if modal_proc is not None and modal_proc.poll() is not None:
                print(f"[Commander] modal_server exited (code={modal_proc.returncode})")
                modal_proc = None
            if START_MODAL_SERVER and modal_proc is None:
                if time.time() - last_modal_restart_try >= MODAL_RESTART_INTERVAL_S:
                    modal_proc = maybe_start_modal_server()
                    last_modal_restart_try = time.time()

            time.sleep(LOOP_SLEEP)

    finally:
        stop_process(detector_proc)
        stop_process(modal_proc)
        eeg_sub.close()
        obj_sub.close()
        send_pose(ser, 0, 90, 90, 90, 90, 0)
        time.sleep(0.5)
        ser.close()
        print("[Commander] stopped.")


if __name__ == "__main__":
    main()
