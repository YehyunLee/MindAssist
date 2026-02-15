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

# ---------- CONFIG ----------
PORT = os.environ.get("MINIARM_PORT", "/dev/cu.usbserial-10")
BAUD = int(os.environ.get("MINIARM_BAUD", "9600"))
TIMEOUT = float(os.environ.get("MINIARM_TIMEOUT", "0.05"))

FOCUS_TRIGGER_THRESHOLD = float(os.environ.get("FOCUS_TRIGGER_THRESHOLD", "65"))
FOCUS_HOLD_S = float(os.environ.get("FOCUS_HOLD_S", "2.0"))
LOOP_SLEEP = float(os.environ.get("COMMANDER_LOOP_SLEEP", "0.05"))

OBJECT_STALE_S = float(os.environ.get("OBJECT_STALE_S", "1.5"))
SEARCH_SWEEP_INTERVAL_S = float(os.environ.get("SEARCH_SWEEP_INTERVAL_S", "1.8"))
APPROACH_CENTER_TOL = float(os.environ.get("APPROACH_CENTER_TOL", "0.12"))
PICKUP_AREA_THRESHOLD = float(os.environ.get("PICKUP_AREA_THRESHOLD", "0.08"))
LOCK_DETECTIONS_REQUIRED = int(os.environ.get("LOCK_DETECTIONS_REQUIRED", "3"))
MAX_OBJECT_MISS = int(os.environ.get("MAX_OBJECT_MISS", "10"))
APPROACH_PICK_STABLE_COUNT = int(os.environ.get("APPROACH_PICK_STABLE_COUNT", "4"))
COMMAND_RESEND_S = float(os.environ.get("COMMAND_RESEND_S", "1.0"))

PICK_HOLD_S = float(os.environ.get("PICK_HOLD_S", "2.0"))
LIFT_HOLD_S = float(os.environ.get("LIFT_HOLD_S", "2.0"))
SERVE_HOLD_S = float(os.environ.get("SERVE_HOLD_S", "8.0"))

START_OBJECT_DETECTION = os.environ.get("START_OBJECT_DETECTION", "1") == "1"
OBJECT_SHOW_WINDOW = os.environ.get("OBJECT_SHOW_WINDOW", "1")
OBJECT_RESTART_INTERVAL_S = float(os.environ.get("OBJECT_RESTART_INTERVAL_S", "3.0"))
OBJECT_DETECTION_SCRIPT = os.environ.get("OBJECT_DETECTION_SCRIPT", "object_detection.py")
OBJECT_DETECTION_FALLBACK = os.environ.get("OBJECT_DETECTION_FALLBACK", "object_detection.py")
START_MODAL_SERVER = os.environ.get("START_MODAL_SERVER", "1") == "1"
MODAL_SERVER_SCRIPT = os.environ.get("MODAL_SERVER_SCRIPT", "modal_server.py")
MODAL_RESTART_INTERVAL_S = float(os.environ.get("MODAL_RESTART_INTERVAL_S", "4.0"))
# ----------------------------


class WorkflowFSM:
    """Triggered workflow: idle -> search -> approach -> pick -> lift/serve."""

    WAIT_TRIGGER = "WAIT_TRIGGER"
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    PICK = "PICK"
    LIFT = "LIFT"
    SERVE = "SERVE"

    def __init__(self):
        self.state = self.WAIT_TRIGGER
        self.state_started = time.time()
        self.focus_since = None
        self.last_sent_step = None
        self.last_search_move = 0.0
        self.search_idx = 0
        self.search_pattern = [0, 1, 2, 1, 0, 2]
        self.obj_seen_streak = 0
        self.obj_miss_streak = 0
        self.pick_ready_streak = 0

    def set_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            self.state_started = time.time()
            self.obj_seen_streak = 0
            self.obj_miss_streak = 0
            self.pick_ready_streak = 0
            print(f"[FSM] -> {self.state}")

    def _is_object_fresh(self, obj):
        if not obj:
            return False
        return (time.time() - float(obj.get("ts", 0.0))) <= OBJECT_STALE_S

    def update(self, eeg, obj):
        now = time.time()
        attn = float(eeg.get("attention", 0.0)) if eeg else 0.0
        obj_fresh = self._is_object_fresh(obj)
        obj_found = bool(obj and obj.get("found", False) and obj_fresh)

        if self.state == self.WAIT_TRIGGER:
            if attn >= FOCUS_TRIGGER_THRESHOLD:
                if self.focus_since is None:
                    self.focus_since = now
                if now - self.focus_since >= FOCUS_HOLD_S:
                    print(
                        f"[TRIGGER] Focus held for {FOCUS_HOLD_S:.1f}s "
                        f"(attn={attn:.1f}) -> workflow start"
                    )
                    self.set_state(self.SEARCH)
                    self.focus_since = None
            else:
                self.focus_since = None
            return 0

        # From here onward, ignore EEG until SERVE completes.
        if self.state == self.SEARCH:
            if obj_found:
                self.obj_seen_streak += 1
                if self.obj_seen_streak >= LOCK_DETECTIONS_REQUIRED:
                    print(
                        f"[SEARCH] Object lock acquired: label={obj.get('label')} "
                        f"score={float(obj.get('score', 0.0)):.2f}"
                    )
                    self.set_state(self.APPROACH)
                    return 1
            else:
                self.obj_seen_streak = 0

            # Keep sweeping continuously until object lock is acquired.
            if now - self.last_search_move >= SEARCH_SWEEP_INTERVAL_S:
                step = self.search_pattern[self.search_idx]
                self.search_idx = (self.search_idx + 1) % len(self.search_pattern)
                self.last_search_move = now
                print(f"[SEARCH] Sweeping pose step={step}")
                return step
            return None

        if self.state == self.APPROACH:
            if not obj_found:
                self.obj_miss_streak += 1
                if self.obj_miss_streak >= MAX_OBJECT_MISS:
                    print("[APPROACH] Lost object -> back to search")
                    self.set_state(self.SEARCH)
                    return None
                return 1
            self.obj_miss_streak = 0

            area = float(obj.get("area", 0.0))
            cx = float(obj.get("cx", 0.5))
            centered = abs(cx - 0.5) <= APPROACH_CENTER_TOL

            if area >= PICKUP_AREA_THRESHOLD and centered:
                self.pick_ready_streak += 1
            else:
                self.pick_ready_streak = 0

            if self.pick_ready_streak >= APPROACH_PICK_STABLE_COUNT:
                print(
                    f"[APPROACH] Close enough (area={area:.3f}) -> pick"
                )
                self.set_state(self.PICK)
                return 2

            if cx < (0.5 - APPROACH_CENTER_TOL):
                return 0
            if cx > (0.5 + APPROACH_CENTER_TOL):
                return 2
            return 1

        if self.state == self.PICK:
            if now - self.state_started >= PICK_HOLD_S:
                self.set_state(self.LIFT)
                return 3
            return 2

        if self.state == self.LIFT:
            if now - self.state_started >= LIFT_HOLD_S:
                self.set_state(self.SERVE)
                return 3
            return 3

        if self.state == self.SERVE:
            if now - self.state_started >= SERVE_HOLD_S:
                print("[SERVE] Hold complete -> reset to wait trigger")
                self.set_state(self.WAIT_TRIGGER)
                return 0
            return 3

        return 0


def send_step(ser, step):
    ser.write(f"{int(step)}\n".encode("ascii"))


def maybe_start_object_detection():
    if not START_OBJECT_DETECTION:
        return None
    script = ROOT / "ObjectDetection" / OBJECT_DETECTION_SCRIPT
    model_path = Path(
        os.environ.get("YOLO_MODEL_PATH", ROOT / "ObjectDetection" / "yolov8n.onnx")
    )
    if not script.exists():
        fallback = ROOT / "ObjectDetection" / OBJECT_DETECTION_FALLBACK
        print(f"[Commander] {script.name} not found, falling back to {fallback.name}")
        script = fallback
    if script.name == "object_detection.py" and not model_path.exists():
        print(
            f"[Commander] YOLO model not found at {model_path}. "
            "modal_server will use its default model download/cache behavior."
        )
    if not script.exists():
        fallback = ROOT / "ObjectDetection" / "object_detection.py"
        print(f"[Commander] {script.name} missing, final fallback to {fallback.name}")
        script = fallback
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
    proc = subprocess.Popen([sys.executable, script.name], cwd=str(script.parent), env=os.environ.copy())
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


def main():
    running = True

    def stop_handler(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    eeg_sub = EEGSubscriber()
    obj_sub = ObjectSubscriber()
    fsm = WorkflowFSM()
    modal_proc = maybe_start_modal_server()
    if modal_proc is not None:
        # Give server a brief warmup before detector starts sending requests.
        time.sleep(1.0)
    detector_proc = maybe_start_object_detection()
    last_restart_try = time.time()
    last_modal_restart_try = time.time()

    latest_eeg = None
    latest_obj = None
    last_eeg_ts = 0.0
    last_cmd_ts = 0.0

    time.sleep(2)
    print("[Commander] Online. Waiting for EEG + object streams...")

    try:
        while running:
            eeg = eeg_sub.recv()
            if eeg:
                latest_eeg = eeg
                last_eeg_ts = float(eeg.get("ts", 0.0))

            # Drain object queue and keep latest object telemetry.
            while True:
                obj = obj_sub.recv()
                if not obj:
                    break
                latest_obj = obj

            if latest_eeg is None:
                time.sleep(LOOP_SLEEP)
                continue

            if time.time() - last_eeg_ts > 5:
                if fsm.last_sent_step != 0:
                    send_step(ser, 0)
                    fsm.last_sent_step = 0
                    print("[Commander] EEG stream stale -> HOME")
                time.sleep(LOOP_SLEEP)
                continue

            step = fsm.update(latest_eeg, latest_obj)
            should_send = False
            if step is not None and step != fsm.last_sent_step:
                should_send = True
            elif step is not None and (time.time() - last_cmd_ts) >= COMMAND_RESEND_S:
                should_send = True

            if should_send:
                send_step(ser, step)
                fsm.last_sent_step = step
                last_cmd_ts = time.time()
                print(
                    "[CMD] step=%d state=%s attn=%.1f obj=%s score=%.2f area=%.3f cx=%.2f"
                    % (
                        step,
                        fsm.state,
                        float(latest_eeg.get("attention", 0.0)),
                        "Y" if latest_obj and latest_obj.get("found") else "N",
                        float(latest_obj.get("score", 0.0)) if latest_obj else 0.0,
                        float(latest_obj.get("area", 0.0)) if latest_obj else 0.0,
                        float(latest_obj.get("cx", 0.5)) if latest_obj else 0.5,
                    )
                )

            if ser.in_waiting:
                reply = ser.readline().decode("utf-8", errors="ignore").strip()
                if reply:
                    print("Arduino:", reply)

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
        obj_sub.close()
        eeg_sub.close()
        ser.close()
        print("[Commander] stopped.")


if __name__ == "__main__":
    main()
