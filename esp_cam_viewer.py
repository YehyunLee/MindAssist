"""
ESP32S3-Cam Live Viewer
-----------------------
Connects to the ESP32S3-Cam video stream and displays it in an OpenCV window.
Press 'q' to quit.

Usage:
    python esp_cam_viewer.py                          # default AP mode (192.168.5.1)
    ESP32_BASE_URL="http://192.168....." python esp_cam_viewer.py   # LAN mode
"""

import os
import time
import urllib.request

import cv2
import numpy as np

ESP32_BASE = os.environ.get("ESP32_BASE_URL", "http://192.168.5.1")
STREAM_URL = os.environ.get("ESP32_STREAM_URL", f"{ESP32_BASE}/stream")
CAPTURE_URL = f"{ESP32_BASE}/capture"
STREAM_CANDIDATES = [
    STREAM_URL,
    f"{ESP32_BASE}:81/stream",
    f"{ESP32_BASE}:8080/?action=stream",
    f"{ESP32_BASE}/mjpeg/1",
]


class FrameGrabber:
    """Falls back to fetching individual JPEG frames via /capture."""

    def __init__(self, url):
        self.url = url

    def isOpened(self):
        return True

    def read(self):
        try:
            resp = urllib.request.urlopen(self.url, timeout=5)
            jpg = np.frombuffer(resp.read(), dtype=np.uint8)
            frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            if frame is not None:
                return True, frame
        except Exception:
            pass
        return False, None

    def release(self):
        pass


def open_camera():
    """Try MJPEG stream candidates, then fall back to single-frame capture."""
    # Verify ESP32-CAM is reachable
    print(f"Checking ESP32-CAM at {ESP32_BASE} ...")
    try:
        urllib.request.urlopen(ESP32_BASE, timeout=5)
        print("ESP32-CAM is reachable.")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach ESP32-CAM at {ESP32_BASE}: {e}\n"
            "Make sure the ESP32-CAM is on and your Mac is connected to its WiFi."
        )

    # Try MJPEG streams
    for candidate in STREAM_CANDIDATES:
        print(f"Trying stream: {candidate} ...")
        test_cap = cv2.VideoCapture()
        test_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        test_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        test_cap.open(candidate)
        if not test_cap.isOpened():
            test_cap.release()
            continue
        ok, frame = test_cap.read()
        if ok and frame is not None:
            print(f"Connected via MJPEG: {candidate}")
            return test_cap
        test_cap.release()

    # Fall back to single-frame capture
    print("MJPEG streams failed. Trying single-frame capture ...")
    grabber = FrameGrabber(CAPTURE_URL)
    ok, frame = grabber.read()
    if ok:
        print(f"Connected via capture: {CAPTURE_URL}")
        return grabber

    raise RuntimeError(
        "Could not open any ESP32-CAM endpoint.\n"
        "Tried:\n  " + "\n  ".join(STREAM_CANDIDATES) + f"\n  {CAPTURE_URL}\n\n"
        "Set ESP32_BASE_URL if the camera IP is different:\n"
        '  ESP32_BASE_URL="http://<IP>" python esp_cam_viewer.py'
    )


def main():
    cap = open_camera()
    fps_time = time.monotonic()
    frame_count = 0

    print("Displaying stream. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Overlay FPS
        frame_count += 1
        elapsed = time.monotonic() - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.monotonic()
        else:
            fps = frame_count / max(elapsed, 0.001)
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        cv2.imshow("ESP32S3-Cam Viewer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except RuntimeError as e:
        print(f"\nError: {e}")
