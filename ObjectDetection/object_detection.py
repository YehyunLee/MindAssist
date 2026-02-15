"""ESP32-CAM object detection client using YOLO modal server.

Pipeline:
  ESP32-CAM stream (via esp_cam_viewer.open_camera) -> /detect (FastAPI YOLO)
  -> best object telemetry over UDP for Commander.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import cv2
import numpy as np
import requests

from esp_cam_viewer import open_camera
from object_stream import UDPBroadcaster


MODAL_ENDPOINT = os.environ.get("MODAL_ENDPOINT", "http://127.0.0.1:8080/detect")
CONF_THRESHOLD = float(os.environ.get("OBJ_CONF_THRESHOLD", "0.25"))
INFER_INTERVAL_S = float(os.environ.get("OBJ_INFER_INTERVAL_S", "0.12"))
SEND_INTERVAL_S = float(os.environ.get("OBJ_SEND_INTERVAL_S", "0.08"))
SHOW_WINDOW = os.environ.get("OBJECT_SHOW_WINDOW", "1") == "1"
CAMERA_RETRY_S = float(os.environ.get("CAMERA_RETRY_S", "2.0"))
MAX_READ_FAILS = int(os.environ.get("MAX_CAMERA_READ_FAILS", "30"))
HTTP_TIMEOUT_S = float(os.environ.get("MODAL_HTTP_TIMEOUT_S", "5.0"))
LOG_OBJECT_EVERY_S = float(os.environ.get("LOG_OBJECT_EVERY_S", "1.0"))

def call_modal_inference(frame: np.ndarray) -> List[Dict]:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return []

    files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    try:
        r = requests.post(MODAL_ENDPOINT, files=files, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
        detections = data.get("detections", [])
        if isinstance(detections, list):
            return detections
        return []
    except Exception as e:
        print(f"[modal] request failed: {e}")
        return []


def best_detection_payload(detections: List[Dict], frame_w: int, frame_h: int) -> Dict:
    payload = {
        "ts": time.time(),
        "found": False,
        "label": "",
        "score": 0.0,
        "cx": 0.5,
        "cy": 0.5,
        "area": 0.0,
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
    }

    best = None
    best_score = -1.0
    for det in detections:
        score = float(det.get("score", 0.0))
        if score < CONF_THRESHOLD:
            continue
        if score > best_score:
            best = det
            best_score = score

    if best is None:
        return payload

    x = float(best.get("origin_x", 0))
    y = float(best.get("origin_y", 0))
    w = float(best.get("width", 0))
    h = float(best.get("height", 0))
    cx = x + w / 2.0
    cy = y + h / 2.0
    area = (w * h) / float(max(1, frame_w * frame_h))

    payload.update(
        {
            "found": True,
            "label": str(best.get("label", "")),
            "score": float(best_score),
            "cx": float(cx / max(1, frame_w)),
            "cy": float(cy / max(1, frame_h)),
            "area": float(area),
        }
    )
    return payload


def open_camera_with_retry():
    while True:
        try:
            return open_camera()
        except RuntimeError as e:
            print(f"[camera] {e}")
            print(f"[camera] Retrying in {CAMERA_RETRY_S:.1f}s ...")
            time.sleep(CAMERA_RETRY_S)


def main():
    cap = open_camera_with_retry()
    stream = UDPBroadcaster()
    read_failures = 0
    last_infer = 0.0
    last_send = 0.0
    last_log = 0.0
    latest_detections: List[Dict] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures >= MAX_READ_FAILS:
                    print("[camera] Stream dropped. Reconnecting camera ...")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = open_camera_with_retry()
                    read_failures = 0
                continue
            read_failures = 0

            now = time.monotonic()
            h, w = frame.shape[:2]

            if now - last_infer >= INFER_INTERVAL_S:
                latest_detections = call_modal_inference(frame)
                last_infer = now
                if latest_detections:
                    for d in latest_detections:
                        tag = ">>>" if float(d.get("score", 0)) >= CONF_THRESHOLD else "   "
                        print(
                            f"[RAW] {tag} {d.get('label')} "
                            f"score={float(d.get('score', 0)):.2f} "
                            f"box=({d.get('origin_x')},{d.get('origin_y')},"
                            f"{d.get('width')}x{d.get('height')})"
                        )

            payload = best_detection_payload(latest_detections, w, h)

            if now - last_send >= SEND_INTERVAL_S:
                stream.send(payload)
                last_send = now

            if now - last_log >= LOG_OBJECT_EVERY_S:
                if payload["found"]:
                    print(
                        "[OBJ] found label=%s score=%.2f cx=%.2f area=%.3f"
                        % (
                            payload["label"],
                            float(payload["score"]),
                            float(payload["cx"]),
                            float(payload["area"]),
                        )
                    )
                else:
                    print("[OBJ] none")
                last_log = now

            if SHOW_WINDOW:
                cv2.imshow("MindAssist Object Detection (YOLO Modal)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        stream.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
