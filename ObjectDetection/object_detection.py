import os
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from object_stream import UDPBroadcaster


MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1.2
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 255)
CONF_THRESHOLD = float(os.environ.get("OBJ_CONF_THRESHOLD", "0.5"))
SEND_INTERVAL_S = float(os.environ.get("OBJ_SEND_INTERVAL_S", "0.08"))
SHOW_WINDOW = os.environ.get("OBJECT_SHOW_WINDOW", "1") == "1"
CAMERA_RETRY_S = float(os.environ.get("CAMERA_RETRY_S", "2.0"))
MAX_READ_FAILS = int(os.environ.get("MAX_CAMERA_READ_FAILS", "30"))

_miniarm_stream = os.environ.get("MINIARM_STREAM_URL")
if os.environ.get("ESP32_BASE_URL"):
    ESP32_BASE = os.environ["ESP32_BASE_URL"]
elif _miniarm_stream:
    s = urlsplit(_miniarm_stream)
    ESP32_BASE = f"{s.scheme}://{s.netloc}" if s.scheme and s.netloc else "http://192.168.5.1"
else:
    ESP32_BASE = "http://192.168.5.1"

STREAM_URL = os.environ.get("ESP32_STREAM_URL", _miniarm_stream or f"{ESP32_BASE}/stream")
CAPTURE_URL = f"{ESP32_BASE}/capture"
STREAM_CANDIDATES = [
    STREAM_URL,
    f"{ESP32_BASE}:81/stream",
    f"{ESP32_BASE}:8080/?action=stream",
    f"{ESP32_BASE}/mjpeg/1",
]

latest_result = None


def result_callback(result, _output_image, _timestamp_ms):
    global latest_result
    latest_result = result


class FrameGrabber:
    """Fallback single-frame JPEG fetcher for /capture endpoint."""

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


def visualize(image, detection_result) -> np.ndarray:
    if not detection_result:
        return image
    for detection in detection_result.detections:
        category = detection.categories[0]
        probability = round(category.score, 2)
        if probability < CONF_THRESHOLD:
            continue

        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 2)
        result_text = f"{category.category_name} ({probability})"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )
    return image


def best_detection_payload(detection_result, frame_w: int, frame_h: int):
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
    if not detection_result:
        return payload

    best = None
    best_score = -1.0
    for det in detection_result.detections:
        score = det.categories[0].score
        if score >= CONF_THRESHOLD and score > best_score:
            best = det
            best_score = score

    if best is None:
        return payload

    bbox = best.bounding_box
    cx = bbox.origin_x + bbox.width / 2.0
    cy = bbox.origin_y + bbox.height / 2.0
    area = float(bbox.width * bbox.height) / float(max(1, frame_w * frame_h))
    payload.update(
        {
            "found": True,
            "label": best.categories[0].category_name,
            "score": float(best_score),
            "cx": float(cx / max(1, frame_w)),
            "cy": float(cy / max(1, frame_h)),
            "area": area,
        }
    )
    return payload


def open_camera():
    """Try MJPEG endpoints, then fall back to /capture polling."""
    print(f"Checking ESP32-CAM at {ESP32_BASE} ...")
    try:
        urllib.request.urlopen(ESP32_BASE, timeout=5)
        print("ESP32-CAM is reachable.")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach ESP32-CAM at {ESP32_BASE}: {e}\n"
            "Make sure this Mac is connected to the camera Wi-Fi/AP."
        )

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

    print("MJPEG streams failed. Trying single-frame capture ...")
    grabber = FrameGrabber(CAPTURE_URL)
    ok, frame = grabber.read()
    if ok and frame is not None:
        print(f"Connected via capture: {CAPTURE_URL}")
        return grabber

    raise RuntimeError(
        "Could not open any ESP32-CAM endpoint.\n"
        "Tried:\n  " + "\n  ".join(STREAM_CANDIDATES) + f"\n  {CAPTURE_URL}\n\n"
        "Set ESP32_BASE_URL or MINIARM_STREAM_URL for your camera IP."
    )


def open_camera_with_retry():
    while True:
        try:
            return open_camera()
        except RuntimeError as e:
            print(f"[camera] {e}")
            print(f"[camera] Retrying in {CAMERA_RETRY_S:.1f}s ...")
            time.sleep(CAMERA_RETRY_S)


def main():
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "efficientdet_lite0.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cap = open_camera_with_retry()
    stream = UDPBroadcaster()
    options = vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback,
        max_results=3,
    )
    t0 = time.monotonic()
    last_send = 0.0

    with vision.ObjectDetector.create_from_options(options) as detector:
        read_failures = 0
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

            h, w = frame.shape[:2]
            timestamp_ms = int((time.monotonic() - t0) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(mp_image, timestamp_ms=timestamp_ms)

            local_result = latest_result
            if (time.monotonic() - last_send) >= SEND_INTERVAL_S:
                payload = best_detection_payload(local_result, w, h)
                stream.send(payload)
                last_send = time.monotonic()

            if SHOW_WINDOW:
                vis = visualize(frame, local_result)
                p = best_detection_payload(local_result, w, h)
                if p["found"]:
                    cv2.putText(
                        vis,
                        f"{p['label']} score={p['score']:.2f} cx={p['cx']:.2f} area={p['area']:.3f}",
                        (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                cv2.imshow("MindAssist Object Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    stream.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
