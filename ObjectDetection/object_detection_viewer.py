"""Object detection built directly on esp_cam_viewer camera pipeline.

Camera behavior is intentionally delegated to esp_cam_viewer.open_camera()
so connectivity matches the known-good viewer script.
"""

import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from esp_cam_viewer import open_camera
from object_stream import UDPBroadcaster


CONF_THRESHOLD = float(os.environ.get("OBJ_CONF_THRESHOLD", "0.5"))
SEND_INTERVAL_S = float(os.environ.get("OBJ_SEND_INTERVAL_S", "0.08"))
SHOW_WINDOW = os.environ.get("OBJECT_SHOW_WINDOW", "1") == "1"
CAMERA_RETRY_S = float(os.environ.get("CAMERA_RETRY_S", "2.0"))
MAX_READ_FAILS = int(os.environ.get("MAX_CAMERA_READ_FAILS", "30"))

latest_result = None


def result_callback(result, _output_image, _timestamp_ms):
    global latest_result
    latest_result = result


def best_detection_payload(detection_result, frame_w, frame_h):
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
        score = float(det.categories[0].score)
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
            "score": best_score,
            "cx": float(cx / max(1, frame_w)),
            "cy": float(cy / max(1, frame_h)),
            "area": area,
        }
    )
    return payload


def draw_overlay(frame, detection_result):
    if not detection_result:
        return frame

    for det in detection_result.detections:
        cat = det.categories[0]
        if cat.score < CONF_THRESHOLD:
            continue
        bbox = det.bounding_box
        x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
        x2, y2 = x1 + int(bbox.width), y1 + int(bbox.height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{cat.category_name} {cat.score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


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
    t0 = time.monotonic()
    last_send = 0.0
    read_failures = 0

    options = vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback,
        max_results=3,
    )

    with vision.ObjectDetector.create_from_options(options) as detector:
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

            local = latest_result
            if (time.monotonic() - last_send) >= SEND_INTERVAL_S:
                stream.send(best_detection_payload(local, w, h))
                last_send = time.monotonic()

            if SHOW_WINDOW:
                vis = draw_overlay(frame.copy(), local)
                p = best_detection_payload(local, w, h)
                if p["found"]:
                    cv2.putText(
                        vis,
                        f"{p['label']} score={p['score']:.2f} cx={p['cx']:.2f} area={p['area']:.3f}",
                        (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                cv2.imshow("MindAssist Object Detection (Viewer Base)", vis)
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
