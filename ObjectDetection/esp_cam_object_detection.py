import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import serial
from mediapipe.tasks.python import vision


MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1.2
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 255)  # BGR red
CONF_THRESHOLD = 0.5
SEND_INTERVAL_S = 0.08

ARDUINO_PORT = os.environ.get("MINIARM_PORT", "/dev/cu.usbmodem211301")
BAUD_RATE = 9600
STREAM_URL = os.environ.get("MINIARM_STREAM_URL", "http://192.168.5.1/stream")
STREAM_CANDIDATES = [
    STREAM_URL,
    "http://192.168.5.1/stream",
    "http://192.168.5.1:81/stream",
    "http://192.168.5.1:8080/?action=stream",
    "http://192.168.5.1/mjpeg/1",
]

latest_result = None


def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


def clamp(value, lo, hi):
    return max(lo, min(hi, int(value)))


def send_servo(ser, servo, angle):
    ser.write(f"{servo}{angle}$".encode())


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


def main():
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "efficientdet_lite0.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)

    cap = None
    selected_stream = None
    for candidate in STREAM_CANDIDATES:
        test_cap = cv2.VideoCapture(candidate)
        test_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        test_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        if not test_cap.isOpened():
            test_cap.release()
            continue
        ok, frame = test_cap.read()
        if ok and frame is not None:
            cap = test_cap
            selected_stream = candidate
            break
        test_cap.release()

    if cap is None:
        raise RuntimeError(
            "Could not open any ESP32 stream endpoint.\n"
            "Tried:\n- " + "\n- ".join(STREAM_CANDIDATES)
        )
    print(f"Using stream: {selected_stream}")

    options = vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback,
        max_results=3,
    )

    last_send = 0.0
    t0 = time.monotonic()

    with vision.ObjectDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]

            timestamp_ms = int((time.monotonic() - t0) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(mp_image, timestamp_ms=timestamp_ms)

            local_result = latest_result
            frame = visualize(frame, local_result)

            if local_result and (time.monotonic() - last_send) >= SEND_INTERVAL_S:
                best = None
                best_score = -1.0
                for detection in local_result.detections:
                    score = detection.categories[0].score
                    if score >= CONF_THRESHOLD and score > best_score:
                        best = detection
                        best_score = score

                if best is not None:
                    bbox = best.bounding_box
                    cx = bbox.origin_x + bbox.width / 2.0
                    cy = bbox.origin_y + bbox.height / 2.0

                    nx = cx / float(w)
                    ny = cy / float(h)

                    base = clamp(180 * nx, 0, 180)
                    shoulder = clamp(120 - ny * 60, 0, 180)
                    elbow = clamp(60 + ny * 60, 0, 180)

                    send_servo(ser, "A", base)
                    send_servo(ser, "B", shoulder)
                    send_servo(ser, "C", elbow)
                    last_send = time.monotonic()

                    cv2.putText(
                        frame,
                        f"A:{base} B:{shoulder} C:{elbow}",
                        (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("MindAssist Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"Could not connect to Arduino on {ARDUINO_PORT}: {e}")
    except KeyboardInterrupt:
        print("Program interrupted by user.")
