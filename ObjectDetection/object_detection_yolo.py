"""YOLO object detection on ESP32-CAM stream + UDP telemetry for Commander.

Built on top of esp_cam_viewer's proven camera connection logic.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

from esp_cam_viewer import open_camera
from object_stream import UDPBroadcaster


CONF_THRESHOLD = float(os.environ.get("OBJ_CONF_THRESHOLD", "0.4"))
NMS_THRESHOLD = float(os.environ.get("OBJ_NMS_THRESHOLD", "0.45"))
SEND_INTERVAL_S = float(os.environ.get("OBJ_SEND_INTERVAL_S", "0.08"))
SHOW_WINDOW = os.environ.get("OBJECT_SHOW_WINDOW", "1") == "1"
CAMERA_RETRY_S = float(os.environ.get("CAMERA_RETRY_S", "2.0"))
MAX_READ_FAILS = int(os.environ.get("MAX_CAMERA_READ_FAILS", "30"))
YOLO_INPUT_SIZE = int(os.environ.get("YOLO_INPUT_SIZE", "640"))


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def letterbox(image, new_size=640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_w = (new_size - nw) // 2
    pad_h = (new_size - nh) // 2
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return canvas, scale, pad_w, pad_h


def load_model():
    script_dir = Path(__file__).resolve().parent
    model_path = Path(os.environ.get("YOLO_MODEL_PATH", script_dir / "yolov8n.onnx"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n"
            "Export/download an ONNX model and set YOLO_MODEL_PATH."
        )
    net = cv2.dnn.readNetFromONNX(str(model_path))
    print(f"[YOLO] Loaded model: {model_path}")
    return net


def decode_yolo_v8(raw_output, frame_shape, scale, pad_w, pad_h):
    """Decode YOLOv8 ONNX output into list of dict detections."""
    fh, fw = frame_shape[:2]
    out = raw_output
    if isinstance(out, tuple):
        out = out[0]
    if out.ndim == 3:
        out = out[0]
    if out.shape[0] < out.shape[1]:
        out = out.T  # (84, 8400) -> (8400, 84)

    # Expected row = [cx, cy, w, h, cls1..clsN]
    if out.shape[1] < 6:
        return []

    boxes = []
    scores = []
    class_ids = []

    for row in out:
        cls_scores = row[4:]
        class_id = int(np.argmax(cls_scores))
        score = float(cls_scores[class_id])
        if score < CONF_THRESHOLD:
            continue

        cx, cy, w, h = row[:4]
        x1 = (cx - w / 2.0 - pad_w) / scale
        y1 = (cy - h / 2.0 - pad_h) / scale
        x2 = (cx + w / 2.0 - pad_w) / scale
        y2 = (cy + h / 2.0 - pad_h) / scale

        x1 = int(clamp(round(x1), 0, fw - 1))
        y1 = int(clamp(round(y1), 0, fh - 1))
        x2 = int(clamp(round(x2), 0, fw - 1))
        y2 = int(clamp(round(y2), 0, fh - 1))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        boxes.append([x1, y1, bw, bh])
        scores.append(score)
        class_ids.append(class_id)

    if not boxes:
        return []

    keep = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)
    if len(keep) == 0:
        return []

    detections = []
    for i in keep.flatten():
        x, y, w, h = boxes[i]
        cid = class_ids[i]
        label = COCO80[cid] if 0 <= cid < len(COCO80) else f"class_{cid}"
        detections.append(
            {
                "label": label,
                "score": float(scores[i]),
                "origin_x": int(x),
                "origin_y": int(y),
                "width": int(w),
                "height": int(h),
                "class_id": int(cid),
            }
        )
    return detections


def best_detection_payload(detections, frame_w, frame_h):
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
    if not detections:
        return payload

    best = max(detections, key=lambda d: d["score"])
    cx = best["origin_x"] + best["width"] / 2.0
    cy = best["origin_y"] + best["height"] / 2.0
    area = float(best["width"] * best["height"]) / float(max(1, frame_w * frame_h))
    payload.update(
        {
            "found": True,
            "label": best["label"],
            "score": float(best["score"]),
            "cx": float(cx / frame_w),
            "cy": float(cy / frame_h),
            "area": area,
        }
    )
    return payload


def draw_detections(frame, detections):
    for d in detections:
        x, y, w, h = d["origin_x"], d["origin_y"], d["width"], d["height"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{d['label']} {d['score']:.2f}",
            (x, max(20, y - 8)),
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
    net = load_model()
    cap = open_camera_with_retry()
    stream = UDPBroadcaster()
    last_send = 0.0
    read_failures = 0

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

            inp, scale, pad_w, pad_h = letterbox(frame, YOLO_INPUT_SIZE)
            blob = cv2.dnn.blobFromImage(inp, scalefactor=1.0 / 255.0, size=(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward()

            detections = decode_yolo_v8(outputs, frame.shape, scale, pad_w, pad_h)

            now = time.monotonic()
            if now - last_send >= SEND_INTERVAL_S:
                p = best_detection_payload(detections, frame.shape[1], frame.shape[0])
                stream.send(p)
                last_send = now

            if SHOW_WINDOW:
                vis = draw_detections(frame.copy(), detections)
                p = best_detection_payload(detections, vis.shape[1], vis.shape[0])
                if p["found"]:
                    cv2.putText(
                        vis,
                        f"{p['label']} score={p['score']:.2f} cx={p['cx']:.2f} area={p['area']:.3f}",
                        (10, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                cv2.imshow("MindAssist YOLO Detection", vis)
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
