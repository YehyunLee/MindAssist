
import cv2
import numpy as np
import os
import requests
import serial
import threading
import time
from typing import List, Dict



# EEG imports
import sys
# from eeg_processor import EEGProcessor, ATTENTION_THRESHOLD
from esp_cam_viewer import FrameGrabber, open_camera

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 4
FONT_THICKNESS = 5
TEXT_COLOR = (255, 0, 0)  # red


MODAL_ENDPOINT = os.environ.get("MODAL_ENDPOINT", "http://localhost:8080/detect")

arduino_port = os.environ.get("ARDUINO_PORT", "/dev/cu.usbmodem211301")
baud_rate = int(os.environ.get("ARDUINO_BAUD", "9600"))


def visualize(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    for det in detections:
        if det["score"] < 0.4:
            continue
        start_point = (det["origin_x"], det["origin_y"]) 
        end_point = (det["origin_x"] + det["width"], det["origin_y"] + det["height"])
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
        result_text = f"{det['label']} ({det['score']:.2f})"
        text_location = (MARGIN + det["origin_x"], MARGIN + ROW_SIZE + det["origin_y"]) 
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image


def send_servo(ser, servo, angle):
    ser.write(f"{servo}{angle}$".encode())


# --- Modal YOLO inference ---
def call_modal_inference(frame: np.ndarray) -> List[Dict]:
    # Encode to JPEG
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        return []
    files = {'file': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
    try:
        r = requests.post(MODAL_ENDPOINT, files=files, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get('detections', [])
    except Exception as e:
        print(f"Error calling modal endpoint: {e}")
        return []


def process_and_control(detections: List[Dict]):
    for det in detections:
        if det['score'] < 0.5:
            continue
        print(det)
        bbox = det
        cx = bbox['origin_x'] + bbox['width'] / 2
        cy = bbox['origin_y'] + bbox['height'] / 2
        nx = cx / (bbox['width'] if bbox['width'] else 1)
        ny = cy / (bbox['height'] if bbox['height'] else 1)
        base = int(180 * nx)
        shoulder = int(120 - ny * 60)
        elbow = int(60 + ny * 60)
        print(f"[ARM] Would move: base={base}, shoulder={shoulder}, elbow={elbow}")


def main():
    cap = open_camera()
    # cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 0)
            detections = call_modal_inference(frame)
            if detections:
                process_and_control(detections)
            vis = visualize(frame, detections)
            cv2.imshow('frame', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Program interrupted by user. Closing.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
