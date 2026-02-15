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
from eeg_processor import EEGProcessor, ATTENTION_THRESHOLD
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
        if det["score"] < 0.5:
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


def call_modal_inference(frame: np.ndarray) -> List[Dict]:
    # Encode to JPEG
    ret, buf = cv2.imencode('.jpg', frame)
    if not ret:
        return []
    files = {'file': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
    try:
        r = requests.post(MODAL_ENDPOINT, files=files, timeout=5)
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
        print(1)
        bbox = det
        cx = bbox['origin_x'] + bbox['width'] / 2
        cy = bbox['origin_y'] + bbox['height'] / 2
        nx = cx / (bbox['width'] if bbox['width'] else 1)
        ny = cy / (bbox['height'] if bbox['height'] else 1)
        base = int(180 * nx)
        shoulder = int(120 - ny * 60)
        elbow = int(60 + ny * 60)
        print(f"[ARM] Would move: base={base}, shoulder={shoulder}, elbow={elbow}")
        # send_servo(ser, 'A', base)
        # send_servo(ser, 'B', shoulder)
        # send_servo(ser, 'C', elbow)


# Shared EEG state
latest_attention = {'value': 0.0}
eeg_lock = threading.Lock()

def eeg_on_data(attn, med, sig, blink, state):
    with eeg_lock:
        latest_attention['value'] = attn

def start_eeg_thread():
    proc = EEGProcessor(on_data=eeg_on_data)
    t = threading.Thread(target=proc.start, daemon=True)
    t.start()
    return proc

def main():
    # try:
    #     ser = serial.Serial(arduino_port, baud_rate)
    # except serial.SerialException as e:
    #     print(f"Could not connect to Arduino on {arduino_port}: {e}")
    #     ser = None

    # Start EEG processor in background
    # start_eeg_thread()

    # cap = cv2.VideoCapture(0)

    cap = open_camera()
    # fps_time = time.monotonic()
    # frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 0)
            detections = call_modal_inference(frame)
            # Only move arm if object detected AND attention high
            # with eeg_lock:
            #     attn = latest_attention['value']
            if detections:  # and attn >= ATTENTION_THRESHOLD:
                print(f"[FUSION] object detected: moving arm.")
                process_and_control(detections)
            # else:
            #     if detections:
            #         print(f"[FUSION] object detected: NOT moving arm.")
            vis = visualize(frame, detections)
            cv2.imshow('frame', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Program interrupted by user. Closing.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # if 'ser' in locals() and ser is not None and ser.isOpen():
        #     ser.close()


if __name__ == '__main__':
    main()
