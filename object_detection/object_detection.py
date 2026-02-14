import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import numpy as np

import serial
import time

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 4
FONT_THICKNESS = 5
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
    Returns:
        Image with bounding boxes.
    """
    for detection in detection_result.detections:

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        if probability < 0.5:  # Skip detections with low confidence
            continue

        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                        MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


arduino_port = "/dev/cu.usbmodem211301"  # Update this to your Arduino's port
baud_rate = 9600

def send_servo(ser, servo, angle):
    ser.write(f"{servo}{angle}$".encode())


try:
    ser = serial.Serial(arduino_port, baud_rate)
    cap = cv2.VideoCapture(0)
    latest_result = None

    options = vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback
    )

    while True:
        with vision.ObjectDetector.create_from_options(options) as landmarker:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = landmarker.detect_async(mp_image, timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))
                if latest_result:
                    for detection in latest_result.detections:
                        category = detection.categories[0]
                        probability = round(category.score, 2)
                        if probability > 0.5:  # Detected with high confidence
                            bbox = detection.bounding_box
                            cx = bbox.origin_x + bbox.width / 2
                            cy = bbox.origin_y + bbox.height / 2

                            nx = cx / bbox.width
                            ny = cy / bbox.height

                            base = int(180 * nx)
                            shoulder = int(120 - ny * 60)
                            elbow = int(60 + ny * 60)

                            send_servo(ser, 'A', base)
                            send_servo(ser, 'B', shoulder)
                            send_servo(ser, 'C', elbow)

                            time.sleep(10)

except serial.SerialException as e:
    print(f"Could not connect to Arduino on {arduino_port}: {e}")
except KeyboardInterrupt:
    print("Program interrupted by user. Closing serial port.")
    if 'ser' in locals() and ser.isOpen():
        cap.release()
        ser.close()
        cv2.destroyAllWindows()


