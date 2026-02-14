import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import time
import pprint


def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


def draw_landmarks(frame, result):
    if result.face_landmarks is None:
        return frame
    
    pprint.pprint(result)
    exit(0)
    
    h, w, _ = frame.shape

    for hand in result.face_landmarks:
        for landmark in hand:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    # cv2.putText(frame, result.face_landmarks[0].category_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    latest_result = None

    # options = vision.HandLandmarkerOptions(
    #     base_options=mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task'),
    #     running_mode=vision.RunningMode.LIVE_STREAM,
    #     min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
    #     min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
    #     min_tracking_confidence = 0.3, # lower than value to get predictions more often
    #     result_callback=result_callback,
    # )

    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback
    )

    while True:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
        # with vision.HandLandmarker.create_from_options(options) as landmarker:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = landmarker.detect_async(mp_image, timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))
                if latest_result and latest_result.face_landmarks: 
                    frame = draw_landmarks(frame, latest_result)
                cv2.imshow('Face Landmarker Live', frame)

            # Exit the loop if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
