from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image
import io

import mediapipe as mp
from mediapipe.tasks.python import vision

app = FastAPI()


def pil_to_bgr_array(pil_image: Image.Image) -> np.ndarray:
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb)
    # Convert RGB to BGR for OpenCV style
    return arr[:, :, ::-1]


# Initialize detector once on startup
detector = None


@app.on_event("startup")
async def startup_event():
    global detector
    options = vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.ObjectDetector.create_from_options(options)


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    global detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")

    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    bgr = pil_to_bgr_array(pil)

    # Mediapipe expects an mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr)
    result = detector.detect(mp_image)

    out = []
    for detection in result.detections:
        cat = detection.categories[0]
        bbox = detection.bounding_box
        out.append({
            "label": cat.category_name,
            "score": float(cat.score),
            "origin_x": int(bbox.origin_x),
            "origin_y": int(bbox.origin_y),
            "width": int(bbox.width),
            "height": int(bbox.height),
        })

    return {"detections": out}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
