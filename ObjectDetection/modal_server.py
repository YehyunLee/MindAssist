from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
from PIL import Image
import io
import os


# YOLO imports
from ultralytics import YOLO

app = FastAPI()


def pil_to_bgr_array(pil_image: Image.Image) -> np.ndarray:
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb)
    # Convert RGB to BGR for OpenCV style
    return arr[:, :, ::-1]



# Initialize YOLO model once on startup
yolo_model = None
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")  # Default to YOLOv8 nano


@app.on_event("startup")
async def startup_event():
    global yolo_model
    yolo_model = YOLO(YOLO_MODEL_PATH)



@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    global yolo_model
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLO model not initialized")

    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_np = np.array(pil)
    results = yolo_model(img_np)
    out = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls] if hasattr(yolo_model, 'names') else str(cls)
            out.append({
                "label": label,
                "score": conf,
                "origin_x": x1,
                "origin_y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            })
    return {"detections": out}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
