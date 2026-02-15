"""FastAPI YOLO inference server for ESP camera frames."""

from __future__ import annotations

import io
import os

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO


app = FastAPI()

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
yolo_model = None


@app.on_event("startup")
async def startup_event():
    global yolo_model
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"[modal_server] Loaded YOLO model: {YOLO_MODEL_PATH}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": YOLO_MODEL_PATH}


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

    try:
        results = yolo_model(img_np, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    out = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            cls = int(box.cls[0].item())
            label = yolo_model.names.get(cls, str(cls))
            out.append(
                {
                    "label": label,
                    "score": conf,
                    "origin_x": x1,
                    "origin_y": y1,
                    "width": max(1, x2 - x1),
                    "height": max(1, y2 - y1),
                }
            )

    return {"detections": out}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
