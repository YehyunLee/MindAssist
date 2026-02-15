"""Modal cloud deployment for YOLO inference.

Deploy with: modal deploy modal_yolo.py
Run locally for testing: modal serve modal_yolo.py

After deployment, you'll get a URL like:
  https://<your-username>--yolo-inference-fastapi-app.modal.run
"""

from __future__ import annotations

import io

import modal

# Define the Modal image with required dependencies and pre-download the model
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV dependencies
    .pip_install(
        "ultralytics>=8.0.0",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "python-multipart>=0.0.6",
    )
    .run_commands("python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
)

app = modal.App("yolo-inference", image=image)

# Global model reference (loaded once per container)
yolo_model = None


def get_model():
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")
        print("[Modal] YOLO model loaded")
    return yolo_model


# FastAPI web endpoint
from fastapi import FastAPI, File, HTTPException, UploadFile

web_app = FastAPI()


@web_app.get("/health")
async def health():
    return {"status": "ok", "platform": "modal"}


@web_app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    import numpy as np
    from PIL import Image

    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_np = np.array(pil)
    model = get_model()

    try:
        results = model(img_np, verbose=False)
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
            label = model.names.get(cls, str(cls))
            out.append({
                "label": label,
                "score": conf,
                "origin_x": x1,
                "origin_y": y1,
                "width": max(1, x2 - x1),
                "height": max(1, y2 - y1),
            })

    return {"detections": out}


@app.function(gpu="T4", scaledown_window=120)
@modal.asgi_app()
def fastapi_app():
    return web_app
