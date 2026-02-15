"""Modal cloud fine-tuning for YOLO model.

Usage (fine-tune with small dataset):
  1. Upload your base model: modal volume put yolo-training-vol ./yolov8n.pt /models/
  2. Upload dataset: modal volume put yolo-training-vol ./my_data /datasets/
  3. Train: modal run modal_train.py --action train --epochs 50
  4. Download: modal run modal_train.py --action download

For small datasets, we use:
  - Lower learning rate
  - Freeze early layers
  - Data augmentation
"""

from __future__ import annotations

import modal

# Persistent volume for datasets and trained models
volume = modal.Volume.from_name("yolo-training-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# Training image with YOLO and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.0.0",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        "opencv-python-headless>=4.8.0",
    )
)

app = modal.App("yolo-training", image=image)


@app.function(
    gpu="T4",  # Use T4 for training (or "A10G", "A100" for faster)
    timeout=3600 * 2,  # 2 hour timeout
    volumes={VOLUME_PATH: volume},
)
def train_yolo(
    data_yaml: str = "data.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    project_name: str = "mindassist",
    freeze_layers: int = 10,
):
    """Fine-tune YOLO model on Modal GPU with small dataset.
    
    Args:
        data_yaml: Path to data.yaml config (relative to /vol/datasets/)
        epochs: Number of training epochs (50 is good for fine-tuning)
        imgsz: Image size for training
        batch: Batch size (smaller for limited data)
        project_name: Name for the training project/output folder
        freeze_layers: Number of backbone layers to freeze (helps prevent overfitting)
    """
    from ultralytics import YOLO
    import os

    # Check for uploaded base model, fallback to downloading
    model_path = f"{VOLUME_PATH}/models/yolov8n.pt"
    if os.path.exists(model_path):
        print(f"[TRAIN] Using uploaded model: {model_path}")
    else:
        print(f"[TRAIN] No uploaded model found, downloading yolov8n.pt")
        model_path = "yolov8n.pt"

    # Check if dataset exists in volume
    dataset_path = f"{VOLUME_PATH}/datasets"
    data_config = f"{dataset_path}/{data_yaml}"
    
    if not os.path.exists(data_config):
        print(f"[ERROR] data.yaml not found at {data_config}")
        print(f"[INFO] Upload dataset: modal volume put yolo-training-vol ./my_data /datasets/")
        print(f"[INFO] Volume contents:")
        for root, dirs, files in os.walk(VOLUME_PATH):
            for f in files:
                print(f"  {os.path.join(root, f)}")
        return None

    print(f"[TRAIN] Loading model: {model_path}")
    model = YOLO(model_path)

    # Fine-tuning settings for small datasets
    print(f"[TRAIN] Fine-tuning: {epochs} epochs, batch={batch}, freeze={freeze_layers} layers")
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=f"{VOLUME_PATH}/runs",
        name=project_name,
        exist_ok=True,
        verbose=True,
        # Fine-tuning optimizations for small datasets
        freeze=freeze_layers,  # Freeze backbone layers
        lr0=0.001,  # Lower initial learning rate
        lrf=0.01,  # Lower final learning rate
        warmup_epochs=3,  # Warmup period
        augment=True,  # Enable augmentation
        hsv_h=0.015,  # HSV augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,  # Rotation augmentation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,  # Horizontal flip
        mosaic=1.0,  # Mosaic augmentation
        patience=10,  # Early stopping patience
    )

    # Save best model path
    best_model_path = f"{VOLUME_PATH}/runs/{project_name}/weights/best.pt"
    print(f"[TRAIN] Training complete!")
    print(f"[TRAIN] Best model saved to: {best_model_path}")
    
    volume.commit()
    return best_model_path


@app.function(volumes={VOLUME_PATH: volume})
def upload_dataset(local_path: str, dataset_name: str = "custom"):
    """Upload a local dataset directory to Modal Volume.
    
    Dataset should be in YOLO format:
    my_dataset/
      data.yaml
      images/
        train/
        val/
      labels/
        train/
        val/
    """
    import os
    import shutil

    dest = f"{VOLUME_PATH}/datasets"
    os.makedirs(dest, exist_ok=True)
    
    print(f"[UPLOAD] Uploading dataset to {dest}")
    # Note: For large datasets, use modal volume put command instead
    print(f"[INFO] For large datasets, use CLI: modal volume put yolo-training-vol {local_path} /datasets/")
    
    volume.commit()
    return dest


@app.function(volumes={VOLUME_PATH: volume})
def download_model(model_path: str = None, project_name: str = "mindassist"):
    """Get the trained model weights."""
    import os

    if model_path is None:
        model_path = f"{VOLUME_PATH}/runs/{project_name}/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        available = []
        runs_dir = f"{VOLUME_PATH}/runs"
        if os.path.exists(runs_dir):
            for d in os.listdir(runs_dir):
                weights = f"{runs_dir}/{d}/weights"
                if os.path.exists(weights):
                    available.append(f"{weights}/best.pt")
        print(f"[INFO] Available models: {available}")
        return None

    with open(model_path, "rb") as f:
        return f.read()


@app.function(volumes={VOLUME_PATH: volume})
def list_volume():
    """List contents of the training volume."""
    import os

    def tree(path, prefix=""):
        if not os.path.exists(path):
            return ["(empty)"]
        items = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                items.append(f"{prefix}{name}/")
                items.extend(tree(full, prefix + "  "))
            else:
                size = os.path.getsize(full) / 1024
                items.append(f"{prefix}{name} ({size:.1f}KB)")
        return items

    contents = tree(VOLUME_PATH)
    for line in contents[:50]:  # Limit output
        print(line)
    return contents


@app.local_entrypoint()
def main(
    action: str = "train",
    epochs: int = 50,
    batch: int = 8,
    freeze: int = 10,
    project: str = "mindassist",
):
    """Local entrypoint for CLI usage.
    
    Examples:
      modal run modal_train.py --action list
      modal run modal_train.py --action train --epochs 50 --batch 8 --freeze 10
      modal run modal_train.py --action download --project mindassist
    
    Setup (run these first):
      modal volume put yolo-training-vol ./yolov8n.pt /models/
      modal volume put yolo-training-vol ./my_dataset /datasets/
    """
    if action == "list":
        list_volume.remote()
    elif action == "train":
        result = train_yolo.remote(
            epochs=epochs,
            batch=batch,
            freeze_layers=freeze,
            project_name=project,
        )
        print(f"Training result: {result}")
    elif action == "download":
        weights = download_model.remote(project_name=project)
        if weights:
            out_path = f"{project}_best.pt"
            with open(out_path, "wb") as f:
                f.write(weights)
            print(f"Downloaded model to: {out_path}")
    else:
        print(f"Unknown action: {action}")
        print("Actions: list, train, download")
