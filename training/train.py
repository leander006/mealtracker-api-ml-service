"""Fine-tunes a YOLOv8 model on your food-class dataset.

Starts from Ultralytics' pretrained COCO weights (yolov8n.pt by default) and
fine-tunes on your 15 food classes, rather than training from scratch -
transfer learning from COCO gives you usable accuracy on far less data than
training from random weights would need (a few hundred images per class
instead of tens of thousands).

Usage:
    python train.py --data data.yaml --epochs 100 --base yolov8n.pt

Model size guide (base weights):
  yolov8n.pt - fastest, least accurate, runs fine on CPU - good for an EC2
               free-tier instance with no GPU.
  yolov8s.pt - better accuracy, still CPU-workable, slower.
  yolov8m.pt+ - needs a GPU to train/infer in reasonable time; skip this on
               free-tier infra.
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml", help="Path to dataset config")
    parser.add_argument("--base", default="yolov8n.pt", help="Pretrained base weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--resume-from", default=None, help="Path to a previous best.pt to continue training from (e.g. after a retrain cycle) instead of --base")
    parser.add_argument("--out", default="../weights/food_yolov8.pt", help="Where to copy the resulting best weights")
    args = parser.parse_args()

    base_weights = args.resume_from or args.base
    model = YOLO(base_weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=15,  # early stop if val loss plateaus for 15 epochs - avoids
                      # overfitting on a small food dataset
        project="runs",
        name="food_detector",
    )

    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.3f}  mAP50: {metrics.box.map50:.3f}")

    best_path = "runs/food_detector/weights/best.pt"
    import shutil, os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shutil.copy(best_path, args.out)
    print(f"Copied best weights to {args.out}")
    print("Set DETECTOR_MODE=real in .env (ml-service root) and restart to use them.")


if __name__ == "__main__":
    main()
