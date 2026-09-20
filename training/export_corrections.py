"""Pulls correction rows from every training DB (all of them, not just the
currently-active one - see feedback/db_router.py's rotation model) and turns
them into a YOLO-format dataset you can use for a retrain cycle.

IMPORTANT LIMITATION, read before relying on this: corrections store a
corrected LABEL and WEIGHT, not a bounding box - the feedback UI never asked
the user to draw one. This script assumes the food fills most of the frame
(true for the close-up single-item photos the app's capture flow encourages)
and generates a bounding box covering the center 80% of each image as a
stand-in. This is a real accuracy ceiling: it teaches the model roughly
where the food is, not precisely. If retrain accuracy plateaus, the next
investment should be adding a lightweight bounding-box step to the
correction UI (even a single draggable rectangle) rather than more data
under this approximation.

Usage:
    python export_corrections.py --out dataset/retrain_batch_1
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys

import requests
import yaml
from PIL import Image
from sqlalchemy import create_engine, text

# Import the training DB URLs the same way ml-service does, so this script
# stays in sync with whatever's configured there. training/ is nested
# directly inside the ml-service repo, so config.py is one level up.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import settings  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Center-crop fraction used as the pseudo-bounding-box (see module docstring).
ASSUMED_BOX_FRACTION = 0.8


def load_class_names(data_yaml_path: str) -> dict[str, int]:
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)
    return {name: idx for idx, name in data["names"].items()}


def fetch_corrections(engine) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, image_url, corrected_label FROM corrections"
        )).mappings().all()
    return [dict(r) for r in rows]


def download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def write_yolo_label(path: str, class_id: int) -> None:
    # Pseudo-box: centered, ASSUMED_BOX_FRACTION of frame width/height.
    # YOLO format is normalized so these numbers don't depend on image size.
    cx, cy = 0.5, 0.5
    w = h = ASSUMED_BOX_FRACTION
    with open(path, "w") as f:
        f.write(f"{class_id} {cx} {cy} {w} {h}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output dataset directory")
    parser.add_argument("--data-yaml", default="data.yaml", help="Existing data.yaml with your class name->id mapping")
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    class_ids = load_class_names(args.data_yaml)

    train_img_dir = os.path.join(args.out, "images", "train")
    val_img_dir = os.path.join(args.out, "images", "val")
    train_lbl_dir = os.path.join(args.out, "labels", "train")
    val_lbl_dir = os.path.join(args.out, "labels", "val")
    for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        os.makedirs(d, exist_ok=True)

    training_urls = settings.training_db_url_list
    logger.info("Exporting corrections from %d training DB(s)", len(training_urls))

    total, skipped = 0, 0
    for db_url in training_urls:
        engine = create_engine(db_url)
        corrections = fetch_corrections(engine)
        logger.info("Found %d correction rows in this DB", len(corrections))

        for i, row in enumerate(corrections):
            label = row["corrected_label"]
            if label not in class_ids:
                logger.warning("Skipping correction %s: label '%s' not in data.yaml classes", row["id"], label)
                skipped += 1
                continue

            try:
                image = download_image(row["image_url"])
            except Exception as e:
                logger.warning("Skipping correction %s: image download failed (%s)", row["id"], e)
                skipped += 1
                continue

            is_val = (i % int(1 / args.val_split)) == 0
            img_dir = val_img_dir if is_val else train_img_dir
            lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            filename = f"correction_{row['id']}"
            image.save(os.path.join(img_dir, f"{filename}.jpg"), "JPEG")
            write_yolo_label(os.path.join(lbl_dir, f"{filename}.txt"), class_ids[label])
            total += 1

    logger.info("Exported %d images (%d skipped) to %s", total, skipped, args.out)
    logger.info("Next: python train.py --data %s --resume-from <previous best.pt>", args.data_yaml)


if __name__ == "__main__":
    main()
