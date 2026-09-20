"""Real YOLOv8-backed detector. Loaded only when DETECTOR_MODE=real, so mock
mode never pays the cost of importing ultralytics or loading weights.
"""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from config import settings
from detection.interface import Detection

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(settings.yolo_weights_path)
    return _model


def detect_real(image_bgr: np.ndarray) -> list[Detection]:
    model = _get_model()
    results = model.predict(
        source=image_bgr,
        conf=settings.yolo_confidence_threshold,
        verbose=False,
    )[0]

    detections: list[Detection] = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        label = model.names[cls_id]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            Detection(label=label, confidence=round(confidence, 3), box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        )
    return detections
