"""Detector interface — the seam your original design already established:
one dataclass, one function signature, two swappable implementations chosen
by config.detector_mode. Nothing downstream (nutrition_routes.py) knows or
cares which one is active.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import settings


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h in pixels


def detect(image_bgr: np.ndarray) -> list[Detection]:
    if settings.detector_mode == "real":
        from detection.yolo_detector import detect_real
        return detect_real(image_bgr)

    from detection.mock_detector import detect_mock
    return detect_mock(image_bgr)
