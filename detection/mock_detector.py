"""Mock detector — no model weights required. Used when DETECTOR_MODE=mock,
which is the default so `docker compose up` works for anyone on the team
without them needing your trained .pt file first.

Returns a plausible-looking single detection from your 15-class list so the
rest of the pipeline (nutrition lookup, portion estimation, meal saving) is
fully exercisable in local dev and in CI.
"""
from __future__ import annotations

import random

import numpy as np

from detection.interface import Detection

CLASSES = [
    "fried_rice", "grilled_chicken_breast", "pizza_slice", "roti", "dal",
    "oats", "paneer_curry", "curd", "protein_shake", "grilled_fish",
    "sweet_potato", "almonds", "chole",
]


def detect_mock(image_bgr: np.ndarray) -> list[Detection]:
    h, w = image_bgr.shape[:2]
    label = random.choice(CLASSES)
    # A plausible centered box, ~40% of frame width/height — close enough to
    # real-world framing for testing portion estimation math end-to-end.
    box_w, box_h = int(w * 0.4), int(h * 0.4)
    x, y = (w - box_w) // 2, (h - box_h) // 2
    return [Detection(label=label, confidence=round(random.uniform(0.55, 0.95), 2), box=(x, y, box_w, box_h))]
