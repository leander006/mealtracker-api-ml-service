"""Detects a standard reference object in frame and computes a pixel->cm scale.

Design decision: use a credit-card / ID-card sized object (ISO/IEC 7810 ID-1,
85.60mm x 53.98mm — the size of every debit/credit/ID card) as the reference,
because it's something almost every user already has in their wallet, unlike
a ruler or a coin of a specific denomination. The capture-flow prompt should
say: "place a card next to your plate."

V1 implementation here is classical CV (contour + aspect-ratio matching), NOT
a second YOLO class. Reasoning: it needs zero additional training data and
ships today. It also fails informatively when it can't find a confident card
— which matters, because feeding a bad scale estimate downstream silently is
worse than telling the user "couldn't find your reference card, try again."

Upgrade path (do this once you have real usage data): fine-tune the same
YOLO model with an added `reference_card` class. Classical CV struggles with
low contrast (white card on white plate) and partial occlusion — a learned
detector handles both far better. Swap this module's internals without
changing its public interface (`find_reference_scale`) when you do.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ISO/IEC 7810 ID-1 card dimensions in mm — true for credit cards, debit
# cards, most national ID cards and driver's licenses.
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CARD_ASPECT_RATIO = CARD_WIDTH_MM / CARD_HEIGHT_MM  # ~1.586

# How much a candidate contour's aspect ratio may deviate from a true card's
# before we reject it. Loose enough to tolerate perspective skew from a
# non-overhead photo angle, tight enough to reject plates/bowls/phones.
ASPECT_RATIO_TOLERANCE = 0.18


@dataclass
class ReferenceScale:
    px_per_cm: float
    confidence: float  # 0-1, based on how well the match fit expected card geometry
    card_bbox_px: tuple[int, int, int, int]  # x, y, w, h — surfaced for debugging/overlay


def find_reference_scale(image_bgr: np.ndarray) -> ReferenceScale | None:
    """Returns None if no confident card match is found — caller must handle
    this (fall back to the old area-heuristic estimate, flagged as low
    confidence, rather than blocking the whole detection response)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best: ReferenceScale | None = None
    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1500:  # discard tiny noise contours
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        if w == 0 or h == 0:
            continue

        long_side, short_side = max(w, h), min(w, h)
        observed_ratio = long_side / short_side
        ratio_error = abs(observed_ratio - CARD_ASPECT_RATIO) / CARD_ASPECT_RATIO

        if ratio_error > ASPECT_RATIO_TOLERANCE:
            continue

        # Score candidates by how close their aspect ratio is to a true
        # card's and how large they are (prefer the card closest to camera /
        # least occluded over tiny spurious matches elsewhere in frame).
        score = (1 - ratio_error) * area
        if score <= best_score:
            continue

        px_per_cm = long_side / (CARD_WIDTH_MM / 10)  # mm -> cm
        x, y, bw, bh = cv2.boundingRect(contour)

        best = ReferenceScale(
            px_per_cm=px_per_cm,
            confidence=round(1 - ratio_error, 3),
            card_bbox_px=(x, y, bw, bh),
        )
        best_score = score

    if best is None:
        logger.info("No reference card found in frame")
    return best
