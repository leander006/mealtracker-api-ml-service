"""POST /estimate-meal

Takes a meal photo, runs food detection (your existing detector interface —
mock or real YOLOv8, unchanged), finds the reference card for scale, and
returns per-item weight + macro estimates plus a meal total.

This is additive to your existing /detect endpoint, not a replacement — keep
/detect for raw bounding-box debugging/testing, and use this one from the
mobile/web client for the actual user-facing flow.
"""
from __future__ import annotations

import io
from dataclasses import asdict

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from nutrition.lookup import lookup_by_label
from portion.reference_object import find_reference_scale
from portion.volume_estimator import estimate_weight_g

# Swap this import for your existing detector interface — this router
# expects a `detect(image: np.ndarray) -> list[Detection]` function where
# Detection has .label, .confidence, .box (x, y, w, h). It's already behind
# an interface per your mock/real config switch, so nothing here cares which
# implementation is active.
from detection.interface import detect as run_detection  # your existing module
from db import get_session  # your existing SQLAlchemy session dependency

router = APIRouter()


def _decode_upload(raw_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@router.post("/estimate-meal")
async def estimate_meal(
    photo: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    raw_bytes = await photo.read()
    try:
        image_bgr = _decode_upload(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detections = run_detection(image_bgr)
    if not detections:
        return {"items": [], "meal_totals": _empty_totals(), "reference_object_found": False}

    reference_scale = find_reference_scale(image_bgr)
    frame_shape = image_bgr.shape[:2]  # (height, width)

    items = []
    for det in detections:
        nutrition = lookup_by_label(session, det.label)
        if nutrition is None:
            # Detected something we have no nutrition data for. Surface it
            # rather than silently dropping it or guessing — the client
            # should let the user manually search/select in this case.
            items.append({
                "label": det.label,
                "detection_confidence": det.confidence,
                "nutrition_found": False,
            })
            continue

        portion = estimate_weight_g(
            box_px=det.box,
            frame_shape=frame_shape,
            model_label=det.label,
            typical_serving_g=nutrition.typical_serving_g,
            density_g_cm3=nutrition.density_g_cm3,
            reference_scale=reference_scale,
        )

        scale_factor = portion.weight_g / 100.0
        macros = {k: round(v * scale_factor, 1) for k, v in nutrition.per_100g.items()}

        items.append({
            "label": det.label,
            "matched_food": nutrition.food_description,
            "match_type": nutrition.match_type,
            "detection_confidence": det.confidence,
            "portion": {
                "weight_g": portion.weight_g,
                "confidence": portion.confidence,
                "method": portion.method,
            },
            "macros": macros,
            "nutrition_found": True,
        })

    return {
        "items": items,
        "meal_totals": _sum_totals(items),
        "reference_object_found": reference_scale is not None,
    }


def _empty_totals() -> dict:
    return {"calories_kcal": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "fiber_g": 0, "sugar_g": 0, "sodium_mg": 0}


def _sum_totals(items: list[dict]) -> dict:
    totals = _empty_totals()
    for item in items:
        if not item.get("nutrition_found"):
            continue
        for key, value in item["macros"].items():
            totals[key] = round(totals.get(key, 0) + value, 1)
    return totals
