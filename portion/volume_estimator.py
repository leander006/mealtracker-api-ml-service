"""Estimates food weight in grams from a detection box, using the
reference-object scale when available, falling back to the old
frame-area heuristic when it's not.

This is deliberately kept honest about uncertainty: every estimate carries
a `confidence` and a `method` field. Don't let the API response collapse
this into a single bare number — the client should show a range or an
explicit "estimated" affordance, not a false-precision integer.
"""
from __future__ import annotations

from dataclasses import dataclass

from portion.reference_object import ReferenceScale

# Assumed food "height" (thickness) in cm when we only have a 2D area and no
# depth sensor. This is the single biggest source of error in monocular
# portion estimation — a flat pizza slice and a mounded rice bowl can have
# the same footprint area but very different volumes. Per-food overrides
# (keyed by model_label) should replace this default as you gather ground-
# truth photos; treat this dict as a living calibration table, not a
# constant.
DEFAULT_HEIGHT_CM = 2.0
HEIGHT_OVERRIDES_CM: dict[str, float] = {
    "fried_rice": 2.5,
    "dal": 3.0,          # served in a bowl, mounds higher than flat items
    "pizza_slice": 1.2,
    "roti": 0.4,
    "grilled_chicken_breast": 2.2,
}

# When no reference card is found, fall back to this: weight is a fraction
# of the food's typical_serving_g, scaled by how much of the frame the
# detection box occupies relative to a "normal" framing. This is the
# original heuristic — kept as a degraded-confidence fallback, not removed,
# per the earlier design decision to keep the app functional without the
# reference object.
ASSUMED_FRAME_COVERAGE_AT_TYPICAL_SERVING = 0.15


@dataclass
class PortionEstimate:
    weight_g: float
    confidence: float  # 0-1
    method: str  # "reference_object" | "frame_area_fallback"


def estimate_weight_g(
    box_px: tuple[int, int, int, int],   # x, y, w, h of the YOLO detection box
    frame_shape: tuple[int, int],         # (height, width) of the source image
    model_label: str,
    typical_serving_g: float | None,
    density_g_cm3: float | None,
    reference_scale: ReferenceScale | None,
) -> PortionEstimate:
    _, _, box_w, box_h = box_px

    if reference_scale is not None and density_g_cm3 is not None:
        return _estimate_via_reference_object(
            box_w, box_h, model_label, density_g_cm3, reference_scale
        )

    return _estimate_via_frame_area_fallback(
        box_w, box_h, frame_shape, typical_serving_g
    )


def _estimate_via_reference_object(
    box_w: int,
    box_h: int,
    model_label: str,
    density_g_cm3: float,
    reference_scale: ReferenceScale,
) -> PortionEstimate:
    px_per_cm = reference_scale.px_per_cm
    width_cm = box_w / px_per_cm
    depth_cm = box_h / px_per_cm
    height_cm = HEIGHT_OVERRIDES_CM.get(model_label, DEFAULT_HEIGHT_CM)

    # Treat the footprint as an ellipse rather than a rectangle — most food
    # items (a scoop of rice, a chicken breast, a slice) are closer to
    # elliptical than to filling their full bounding box edge-to-edge.
    # This alone measurably reduces overestimation vs. a naive box volume.
    area_cm2 = 3.14159 * (width_cm / 2) * (depth_cm / 2)
    volume_cm3 = area_cm2 * height_cm
    weight_g = volume_cm3 * density_g_cm3

    # Confidence combines the card-match confidence with a penalty for
    # relying on a generic (non-per-food-calibrated) height assumption.
    height_is_calibrated = model_label in HEIGHT_OVERRIDES_CM
    height_confidence_penalty = 0.0 if height_is_calibrated else 0.15
    confidence = max(0.0, reference_scale.confidence - height_confidence_penalty)

    return PortionEstimate(
        weight_g=round(weight_g, 1),
        confidence=round(confidence, 3),
        method="reference_object",
    )


def _estimate_via_frame_area_fallback(
    box_w: int,
    box_h: int,
    frame_shape: tuple[int, int],
    typical_serving_g: float | None,
) -> PortionEstimate:
    frame_h, frame_w = frame_shape
    frame_coverage = (box_w * box_h) / (frame_w * frame_h)
    baseline = typical_serving_g or 150.0  # generic default if label has no prior

    weight_g = baseline * (frame_coverage / ASSUMED_FRAME_COVERAGE_AT_TYPICAL_SERVING)
    # Clamp to a sane range — this heuristic has no real floor/ceiling logic
    # and can produce absurd values (e.g. 2000g) on an oddly-cropped photo.
    weight_g = max(20.0, min(weight_g, baseline * 3))

    return PortionEstimate(
        weight_g=round(weight_g, 1),
        confidence=0.35,  # deliberately low and fixed — this path is a guess, say so
        method="frame_area_fallback",
    )
