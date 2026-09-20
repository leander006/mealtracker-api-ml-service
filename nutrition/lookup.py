"""Resolves a model_label (YOLO class name) to nutrition data.

Lookup strategy, in order:
  1. Exact match in food_label_map (the common, fast path — this is what the
     usda_loader seeds).
  2. Fuzzy fallback via pg_trgm similarity on foods.description, for labels
     that were never explicitly seeded (keeps the system degrading
     gracefully instead of hard-failing when the model recognizes something
     new before you've re-run the seeder).
  3. Live USDA fallback — only reached if both local checks miss. Calls
     FoodData Central on the spot, and writes the result back into
     foods/food_nutrients/food_label_map (source="usda_live", verified=False)
     so the *next* scan of this same label is a fast local exact-match hit
     again. This only runs if USDA_API_KEY is configured; if it's not set,
     or the API call fails for any reason (network, rate limit, no match),
     this tier is skipped silently and the caller gets None — same
     "unrecognized item" behavior as before this tier existed.

Returns None (never raises) on total miss — caller decides how to handle
"we detected food but have no macro data for it" (e.g. surface it to the
user as "unrecognized item" rather than silently guessing).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import settings
from nutrition.models import Food, FoodLabelMap, FoodNutrient
from nutrition.usda_client import fetch_usda_food, find_or_create_food

logger = logging.getLogger(__name__)


@dataclass
class NutritionResult:
    food_description: str
    fdc_id: int
    per_100g: dict  # calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg
    density_g_cm3: float | None
    typical_serving_g: float | None
    match_type: str  # "exact" | "fuzzy" | "external_api" — surface this in the API
                      # response so the client can show lower confidence / a
                      # "from online lookup" badge for the non-exact matches


def _to_result(food: Food, nutrients: FoodNutrient, typical_serving_g: float | None, match_type: str) -> NutritionResult:
    return NutritionResult(
        food_description=food.description,
        fdc_id=food.fdc_id,
        per_100g={
            "calories_kcal": float(nutrients.calories_kcal),
            "protein_g": float(nutrients.protein_g),
            "carbs_g": float(nutrients.carbs_g),
            "fat_g": float(nutrients.fat_g),
            "fiber_g": float(nutrients.fiber_g),
            "sugar_g": float(nutrients.sugar_g),
            "sodium_mg": float(nutrients.sodium_mg),
        },
        density_g_cm3=float(nutrients.density_g_cm3) if nutrients.density_g_cm3 else None,
        typical_serving_g=float(typical_serving_g) if typical_serving_g else None,
        match_type=match_type,
    )


def lookup_by_label(session: Session, model_label: str) -> NutritionResult | None:
    row = (
        session.query(FoodLabelMap)
        .filter(FoodLabelMap.model_label == model_label)
        .order_by(FoodLabelMap.priority.asc())
        .first()
    )
    if row is not None:
        return _to_result(row.food, row.food.nutrients, row.typical_serving_g, "exact")

    fuzzy = _fuzzy_fallback(session, model_label)
    if fuzzy is not None:
        return fuzzy

    return _live_usda_fallback(session, model_label)


def _fuzzy_fallback(session: Session, model_label: str) -> NutritionResult | None:
    # pg_trgm similarity search. Requires: CREATE EXTENSION pg_trgm;
    # Threshold of 0.3 is a reasonable starting point for short food-label
    # strings; tune based on false-positive rate once you have real traffic.
    query = text(
        """
        SELECT f.id, f.description, f.fdc_id, similarity(f.description, :label) AS sim
        FROM foods f
        WHERE f.description % :label
        ORDER BY sim DESC
        LIMIT 1
        """
    )
    row = session.execute(query, {"label": model_label.replace("_", " ")}).first()
    if row is None:
        return None

    food = session.query(Food).get(row.id)
    if food is None or food.nutrients is None:
        return None
    return _to_result(food, food.nutrients, None, "fuzzy")


def _live_usda_fallback(session: Session, model_label: str) -> NutritionResult | None:
    if not settings.usda_api_key:
        return None  # not configured - skip silently, same as "no match"

    query = model_label.replace("_", " ")
    try:
        usda_food = fetch_usda_food(query, settings.usda_api_key, timeout=6)
    except Exception:
        # Network hiccup, timeout, rate limit, malformed response, whatever -
        # a scan should never 500 because an external API had a bad moment.
        # Worst case the user falls through to manual entry, same as today.
        logger.warning("Live USDA fallback failed for label=%s", model_label, exc_info=True)
        return None

    if usda_food is None:
        return None

    food = find_or_create_food(session, usda_food, source="usda_live", verified=False)

    # Cache the mapping too, not just the food row - so the *next* scan of
    # this exact label is a fast local exact-match hit, no USDA call at all.
    session.add(FoodLabelMap(model_label=model_label, food_id=food.id, priority=2))
    session.commit()

    return _to_result(food, food.nutrients, None, "external_api")
