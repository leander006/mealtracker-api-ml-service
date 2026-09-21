"""Resolves a model_label (YOLO class name) to nutrition data.

Lookup strategy, in order:
  1. Exact match in food_label_map (the common, fast path — this is what the
     usda_loader seeds).
  2. Fuzzy fallback via pg_trgm similarity on foods.description, for labels
     that were never explicitly seeded (keeps the system degrading
     gracefully instead of hard-failing when the model recognizes something
     new before you've re-run the seeder).

Returns None (never raises) on total miss — caller decides how to handle
"we detected food but have no macro data for it" (e.g. surface it to the
user as "unrecognized item" rather than silently guessing).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from nutrition.models import Food, FoodLabelMap, FoodNutrient


@dataclass
class NutritionResult:
    food_description: str
    fdc_id: int
    per_100g: dict  # calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg
    density_g_cm3: float | None
    typical_serving_g: float | None
    match_type: str  # "exact" | "fuzzy" — surface this in the API response so the
                      # client can show lower confidence for fuzzy matches


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

    return _fuzzy_fallback(session, model_label)


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
