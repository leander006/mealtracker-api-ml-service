"""USDA FoodData Central client - the search/fetch/parse logic shared by
both the batch seeder (usda_loader.py) and the live on-demand fallback
(nutrition/lookup.py's third lookup tier).

Kept deliberately dumb and stateless: functions here just call USDA and
return data, they don't know about your DB session or your app's models
except for find_or_create_food, which is the one shared "persist what USDA
gave us" step both callers need.
"""
from __future__ import annotations

import logging

import requests
from sqlalchemy.orm import Session

from nutrition.models import Food, FoodNutrient

logger = logging.getLogger(__name__)

FDC_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# USDA nutrient IDs we care about (stable across the API, documented at
# https://fdc.nal.usda.gov/api-guide.html). Mapped to our column names.
NUTRIENT_ID_MAP = {
    1008: "calories_kcal",  # Energy (kcal)
    1003: "protein_g",
    1005: "carbs_g",
    1004: "fat_g",
    1079: "fiber_g",
    2000: "sugar_g",
    1093: "sodium_mg",
}

# Foundation/SR Legacy = raw/whole ingredients (chicken breast, banana, ...).
# Survey (FNDDS) = "what people actually reported eating" - much better
# coverage of prepared/mixed dishes (curries, fried rice, etc.) that
# Foundation/SR Legacy mostly doesn't have. Branded is deliberately excluded
# - packaged-product data is noisy for this use case and not what we want.
DEFAULT_DATA_TYPES = ["Foundation", "SR Legacy", "Survey (FNDDS)"]


def fetch_usda_food(query: str, api_key: str, data_types: list[str] | None = None, timeout: float = 10) -> dict | None:
    """Searches FDC and returns the best (first) match, or None.

    Uses POST with a JSON body rather than GET with query params: USDA's own
    docs only show dataType-filtered examples via POST with a real JSON
    array (https://fdc.nal.usda.gov/api-guide.html). GET technically accepts
    multiple values for the same param too, but that's not how USDA
    documents or appears to test it — repeated dataType= GET params return
    a 400 in practice, so don't fight that; use the documented shape.
    """
    resp = requests.post(
        FDC_SEARCH_URL,
        params={"api_key": api_key},
        json={
            "query": query,
            "dataType": data_types or DEFAULT_DATA_TYPES,
            "pageSize": 1,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    foods = resp.json().get("foods", [])
    return foods[0] if foods else None


def extract_nutrients(usda_food: dict) -> dict:
    """Pulls the fields we track out of USDA's nutrient array, defaulting
    anything missing to 0 rather than failing the whole row - a food
    missing e.g. fiber data shouldn't block getting calories/protein/carbs/fat."""
    values = {col: 0.0 for col in NUTRIENT_ID_MAP.values()}
    for n in usda_food.get("foodNutrients", []):
        nutrient_id = n.get("nutrientId")
        if nutrient_id in NUTRIENT_ID_MAP:
            values[NUTRIENT_ID_MAP[nutrient_id]] = n.get("value", 0.0)
    return values


def find_or_create_food(
    session: Session,
    usda_food: dict,
    density_g_cm3: float | None = None,
    source: str = "usda",
    verified: bool = False,
) -> Food:
    """Idempotent on fdc_id - if this USDA food is already in our library
    (e.g. the batch loader seeded it, or a previous live fallback already
    fetched it), returns the existing row instead of creating a duplicate."""
    fdc_id = usda_food["fdcId"]
    food = session.query(Food).filter(Food.fdc_id == fdc_id).first()
    if food:
        return food

    food = Food(
        fdc_id=fdc_id,
        description=usda_food.get("description", "Unknown"),
        data_type=usda_food.get("dataType", "unknown"),
        food_category=usda_food.get("foodCategory"),
        source=source,
        verified=verified,
    )
    session.add(food)
    session.flush()  # need food.id before writing the dependent row

    nutrients = extract_nutrients(usda_food)
    session.add(FoodNutrient(food_id=food.id, density_g_cm3=density_g_cm3, **nutrients))
    return food