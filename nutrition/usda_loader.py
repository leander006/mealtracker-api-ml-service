"""Seeds the nutrition database from USDA FoodData Central (FDC).

Two ways to source data, pick based on scale:

1. API mode (this script, default) — good for seeding a curated list of a
   few hundred foods (i.e. exactly the classes your YOLO model recognizes).
   Free API key: https://fdc.nal.usda.gov/api-key-signup.html
   Rate limit on the free tier is 1,000 req/hour, which is fine for a
   one-time seed of a constrained food-class list but NOT for bulk-loading
   the full ~2M-row FDC dataset.

2. Bulk CSV mode — if you later want the full FDC dataset (e.g. to support
   open-ended text search instead of a fixed class list), download the
   "Full Download" CSV dump from https://fdc.nal.usda.gov/download-datasets.html
   and bulk-COPY into Postgres directly. Not implemented here since your
   current design (fixed YOLO class list) doesn't need it — start with API
   mode and only build the CSV pipeline if you outgrow it.

Run:
    export USDA_API_KEY=your_key
    export DATABASE_URL=postgresql://user:pass@host/dbname
    python -m nutrition.usda_loader --labels food_labels.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from nutrition.models import Base, Food, FoodLabelMap, FoodNutrient

logging.basicConfig(level=logging.INFO)
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


@dataclass
class LabelSpec:
    """One entry in the input labels file — a YOLO class we need mapped."""

    model_label: str          # must exactly match the YOLO model's class name
    search_query: str         # what to search USDA for, e.g. "fried rice"
    typical_serving_g: float  # prior used before reference-object correction
    density_g_cm3: float | None = None  # optional, improves volume->weight step


def load_label_specs(path: str) -> list[LabelSpec]:
    with open(path) as f:
        raw = json.load(f)
    return [LabelSpec(**entry) for entry in raw]


def fetch_usda_food(query: str, api_key: str) -> dict | None:
    """Searches FDC and returns the best (first) Foundation/SR Legacy match."""
    resp = requests.get(
        FDC_SEARCH_URL,
        params={
            "api_key": api_key,
            "query": query,
            "dataType": ["Foundation", "SR Legacy"],
            "pageSize": 1,
        },
        timeout=10,
    )
    resp.raise_for_status()
    foods = resp.json().get("foods", [])
    return foods[0] if foods else None


def extract_nutrients(usda_food: dict) -> dict:
    """Pulls the fields we track out of USDA's nutrient array, defaulting
    anything missing to 0 rather than failing the whole row — a food
    missing e.g. fiber data shouldn't block seeding calories/protein/carbs/fat."""
    values = {col: 0.0 for col in NUTRIENT_ID_MAP.values()}
    for n in usda_food.get("foodNutrients", []):
        nutrient_id = n.get("nutrientId")
        if nutrient_id in NUTRIENT_ID_MAP:
            values[NUTRIENT_ID_MAP[nutrient_id]] = n.get("value", 0.0)
    return values


def seed(session: Session, specs: list[LabelSpec], api_key: str) -> None:
    for spec in specs:
        existing = (
            session.query(FoodLabelMap)
            .filter(FoodLabelMap.model_label == spec.model_label)
            .first()
        )
        if existing:
            logger.info("Skipping %s — already mapped", spec.model_label)
            continue

        usda_food = fetch_usda_food(spec.search_query, api_key)
        if not usda_food:
            logger.warning(
                "No USDA match for label=%s query=%r — SKIPPED, needs manual entry",
                spec.model_label, spec.search_query,
            )
            continue

        fdc_id = usda_food["fdcId"]
        food = session.query(Food).filter(Food.fdc_id == fdc_id).first()
        if not food:
            food = Food(
                fdc_id=fdc_id,
                description=usda_food.get("description", spec.search_query),
                data_type=usda_food.get("dataType", "unknown"),
                food_category=usda_food.get("foodCategory"),
            )
            session.add(food)
            session.flush()  # need food.id before writing dependent rows

            nutrients = extract_nutrients(usda_food)
            session.add(FoodNutrient(food_id=food.id, density_g_cm3=spec.density_g_cm3, **nutrients))

        session.add(
            FoodLabelMap(
                model_label=spec.model_label,
                food_id=food.id,
                priority=1,
                typical_serving_g=spec.typical_serving_g,
            )
        )
        logger.info("Mapped %s -> %s (fdc_id=%s)", spec.model_label, food.description, fdc_id)

        # USDA free tier: 1000 req/hour ≈ ~3.6s/req sustainable. Small sleep
        # avoids bursting into a 429 on a multi-hundred-label seed run.
        time.sleep(1.0)

    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Path to food_labels.json")
    args = parser.parse_args()

    api_key = os.environ["USDA_API_KEY"]
    database_url = os.environ["DATABASE_URL"]

    engine = create_engine(database_url)
    Base.metadata.create_all(engine)  # no-op if schema.sql already applied

    specs = load_label_specs(args.labels)
    with Session(engine) as session:
        seed(session, specs, api_key)

    logger.info("Seeding complete: %d labels processed", len(specs))


if __name__ == "__main__":
    main()
