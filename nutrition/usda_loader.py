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

Note: the actual USDA search/parse logic lives in usda_client.py now, shared
with nutrition/lookup.py's live on-demand fallback tier — this script is
just the batch-orchestration loop around it.

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

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from nutrition.models import Base, FoodLabelMap
from nutrition.usda_client import fetch_usda_food, find_or_create_food

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # Batch-seeded rows are curated by whoever wrote food_labels.json,
        # so mark them verified — unlike the live fallback's on-the-fly
        # writes, which default to verified=false.
        food = find_or_create_food(session, usda_food, density_g_cm3=spec.density_g_cm3, source="usda", verified=True)

        session.add(
            FoodLabelMap(
                model_label=spec.model_label,
                food_id=food.id,
                priority=1,
                typical_serving_g=spec.typical_serving_g,
            )
        )
        logger.info("Mapped %s -> %s (fdc_id=%s)", spec.model_label, food.description, food.fdc_id)

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
