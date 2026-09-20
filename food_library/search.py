"""Community food library: search-by-name and add-your-own.

Search ranks USDA-verified foods above unverified user-added ones (a user
food with a typo or bad macro entry shouldn't outrank a vetted USDA row for
the same query) but still surfaces user-added foods, since that's the whole
point of the library covering foods USDA doesn't have (regional/home-cooked
dishes especially).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from nutrition.models import Food, FoodNutrient


@dataclass
class FoodSearchResult:
    food_id: int
    description: str
    source: str
    verified: bool
    per_100g: dict


def search_foods(session: Session, query: str, limit: int = 15) -> list[FoodSearchResult]:
    # Trigram similarity search, ordered verified-first then by relevance.
    # Requires: CREATE EXTENSION pg_trgm;
    rows = session.execute(
        text(
            """
            SELECT f.id, f.description, f.source, f.verified,
                   similarity(f.description, :q) AS sim
            FROM foods f
            WHERE f.description % :q OR f.description ILIKE :like_q
            ORDER BY f.verified DESC, sim DESC
            LIMIT :limit
            """
        ),
        {"q": query, "like_q": f"%{query}%", "limit": limit},
    ).all()

    results = []
    for row in rows:
        nutrients = session.query(FoodNutrient).filter(FoodNutrient.food_id == row.id).first()
        if nutrients is None:
            continue
        results.append(
            FoodSearchResult(
                food_id=row.id,
                description=row.description,
                source=row.source,
                verified=row.verified,
                per_100g={
                    "calories_kcal": float(nutrients.calories_kcal),
                    "protein_g": float(nutrients.protein_g),
                    "carbs_g": float(nutrients.carbs_g),
                    "fat_g": float(nutrients.fat_g),
                    "fiber_g": float(nutrients.fiber_g),
                    "sugar_g": float(nutrients.sugar_g),
                    "sodium_mg": float(nutrients.sodium_mg),
                },
            )
        )
    return results


@dataclass
class NewFoodInput:
    description: str
    calories_kcal: float
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    fiber_g: float = 0
    sugar_g: float = 0
    sodium_mg: float = 0
    added_by_user_id: int | None = None


def add_user_food(session: Session, data: NewFoodInput) -> int:
    """Adds a user-submitted food. Starts unverified — surface these lower
    in search until you build a review step (even a simple "5 other users
    used this without complaint" auto-verify rule works as a v1)."""
    food = Food(
        fdc_id=None,
        description=data.description,
        data_type="user_added",
        source="user_added",
        added_by_user_id=data.added_by_user_id,
        verified=False,
    )
    session.add(food)
    session.flush()

    session.add(
        FoodNutrient(
            food_id=food.id,
            calories_kcal=data.calories_kcal,
            protein_g=data.protein_g,
            carbs_g=data.carbs_g,
            fat_g=data.fat_g,
            fiber_g=data.fiber_g,
            sugar_g=data.sugar_g,
            sodium_mg=data.sodium_mg,
        )
    )
    session.commit()
    return food.id
