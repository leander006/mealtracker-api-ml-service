from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_session
from food_library.search import NewFoodInput, add_user_food, search_foods

router = APIRouter(prefix="/food-library", tags=["food-library"])


class NewFoodRequest(BaseModel):
    description: str
    calories_kcal: float
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0
    fiber_g: float = 0
    sugar_g: float = 0
    sodium_mg: float = 0
    added_by_user_id: int | None = None


@router.get("/search")
def search(q: str, session: Session = Depends(get_session)):
    results = search_foods(session, q)
    return {"results": [r.__dict__ for r in results]}


@router.post("/foods")
def add_food(payload: NewFoodRequest, session: Session = Depends(get_session)):
    food_id = add_user_food(session, NewFoodInput(**payload.model_dump()))
    return {"food_id": food_id, "verified": False}
