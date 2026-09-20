from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.nutrition_routes import router as meal_estimate_router
from config import settings
from db import engine as main_engine
from feedback.db_router import TrainingDBRouter
from feedback.models import TrainingBase
from food_library.routes import router as food_library_router
from feedback.routes import router as feedback_router
from nutrition.models import Base as NutritionBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Main DB: create food-library / nutrition tables if not already applied
    # via schema.sql.
    NutritionBase.metadata.create_all(main_engine)

    training_urls = settings.training_db_url_list
    if not training_urls:
        raise RuntimeError(
            "TRAINING_DB_URLS is empty. Set 3-4 comma-separated Postgres "
            "connection URLs (see .env.example) before starting the service."
        )

    training_router = TrainingDBRouter(
        main_engine=main_engine,
        training_urls=training_urls,
        capacity_mb_per_db=settings.training_db_capacity_mb,
        threshold_pct=settings.training_db_rotation_threshold,
    )
    # corrections table must exist on every training DB, since rotation
    # means any of them could become the active write target.
    for eng in training_router.get_read_engines():
        TrainingBase.metadata.create_all(eng)

    app.state.training_db_router = training_router
    logger.info("Started with %d training DB(s) configured, detector_mode=%s", len(training_urls), settings.detector_mode)

    yield


app = FastAPI(title="Meal Tracker ML Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meal_estimate_router)
app.include_router(food_library_router)
app.include_router(feedback_router)


@app.get("/health")
def health():
    return {"status": "ok", "detector_mode": settings.detector_mode}
