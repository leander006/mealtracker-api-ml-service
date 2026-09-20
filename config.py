"""Central config, loaded from environment variables / .env.

DETECTOR_MODE is the switch mentioned in your original design: "mock" runs
without any model weights (useful for local dev and CI), "real" loads the
YOLOv8 .pt file and runs actual inference. Nothing else in the codebase
branches on this — it's read once here.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres. Works with any Postgres, including Neon — Neon just gives you
    # a connection string in this exact format, sslmode=require included.
    database_url: str = "postgresql://postgres:postgres@localhost:5432/mealtracker"

    detector_mode: str = "mock"  # "mock" | "real"
    yolo_weights_path: str = "weights/food_yolov8.pt"
    yolo_confidence_threshold: float = 0.4

    usda_api_key: str = ""

    cors_allow_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Cloudinary - stores correction images referenced by feedback/models.py
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""

    # Round-robin training DBs. Comma-separated connection URLs, e.g.
    # "postgresql://...db1,postgresql://...db2,postgresql://...db3"
    training_db_urls: str = ""
    training_db_capacity_mb: float = 450  # set a bit under your provider's free-tier cap
    training_db_rotation_threshold: float = 0.8

    @property
    def training_db_url_list(self) -> list[str]:
        return [u.strip() for u in self.training_db_urls.split(",") if u.strip()]


settings = Settings()
