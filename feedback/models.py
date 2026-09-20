"""Correction records — one row per user-corrected prediction. This is your
training data: predicted vs. corrected label/weight, plus the image that
produced the prediction. Lives in whichever training DB is currently active
per db_router.py, NOT in the main food-library DB.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class TrainingBase(DeclarativeBase):
    pass


class Correction(TrainingBase):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    image_url: Mapped[str] = mapped_column(Text, nullable=False)  # Cloudinary URL

    predicted_label: Mapped[str | None] = mapped_column(String, nullable=True)
    predicted_confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)

    # What the user says is actually true - this is the label your next
    # training run learns from.
    corrected_label: Mapped[str] = mapped_column(String, nullable=False)
    corrected_weight_g: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)

    # True when the model's prediction was low-confidence and the user typed
    # both fields from scratch, vs. just confirming/adjusting a high-confidence
    # prediction. Useful later to weight "hard" examples differently in training.
    was_low_confidence_flow: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
