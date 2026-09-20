"""SQLAlchemy models for the nutrition database.

Mirrors schema.sql exactly. Keep these two in sync manually (no ORM-driven
migrations here yet) — if you add Alembic later, generate the first migration
from schema.sql rather than from these models to avoid drift.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Food(Base):
    __tablename__ = "foods"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # fdc_id is nullable now: USDA-seeded rows have one, user-added rows don't.
    fdc_id: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[str] = mapped_column(String, nullable=False)
    food_category: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Community food library fields
    source: Mapped[str] = mapped_column(String, default="usda")  # 'usda' | 'user_added'
    added_by_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    verified: Mapped[bool] = mapped_column(default=False)  # true for USDA rows; user rows start false

    nutrients: Mapped["FoodNutrient"] = relationship(
        back_populates="food", uselist=False, cascade="all, delete-orphan"
    )
    label_maps: Mapped[list["FoodLabelMap"]] = relationship(
        back_populates="food", cascade="all, delete-orphan"
    )


class FoodNutrient(Base):
    __tablename__ = "food_nutrients"

    food_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("foods.id", ondelete="CASCADE"), primary_key=True
    )
    calories_kcal: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    protein_g: Mapped[float] = mapped_column(Numeric(8, 2), default=0)
    carbs_g: Mapped[float] = mapped_column(Numeric(8, 2), default=0)
    fat_g: Mapped[float] = mapped_column(Numeric(8, 2), default=0)
    fiber_g: Mapped[float] = mapped_column(Numeric(8, 2), default=0)
    sugar_g: Mapped[float] = mapped_column(Numeric(8, 2), default=0)
    sodium_mg: Mapped[float] = mapped_column(Numeric(8, 2), default=0)
    density_g_cm3: Mapped[float | None] = mapped_column(Numeric(6, 3), nullable=True)

    food: Mapped["Food"] = relationship(back_populates="nutrients")


class FoodLabelMap(Base):
    __tablename__ = "food_label_map"
    __table_args__ = (
        UniqueConstraint("model_label", "food_id", name="uq_label_food"),
        Index("idx_food_label_map_label", "model_label", "priority"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_label: Mapped[str] = mapped_column(String, nullable=False)
    food_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("foods.id", ondelete="CASCADE"), nullable=False
    )
    priority: Mapped[int] = mapped_column(SmallInteger, default=1)
    typical_serving_g: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)

    food: Mapped["Food"] = relationship(back_populates="label_maps")
