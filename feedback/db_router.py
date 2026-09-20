"""Routes writes for training/feedback data across multiple free-tier Postgres
instances, rotating to the next one once the current one nears its storage
cap. Rotation pointer is stored as a single row in the MAIN database (the
one Neon instance holding the food library) — that DB isn't itself subject
to rotation, so it's a safe, always-available place to keep shared state
across however many app instances you're running.

This does NOT move old data between databases. Rotation only changes where
NEW rows go. Reads for training therefore need to fan out across all
configured training DBs, not just the current one — see `get_read_engines`.
"""
from __future__ import annotations

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class TrainingDBRouter:
    def __init__(self, main_engine: Engine, training_urls: list[str], capacity_mb_per_db: float, threshold_pct: float = 0.8):
        if not training_urls:
            raise ValueError("At least one training DB URL is required")
        self.main_engine = main_engine
        self.training_engines = [create_engine(url) for url in training_urls]
        self.capacity_mb = capacity_mb_per_db
        self.threshold_pct = threshold_pct
        self._ensure_state_table()

    def _ensure_state_table(self) -> None:
        with self.main_engine.begin() as conn:
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS training_db_state (
                    id SMALLINT PRIMARY KEY DEFAULT 1,
                    current_index SMALLINT NOT NULL DEFAULT 0,
                    CHECK (id = 1)
                )
                """
            ))
            conn.execute(text(
                "INSERT INTO training_db_state (id, current_index) VALUES (1, 0) ON CONFLICT (id) DO NOTHING"
            ))

    def _get_current_index(self) -> int:
        with self.main_engine.connect() as conn:
            row = conn.execute(text("SELECT current_index FROM training_db_state WHERE id = 1")).first()
            return row.current_index if row else 0

    def _advance_index(self, new_index: int) -> None:
        with self.main_engine.begin() as conn:
            conn.execute(text("UPDATE training_db_state SET current_index = :i WHERE id = 1"), {"i": new_index})

    def _db_size_mb(self, engine: Engine) -> float:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT pg_database_size(current_database()) AS size_bytes")).first()
            return row.size_bytes / (1024 * 1024)

    def get_write_engine(self) -> Engine:
        """Returns the engine new feedback rows should be written to,
        rotating first if the current DB has crossed the capacity threshold."""
        idx = self._get_current_index()
        if idx >= len(self.training_engines):
            raise RuntimeError(
                f"All {len(self.training_engines)} configured training DBs are full. "
                "Provision another free-tier Postgres instance and add its URL to TRAINING_DB_URLS."
            )

        engine = self.training_engines[idx]
        size_mb = self._db_size_mb(engine)
        usage_pct = size_mb / self.capacity_mb

        if usage_pct >= self.threshold_pct:
            next_idx = idx + 1
            if next_idx >= len(self.training_engines):
                logger.warning(
                    "Training DB %d is at %.0f%% capacity and no further DB is configured "
                    "- continuing to write here. Add another TRAINING_DB_URL soon.",
                    idx, usage_pct * 100,
                )
                return engine
            logger.info("Training DB %d at %.0f%% capacity - rotating writes to DB %d", idx, usage_pct * 100, next_idx)
            self._advance_index(next_idx)
            engine = self.training_engines[next_idx]

        return engine

    def get_read_engines(self) -> list[Engine]:
        """All training DBs currently holding data - use when querying
        across the full feedback dataset (e.g. exporting a training run)."""
        return self.training_engines
