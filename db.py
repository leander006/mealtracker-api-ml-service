"""Session dependency for the MAIN database — the single Neon Postgres
instance holding the food library, nutrition data, and the training-DB
rotation pointer. This is deliberately separate from the training/feedback
DBs (see feedback/db_router.py), which rotate across multiple instances.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session():
    session: Session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
