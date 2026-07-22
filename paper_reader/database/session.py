"""SQLAlchemy engine and session helpers."""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from paper_reader.database.config import database_url_from_env, safe_database_label, sanitize_error
from paper_reader.database.models import Base

logger = logging.getLogger(__name__)


def create_app_engine(database_url: str | None = None) -> Engine:
    url = database_url or database_url_from_env()
    kwargs = {"pool_pre_ping": True, "future": True}
    if str(url).startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_database(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def check_database_connection(engine: Engine) -> tuple[bool, str]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, f"Connected to {safe_database_label()}."
    except SQLAlchemyError as exc:
        detail = sanitize_error(exc)
        logger.error(
            "Database connection failed for %s: %s: %s",
            safe_database_label(),
            type(exc).__name__,
            detail,
        )
        return False, f"Database unavailable. {type(exc).__name__}: {detail}"


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
