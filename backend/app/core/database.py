"""Synchronous SQLAlchemy 2.0 database setup and health check helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""


def _is_in_memory_sqlite(database_url: str, db_name: str | None) -> bool:
    if database_url in ("sqlite://", "sqlite:///:memory:"):
        return True
    if not db_name or db_name == ":memory:":
        return True
    return db_name.startswith("file:") and "mode=memory" in db_name


def create_db_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine with SQLite foreign-key enforcement when applicable."""
    if database_url.startswith("sqlite"):
        engine_kwargs: dict[str, Any] = {
            "connect_args": {"check_same_thread": False},
        }
        if database_url in ("sqlite://", "sqlite:///:memory:"):
            engine_kwargs["poolclass"] = StaticPool

        engine = create_engine(database_url, **engine_kwargs)

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    return create_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a configured SQLAlchemy session factory bound to `engine`."""
    return sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def init_db(engine: Engine) -> None:
    """Ensure SQLite parent directory exists and create all declared tables."""
    if engine.url.drivername.startswith("sqlite"):
        db_name = engine.url.database
        if not _is_in_memory_sqlite(str(engine.url), db_name) and db_name is not None:
            Path(db_name).parent.mkdir(parents=True, exist_ok=True)

    # Import ORM models so they register on Base.metadata before create_all.
    from ..models import chat as _chat_models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def check_db_health(engine: Engine) -> bool:
    """Execute a lightweight connectivity check against the database."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except (SQLAlchemyError, OSError):
        return False
