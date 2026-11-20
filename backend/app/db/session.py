"""Database engine/session setup. SQLite for the MVP; swap DATABASE_URL for PostgreSQL later."""

import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect as sa_inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine() -> Engine:
    url = get_settings().resolved_database_url
    parsed = make_url(url)
    kwargs: dict[str, object] = {}

    if parsed.get_backend_name() == "sqlite":
        # Sessions are used from FastAPI's worker threads.
        kwargs["connect_args"] = {"check_same_thread": False}
        if parsed.database in (None, "", ":memory:"):
            kwargs["poolclass"] = StaticPool  # one shared in-memory DB
        else:
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, **kwargs)

    if parsed.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create tables that don't exist yet (no migrations in the MVP)."""
    from app.db import models  # noqa: F401  (registers tables on Base.metadata)

    engine = get_engine()
    Base.metadata.create_all(engine)
    _sqlite_add_missing_inspection_columns(engine)
    _sqlite_drop_retired_inspection_columns(engine)
    logger.info("Database ready")


def _sqlite_add_missing_inspection_columns(engine: Engine) -> None:
    """Existing local SQLite files predate decision/vision columns; create_all will not add them."""
    if engine.dialect.name != "sqlite":
        return
    inspector = sa_inspect(engine)
    if "inspections" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("inspections")}
    additions = {
        "decision": "VARCHAR(32)",
        "vision_status": "VARCHAR(16)",
        "vision_model": "VARCHAR(64)",
        "vision_error": "VARCHAR(500)",
        "review_mode": "VARCHAR(16) NOT NULL DEFAULT 'semantic'",
    }
    with engine.begin() as connection:
        for name, coltype in additions.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE inspections ADD COLUMN {name} {coltype}"))


def _sqlite_drop_retired_inspection_columns(engine: Engine) -> None:
    """Drop detector columns left by the previous local-model pipeline."""
    if engine.dialect.name != "sqlite":
        return
    inspector = sa_inspect(engine)
    if "inspections" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("inspections")}
    retired = ("yolo_model", "yolo_conf_threshold", "yolo_iou_threshold", "matching_version")
    with engine.begin() as connection:
        for name in retired:
            if name in existing:
                connection.execute(text(f"ALTER TABLE inspections DROP COLUMN {name}"))


def reset_db_caches() -> None:
    """Dispose the cached engine so a changed DATABASE_URL takes effect (used by tests)."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_session_factory.cache_clear()
    get_engine.cache_clear()
