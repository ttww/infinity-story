"""Async SQLite database setup with SQLAlchemy + aiosqlite.

Provides:
  - ``Base`` — declarative base for all ORM models
  - ``get_engine()`` — lazily-created async engine
  - ``get_session()`` — FastAPI dependency yielding ``AsyncSession``
  - ``init_db()`` — create all tables on startup
  - ``close_db()`` — dispose engine on shutdown
  - ``DB_PATH`` — filesystem path for raw aiosqlite access

Supports both ``database_url`` (SQLAlchemy format) and ``database_path``
(raw file path) from settings.  Tests can override via ``DATABASE_URL``
env var.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ── database URL resolution ───────────────────────────────────────────

def _get_database_url() -> str:
    """Determine the SQLAlchemy async database URL.

    Priority:
      1. ``DATABASE_URL`` env var (used by tests for in-memory DB)
      2. ``Settings.database_url`` if present
      3. ``Settings.database_path`` converted to a SQLAlchemy URL
    """
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url

    from app.core.config import get_settings
    s = get_settings()

    db_url = getattr(s, "database_url", None)
    if db_url:
        return db_url

    db_path = getattr(s, "database_path", None)
    if db_path:
        if not db_path.startswith("/"):
            import app.core.config as cfg
            if hasattr(cfg, "BASE_DIR"):
                db_path = str(cfg.BASE_DIR / db_path)
        return f"sqlite+aiosqlite:///{db_path}"

    return "sqlite+aiosqlite:///./story.db"


def _resolve_db_path() -> Path:
    """Return the filesystem path for the SQLite database file.

    Used by legacy code that accesses the DB via raw ``aiosqlite``
    (repositories.py).  Derived from the same URL logic so both layers
    hit the same database.
    """
    url = _get_database_url()
    if ":///" in url:
        path_part = url.split(":///", 1)[1]
        if path_part in ("", ":memory:") or "mode=memory" in path_part:
            return Path("/tmp/infinity_story.db")
        return Path(path_part)
    return Path("./story.db")


# Backward-compatible DB_PATH for repositories.py
DB_PATH = _resolve_db_path()


# ── engine / session management ───────────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the lazily-created async engine."""
    global _engine
    if _engine is None:
        url = _get_database_url()
        from app.core.config import get_settings
        s = get_settings()
        # Use StaticPool for in-memory SQLite so all sessions share one connection
        pool_kwargs: dict[str, Any] = {}
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool
            pool_kwargs["poolclass"] = StaticPool
        _engine = create_async_engine(
            url,
            echo=getattr(s, "debug", False),
            future=True,
            **pool_kwargs,
        )

        # Enable SQLite foreign-key enforcement (required for ON DELETE CASCADE)
        @event.listens_for(_engine.sync_engine, "connect")
        def _enable_fk(dbapi_conn, _conn_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the lazily-created session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an async DB session (compatible with FastAPI Depends)."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def _migrate_sqlite_schema(conn) -> None:
    """Apply lightweight SQLite migrations for existing databases.

    ``Base.metadata.create_all()`` creates missing tables but intentionally does
    not ALTER existing tables. Keep small additive migrations here so persisted
    SQLite deployments stay compatible when ORM columns are added.
    """
    if conn.dialect.name != "sqlite":
        return

    result = await conn.exec_driver_sql("PRAGMA table_info(story_drafts)")
    existing_columns = {row[1] for row in result}
    if not existing_columns:
        return

    required_columns = {
        "min_sentences_per_node": 3,
        "max_sentences_per_node": 8,
        "min_node_connections": 2,
        "max_node_connections": 5,
    }
    for column_name, default_value in required_columns.items():
        if column_name not in existing_columns:
            await conn.exec_driver_sql(
                "ALTER TABLE story_drafts "
                f"ADD COLUMN {column_name} INTEGER NOT NULL DEFAULT {default_value}"
            )


async def init_db() -> None:
    """Create all tables and apply lightweight migrations. Call on startup."""
    global _engine, _session_factory
    # Reset globals if they exist (for test isolation)
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None

    engine = get_engine()
    # Import models so they are registered on Base.metadata
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _migrate_sqlite_schema(conn)


async def close_db() -> None:
    """Dispose engine on shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
