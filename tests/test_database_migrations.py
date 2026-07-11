"""Regression tests for lightweight SQLite schema migrations."""

from __future__ import annotations

import sqlite3

import pytest


@pytest.mark.asyncio
async def test_init_db_adds_story_draft_config_columns_to_existing_sqlite_schema(
    tmp_path, monkeypatch,
):
    """Existing SQLite DBs get new StoryDraft config columns during init_db."""
    db_path = tmp_path / "legacy_story.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE story_drafts (
            id VARCHAR(64) NOT NULL PRIMARY KEY,
            title VARCHAR(256) NOT NULL,
            genre VARCHAR(64) NOT NULL,
            tone VARCHAR(64) NOT NULL,
            language VARCHAR(16) NOT NULL,
            target_age VARCHAR(16) NOT NULL,
            brief_json TEXT NOT NULL,
            status VARCHAR(32) NOT NULL,
            quality_score FLOAT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            approved_at DATETIME,
            published_at DATETIME
        )
        """
    )
    conn.execute(
        """
        INSERT INTO story_drafts (
            id, title, genre, tone, language, target_age, brief_json,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "draft_legacy",
            "Legacy Draft",
            "mystery",
            "tense",
            "de",
            "16+",
            "{}",
            "draft",
            "2026-01-01 00:00:00",
            "2026-01-01 00:00:00",
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")

    from app.core.config import get_settings
    from app.persistence.database import close_db, get_session_factory, init_db
    from app.persistence.authoring_repositories import StoryDraftRepository

    get_settings.cache_clear()
    await close_db()
    try:
        await init_db()
        async with get_session_factory()() as session:
            drafts = await StoryDraftRepository(session).list_all()
    finally:
        await close_db()
        get_settings.cache_clear()

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.id == "draft_legacy"
    assert draft.min_sentences_per_node == 3
    assert draft.max_sentences_per_node == 8
    assert draft.min_node_connections == 2
    assert draft.max_node_connections == 5

    migrated_columns = {
        row[1]: row[4]
        for row in sqlite3.connect(db_path).execute("PRAGMA table_info(story_drafts)")
    }
    assert migrated_columns["min_sentences_per_node"] == "3"
    assert migrated_columns["max_sentences_per_node"] == "8"
    assert migrated_columns["min_node_connections"] == "2"
    assert migrated_columns["max_node_connections"] == "5"
