"""Repository pattern for database access."""

import json
import uuid
from datetime import datetime
from typing import Any

import aiosqlite

from app.persistence.database import DB_PATH


def gen_id(prefix: str = "id") -> str:
    """Generate a unique ID with a prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class UserRepository:
    """User-related database operations."""

    @staticmethod
    async def get_or_create(channel_user_id: str, plan: str = "free") -> dict:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM users WHERE channel_user_id = ?", (channel_user_id,)
            )
            row = await cursor.fetchone()
            if row:
                return dict(row)
            user_id = gen_id("user")
            await db.execute(
                "INSERT INTO users (id, channel_user_id, plan) VALUES (?, ?, ?)",
                (user_id, channel_user_id, plan),
            )
            await db.commit()
            return {"id": user_id, "channel_user_id": channel_user_id, "plan": plan}


class SessionRepository:
    """Story session database operations."""

    @staticmethod
    async def create(user_id: str, scenario_id: str | None = None) -> dict:
        session_id = gen_id("session")
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """INSERT INTO story_sessions (id, user_id, scenario_id, status)
                VALUES (?, ?, ?, 'new')""",
                (session_id, user_id, scenario_id),
            )
            await db.commit()
        return await SessionRepository.get(session_id)

    @staticmethod
    async def get(session_id: str) -> dict | None:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM story_sessions WHERE id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def get_active_for_user(user_id: str) -> dict | None:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM story_sessions
                WHERE user_id = ? AND status IN ('new', 'selecting_scenario', 'collecting_parameters', 'running', 'paused')
                ORDER BY updated_at DESC LIMIT 1""",
                (user_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def update(session_id: str, **fields) -> dict | None:
        if not fields:
            return await SessionRepository.get(session_id)
        sets = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [session_id]
        async with aiosqlite.connect(str(DB_PATH)) as db:
            await db.execute(
                f"UPDATE story_sessions SET {sets}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                values,
            )
            await db.commit()
        return await SessionRepository.get(session_id)

    @staticmethod
    async def add_message(session_id: str, direction: str, text: str) -> None:
        msg_id = gen_id("msg")
        async with aiosqlite.connect(str(DB_PATH)) as db:
            await db.execute(
                "INSERT INTO messages (id, session_id, direction, text) VALUES (?, ?, ?, ?)",
                (msg_id, session_id, direction, text),
            )
            await db.commit()

    @staticmethod
    async def get_messages(session_id: str, limit: int = 50) -> list[dict]:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in reversed(rows)]

    @staticmethod
    async def save_scene(
        session_id: str,
        node_id: str | None,
        scene_text: str,
        choices: list,
        state_updates: dict,
        llm_provider: str | None = None,
        token_usage: int = 0,
    ) -> None:
        scene_id = gen_id("scene")
        async with aiosqlite.connect(str(DB_PATH)) as db:
            await db.execute(
                """INSERT INTO generated_scenes
                (id, session_id, node_id, scene_text, choices_json, state_updates_json, llm_provider, token_usage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scene_id, session_id, node_id, scene_text,
                    json.dumps(choices), json.dumps(state_updates),
                    llm_provider, token_usage,
                ),
            )
            await db.commit()


class DraftRepository:
    """Story draft database operations."""

    @staticmethod
    async def create(brief: dict) -> dict:
        draft_id = gen_id("draft")
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """INSERT INTO story_drafts (id, title, genre, tone, language, target_age, brief_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'draft')""",
                (
                    draft_id,
                    brief.get("title", "Untitled"),
                    brief.get("genre", ""),
                    brief.get("tone", ""),
                    brief.get("language", "de"),
                    brief.get("target_age", "16+"),
                    json.dumps(brief),
                ),
            )
            await db.commit()
        return await DraftRepository.get(draft_id)

    @staticmethod
    async def get(draft_id: str) -> dict | None:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM story_drafts WHERE id = ?", (draft_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def list_all() -> list[dict]:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM story_drafts ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    async def update(draft_id: str, **fields) -> dict | None:
        if not fields:
            return await DraftRepository.get(draft_id)
        sets = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [draft_id]
        async with aiosqlite.connect(str(DB_PATH)) as db:
            await db.execute(
                f"UPDATE story_drafts SET {sets}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                values,
            )
            await db.commit()
        return await DraftRepository.get(draft_id)

    @staticmethod
    async def add_version(
        draft_id: str, outline: dict, graph: dict, created_by: str = "system", notes: str = ""
    ) -> dict:
        version_id = gen_id("ver")
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT COUNT(*) as cnt FROM story_draft_versions WHERE draft_id = ?",
                (draft_id,),
            )
            row = await cursor.fetchone()
            version_number = (row["cnt"] if row else 0) + 1
            await db.execute(
                """INSERT INTO story_draft_versions
                (id, draft_id, version_number, outline_json, graph_json, created_by, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    version_id, draft_id, version_number,
                    json.dumps(outline), json.dumps(graph), created_by, notes,
                ),
            )
            await db.commit()
        return await DraftRepository.get_version(version_id)

    @staticmethod
    async def get_version(version_id: str) -> dict | None:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM story_draft_versions WHERE id = ?", (version_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def get_latest_version(draft_id: str) -> dict | None:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM story_draft_versions WHERE draft_id = ? ORDER BY version_number DESC LIMIT 1",
                (draft_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def add_review_report(
        draft_id: str, version_id: str, score: float, issues: list, summary: str
    ) -> str:
        report_id = gen_id("review")
        async with aiosqlite.connect(str(DB_PATH)) as db:
            await db.execute(
                """INSERT INTO story_review_reports (id, draft_id, version_id, score, issues_json, summary)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (report_id, draft_id, version_id, score, json.dumps(issues), summary),
            )
            await db.commit()
        return report_id

    @staticmethod
    async def add_validation_report(
        draft_id: str, version_id: str, is_valid: bool, errors: list, warnings: list
    ) -> str:
        report_id = gen_id("val")
        async with aiosqlite.connect(str(DB_PATH)) as db:
            await db.execute(
                """INSERT INTO story_validation_reports (id, draft_id, version_id, is_valid, errors_json, warnings_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (report_id, draft_id, version_id, int(is_valid), json.dumps(errors), json.dumps(warnings)),
            )
            await db.commit()
        return report_id

    @staticmethod
    async def get_latest_review(draft_id: str) -> dict | None:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM story_review_reports WHERE draft_id = ? ORDER BY created_at DESC LIMIT 1",
                (draft_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def get_latest_validation(draft_id: str) -> dict | None:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM story_validation_reports WHERE draft_id = ? ORDER BY created_at DESC LIMIT 1",
                (draft_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None


class PublishedScenarioRepository:
    """Published scenario database operations."""

    @staticmethod
    async def list_all() -> list[dict]:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM published_scenarios ORDER BY published_at DESC"
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    async def get(scenario_id: str) -> dict | None:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM published_scenarios WHERE id = ?", (scenario_id,)
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    async def create(draft_id: str, title: str, genre: str, graph: dict) -> dict:
        scenario_id = gen_id("scn")
        async with aiosqlite.connect(str(DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """INSERT INTO published_scenarios (id, draft_id, title, genre, graph_json)
                VALUES (?, ?, ?, ?, ?)""",
                (scenario_id, draft_id, title, genre, json.dumps(graph)),
            )
            await db.commit()
        return await PublishedScenarioRepository.get(scenario_id)
