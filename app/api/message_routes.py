"""POST /api/message — process incoming user message (Spec §13.1, §5.7).

Wires the channel gateway → session manager → story orchestrator pipeline.

Supports these user message flows:
  - "Start" / "start" → list scenarios
  - Scenario selection by number or name → start new session with opening scene
  - Choice selection (number, letter, id, or label) → generate next scene
  - Free-form text → generate scene with user input incorporated
"""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import get_session
from app.services.session_manager import SessionManager
from app.services.story_orchestrator import (
    GeneratedScene,
    StoryContext,
    StoryOrchestrator,
)
from app.services.llm_service import get_llm_service
from app.story.scenario_loader import (
    build_initial_world_state,
    get_node,
    get_start_node,
    list_all_scenarios,
    load_scenario_unified,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["message"])

_session_manager = SessionManager()


class IncomingMessage(BaseModel):
    channel: str = Field(default="whatsapp_mock")
    user_id: str
    message: str


class OutgoingMessage(BaseModel):
    messages: list[str]
    session_id: str | None = None
    scene: dict | None = None


@router.post("/message", response_model=OutgoingMessage)
async def process_message(
    payload: IncomingMessage,
    session: AsyncSession = Depends(get_session),
) -> OutgoingMessage:
    """Receive a user message from any channel and return system responses."""
    text = payload.message.strip()

    # ── "Start" → show scenario selection ──────────────────────
    if text.lower() in ("start", "starten", "neu", "new"):
        scenarios = await list_all_scenarios(session)
        lines = ["Wähle ein Szenario:"]
        for i, sc in enumerate(scenarios, 1):
            lines.append(f"{i}. {sc['title']} — {sc['genre']}")
        lines.append("\n(Antworte mit der Nummer oder dem Namen)")
        return OutgoingMessage(messages=["\n".join(lines)])

    # ── Scenario selection → start new session ─────────────────
    scenarios = await list_all_scenarios(session)
    selected = None
    # By number
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(scenarios):
            selected = scenarios[idx]
    # By name/id
    if not selected:
        text_lower = text.lower()
        for sc in scenarios:
            if sc["id"].lower() == text_lower or sc["title"].lower() == text_lower:
                selected = sc
                break

    if selected:
        scenario = await load_scenario_unified(selected["id"], session)
        start_node = get_start_node(scenario)
        if not start_node:
            return OutgoingMessage(messages=["Fehler: Szenario hat keinen Startknoten."])

        # Create or get user via ORM
        from sqlalchemy import select as sa_select
        from app.models.user import User as UserORM
        result_set = await session.execute(
            sa_select(UserORM).where(UserORM.channel_user_id == payload.user_id)
        )
        user_obj = result_set.scalar_one_or_none()
        if not user_obj:
            from uuid import uuid4
            user_obj = UserORM(id=str(uuid4()), channel_user_id=payload.user_id)
            session.add(user_obj)
            await session.flush()

        # Create session via ORM
        from app.models.story_session import StorySession as SessionORM
        from uuid import uuid4 as _uuid4
        ss = SessionORM(
            id=str(_uuid4()),
            user_id=user_obj.id,
            scenario_id=selected["id"],
            current_node_id=start_node.get("id"),
            status="running",
        )
        world_state = build_initial_world_state(scenario)
        world_state["current_location"] = start_node.get("location", "")
        ss.world_state_json = json.dumps(world_state, ensure_ascii=False)
        session.add(ss)
        await session.commit()
        await session.refresh(ss)

        # Generate opening scene
        orchestrator = StoryOrchestrator(get_llm_service())
        ctx = orchestrator.build_context(
            session_id=ss.id,
            node=start_node,
            world_state=world_state,
            scenario_id=selected["id"],
        )
        result = await orchestrator.generate_opening_scene(ctx)

        # Update session with new state
        ss.world_state_json = json.dumps(result.updated_world_state, ensure_ascii=False)
        if result.next_node_id:
            ss.current_node_id = result.next_node_id
        await session.commit()

        # Format response
        msg = _format_scene(result.scene)
        scene_dict = _scene_to_dict(result.scene)
        scene_dict["is_ending"] = result.is_ending
        return OutgoingMessage(
            messages=[msg],
            session_id=ss.id,
            scene=scene_dict,
        )

    # ── Existing session → process choice or free input ────────
    # Look up the user's most recent active session
    from sqlalchemy import select
    from app.models.story_session import StorySession
    from app.models.user import User as UserORM

    # Find the internal user by channel_user_id
    user_result = await session.execute(
        select(UserORM).where(UserORM.channel_user_id == payload.user_id)
    )
    user_obj = user_result.scalar_one_or_none()

    if user_obj:
        result_set = await session.execute(
            select(StorySession)
            .where(StorySession.user_id == user_obj.id)
            .where(StorySession.status == "running")
            .order_by(StorySession.updated_at.desc())
            .limit(1)
        )
    else:
        result_set = await session.execute(
            select(StorySession)
            .where(StorySession.user_id == payload.user_id)
            .where(StorySession.status == "running")
            .order_by(StorySession.updated_at.desc())
            .limit(1)
        )
    ss = result_set.scalar_one_or_none()

    if not ss:
        # No active session — prompt to start
        return OutgoingMessage(
            messages=["Du hast keine aktive Story. Sende 'Start' um zu beginnen."]
        )

    # Load scenario and current node
    scenario = await load_scenario_unified(ss.scenario_id, session)
    current_node = get_node(scenario, ss.current_node_id or "")
    if not current_node:
        return OutgoingMessage(messages=["Fehler: Aktueller Knoten nicht gefunden."])

    world_state = json.loads(ss.world_state_json or "{}")

    orchestrator = StoryOrchestrator(get_llm_service())
    ctx = orchestrator.build_context(
        session_id=ss.id,
        node=current_node,
        world_state=world_state,
        user_input=text,
        scenario_id=ss.scenario_id,
    )
    result = await orchestrator.process_user_input(ctx, scenario=scenario)

    # Update session
    ss.world_state_json = json.dumps(result.updated_world_state, ensure_ascii=False)
    if result.next_node_id:
        next_node = get_node(scenario, result.next_node_id)
        if next_node:
            ss.current_node_id = result.next_node_id
            if next_node.get("is_end") or next_node.get("type") == "end":
                ss.status = "completed"
    await session.commit()

    msg = _format_scene(result.scene)
    if result.is_ending:
        msg += "\n\n— Ende —"

    scene_dict = _scene_to_dict(result.scene)
    scene_dict["is_ending"] = result.is_ending
    return OutgoingMessage(
        messages=[msg],
        session_id=ss.id,
        scene=scene_dict,
    )


def _format_scene(scene: GeneratedScene) -> str:
    """Format a GeneratedScene into a WhatsApp-friendly message."""
    parts = [scene.scene_text]

    if scene.choices:
        parts.append("\nWas möchtest du tun?\n")
        for i, ch in enumerate(scene.choices, 1):
            label = ch.get("label", ch.get("id", "?"))
            parts.append(f"{chr(64 + i)}) {label}")

    return "\n".join(parts)


def _scene_to_dict(scene: GeneratedScene) -> dict:
    """Serialize a GeneratedScene to a dict for the API response."""
    return {
        "scene_text": scene.scene_text,
        "choices": scene.choices,
        "state_updates": scene.state_updates,
        "suggested_next_node": scene.suggested_next_node,
    }
