"""GET /api/scenarios — list published scenarios (Spec §13.1, §15).

Returns all available scenarios: file-based scenarios from the
scenarios directory AND DB-published scenarios from the
``published_scenarios`` table.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import get_session
from app.story.scenario_loader import list_all_scenarios

router = APIRouter(prefix="/api", tags=["scenario"])


class ScenarioSummary(BaseModel):
    """Lightweight scenario summary for the runtime scenario list."""
    id: str
    title: str
    genre: str


@router.get("/scenarios", response_model=list[ScenarioSummary])
async def list_scenarios(
    session: AsyncSession = Depends(get_session),
) -> list[ScenarioSummary]:
    """Return all available scenarios — file-based + DB-published (Spec §15).

    Merges scenarios from the scenarios directory with those
    published via the Admin UI (``published_scenarios`` table).
    File-based scenarios take precedence on id collisions.
    """
    all_scenarios = await list_all_scenarios(session)
    return [
        ScenarioSummary(
            id=s["id"],
            title=s["title"],
            genre=s.get("genre", ""),
        )
        for s in all_scenarios
    ]
