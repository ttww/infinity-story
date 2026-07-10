"""GET /api/sessions/{session_id} — retrieve session state (Spec §13.1)."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import get_session
from app.persistence.repositories import SessionRepository

router = APIRouter(prefix="/api", tags=["session"])


class SessionResponse(BaseModel):
    id: str
    user_id: str
    scenario_id: str | None
    current_node_id: str | None
    world_state_json: str | None
    status: str


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    session: AsyncSession = Depends(get_session),
) -> SessionResponse:
    sess = await SessionRepository.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        id=sess["id"],
        user_id=sess["user_id"],
        scenario_id=sess.get("scenario_id"),
        current_node_id=sess.get("current_node_id"),
        world_state_json=sess.get("world_state_json"),
        status=sess["status"],
    )
