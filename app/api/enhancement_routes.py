"""Story Enhancement API — Multi-Pass Story Enhancement.

Endpoints:
  POST /api/admin/story-drafts/{id}/enhance  — enhance the story graph

Request body:
  {
    "mode": "atmosphere" | "characters" | "choices" | "arc_expansion" | "thematic" | "critic_based",
    "instruction": "optional free-text instruction",
    "target_act": 2,           // optional, for arc_expansion
    "add_node_count": 3,       // optional, for arc_expansion
  }

Response:
  {
    "draft_id": "...",
    "version_id": "...",
    "version_number": N,
    "mode": "...",
    "changes": [...],
    "summary": "...",
    "diff": {...}
  }
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.enums import JobType
from app.persistence.database import get_session
from app.persistence.authoring_repositories import (
    StoryDraftRepository,
    StoryDraftVersionRepository,
    StoryGenerationJobRepository,
    StoryReviewReportRepository,
)
from app.services.event_log import event_log
from app.services.story_enhancement_agent import (
    ENHANCEMENT_MODES,
    StoryEnhancementAgent,
    StoryEnhancementError,
)
from app.story.graph_diff import compute_graph_diff

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin-enhancement"])


# ── Request / Response schemas ─────────────────────────────────────────


class EnhancementRequest(BaseModel):
    """Request body for POST /api/admin/story-drafts/{id}/enhance."""

    mode: str = Field(
        ...,
        description=f"Enhancement mode. One of: {', '.join(ENHANCEMENT_MODES)}",
    )
    instruction: str = Field(
        default="",
        description="Optional free-text instruction for the enhancement.",
    )
    target_act: int | None = Field(
        default=None,
        description="Target act for arc_expansion mode.",
    )
    add_node_count: int | None = Field(
        default=None,
        description="Number of nodes to add for arc_expansion mode.",
    )


class EnhancementResponse(BaseModel):
    """Response after an enhancement pass."""

    draft_id: str
    version_id: str
    version_number: int
    mode: str
    changes: list[str] = Field(default_factory=list)
    summary: str = ""
    diff: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


# ── Endpoint ───────────────────────────────────────────────────────────


@router.post(
    "/story-drafts/{draft_id}/enhance",
    response_model=EnhancementResponse,
)
async def enhance_story(
    draft_id: str,
    req: EnhancementRequest,
    session: AsyncSession = Depends(get_session),
) -> EnhancementResponse:
    """Enhance a story graph using multi-pass LLM-driven improvement.

    Creates a new version with the enhanced graph and returns the diff
    against the previous version.
    """
    # Validate mode
    if req.mode not in ENHANCEMENT_MODES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid enhancement mode '{req.mode}'. "
                f"Valid modes: {', '.join(ENHANCEMENT_MODES)}"
            ),
        )

    # Fetch draft and latest version
    draft_repo = StoryDraftRepository(session)
    version_repo = StoryDraftVersionRepository(session)
    job_repo = StoryGenerationJobRepository(session)

    draft = await draft_repo.get_by_id(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"Draft '{draft_id}' not found")

    latest_version = await version_repo.latest_for_draft(draft_id)
    if latest_version is None:
        raise HTTPException(
            status_code=409,
            detail="Cannot enhance a draft with no versions.",
        )

    # For critic_based mode, fetch the latest review report
    review_report: dict[str, Any] | None = None
    if req.mode == "critic_based":
        review_repo = StoryReviewReportRepository(session)
        reviews = await review_repo.list_by_draft(draft_id)
        if not reviews:
            raise HTTPException(
                status_code=409,
                detail="critic_based mode requires a review. Run review first.",
            )
        latest_review = reviews[-1]
        review_report = {
            "score": latest_review.score,
            "issues": review_repo.parse_issues(latest_review),
            "summary": latest_review.summary or "",
        }

    # Create and run enhancement job
    job = await job_repo.create(
        draft_id=draft_id,
        job_type="enhancement",
        llm_provider=get_settings().llm_provider,
    )
    await job_repo.mark_running(job.id)

    event_log.emit_start(
        "enhancement",
        f"enhance_{req.mode}",
        f"Enhancing '{draft.title}' (mode: {req.mode})",
        draft_id=draft_id,
        detail={"mode": req.mode, "instruction": req.instruction[:200] if req.instruction else ""},
    )

    graph = version_repo.parse_graph(latest_version)
    outline = version_repo.parse_outline(latest_version)
    old_graph = copy.deepcopy(graph)

    try:
        agent = StoryEnhancementAgent()
        result = await agent.enhance(
            graph=graph,
            mode=req.mode,
            instruction=req.instruction,
            review_report=review_report,
            target_act=req.target_act,
            add_node_count=req.add_node_count,
        )
    except StoryEnhancementError as exc:
        await job_repo.mark_failed(job.id, error_message=str(exc))
        event_log.emit_error(
            "enhancement",
            f"enhance_{req.mode}",
            f"Enhancement failed: {exc}",
            draft_id=draft_id,
        )
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {exc}")

    new_graph = result.get("graph", graph)
    changes = result.get("changes", [])
    summary = result.get("summary", "")

    # Create new version with enhanced graph
    new_version = await version_repo.create(
        draft_id=draft_id,
        graph=new_graph,
        outline=outline,
        created_by="enhancement_agent",
        notes=f"Enhancement ({req.mode}): {summary}"[:500],
    )

    # Compute diff
    diff = compute_graph_diff(old_graph, new_graph)

    await job_repo.mark_completed(job.id)

    new_node_count = len(new_graph.get("nodes", {}))
    event_log.emit_done(
        "enhancement",
        f"enhance_{req.mode}",
        f"Enhancement completed ({req.mode}): {new_node_count} nodes, {len(changes)} changes",
        draft_id=draft_id,
        detail={
            "mode": req.mode,
            "node_count": new_node_count,
            "changes_count": len(changes),
            "summary": summary[:200] if summary else "",
        },
    )

    return EnhancementResponse(
        draft_id=draft_id,
        version_id=new_version.id,
        version_number=new_version.version_number,
        mode=req.mode,
        changes=changes,
        summary=summary,
        diff=diff,
        message=f"Story enhanced (mode: {req.mode}).",
    )
