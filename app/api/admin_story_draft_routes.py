"""Admin / Story Authoring API routes (Spec §13.2).

Implements all nine endpoints for the authoring pipeline:
  POST   /api/admin/story-drafts              — create draft + start generation
  GET    /api/admin/story-drafts              — list drafts
  GET    /api/admin/story-drafts/{id}         — get draft detail
  GET    /api/admin/story-drafts/{id}/graph   — get story graph
  POST   /api/admin/story-drafts/{id}/review  — run critic review
  POST   /api/admin/story-drafts/{id}/repair  — run repair pass
  POST   /api/admin/story-drafts/{id}/validate — run deterministic validation
  POST   /api/admin/story-drafts/{id}/approve — manually approve
  POST   /api/admin/story-drafts/{id}/publish — publish to runtime
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.enums import DraftStatus, JobType
from app.persistence.database import get_session
from app.persistence.authoring_repositories import (
    StoryDraftRepository,
    StoryDraftVersionRepository,
    StoryGenerationJobRepository,
    StoryReviewReportRepository,
    StoryValidationReportRepository,
)
from app.services.publishing_service import PublishingService
from app.services.story_authoring_agent import get_authoring_agent
from app.services.story_critic_agent import StoryCriticAgent
from app.services.story_repair_agent import StoryRepairAgent
from app.services.story_validation_service import StoryValidationService
from app.story.authoring_schemas import (
    DraftActionResponse,
    DraftCreateResponse,
    DraftDetailResponse,
    DraftSummaryResponse,
    GraphResponse,
    ReviewReportResponse,
    StoryBriefCreate,
    ValidationReportResponse,
    VersionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ── helpers ────────────────────────────────────────────────────────────

def _draft_to_summary(draft: Any) -> DraftSummaryResponse:
    """Convert a StoryDraft ORM instance to a summary response."""
    return DraftSummaryResponse(
        id=draft.id,
        title=draft.title,
        genre=draft.genre,
        tone=draft.tone,
        language=draft.language,
        target_age=draft.target_age,
        status=draft.status,
        quality_score=draft.quality_score,
        min_sentences_per_node=draft.min_sentences_per_node,
        max_sentences_per_node=draft.max_sentences_per_node,
        min_node_connections=draft.min_node_connections,
        max_node_connections=draft.max_node_connections,
        version_count=len(draft.versions) if draft.versions else 0,
        created_at=draft.created_at,
        approved_at=draft.approved_at,
        published_at=draft.published_at,
    )


def _draft_to_detail(draft: Any, version_repo: StoryDraftVersionRepository) -> DraftDetailResponse:
    """Convert a StoryDraft ORM instance to a full detail response."""
    versions_list = [
        VersionResponse(
            id=v.id,
            version_number=v.version_number,
            created_by=v.created_by,
            created_at=v.created_at,
            notes=v.notes,
            has_outline=v.outline_json is not None,
            has_graph=True,
        )
        for v in sorted(draft.versions, key=lambda v: v.version_number)
    ] if draft.versions else []

    brief = {}
    if draft.brief_json:
        try:
            brief = json.loads(draft.brief_json)
        except (json.JSONDecodeError, TypeError):
            pass

    return DraftDetailResponse(
        id=draft.id,
        title=draft.title,
        genre=draft.genre,
        tone=draft.tone,
        language=draft.language,
        target_age=draft.target_age,
        status=draft.status,
        quality_score=draft.quality_score,
        brief=brief,
        min_sentences_per_node=draft.min_sentences_per_node,
        max_sentences_per_node=draft.max_sentences_per_node,
        min_node_connections=draft.min_node_connections,
        max_node_connections=draft.max_node_connections,
        created_at=draft.created_at,
        updated_at=draft.updated_at,
        approved_at=draft.approved_at,
        published_at=draft.published_at,
        versions=versions_list,
    )


async def _get_draft_or_404(
    session: AsyncSession, draft_id: str
) -> Any:
    """Fetch a draft or raise 404."""
    repo = StoryDraftRepository(session)
    draft = await repo.get_by_id(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"Draft '{draft_id}' not found")
    return draft


# ── endpoints ──────────────────────────────────────────────────────────

@router.post("/story-drafts", response_model=DraftCreateResponse, status_code=201)
async def create_draft(
    brief: StoryBriefCreate,
    session: AsyncSession = Depends(get_session),
) -> DraftCreateResponse:
    """Create a new story draft from a brief.

    Creates the draft record in DRAFT status, then kicks off the
    outline + graph generation synchronously (mock provider) or
    as a background job (real LLM).  Returns immediately with the
    draft ID and job ID.
    """
    repo = StoryDraftRepository(session)
    job_repo = StoryGenerationJobRepository(session)
    version_repo = StoryDraftVersionRepository(session)

    # 1. Create draft record
    draft = await repo.create(
        title=brief.title,
        genre=brief.genre,
        tone=brief.tone,
        language=brief.language,
        target_age=brief.target_age,
        brief=brief.to_storage_dict(),
        min_sentences_per_node=brief.min_sentences_per_node,
        max_sentences_per_node=brief.max_sentences_per_node,
        min_node_connections=brief.min_node_connections,
        max_node_connections=brief.max_node_connections,
    )

    # 2. Create a generation job and mark it running
    job = await job_repo.create(
        draft_id=draft.id,
        job_type=JobType.GRAPH.value,
        llm_provider=get_settings().llm_provider,
    )
    await job_repo.mark_running(job.id)

    # 3. Run outline + graph generation
    try:
        agent = get_authoring_agent(dummy=True)
        outline = await agent.generate_outline(brief.to_storage_dict())

        # StoryAuthoringAgent supports limit kwargs; DummyStoryAuthoringAgent
        # accepts only outline (kwargs are ignored at runtime by duck-typing).
        try:
            graph = await agent.generate_graph(  # type: ignore[call-arg]
                outline,
                min_sentences=brief.min_sentences_per_node,
                max_sentences=brief.max_sentences_per_node,
                min_node_connections=brief.min_node_connections,
                max_node_connections=brief.max_node_connections,
            )
        except TypeError:
            graph = await agent.generate_graph(outline)

        # Store the first version
        await version_repo.create(
            draft_id=draft.id,
            graph=graph,
            outline=outline,
            created_by="authoring_agent",
            notes="Initial generation",
        )

        # Transition to needs_review and mark job completed
        await repo.update_status(draft.id, DraftStatus.GENERATING)
        await repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
        await job_repo.mark_completed(job.id)

    except Exception as exc:
        logger.error("Generation failed for draft %s: %s", draft.id, exc)
        await job_repo.mark_failed(job.id, error_message=str(exc))
        try:
            await repo.update_status(draft.id, DraftStatus.GENERATING)
            await repo.update_status(draft.id, DraftStatus.FAILED)
        except ValueError:
            pass  # transition may fail if already in a terminal state
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {exc}",
        )

    return DraftCreateResponse(
        draft_id=draft.id,
        status=DraftStatus.NEEDS_REVIEW.value,
        job_id=job.id,
    )


@router.get("/story-drafts", response_model=list[DraftSummaryResponse])
async def list_drafts(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> list[DraftSummaryResponse]:
    """List all story drafts, optionally filtered by status."""
    repo = StoryDraftRepository(session)
    status_filter = None
    if status is not None:
        try:
            status_filter = DraftStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status '{status}'. Valid: {[s.value for s in DraftStatus]}",
            )
    drafts = await repo.list_all(status=status_filter, limit=limit, offset=offset)
    return [_draft_to_summary(d) for d in drafts]


@router.get("/story-drafts/{draft_id}", response_model=DraftDetailResponse)
async def get_draft(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
) -> DraftDetailResponse:
    """Get a single story draft with all versions."""
    draft = await _get_draft_or_404(session, draft_id)
    version_repo = StoryDraftVersionRepository(session)
    return _draft_to_detail(draft, version_repo)


@router.get("/story-drafts/{draft_id}/graph", response_model=GraphResponse)
async def get_draft_graph(
    draft_id: str,
    version_id: str | None = Query(None, description="Specific version ID (default: latest)"),
    session: AsyncSession = Depends(get_session),
) -> GraphResponse:
    """Get the story graph for a draft, optionally a specific version."""
    await _get_draft_or_404(session, draft_id)
    version_repo = StoryDraftVersionRepository(session)

    if version_id is not None:
        version = await version_repo.get_by_id(version_id)
        if version is None or version.draft_id != draft_id:
            raise HTTPException(
                status_code=404,
                detail=f"Version '{version_id}' not found for draft '{draft_id}'",
            )
    else:
        version = await version_repo.latest_for_draft(draft_id)
        if version is None:
            raise HTTPException(
                status_code=404,
                detail=f"No versions found for draft '{draft_id}'",
            )

    graph = version_repo.parse_graph(version)
    return GraphResponse(
        draft_id=draft_id,
        version_id=version.id,
        version_number=version.version_number,
        graph=graph,
    )


@router.post("/story-drafts/{draft_id}/review", response_model=ReviewReportResponse)
async def start_review(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
) -> ReviewReportResponse:
    """Run a critic review pass on the latest version of the draft.

    Transitions the draft to needs_review (if not already there),
    runs the StoryCriticAgent, stores the review report, and
    updates the draft's quality_score.
    """
    draft = await _get_draft_or_404(session, draft_id)
    version_repo = StoryDraftVersionRepository(session)
    job_repo = StoryGenerationJobRepository(session)
    review_repo = StoryReviewReportRepository(session)
    draft_repo = StoryDraftRepository(session)

    # Must have at least one version
    latest_version = await version_repo.latest_for_draft(draft_id)
    if latest_version is None:
        raise HTTPException(
            status_code=409,
            detail="Cannot review a draft with no versions.",
        )

    # Create and run review job
    job = await job_repo.create(
        draft_id=draft_id,
        job_type=JobType.REVIEW.value,
        llm_provider=get_settings().llm_provider,
    )
    await job_repo.mark_running(job.id)

    try:
        outline = version_repo.parse_outline(latest_version) or {}
        graph = version_repo.parse_graph(latest_version)

        critic = StoryCriticAgent()
        result = await critic.review(outline, graph)

        # Store review report
        report = await review_repo.create(
            draft_id=draft_id,
            version_id=latest_version.id,
            score=result["score"],
            issues=result["issues"],
            summary=result.get("summary"),
        )

        # Update draft quality_score (update directly to avoid
        # invalid self-transition if already in needs_review)
        from sqlalchemy import update as sa_update
        from app.models.story_draft import StoryDraft as _SD
        stmt = (
            sa_update(_SD)
            .where(_SD.id == draft_id)
            .values(quality_score=result["score"])
        )
        await session.execute(stmt)
        await session.commit()

        await job_repo.mark_completed(job.id)

        return ReviewReportResponse(
            id=report.id,
            draft_id=draft_id,
            version_id=report.version_id,
            score=report.score,
            issues=json.loads(report.issues_json),
            summary=report.summary,
            created_at=report.created_at,
        )

    except Exception as exc:
        logger.error("Review failed for draft %s: %s", draft_id, exc)
        await job_repo.mark_failed(job.id, error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"Review failed: {exc}")


@router.post("/story-drafts/{draft_id}/repair", response_model=GraphResponse)
async def start_repair(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
) -> GraphResponse:
    """Run a repair pass on the latest version.

    Uses the latest review report to guide repairs, creates a new
    version with the improved graph, and transitions the draft to
    needs_review for re-evaluation.
    """
    draft = await _get_draft_or_404(session, draft_id)
    version_repo = StoryDraftVersionRepository(session)
    job_repo = StoryGenerationJobRepository(session)
    review_repo = StoryReviewReportRepository(session)
    draft_repo = StoryDraftRepository(session)

    latest_version = await version_repo.latest_for_draft(draft_id)
    if latest_version is None:
        raise HTTPException(
            status_code=409,
            detail="Cannot repair a draft with no versions.",
        )

    # Get latest review report for guidance
    review_reports = await review_repo.list_by_draft(draft_id)
    review_data: dict[str, Any] = {}
    if review_reports:
        latest_review = review_reports[-1]
        review_data = {
            "score": latest_review.score,
            "issues": review_repo.parse_issues(latest_review),
            "summary": latest_review.summary or "",
        }

    # Create repair job
    job = await job_repo.create(
        draft_id=draft_id,
        job_type=JobType.REPAIR.value,
        llm_provider=get_settings().llm_provider,
    )
    await job_repo.mark_running(job.id)

    try:
        graph = version_repo.parse_graph(latest_version)
        repair_agent = StoryRepairAgent()
        result = await repair_agent.repair(graph, review_data)

        new_graph = result.get("graph", graph)

        # Create new version with repaired graph
        outline = version_repo.parse_outline(latest_version)
        new_version = await version_repo.create(
            draft_id=draft_id,
            graph=new_graph,
            outline=outline,
            created_by="repair_agent",
            notes=result.get("summary", "Repair pass"),
        )

        # Transition: if in needs_review, go to needs_repair, then back to needs_review
        if draft.status == DraftStatus.NEEDS_REVIEW.value:
            await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REPAIR)
            await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REVIEW)
        elif draft.status == DraftStatus.VALIDATED.value:
            await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REPAIR)
            await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REVIEW)

        await job_repo.mark_completed(job.id)

        return GraphResponse(
            draft_id=draft_id,
            version_id=new_version.id,
            version_number=new_version.version_number,
            graph=new_graph,
        )

    except Exception as exc:
        logger.error("Repair failed for draft %s: %s", draft_id, exc)
        await job_repo.mark_failed(job.id, error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"Repair failed: {exc}")


@router.post("/story-drafts/{draft_id}/validate", response_model=ValidationReportResponse)
async def start_validation(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
) -> ValidationReportResponse:
    """Run deterministic validation on the latest version's graph.

    If validation passes and the draft is in needs_review, it transitions
    to validated.  If validation fails, the draft stays in its current
    status so repairs can be applied.
    """
    draft = await _get_draft_or_404(session, draft_id)
    version_repo = StoryDraftVersionRepository(session)
    job_repo = StoryGenerationJobRepository(session)
    validation_repo = StoryValidationReportRepository(session)
    draft_repo = StoryDraftRepository(session)

    latest_version = await version_repo.latest_for_draft(draft_id)
    if latest_version is None:
        raise HTTPException(
            status_code=409,
            detail="Cannot validate a draft with no versions.",
        )

    # Create validation job
    job = await job_repo.create(
        draft_id=draft_id,
        job_type=JobType.VALIDATE.value,
        llm_provider=None,  # deterministic, no LLM
    )
    await job_repo.mark_running(job.id)

    try:
        graph = version_repo.parse_graph(latest_version)
        validator = StoryValidationService()
        result = await validator.validate(graph)

        # Store validation report
        report = await validation_repo.create(
            draft_id=draft_id,
            version_id=latest_version.id,
            is_valid=result["is_valid"],
            errors=result["errors"],
            warnings=result["warnings"],
        )

        # Transition to validated if valid and currently needs_review
        if result["is_valid"] and draft.status == DraftStatus.NEEDS_REVIEW.value:
            await draft_repo.update_status(draft_id, DraftStatus.VALIDATED)

        await job_repo.mark_completed(job.id)

        return ValidationReportResponse(
            id=report.id,
            draft_id=draft_id,
            version_id=report.version_id,
            is_valid=report.is_valid,
            errors=json.loads(report.errors_json),
            warnings=json.loads(report.warnings_json),
            created_at=report.created_at,
        )

    except Exception as exc:
        logger.error("Validation failed for draft %s: %s", draft_id, exc)
        await job_repo.mark_failed(job.id, error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}")


@router.post("/story-drafts/{draft_id}/approve", response_model=DraftActionResponse)
async def approve_draft(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
) -> DraftActionResponse:
    """Manually approve a draft.

    The draft must be in 'validated' status (or already 'approved').
    Transitions to 'approved' and sets the approved_at timestamp.
    """
    draft = await _get_draft_or_404(session, draft_id)
    draft_repo = StoryDraftRepository(session)

    if draft.status == DraftStatus.APPROVED.value:
        return DraftActionResponse(
            draft_id=draft_id,
            status=DraftStatus.APPROVED.value,
            message="Draft is already approved.",
        )

    if draft.status != DraftStatus.VALIDATED.value:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Draft must be in 'validated' status to approve, "
                f"got '{draft.status}'. Run validation first."
            ),
        )

    await draft_repo.update_status(draft_id, DraftStatus.APPROVED)
    return DraftActionResponse(
        draft_id=draft_id,
        status=DraftStatus.APPROVED.value,
        message="Draft approved successfully.",
    )


@router.post("/story-drafts/{draft_id}/delete", response_model=DraftActionResponse)
async def delete_draft(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
) -> DraftActionResponse:
    """Delete a draft and all cascade children (versions, reviews, validations, jobs).

    Published scenarios are NOT deleted — only the authoring draft data.
    Returns a confirmation with the draft ID and title.
    """
    draft = await _get_draft_or_404(session, draft_id)
    draft_repo = StoryDraftRepository(session)
    title = draft.title

    deleted = await draft_repo.delete(draft_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete draft")

    # Emit event log entry
    from app.services.event_log import event_log
    event_log.emit_done(
        "delete", "delete_draft",
        f"Draft '{title}' ({draft_id}) deleted with all versions, reviews, and jobs",
        draft_id=draft_id,
        detail={"title": title, "draft_id": draft_id},
    )

    return DraftActionResponse(
        draft_id=draft_id,
        status="deleted",
        message=f"Draft '{title}' deleted with all versions, reviews, validations, and jobs.",
    )


@router.post("/story-drafts/{draft_id}/publish", response_model=DraftActionResponse)
async def publish_draft(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
) -> DraftActionResponse:
    """Publish an approved draft — makes it available in runtime.

    Checks all publish criteria (Spec §15), copies the graph into the
    story_nodes table, and transitions the draft to 'published'.
    """
    draft = await _get_draft_or_404(session, draft_id)
    job_repo = StoryGenerationJobRepository(session)

    if draft.status == DraftStatus.PUBLISHED.value:
        return DraftActionResponse(
            draft_id=draft_id,
            status=DraftStatus.PUBLISHED.value,
            message="Draft is already published.",
        )

    if draft.status != DraftStatus.APPROVED.value:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Draft must be in 'approved' status to publish, "
                f"got '{draft.status}'."
            ),
        )

    # Create publish job
    job = await job_repo.create(
        draft_id=draft_id,
        job_type=JobType.PUBLISH.value,
        llm_provider=None,
    )
    await job_repo.mark_running(job.id)

    try:
        pub_service = PublishingService(session)
        result = await pub_service.publish(draft_id)
        await job_repo.mark_completed(job.id)

        return DraftActionResponse(
            draft_id=draft_id,
            status=DraftStatus.PUBLISHED.value,
            message=(
                f"Published as scenario '{result['scenario_id']}' "
                f"with {result['nodes_published']} nodes."
            ),
        )

    except ValueError as exc:
        await job_repo.mark_failed(job.id, error_message=str(exc))
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.error("Publish failed for draft %s: %s", draft_id, exc)
        await job_repo.mark_failed(job.id, error_message=str(exc))
        raise HTTPException(status_code=500, detail=f"Publish failed: {exc}")
