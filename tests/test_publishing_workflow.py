"""Tests for Quality Thresholds & Publishing Workflow (Spec §15).

Tests cover:
  - can_publish() quality threshold checks (all 7 criteria)
  - publish() workflow: Draft → approved → published
  - PublishedScenario row creation
  - GET /api/scenarios returns only published scenarios
  - Status transition enforcement

Uses in-memory SQLite + mock LLM provider.
"""

import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.enums import DraftStatus
from app.models.published_scenario import PublishedScenario
from app.persistence.authoring_repositories import (
    StoryDraftRepository,
    StoryDraftVersionRepository,
    StoryReviewReportRepository,
    StoryValidationReportRepository,
)
from app.persistence.database import get_session_factory, init_db
from app.services.publishing_service import PublishingService


# ── Test graph helpers ─────────────────────────────────────────────────

def _make_graph(
    num_nodes: int = 20,
    num_endings: int = 3,
    reachable_ends: bool = True,
) -> dict:
    """Build a valid story graph with configurable size and endings.

    The graph has one start node, ``num_endings`` end nodes, and
    enough intermediate nodes to reach ``num_nodes`` total.
    If ``reachable_ends`` is False, end nodes are disconnected.
    """
    nodes: dict[str, dict] = {}

    # Start node
    nodes["start"] = {
        "id": "start",
        "title": "Start",
        "type": "start",
        "scene_goal": "Beginning",
        "choices": [{"id": "c1", "label": "Go", "next_node_id": "n1"}],
        "quality_notes": ["start note"],
        "is_start": True,
        "is_end": False,
    }

    # Intermediate chain: n1 → n2 → ... → nK
    intermediates_needed = num_nodes - 1 - num_endings
    prev = "n1"
    for i in range(1, intermediates_needed + 1):
        nid = f"n{i}"
        next_nid = f"n{i+1}" if i < intermediates_needed else "bridge"
        nodes[nid] = {
            "id": nid,
            "title": f"Node {i}",
            "type": "scene",
            "scene_goal": f"Goal {i}",
            "choices": [{"id": f"c_n{i}", "label": "Next", "next_node_id": next_nid}],
            "quality_notes": [f"note {i}"],
            "is_start": False,
            "is_end": False,
        }
        prev = nid

    # Bridge node that fans out to endings (or not)
    if intermediates_needed > 0:
        bridge_targets = [f"end_{j}" for j in range(num_endings)] if reachable_ends else []
        nodes["bridge"] = {
            "id": "bridge",
            "title": "Bridge",
            "type": "decision",
            "scene_goal": "Choose your path",
            "choices": [
                {"id": f"cb_{j}", "label": f"Path {j}", "next_node_id": tgt}
                for j, tgt in enumerate(bridge_targets)
            ],
            "quality_notes": ["bridge note"],
            "is_start": False,
            "is_end": False,
        }
    else:
        # If no intermediates, start points to endings directly
        targets = [f"end_{j}" for j in range(num_endings)] if reachable_ends else []
        nodes["start"]["choices"] = [
            {"id": f"cs_{j}", "label": f"Path {j}", "next_node_id": tgt}
            for j, tgt in enumerate(targets)
        ]

    # End nodes
    for j in range(num_endings):
        eid = f"end_{j}"
        nodes[eid] = {
            "id": eid,
            "title": f"Ending {j}",
            "type": "end",
            "scene_goal": f"Ending {j}",
            "choices": [],
            "quality_notes": [f"ending note {j}"],
            "is_start": False,
            "is_end": True,
        }

    return {
        "title": "Test Story",
        "genre": "test",
        "tone": "neutral",
        "start_node_id": "start",
        "nodes": nodes,
    }


# ── Shared fixtures ────────────────────────────────────────────────────

@pytest.fixture
async def db_session():
    """Provide an async DB session for direct service tests."""
    await init_db()
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def _create_approved_draft(
    session: AsyncSession,
    *,
    graph: dict | None = None,
    quality_score: float | None = 7.5,
    validation_valid: bool = True,
    review_issues: list | None = None,
    review_score: float = 7.5,
) -> str:
    """Create a draft that is fully approved and ready to publish.

    Returns the draft_id.
    """
    draft_repo = StoryDraftRepository(session)
    version_repo = StoryDraftVersionRepository(session)
    validation_repo = StoryValidationReportRepository(session)
    review_repo = StoryReviewReportRepository(session)

    if graph is None:
        graph = _make_graph(num_nodes=20, num_endings=3)

    if review_issues is None:
        review_issues = [
            {"severity": "medium", "node_id": "n1", "problem": "Minor", "suggestion": "Fix"},
        ]

    # Create draft
    draft = await draft_repo.create(
        title="Test Story",
        genre="test",
        tone="neutral",
        brief={"title": "Test"},
    )

    # Create version
    version = await version_repo.create(
        draft_id=draft.id,
        graph=graph,
        outline={"premise": "test"},
        created_by="test",
        notes="test version",
    )

    # Create review report
    await review_repo.create(
        draft_id=draft.id,
        version_id=version.id,
        score=review_score,
        issues=review_issues,
        summary="Test review",
    )

    # Create validation report
    await validation_repo.create(
        draft_id=draft.id,
        version_id=version.id,
        is_valid=validation_valid,
        errors=[] if validation_valid else ["Some error"],
        warnings=[],
    )

    # Transition: draft → generating → needs_review → validated → approved
    await draft_repo.update_status(draft.id, DraftStatus.GENERATING)
    await draft_repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
    await draft_repo.update_status(draft.id, DraftStatus.VALIDATED)
    await draft_repo.update_status(draft.id, DraftStatus.APPROVED)

    # Set quality_score directly (update_status doesn't set it on approved transition)
    from sqlalchemy import update as sa_update
    from app.models.story_draft import StoryDraft as _SD
    stmt = sa_update(_SD).where(_SD.id == draft.id).values(quality_score=quality_score)
    await session.execute(stmt)
    await session.commit()

    return draft.id


# ── can_publish: quality threshold checks ──────────────────────────────

class TestCanPublishThresholds:
    """Test each quality threshold criterion from Spec §15."""

    @pytest.mark.asyncio
    async def test_can_publish_all_criteria_met(self, db_session):
        """All criteria met → can_publish returns True."""
        draft_id = await _create_approved_draft(db_session)
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is True
        assert reasons == []

    @pytest.mark.asyncio
    async def test_reject_not_approved_status(self, db_session):
        """Draft not in 'approved' status → rejection."""
        draft_id = await _create_approved_draft(db_session)
        # Move it back to validated
        draft_repo = StoryDraftRepository(db_session)
        # APPROVED → NEEDS_REPAIR → NEEDS_REVIEW → VALIDATED
        await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REPAIR)
        await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REVIEW)
        await draft_repo.update_status(draft_id, DraftStatus.VALIDATED)

        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("approved" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_reject_no_versions(self, db_session):
        """Draft with no versions → rejection."""
        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.create(
            title="Empty", genre="g", tone="t", brief={},
        )
        # Force to approved via raw update (bypassing transition checks)
        from sqlalchemy import update as sa_update
        from app.models.story_draft import StoryDraft as _SD
        stmt = sa_update(_SD).where(_SD.id == draft.id).values(status="approved")
        await db_session.execute(stmt)
        await db_session.commit()

        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft.id)
        assert can is False
        assert any("no versions" in r.lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_reject_too_few_nodes(self, db_session):
        """Graph has fewer than min_node_count nodes → rejection."""
        settings = get_settings()
        # Create a graph with fewer nodes than the minimum
        small_graph = _make_graph(
            num_nodes=max(2, settings.min_node_count - 1),
            num_endings=1,
        )
        # If the graph still has >= min_node_count nodes (due to
        # minimum node construction), trim it manually
        if len(small_graph["nodes"]) >= settings.min_node_count:
            # Keep only start + one end node
            small_graph = {
                "title": "Tiny",
                "genre": "test",
                "tone": "neutral",
                "start_node_id": "start",
                "nodes": {
                    "start": {
                        "id": "start",
                        "title": "Start",
                        "type": "start",
                        "scene_goal": "Begin",
                        "choices": [{"id": "c1", "label": "End", "next_node_id": "end_0"}],
                        "quality_notes": ["n"],
                        "is_start": True,
                        "is_end": False,
                    },
                    "end_0": {
                        "id": "end_0",
                        "title": "End",
                        "type": "end",
                        "scene_goal": "Done",
                        "choices": [],
                        "quality_notes": ["n"],
                        "is_start": False,
                        "is_end": True,
                    },
                },
            }
        draft_id = await _create_approved_draft(db_session, graph=small_graph)
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("nodes" in r and "minimum" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_reject_too_few_endings(self, db_session):
        """Graph has fewer than min_ending_count endings → rejection."""
        settings = get_settings()
        graph = _make_graph(
            num_nodes=settings.min_node_count,
            num_endings=max(0, settings.min_ending_count - 1),
        )
        draft_id = await _create_approved_draft(db_session, graph=graph)
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("endings" in r and "minimum" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_reject_no_quality_score(self, db_session):
        """quality_score is None → rejection."""
        draft_id = await _create_approved_draft(db_session, quality_score=None)
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("quality_score" in r or "no quality" in r.lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_reject_low_quality_score(self, db_session):
        """quality_score < 7.0 → rejection."""
        draft_id = await _create_approved_draft(db_session, quality_score=5.5)
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("quality score" in r.lower() and "threshold" in r.lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_reject_no_validation_report(self, db_session):
        """No validation report → rejection."""
        draft_id = await _create_approved_draft(db_session)
        # Delete validation reports
        val_repo = StoryValidationReportRepository(db_session)
        reports = await val_repo.list_by_draft(draft_id)
        for r in reports:
            await db_session.delete(r)
        await db_session.commit()

        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("validation" in r.lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_reject_invalid_validation(self, db_session):
        """Latest validation report is_valid=False → rejection."""
        draft_id = await _create_approved_draft(db_session, validation_valid=False)
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("validation" in r.lower() and "invalid" in r.lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_reject_no_review_report(self, db_session):
        """No review report → rejection."""
        draft_id = await _create_approved_draft(db_session)
        # Delete review reports
        review_repo = StoryReviewReportRepository(db_session)
        reports = await review_repo.list_by_draft(draft_id)
        for r in reports:
            await db_session.delete(r)
        await db_session.commit()

        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("review" in r.lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_reject_high_severity_issues(self, db_session):
        """Latest review has high-severity issues → rejection."""
        issues = [
            {"severity": "high", "node_id": "n1", "problem": "Bad", "suggestion": "Fix it"},
            {"severity": "medium", "node_id": "n2", "problem": "Meh", "suggestion": "Whatever"},
        ]
        draft_id = await _create_approved_draft(
            db_session, review_issues=issues, review_score=8.0,
        )
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("high-severity" in r for r in reasons)

    @pytest.mark.asyncio
    async def test_accept_medium_severity_issues(self, db_session):
        """Medium/low/info severity issues do NOT block publishing."""
        issues = [
            {"severity": "medium", "node_id": "n1", "problem": "OK", "suggestion": "Fix"},
            {"severity": "low", "node_id": "n2", "problem": "Meh", "suggestion": "Meh"},
            {"severity": "info", "node_id": "n3", "problem": "FYI", "suggestion": "FYI"},
        ]
        draft_id = await _create_approved_draft(
            db_session, review_issues=issues, review_score=7.0,
        )
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is True, f"Expected publishable, got reasons: {reasons}"

    @pytest.mark.asyncio
    async def test_reject_unreachable_end_nodes(self, db_session):
        """End nodes not reachable from start → rejection."""
        graph = _make_graph(num_nodes=20, num_endings=3, reachable_ends=False)
        draft_id = await _create_approved_draft(db_session, graph=graph)
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("not reachable" in r.lower() or "reachability" in r.lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_quality_score_exactly_at_threshold(self, db_session):
        """quality_score == 7.0 (exactly the threshold) → accepted."""
        draft_id = await _create_approved_draft(db_session, quality_score=7.0)
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is True, f"Score 7.0 should be publishable, got: {reasons}"

    @pytest.mark.asyncio
    async def test_quality_score_just_below_threshold(self, db_session):
        """quality_score == 6.9 (just below threshold) → rejected."""
        draft_id = await _create_approved_draft(db_session, quality_score=6.9)
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        assert any("quality score" in r.lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_draft_not_found(self, db_session):
        """Non-existent draft → rejection with 'not found'."""
        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish("draft_nonexistent")
        assert can is False
        assert any("not found" in r.lower() for r in reasons)

    @pytest.mark.asyncio
    async def test_multiple_failure_reasons(self, db_session):
        """Multiple criteria fail → all reasons returned."""
        # No quality_score, no validation, no review, too few nodes
        graph = _make_graph(num_nodes=2, num_endings=1)
        draft_id = await _create_approved_draft(
            db_session,
            graph=graph,
            quality_score=None,
        )
        # Delete validation and review reports
        val_repo = StoryValidationReportRepository(db_session)
        for r in await val_repo.list_by_draft(draft_id):
            await db_session.delete(r)
        review_repo = StoryReviewReportRepository(db_session)
        for r in await review_repo.list_by_draft(draft_id):
            await db_session.delete(r)
        await db_session.commit()

        svc = PublishingService(db_session)
        can, reasons = await svc.can_publish(draft_id)
        assert can is False
        # Should have multiple reasons
        assert len(reasons) >= 3


# ── publish() workflow ─────────────────────────────────────────────────

class TestPublishWorkflow:
    """Test the publish() method end-to-end."""

    @pytest.mark.asyncio
    async def test_publish_success(self, db_session):
        """Publish creates PublishedScenario row and transitions to PUBLISHED."""
        draft_id = await _create_approved_draft(db_session)
        svc = PublishingService(db_session)
        result = await svc.publish(draft_id)

        assert result["draft_id"] == draft_id
        assert result["scenario_id"] == f"published_{draft_id}"
        assert result["nodes_published"] > 0

        # Verify PublishedScenario row exists
        stmt = select(PublishedScenario).where(PublishedScenario.id == result["scenario_id"])
        sc_result = await db_session.execute(stmt)
        scenario = sc_result.scalar_one_or_none()
        assert scenario is not None
        assert scenario.draft_id == draft_id
        assert scenario.title == "Test Story"
        assert scenario.genre == "test"
        # graph_json should be valid JSON with nodes
        graph = json.loads(scenario.graph_json)
        assert "nodes" in graph
        assert len(graph["nodes"]) == result["nodes_published"]

    @pytest.mark.asyncio
    async def test_publish_transitions_to_published(self, db_session):
        """Draft status transitions to PUBLISHED after publish()."""
        draft_id = await _create_approved_draft(db_session)
        svc = PublishingService(db_session)
        await svc.publish(draft_id)

        draft_repo = StoryDraftRepository(db_session)
        draft = await draft_repo.get_by_id(draft_id)
        assert draft.status == DraftStatus.PUBLISHED.value
        assert draft.published_at is not None

    @pytest.mark.asyncio
    async def test_publish_raises_on_criteria_failure(self, db_session):
        """publish() raises ValueError when criteria not met."""
        draft_id = await _create_approved_draft(db_session, quality_score=5.0)
        svc = PublishingService(db_session)
        with pytest.raises(ValueError, match="Cannot publish"):
            await svc.publish(draft_id)

    @pytest.mark.asyncio
    async def test_publish_idempotent_overwrites(self, db_session):
        """Re-publishing replaces the old PublishedScenario row."""
        draft_id = await _create_approved_draft(db_session)
        svc = PublishingService(db_session)

        # First publish
        result1 = await svc.publish(draft_id)
        scenario_id = result1["scenario_id"]

        # Move back to approved (via the transition: PUBLISHED is terminal,
        # so we use raw update to test the overwrite path)
        from sqlalchemy import update as sa_update
        from app.models.story_draft import StoryDraft as _SD
        stmt = sa_update(_SD).where(_SD.id == draft_id).values(status="approved")
        await db_session.execute(stmt)
        await db_session.commit()

        # Second publish — should overwrite, not duplicate
        result2 = await svc.publish(draft_id)
        assert result2["scenario_id"] == scenario_id

        # Only one PublishedScenario row for this scenario_id
        stmt = select(PublishedScenario).where(PublishedScenario.id == scenario_id)
        rows = (await db_session.execute(stmt)).scalars().all()
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_publish_nonexistent_draft(self, db_session):
        """publish() on non-existent draft raises ValueError."""
        svc = PublishingService(db_session)
        with pytest.raises(ValueError, match="not found"):
            await svc.publish("draft_nonexistent")


# ── API integration: scenario routes ──────────────────────────────────

class TestScenarioRoutes:
    """Test GET /api/scenarios returns only published scenarios."""

    @pytest.mark.asyncio
    async def test_empty_scenarios(self, client):
        """No published scenarios → only file-based scenarios returned."""
        resp = await client.get("/api/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        # File-based scenarios (e.g. helios) are always present.
        # No DB-published scenarios should exist yet.
        ids = [s["id"] for s in data]
        assert "helios" in ids  # file-based
        # No published_ IDs from the DB
        assert not any(i.startswith("published_") for i in ids)

    @pytest.mark.asyncio
    async def test_published_scenario_visible(self, client):
        """After publishing, the scenario appears in GET /api/scenarios."""
        # Full pipeline: create → review → validate → approve → publish
        resp = await client.post("/api/admin/story-drafts", json={
            "title": "Published Test",
            "genre": "adventure",
            "tone": "heroic",
        })
        assert resp.status_code == 201
        draft_id = resp.json()["draft_id"]

        await client.post(f"/api/admin/story-drafts/{draft_id}/review")
        await client.post(f"/api/admin/story-drafts/{draft_id}/validate")
        await client.post(f"/api/admin/story-drafts/{draft_id}/approve")
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/publish")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

        # Now check scenarios endpoint
        resp = await client.get("/api/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        # Find our published scenario
        found = [s for s in data if s["title"] == "Published Test"]
        assert len(found) == 1
        assert found[0]["genre"] == "adventure"
        assert found[0]["id"].startswith("published_")

    @pytest.mark.asyncio
    async def test_unpublished_draft_not_visible(self, client):
        """Drafts that are not published must NOT appear in scenarios."""
        # Create a draft but don't publish it
        resp = await client.post("/api/admin/story-drafts", json={
            "title": "Unpublished Draft",
            "genre": "horror",
            "tone": "eerie",
        })
        assert resp.status_code == 201

        resp = await client.get("/api/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        # Should not contain the unpublished draft
        titles = [s["title"] for s in data]
        assert "Unpublished Draft" not in titles

    @pytest.mark.asyncio
    async def test_multiple_published_scenarios(self, client):
        """Multiple published scenarios all appear in the list."""
        for i in range(3):
            resp = await client.post("/api/admin/story-drafts", json={
                "title": f"Multi Publish {i}",
                "genre": "test",
                "tone": "neutral",
            })
            draft_id = resp.json()["draft_id"]
            await client.post(f"/api/admin/story-drafts/{draft_id}/review")
            await client.post(f"/api/admin/story-drafts/{draft_id}/validate")
            await client.post(f"/api/admin/story-drafts/{draft_id}/approve")
            await client.post(f"/api/admin/story-drafts/{draft_id}/publish")

        resp = await client.get("/api/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 3
        titles = [s["title"] for s in data]
        for i in range(3):
            assert f"Multi Publish {i}" in titles


# ── API integration: publish endpoint ─────────────────────────────────

class TestPublishEndpoint:
    """Test POST /api/admin/story-drafts/{id}/publish with quality thresholds."""

    @pytest.mark.asyncio
    async def test_publish_without_review_rejected(self, client):
        """Publish without running review → 409 (quality_score is None)."""
        resp = await client.post("/api/admin/story-drafts", json={
            "title": "No Review",
            "genre": "g",
            "tone": "t",
        })
        draft_id = resp.json()["draft_id"]

        # Validate and approve, but no review → quality_score is None
        await client.post(f"/api/admin/story-drafts/{draft_id}/validate")
        await client.post(f"/api/admin/story-drafts/{draft_id}/approve")

        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/publish")
        assert resp.status_code == 409
        assert "quality_score" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_publish_without_validation_rejected(self, client):
        """Publish without validation → cannot reach approved status."""
        resp = await client.post("/api/admin/story-drafts", json={
            "title": "No Validation",
            "genre": "g",
            "tone": "t",
        })
        draft_id = resp.json()["draft_id"]

        # Review only, no validation
        await client.post(f"/api/admin/story-drafts/{draft_id}/review")

        # Try to approve — should fail (needs validation first)
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/approve")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_publish_without_approval_rejected(self, client):
        """Publish without approval → 409."""
        resp = await client.post("/api/admin/story-drafts", json={
            "title": "No Approval",
            "genre": "g",
            "tone": "t",
        })
        draft_id = resp.json()["draft_id"]

        # Review + validate but no approve
        await client.post(f"/api/admin/story-drafts/{draft_id}/review")
        await client.post(f"/api/admin/story-drafts/{draft_id}/validate")

        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/publish")
        assert resp.status_code == 409
        assert "approved" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_full_pipeline_publish_success(self, client):
        """Full pipeline: create → review → validate → approve → publish."""
        resp = await client.post("/api/admin/story-drafts", json={
            "title": "Full Pipeline",
            "genre": "sci_fi",
            "tone": "mystery",
        })
        draft_id = resp.json()["draft_id"]

        await client.post(f"/api/admin/story-drafts/{draft_id}/review")
        await client.post(f"/api/admin/story-drafts/{draft_id}/validate")
        await client.post(f"/api/admin/story-drafts/{draft_id}/approve")
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/publish")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "published"
        assert "scenario" in data["message"]
        assert "nodes" in data["message"]

        # Verify draft has published_at timestamp
        detail = await client.get(f"/api/admin/story-drafts/{draft_id}")
        assert detail.json()["published_at"] is not None

    @pytest.mark.asyncio
    async def test_publish_not_found(self, client):
        """Publish non-existent draft → 404."""
        resp = await client.post("/api/admin/story-drafts/draft_nonexistent/publish")
        assert resp.status_code == 404
