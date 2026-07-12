"""Tests for targeted FixThis repair and re-review flow."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
async def db_session():
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        future=True,
        echo=False,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_fk(dbapi_conn, _conn_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    import app.models  # noqa: F401
    from app.persistence.database import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def admin_client():
    from app.admin_ui.app import admin_app
    from app.persistence.database import close_db, init_db

    await init_db()
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await close_db()


async def _create_draft_with_review() -> tuple[str, str]:
    from app.persistence.authoring_repositories import (
        StoryDraftRepository,
        StoryDraftVersionRepository,
        StoryReviewReportRepository,
    )
    from app.persistence.database import get_session_factory

    graph = {
        "start_node_id": "node_001",
        "nodes": {
            "node_001": {
                "type": "start",
                "title": "Opening",
                "choices": [],
            }
        },
    }
    outline = {"premise": "Test premise"}
    async with get_session_factory()() as session:
        draft_repo = StoryDraftRepository(session)
        version_repo = StoryDraftVersionRepository(session)
        review_repo = StoryReviewReportRepository(session)
        draft = await draft_repo.create(title="FixThis Test", genre="sci-fi", tone="tense")
        version = await version_repo.create(
            draft_id=draft.id,
            graph=graph,
            outline=outline,
            created_by="test",
        )
        report = await review_repo.create(
            draft_id=draft.id,
            version_id=version.id,
            score=5.0,
            issues=[
                {
                    "severity": "high",
                    "node_id": "node_001",
                    "problem": "Opening lacks stakes",
                    "suggestion": "Add immediate danger",
                },
                {
                    "severity": "low",
                    "node_id": None,
                    "problem": "Summary is bland",
                    "suggestion": "Sharpen theme",
                },
            ],
            summary="Needs work",
        )
        return draft.id, report.id


@pytest.mark.asyncio
async def test_update_issue_status_marks_single_issue(db_session):
    from app.persistence.authoring_repositories import (
        StoryDraftRepository,
        StoryReviewReportRepository,
    )

    draft_repo = StoryDraftRepository(db_session)
    draft = await draft_repo.create(title="R", genre="g", tone="t")
    review_repo = StoryReviewReportRepository(db_session)
    report = await review_repo.create(
        draft_id=draft.id,
        score=5.0,
        issues=[
            {"severity": "high", "problem": "first"},
            {"severity": "low", "problem": "second"},
        ],
    )

    updated = await review_repo.update_issue_status(report.id, 0, "fix_requested")

    assert updated is not None
    issues = review_repo.parse_issues(updated)
    assert issues[0]["fix_status"] == "fix_requested"
    assert "fix_requested_at" in issues[0]
    assert "fix_status" not in issues[1]


@pytest.mark.asyncio
async def test_fix_issue_route_repairs_single_issue_rereviews_and_closes(monkeypatch, admin_client):
    from app.persistence.authoring_repositories import (
        StoryDraftVersionRepository,
        StoryReviewReportRepository,
    )
    from app.persistence.database import get_session_factory

    draft_id, report_id = await _create_draft_with_review()
    calls: dict[str, object] = {}

    class FakeRepairAgent:
        async def repair(self, graph, review_report):
            calls["repair_report"] = review_report
            repaired = dict(graph)
            repaired["nodes"] = dict(graph["nodes"])
            repaired["nodes"]["node_001"] = {
                **graph["nodes"]["node_001"],
                "scene_text": "Immediate danger added.",
            }
            return {"graph": repaired, "summary": "Fixed opening stakes"}

    class FakeCriticAgent:
        async def review(self, outline, graph):
            calls["review_graph"] = graph
            return {"score": 8.2, "issues": [], "summary": "Looks fixed"}

    monkeypatch.setattr("app.admin_ui.app.StoryRepairAgent", FakeRepairAgent)
    monkeypatch.setattr("app.admin_ui.app.StoryCriticAgent", FakeCriticAgent)

    resp = await admin_client.post(f"/draft/{draft_id}/fix-issue", json={"finding_id": 1})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["finding_id"] == 1
    assert payload["fix_status"] == "fixed"
    assert payload["review"]["score"] == 8.2
    assert payload["review"]["issues"] == []

    repair_report = calls["repair_report"]
    assert repair_report["issues"] == [
        {
            "severity": "high",
            "node_id": "node_001",
            "problem": "Opening lacks stakes",
            "suggestion": "Add immediate danger",
            "fix_status": "fix_requested",
            "fix_requested_at": repair_report["issues"][0]["fix_requested_at"],
        }
    ]

    async with get_session_factory()() as session:
        review_repo = StoryReviewReportRepository(session)
        version_repo = StoryDraftVersionRepository(session)
        reports = await review_repo.list_by_draft(draft_id)
        versions = await version_repo.list_by_draft(draft_id)
        original = next(r for r in reports if r.id == report_id)
        original_issues = review_repo.parse_issues(original)
        latest_issues = review_repo.parse_issues(reports[-1])

    assert original_issues[0]["fix_status"] == "fixed"
    assert latest_issues == []
    assert len(versions) == 2
    assert versions[-1].created_by == "repair_agent"


@pytest.mark.asyncio
async def test_draft_detail_wires_fixthis_to_backend(admin_client):
    draft_id, _ = await _create_draft_with_review()

    resp = await admin_client.get(f"/draft/{draft_id}")

    assert resp.status_code == 200
    assert "/fix-issue" in resp.text
    assert "fetch('/admin/draft/' + draftId + '/fix-issue'" in resp.text


@pytest.mark.asyncio
async def test_fix_issue_with_report_id_uses_correct_report(monkeypatch, admin_client):
    """When a newer review report appears between render and click, the
    report_id sent with the request must target the original report, not
    reviews[-1]."""
    from app.persistence.authoring_repositories import (
        StoryDraftRepository,
        StoryDraftVersionRepository,
        StoryReviewReportRepository,
    )
    from app.persistence.database import get_session_factory

    graph = {
        "start_node_id": "node_001",
        "nodes": {"node_001": {"type": "start", "title": "X", "choices": []}},
    }
    outline = {"premise": "P"}
    async with get_session_factory()() as session:
        draft_repo = StoryDraftRepository(session)
        version_repo = StoryDraftVersionRepository(session)
        review_repo = StoryReviewReportRepository(session)
        draft = await draft_repo.create(title="Stale Test", genre="sci-fi", tone="tense")
        version = await version_repo.create(
            draft_id=draft.id, graph=graph, outline=outline, created_by="test"
        )
        # Original report with 2 issues — the one that "rendered" the button
        original_report = await review_repo.create(
            draft_id=draft.id,
            version_id=version.id,
            score=5.0,
            issues=[
                {"severity": "high", "node_id": "node_001", "problem": "A", "suggestion": "B"},
                {"severity": "low", "node_id": None, "problem": "C", "suggestion": "D"},
            ],
            summary="Old",
        )
        # A newer report with 0 issues — this would be reviews[-1]
        await review_repo.create(
            draft_id=draft.id,
            version_id=version.id,
            score=9.0,
            issues=[],
            summary="New empty",
        )
        draft_id = draft.id
        original_report_id = original_report.id

    class FakeRepairAgent:
        async def repair(self, graph, review_report):
            return {"graph": graph, "summary": "ok"}

    class FakeCriticAgent:
        async def review(self, outline, graph):
            return {"score": 8.0, "issues": [], "summary": "good"}

    monkeypatch.setattr("app.admin_ui.app.StoryRepairAgent", FakeRepairAgent)
    monkeypatch.setattr("app.admin_ui.app.StoryCriticAgent", FakeCriticAgent)

    # Without report_id: finding_id=1 would hit the newer empty report -> 404
    resp_stale = await admin_client.post(
        f"/draft/{draft_id}/fix-issue", json={"finding_id": 1}
    )
    assert resp_stale.status_code == 404
    assert "Finding 1 not found" in resp_stale.json()["detail"]

    # With report_id: targets the original report with 2 issues -> 200
    resp_ok = await admin_client.post(
        f"/draft/{draft_id}/fix-issue",
        json={"finding_id": 1, "report_id": original_report_id},
    )
    assert resp_ok.status_code == 200
    payload = resp_ok.json()
    assert payload["ok"] is True
    assert payload["finding_id"] == 1
    assert payload["fix_status"] == "fixed"


@pytest.mark.asyncio
async def test_draft_detail_renders_report_id_in_button(admin_client):
    """The FixThis button must include data-report-id so the frontend can
    send it with the fix request."""
    draft_id, report_id = await _create_draft_with_review()

    resp = await admin_client.get(f"/draft/{draft_id}")
    assert resp.status_code == 200
    assert f'data-report-id="{report_id}"' in resp.text
