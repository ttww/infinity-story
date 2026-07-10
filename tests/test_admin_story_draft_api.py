"""Tests for Admin / Story Authoring API endpoints (Spec §13.2).

Tests all nine endpoints end-to-end through the FastAPI test client:
  POST   /api/admin/story-drafts
  GET    /api/admin/story-drafts
  GET    /api/admin/story-drafts/{id}
  GET    /api/admin/story-drafts/{id}/graph
  POST   /api/admin/story-drafts/{id}/review
  POST   /api/admin/story-drafts/{id}/repair
  POST   /api/admin/story-drafts/{id}/validate
  POST   /api/admin/story-drafts/{id}/approve
  POST   /api/admin/story-drafts/{id}/publish

Uses in-memory SQLite + mock LLM provider.
"""

import pytest


# ── helpers ────────────────────────────────────────────────────────────

VALID_BRIEF = {
    "title": "Signal von Helios",
    "genre": "science_fiction",
    "tone": "dark_mystery",
    "language": "de",
    "target_age": "16+",
    "node_count": 25,
    "ending_count": 3,
    "branching_level": "medium",
    "themes": ["isolation", "identity"],
    "forbidden_content": ["excessive_violence"],
}


async def _create_draft(client) -> dict:
    """Helper: create a draft and return the response JSON."""
    resp = await client.post("/api/admin/story-drafts", json=VALID_BRIEF)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── POST /api/admin/story-drafts ───────────────────────────────────────

class TestCreateDraft:
    @pytest.mark.asyncio
    async def test_create_draft_success(self, client):
        data = VALID_BRIEF.copy()
        data["title"] = "Test Story Alpha"
        resp = await client.post("/api/admin/story-drafts", json=data)
        assert resp.status_code == 201
        body = resp.json()
        assert body["draft_id"].startswith("draft_")
        assert body["status"] == "needs_review"
        assert body["job_id"] is not None
        assert body["job_id"].startswith("job_")

    @pytest.mark.asyncio
    async def test_create_draft_minimal_brief(self, client):
        resp = await client.post("/api/admin/story-drafts", json={
            "title": "Minimal",
            "genre": "horror",
            "tone": "eerie",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "needs_review"

    @pytest.mark.asyncio
    async def test_create_draft_empty_title_rejected(self, client):
        resp = await client.post("/api/admin/story-drafts", json={
            "title": "",
            "genre": "g",
            "tone": "t",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_draft_invalid_node_count(self, client):
        resp = await client.post("/api/admin/story-drafts", json={
            "title": "Bad",
            "genre": "g",
            "tone": "t",
            "node_count": 1,
        })
        assert resp.status_code == 422


# ── GET /api/admin/story-drafts ────────────────────────────────────────

class TestListDrafts:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        resp = await client.get("/api/admin/story-drafts")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_after_create(self, client):
        await _create_draft(client)
        await _create_draft(client)
        resp = await client.get("/api/admin/story-drafts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        for item in data:
            assert "id" in item
            assert "title" in item
            assert "status" in item
            assert "version_count" in item

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, client):
        await _create_draft(client)
        resp = await client.get("/api/admin/story-drafts?status=needs_review")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "needs_review"

    @pytest.mark.asyncio
    async def test_list_invalid_status(self, client):
        resp = await client.get("/api/admin/story-drafts?status=nonexistent")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_list_with_limit(self, client):
        for _ in range(3):
            await _create_draft(client)
        resp = await client.get("/api/admin/story-drafts?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ── GET /api/admin/story-drafts/{id} ───────────────────────────────────

class TestGetDraft:
    @pytest.mark.asyncio
    async def test_get_draft_success(self, client):
        created = await _create_draft(client)
        resp = await client.get(f"/api/admin/story-drafts/{created['draft_id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == created["draft_id"]
        assert data["title"] == "Signal von Helios"
        assert data["genre"] == "science_fiction"
        assert data["status"] == "needs_review"
        assert len(data["versions"]) == 1
        assert data["versions"][0]["version_number"] == 1
        assert data["brief"]["title"] == "Signal von Helios"

    @pytest.mark.asyncio
    async def test_get_draft_not_found(self, client):
        resp = await client.get("/api/admin/story-drafts/draft_nonexistent")
        assert resp.status_code == 404


# ── GET /api/admin/story-drafts/{id}/graph ─────────────────────────────

class TestGetDraftGraph:
    @pytest.mark.asyncio
    async def test_get_graph_latest(self, client):
        created = await _create_draft(client)
        resp = await client.get(f"/api/admin/story-drafts/{created['draft_id']}/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert data["draft_id"] == created["draft_id"]
        assert data["version_number"] == 1
        assert "nodes" in data["graph"]
        assert "start_node_id" in data["graph"]
        assert len(data["graph"]["nodes"]) > 0

    @pytest.mark.asyncio
    async def test_get_graph_not_found(self, client):
        resp = await client.get("/api/admin/story-drafts/draft_nonexistent/graph")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_graph_specific_version(self, client):
        created = await _create_draft(client)
        # Get the draft to find version ID
        detail = await client.get(f"/api/admin/story-drafts/{created['draft_id']}")
        version_id = detail.json()["versions"][0]["id"]
        resp = await client.get(
            f"/api/admin/story-drafts/{created['draft_id']}/graph?version_id={version_id}"
        )
        assert resp.status_code == 200
        assert resp.json()["version_id"] == version_id

    @pytest.mark.asyncio
    async def test_get_graph_invalid_version(self, client):
        created = await _create_draft(client)
        resp = await client.get(
            f"/api/admin/story-drafts/{created['draft_id']}/graph?version_id=ver_nonexistent"
        )
        assert resp.status_code == 404


# ── POST /api/admin/story-drafts/{id}/review ───────────────────────────

class TestStartReview:
    @pytest.mark.asyncio
    async def test_review_success(self, client):
        created = await _create_draft(client)
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/review")
        assert resp.status_code == 200
        data = resp.json()
        assert data["draft_id"] == created["draft_id"]
        assert data["id"].startswith("review_")
        assert data["score"] == 7.5
        assert len(data["issues"]) == 2
        assert data["summary"] is not None
        assert data["version_id"] is not None

    @pytest.mark.asyncio
    async def test_review_draft_not_found(self, client):
        resp = await client.post("/api/admin/story-drafts/draft_nonexistent/review")
        assert resp.status_code == 404


# ── POST /api/admin/story-drafts/{id}/repair ───────────────────────────

class TestStartRepair:
    @pytest.mark.asyncio
    async def test_repair_success(self, client):
        created = await _create_draft(client)
        # Review first to get a review report
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/review")
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/repair")
        assert resp.status_code == 200
        data = resp.json()
        assert data["draft_id"] == created["draft_id"]
        assert data["version_number"] == 2  # new version created
        assert "nodes" in data["graph"]
        assert "start_node_id" in data["graph"]

    @pytest.mark.asyncio
    async def test_repair_without_review(self, client):
        """Repair should work even without a prior review report."""
        created = await _create_draft(client)
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/repair")
        assert resp.status_code == 200
        assert resp.json()["version_number"] == 2

    @pytest.mark.asyncio
    async def test_repair_not_found(self, client):
        resp = await client.post("/api/admin/story-drafts/draft_nonexistent/repair")
        assert resp.status_code == 404


# ── POST /api/admin/story-drafts/{id}/validate ─────────────────────────

class TestStartValidation:
    @pytest.mark.asyncio
    async def test_validate_success(self, client):
        created = await _create_draft(client)
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["draft_id"] == created["draft_id"]
        assert data["id"].startswith("val_")
        assert data["is_valid"] is True
        assert isinstance(data["errors"], list)
        assert isinstance(data["warnings"], list)

    @pytest.mark.asyncio
    async def test_validate_transitions_to_validated(self, client):
        created = await _create_draft(client)
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/validate")
        resp = await client.get(f"/api/admin/story-drafts/{created['draft_id']}")
        assert resp.json()["status"] == "validated"

    @pytest.mark.asyncio
    async def test_validate_not_found(self, client):
        resp = await client.post("/api/admin/story-drafts/draft_nonexistent/validate")
        assert resp.status_code == 404


# ── POST /api/admin/story-drafts/{id}/approve ──────────────────────────

class TestApproveDraft:
    @pytest.mark.asyncio
    async def test_approve_success(self, client):
        created = await _create_draft(client)
        # Must validate before approve
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/validate")
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/approve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["draft_id"] == created["draft_id"]
        assert data["status"] == "approved"
        assert "message" in data

    @pytest.mark.asyncio
    async def test_approve_without_validation_rejected(self, client):
        created = await _create_draft(client)
        # Don't validate first — should fail
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/approve")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_approve_already_approved(self, client):
        created = await _create_draft(client)
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/validate")
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/approve")
        # Second approve should be idempotent
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/approve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    @pytest.mark.asyncio
    async def test_approve_not_found(self, client):
        resp = await client.post("/api/admin/story-drafts/draft_nonexistent/approve")
        assert resp.status_code == 404


# ── POST /api/admin/story-drafts/{id}/publish ──────────────────────────

class TestPublishDraft:
    @pytest.mark.asyncio
    async def test_publish_success(self, client):
        created = await _create_draft(client)
        # Full pipeline: review → validate → approve → publish
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/review")
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/validate")
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/approve")
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/publish")
        assert resp.status_code == 200
        data = resp.json()
        assert data["draft_id"] == created["draft_id"]
        assert data["status"] == "published"
        assert "scenario" in data["message"]
        assert "nodes" in data["message"]

    @pytest.mark.asyncio
    async def test_publish_without_approval_rejected(self, client):
        created = await _create_draft(client)
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/publish")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_publish_already_published(self, client):
        created = await _create_draft(client)
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/review")
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/validate")
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/approve")
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/publish")
        resp = await client.post(f"/api/admin/story-drafts/{created['draft_id']}/publish")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

    @pytest.mark.asyncio
    async def test_publish_not_found(self, client):
        resp = await client.post("/api/admin/story-drafts/draft_nonexistent/publish")
        assert resp.status_code == 404


# ── Full pipeline integration test ─────────────────────────────────────

class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_complete_authoring_pipeline(self, client):
        """End-to-end: create → review → repair → review → validate → approve → publish."""
        # 1. Create
        resp = await client.post("/api/admin/story-drafts", json=VALID_BRIEF)
        assert resp.status_code == 201
        draft_id = resp.json()["draft_id"]

        # 2. Review
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/review")
        assert resp.status_code == 200
        assert resp.json()["score"] == 7.5

        # 3. Repair (creates new version)
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/repair")
        assert resp.status_code == 200
        assert resp.json()["version_number"] == 2

        # 4. Review again
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/review")
        assert resp.status_code == 200

        # 5. Validate
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/validate")
        assert resp.status_code == 200
        assert resp.json()["is_valid"] is True

        # 6. Approve
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/approve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

        # 7. Publish
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/publish")
        assert resp.status_code == 200
        assert resp.json()["status"] == "published"

        # 8. Verify final state
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "published"
        assert data["approved_at"] is not None
        assert data["published_at"] is not None
        assert len(data["versions"]) == 2  # original + repair

    @pytest.mark.asyncio
    async def test_draft_has_quality_score_after_review(self, client):
        created = await _create_draft(client)
        await client.post(f"/api/admin/story-drafts/{created['draft_id']}/review")
        resp = await client.get(f"/api/admin/story-drafts/{created['draft_id']}")
        assert resp.json()["quality_score"] == 7.5

    @pytest.mark.asyncio
    async def test_versions_increment(self, client):
        created = await _create_draft(client)
        draft_id = created["draft_id"]

        # Initial version
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}")
        assert resp.json()["versions"][0]["version_number"] == 1

        # After repair, should have 2 versions
        await client.post(f"/api/admin/story-drafts/{draft_id}/repair")
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}")
        versions = resp.json()["versions"]
        assert len(versions) == 2
        assert versions[1]["version_number"] == 2
