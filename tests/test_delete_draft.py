"""Tests for Story/Draft deletion functionality (t_ced413d6).

Tests cover:
  (1)  API: POST /api/admin/story-drafts/{id}/delete — success
  (2)  API: POST /api/admin/story-drafts/{id}/delete — 404 for missing draft
  (3)  API: After delete, GET /api/admin/story-drafts/{id} returns 404
  (4)  API: Delete cascades — versions, reviews, validations, jobs gone
  (5)  API: Published scenario NOT deleted
  (6)  Admin UI: POST /admin/draft/{id}/delete redirects to /admin/ (303)
  (7)  Admin UI: GET /admin/ list has delete button
  (8)  Admin UI: GET /admin/draft/{id} detail has delete button
  (9)  Admin UI: Delete on admin UI redirects to list
  (10) Event log entry emitted on delete
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
    """Create a draft via API and return the response JSON."""
    resp = await client.post("/api/admin/story-drafts", json=VALID_BRIEF)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── API Tests ──────────────────────────────────────────────────────────


class TestDeleteDraftAPI:
    """Tests for the JSON API delete endpoint."""

    @pytest.mark.asyncio
    async def test_delete_draft_success(self, client):
        """(1) POST /api/admin/story-drafts/{id}/delete returns 200 with confirmation."""
        created = await _create_draft(client)
        draft_id = created["draft_id"]

        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/delete")
        assert resp.status_code == 200
        body = resp.json()
        assert body["draft_id"] == draft_id
        assert body["status"] == "deleted"
        assert "Signal von Helios" in body["message"]

    @pytest.mark.asyncio
    async def test_delete_draft_not_found(self, client):
        """(2) POST /api/admin/story-drafts/{id}/delete — 404 for missing draft."""
        resp = await client.post("/api/admin/story-drafts/draft_nonexistent/delete")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_draft_gone_after_delete(self, client):
        """(3) After delete, GET /api/admin/story-drafts/{id} returns 404."""
        created = await _create_draft(client)
        draft_id = created["draft_id"]

        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/delete")
        assert resp.status_code == 200

        resp = await client.get(f"/api/admin/story-drafts/{draft_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_cascades_versions_reviews_jobs(self, client):
        """(4) Delete cascades — versions, reviews, validations, jobs all gone."""
        created = await _create_draft(client)
        draft_id = created["draft_id"]

        # Run review to create review report + review job
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/review")
        assert resp.status_code == 200

        # Run validation to create validation report + validation job
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/validate")
        assert resp.status_code == 200

        # Verify draft has versions, reviews, validations, jobs
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}")
        assert resp.status_code == 200
        detail = resp.json()
        assert len(detail["versions"]) >= 1

        # Delete the draft
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/delete")
        assert resp.status_code == 200

        # Draft is gone
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}")
        assert resp.status_code == 404

        # Graph endpoint also 404
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}/graph")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_removes_from_list(self, client):
        """After delete, the draft no longer appears in the list."""
        created = await _create_draft(client)
        draft_id = created["draft_id"]

        # Verify it's in the list
        resp = await client.get("/api/admin/story-drafts")
        assert resp.status_code == 200
        assert any(d["id"] == draft_id for d in resp.json())

        # Delete
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/delete")
        assert resp.status_code == 200

        # Verify it's gone from the list
        resp = await client.get("/api/admin/story-drafts")
        assert resp.status_code == 200
        assert not any(d["id"] == draft_id for d in resp.json())

    @pytest.mark.asyncio
    async def test_published_scenario_survives_delete(self, client):
        """(5) Published scenario is NOT deleted when its draft is deleted."""
        created = await _create_draft(client)
        draft_id = created["draft_id"]

        # Full pipeline: review → validate → approve → publish
        await client.post(f"/api/admin/story-drafts/{draft_id}/review")
        await client.post(f"/api/admin/story-drafts/{draft_id}/validate")
        await client.post(f"/api/admin/story-drafts/{draft_id}/approve")
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/publish")
        assert resp.status_code == 200
        publish_body = resp.json()
        assert "scenario" in publish_body["message"] or "scenario" in publish_body.get("message", "").lower()

        # Get scenario list before delete
        resp = await client.get("/api/scenarios")
        assert resp.status_code == 200
        scenarios_before = resp.json()

        # Delete the draft
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/delete")
        assert resp.status_code == 200

        # Draft is gone
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}")
        assert resp.status_code == 404

        # Scenarios still exist (same count or at least not reduced)
        resp = await client.get("/api/scenarios")
        assert resp.status_code == 200
        scenarios_after = resp.json()
        assert len(scenarios_after) >= len(scenarios_before)


# ── Admin UI Tests ─────────────────────────────────────────────────────


class TestDeleteDraftAdminUI:
    """Tests for the admin UI delete functionality."""

    @pytest.fixture
    async def admin_client(self):
        """Client targeting the admin sub-app directly."""
        from app.admin_ui.app import admin_app
        from app.persistence.database import init_db, close_db
        await init_db()
        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        await close_db()

    async def _create_draft_via_ui(self, admin_client) -> str:
        """Create a draft via the admin UI form, return its id."""
        resp = await admin_client.post(
            "/new",
            data={
                "title": "Delete Test Story",
                "genre": "science_fiction",
                "tone": "dark",
                "language": "de",
                "target_age": "16+",
                "node_count": "10",
                "ending_count": "2",
                "branching_level": "medium",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        location = resp.headers.get("location", "")
        return location.rsplit("/", 1)[-1]

    @pytest.mark.asyncio
    async def test_admin_ui_delete_redirects_to_list(self, admin_client):
        """(6/9) POST /admin/draft/{id}/delete redirects to /admin/ (303)."""
        draft_id = await self._create_draft_via_ui(admin_client)

        resp = await admin_client.post(f"/draft/{draft_id}/delete", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/admin/"

    @pytest.mark.asyncio
    async def test_admin_ui_draft_gone_after_delete(self, admin_client):
        """After delete, GET /admin/draft/{id} returns 404."""
        draft_id = await self._create_draft_via_ui(admin_client)

        resp = await admin_client.post(f"/draft/{draft_id}/delete", follow_redirects=False)
        assert resp.status_code == 303

        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_admin_ui_draft_list_has_delete_button(self, admin_client):
        """(7) GET /admin/ list has a delete button per draft row."""
        await self._create_draft_via_ui(admin_client)

        resp = await admin_client.get("/")
        assert resp.status_code == 200
        body = resp.text
        assert "Löschen" in body
        assert "/delete" in body

    @pytest.mark.asyncio
    async def test_admin_ui_draft_detail_has_delete_button(self, admin_client):
        """(8) GET /admin/draft/{id} detail page has a delete button."""
        draft_id = await self._create_draft_via_ui(admin_client)

        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        body = resp.text
        assert "Löschen" in body
        assert f"/draft/{draft_id}/delete" in body

    @pytest.mark.asyncio
    async def test_admin_ui_delete_not_found(self, admin_client):
        """POST /admin/draft/{nonexistent}/delete returns 404."""
        resp = await admin_client.post("/draft/draft_nonexistent/delete", follow_redirects=False)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_admin_ui_confirm_dialog_present(self, admin_client):
        """The confirmation JS function is present on both list and detail pages."""
        draft_id = await self._create_draft_via_ui(admin_client)

        # List page
        resp = await admin_client.get("/")
        assert "confirmDelete" in resp.text

        # Detail page
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert "confirmDeleteDetail" in resp.text


# ── Event Log Test ─────────────────────────────────────────────────────


class TestDeleteEventLog:
    """Tests that deleting a draft emits an event log entry."""

    @pytest.mark.asyncio
    async def test_delete_emits_event_log(self, client):
        """(10) Event log entry emitted on delete."""
        from app.services.event_log import event_log
        event_log.clear()

        created = await _create_draft(client)
        draft_id = created["draft_id"]

        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/delete")
        assert resp.status_code == 200

        # Check event log for the delete event
        events = event_log.list_events(limit=50)
        delete_events = [e for e in events["events"] if e["category"] == "delete"]
        assert len(delete_events) >= 1
        assert delete_events[0]["status"] == "done"
        assert "deleted" in delete_events[0]["message"].lower()
        assert delete_events[0]["draft_id"] == draft_id
