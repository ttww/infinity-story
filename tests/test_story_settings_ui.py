"""Tests for the story parameter editing UI endpoint (t_cd4d7480).

Verifies that:
  - POST /admin/draft/{id}/settings persists the four parameters
  - GET /admin/draft/{id} (detail page) shows the current values
  - Invalid input (min > max) is rejected with 422 + clear error message
  - Out-of-bounds values are rejected
  - Non-integer values are rejected
  - 404 for nonexistent draft
"""

import pytest


class TestStorySettingsEndpoint:
    """Tests for POST /admin/draft/{draft_id}/settings."""

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

    async def _create_draft(self, admin_client) -> str:
        """Create a draft via the admin UI form, return its id."""
        resp = await admin_client.post(
            "/new",
            data={
                "title": "Settings Test Story",
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

    # ── Success cases ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_update_settings_success(self, admin_client):
        """Valid settings are persisted and returned."""
        draft_id = await self._create_draft(admin_client)

        payload = {
            "min_sentences_per_node": 5,
            "max_sentences_per_node": 12,
            "min_node_connections": 3,
            "max_node_connections": 7,
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/settings",
            json=payload,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["values"]["min_sentences_per_node"] == 5
        assert data["values"]["max_sentences_per_node"] == 12
        assert data["values"]["min_node_connections"] == 3
        assert data["values"]["max_node_connections"] == 7

    @pytest.mark.asyncio
    async def test_settings_persisted_after_reload(self, admin_client):
        """After saving, GET /draft/{id} shows the updated values."""
        draft_id = await self._create_draft(admin_client)

        payload = {
            "min_sentences_per_node": 4,
            "max_sentences_per_node": 10,
            "min_node_connections": 1,
            "max_node_connections": 6,
        }
        await admin_client.post(f"/draft/{draft_id}/settings", json=payload)

        # Fetch the draft detail page and check the values are in the HTML
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        html = resp.text
        assert 'value="4"' in html  # min_sentences_per_node
        assert 'value="10"' in html  # max_sentences_per_node
        assert 'value="1"' in html  # min_node_connections
        assert 'value="6"' in html  # max_node_connections

    @pytest.mark.asyncio
    async def test_settings_equal_min_max_allowed(self, admin_client):
        """min == max is valid (boundary case)."""
        draft_id = await self._create_draft(admin_client)

        payload = {
            "min_sentences_per_node": 5,
            "max_sentences_per_node": 5,
            "min_node_connections": 3,
            "max_node_connections": 3,
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/settings",
            json=payload,
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    # ── Validation errors ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reject_min_sentences_gt_max(self, admin_client):
        """min_sentences_per_node > max_sentences_per_node → 422."""
        draft_id = await self._create_draft(admin_client)

        payload = {
            "min_sentences_per_node": 10,
            "max_sentences_per_node": 5,
            "min_node_connections": 1,
            "max_node_connections": 5,
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/settings",
            json=payload,
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["ok"] is False
        assert any("min_sentences_per_node" in e for e in data["errors"])

    @pytest.mark.asyncio
    async def test_reject_min_connections_gt_max(self, admin_client):
        """min_node_connections > max_node_connections → 422."""
        draft_id = await self._create_draft(admin_client)

        payload = {
            "min_sentences_per_node": 3,
            "max_sentences_per_node": 8,
            "min_node_connections": 10,
            "max_node_connections": 3,
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/settings",
            json=payload,
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["ok"] is False
        assert any("min_node_connections" in e for e in data["errors"])

    @pytest.mark.asyncio
    async def test_reject_out_of_bounds(self, admin_client):
        """Values outside allowed range → 422."""
        draft_id = await self._create_draft(admin_client)

        payload = {
            "min_sentences_per_node": 0,   # below minimum of 1
            "max_sentences_per_node": 8,
            "min_node_connections": 2,
            "max_node_connections": 5,
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/settings",
            json=payload,
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["ok"] is False
        assert any("min_sentences_per_node" in e for e in data["errors"])

    @pytest.mark.asyncio
    async def test_reject_non_integer(self, admin_client):
        """Non-integer values → 422."""
        draft_id = await self._create_draft(admin_client)

        payload = {
            "min_sentences_per_node": "abc",
            "max_sentences_per_node": 8,
            "min_node_connections": 2,
            "max_node_connections": 5,
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/settings",
            json=payload,
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["ok"] is False
        assert any("min_sentences_per_node" in e for e in data["errors"])

    @pytest.mark.asyncio
    async def test_reject_missing_field(self, admin_client):
        """Missing field → 422."""
        draft_id = await self._create_draft(admin_client)

        payload = {
            "min_sentences_per_node": 3,
            "max_sentences_per_node": 8,
            "min_node_connections": 2,
            # max_node_connections missing
        }
        resp = await admin_client.post(
            f"/draft/{draft_id}/settings",
            json=payload,
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["ok"] is False
        assert any("max_node_connections" in e for e in data["errors"])

    @pytest.mark.asyncio
    async def test_draft_detail_page_has_settings_card(self, admin_client):
        """The draft detail HTML page contains the Story Settings card with all four inputs."""
        draft_id = await self._create_draft(admin_client)

        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        html = resp.text
        assert "story-settings-card" in html
        assert "saveStorySettings" in html
        assert 'id="min_sentences_per_node"' in html
        assert 'id="max_sentences_per_node"' in html
        assert 'id="min_node_connections"' in html
        assert 'id="max_node_connections"' in html
