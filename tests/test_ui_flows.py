"""Integration tests for UI flows — Admin + Chat + Dashboard (t_ui_006).

Tests cover:
  (1)  GET /admin/ returns draft list (200)
  (2)  GET /admin/new shows brief form (200)
  (3)  POST /admin/new creates draft and redirects (303)
  (4)  GET /chat/ loads chat page (200)
  (5)  GET / (dashboard) renders with stats + scenarios (200)
  (6)  GET /health returns ok (200)
  (7)  POST /api/message "Start" lists merged scenarios
  (8)  GET /api/scenarios returns merged list (file + DB)
  (9)  GET /admin/draft/{id} renders detail with graph (200)
  (10) GET /admin/draft/{id}/simulate renders simulation (200)
  (11) POST /api/message scenario selection starts session
  (12) GET /chat/ contains key chat UI elements (input, send button)
  (13) GET / (dashboard) contains links to /chat/ and /admin/
  (14) GET /admin/ links have /admin/ prefix
  (15) GET /admin/new form action has /admin/ prefix
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("MIN_NODE_COUNT", "3")
os.environ.setdefault("MIN_ENDING_COUNT", "1")

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
get_settings.cache_clear()


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
async def app():
    from app.main import app as application
    from app.persistence.database import init_db, close_db
    await init_db()
    yield application
    await close_db()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def admin_client():
    """Client targeting the admin sub-app directly."""
    from app.admin_ui.app import admin_app
    from app.persistence.database import init_db, close_db
    await init_db()
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await close_db()


async def _create_draft(client: AsyncClient) -> str:
    """Create a draft via the admin UI form, return its id."""
    resp = await client.post(
        "/admin/new",
        data={
            "title": "UI Test Story",
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
    # Location is like /admin/draft/draft_xxx
    return location.rsplit("/", 1)[-1]


# ── Admin UI Tests ────────────────────────────────────────────────────


class TestAdminUIFlows:
    """Tests for the admin UI endpoints (Spec §8)."""

    @pytest.mark.asyncio
    async def test_admin_draft_list_renders(self, client):
        """(1) GET /admin/ returns draft list with 200."""
        resp = await client.get("/admin/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Story Drafts" in resp.text or "Infinity Story" in resp.text

    @pytest.mark.asyncio
    async def test_admin_brief_form_renders(self, client):
        """(2) GET /admin/new shows brief form with 200."""
        resp = await client.get("/admin/new")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "<form" in resp.text.lower()
        assert "title" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_admin_create_draft_redirects(self, client):
        """(3) POST /admin/new creates draft and redirects (303)."""
        resp = await client.post(
            "/admin/new",
            data={
                "title": "Test Draft",
                "genre": "mystery",
                "tone": "suspenseful",
                "language": "de",
                "target_age": "16+",
                "node_count": "8",
                "ending_count": "2",
                "branching_level": "low",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        location = resp.headers.get("location", "")
        assert "/admin/draft/" in location

    @pytest.mark.asyncio
    async def test_admin_draft_detail_renders(self, client):
        """(9) GET /admin/draft/{id} renders detail with graph."""
        draft_id = await _create_draft(client)
        resp = await client.get(f"/admin/draft/{draft_id}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "graph" in resp.text.lower() or "Graph" in resp.text

    @pytest.mark.asyncio
    async def test_admin_simulate_renders(self, client):
        """(10) GET /admin/draft/{id}/simulate renders simulation page."""
        draft_id = await _create_draft(client)
        resp = await client.get(f"/admin/draft/{draft_id}/simulate")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_admin_links_have_prefix(self, client):
        """(14) GET /admin/ links have /admin/ prefix."""
        resp = await client.get("/admin/")
        assert resp.status_code == 200
        # Should NOT have root-absolute paths like href="/new"
        assert 'href="/new"' not in resp.text
        assert 'href="/admin/new"' in resp.text

    @pytest.mark.asyncio
    async def test_admin_form_action_has_prefix(self, client):
        """(15) GET /admin/new form action has /admin/ prefix."""
        resp = await client.get("/admin/new")
        assert resp.status_code == 200
        assert 'action="/admin/new"' in resp.text
        assert 'action="/new"' not in resp.text


# ── Chat UI Tests ─────────────────────────────────────────────────────


class TestChatUIFlows:
    """Tests for the runtime chat UI (Spec §13.2)."""

    @pytest.mark.asyncio
    async def test_chat_page_renders(self, client):
        """(4) GET /chat/ loads chat page with 200."""
        resp = await client.get("/chat/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_chat_page_contains_input(self, client):
        """(12) GET /chat/ contains input field and send button."""
        resp = await client.get("/chat/")
        assert resp.status_code == 200
        assert "input" in resp.text.lower()
        assert "send" in resp.text.lower() or "Senden" in resp.text

    @pytest.mark.asyncio
    async def test_chat_page_has_api_endpoint(self, client):
        """Chat page references /api/message for fetch()."""
        resp = await client.get("/chat/")
        assert resp.status_code == 200
        assert "/api/message" in resp.text

    @pytest.mark.asyncio
    async def test_chat_page_has_status_badge(self, client):
        """Chat page has a status indicator."""
        resp = await client.get("/chat/")
        assert resp.status_code == 200
        assert "status-badge" in resp.text or "status" in resp.text.lower()


# ── Dashboard Tests ───────────────────────────────────────────────────


class TestDashboard:
    """Tests for the dashboard / landing page (t_ui_004)."""

    @pytest.mark.asyncio
    async def test_dashboard_renders(self, client):
        """(5) GET / renders dashboard with 200."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_dashboard_contains_links(self, client):
        """(13) Dashboard contains links to /chat/ and /admin/."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert 'href="/chat/"' in resp.text
        assert 'href="/admin/"' in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_contains_stats(self, client):
        """Dashboard shows statistics (drafts, published, sessions)."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "stat-value" in resp.text or "stat" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """(6) GET /health returns ok."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ── API / Scenario Merge Tests ────────────────────────────────────────


class TestScenarioMerge:
    """Tests for the merged file-based + DB-published scenario list."""

    @pytest.mark.asyncio
    async def test_message_start_lists_scenarios(self, client):
        """(7) POST /api/message 'Start' lists merged scenarios."""
        resp = await client.post(
            "/api/message",
            json={"user_id": "test_ui_user", "message": "Start"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) > 0
        # Should contain the file-based helios scenario
        assert "helios" in data["messages"][0].lower() or "Helios" in data["messages"][0]

    @pytest.mark.asyncio
    async def test_api_scenarios_returns_list(self, client):
        """(8) GET /api/scenarios returns merged list."""
        resp = await client.get("/api/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Should contain the file-based helios scenario
        ids = [s["id"] for s in data]
        assert "helios" in ids

    @pytest.mark.asyncio
    async def test_message_scenario_select_starts_session(self, client):
        """(11) POST /api/message with scenario name starts a session."""
        resp = await client.post(
            "/api/message",
            json={"user_id": "test_ui_user_2", "message": "helios"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should have a session_id and scene
        assert data.get("session_id") is not None
        assert data.get("scene") is not None
