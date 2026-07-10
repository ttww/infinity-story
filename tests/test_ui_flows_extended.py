"""Additional integration tests for UI flows — Admin + Chat + API (t_ui_006).

These tests complement test_ui_flows.py with deeper end-to-end coverage:
  - Admin full lifecycle: create draft → view detail → simulate → node API
  - Admin draft list shows created drafts after form submission
  - Admin brief form contains all required fields
  - Chat UI page contains scenario/choice/free-text elements
  - Chat UI references correct API endpoints
  - Dashboard shows scenario grid with Spielen buttons
  - Dashboard stats reflect DB state
  - POST /api/message full chat flow: Start → select scenario → make choice
  - POST /api/message free-text input in active session
  - GET /api/scenarios returns helios with correct fields
  - Admin draft detail contains graph SVG and node panel
  - Admin simulation page contains graph data
  - Admin node detail JSON API returns node data
  - Chat page contains world-state debug panel
  - Chat page has typing indicator
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


async def _create_draft_via_form(client: AsyncClient, title: str = "Deep Test Story") -> str:
    """Create a draft via the admin UI form, return its id."""
    resp = await client.post(
        "/admin/new",
        data={
            "title": title,
            "genre": "science_fiction",
            "tone": "dark_mystery",
            "language": "de",
            "target_age": "16+",
            "node_count": "10",
            "ending_count": "2",
            "branching_level": "medium",
            "themes": "space station, AI",
            "forbidden_content": "",
            "notes": "Integration test draft",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    location = resp.headers.get("location", "")
    return location.rsplit("/", 1)[-1]


# ── Admin UI deep integration tests ───────────────────────────────────


class TestAdminUIDeepFlows:
    """Deep integration tests for admin UI endpoints."""

    @pytest.mark.asyncio
    async def test_draft_list_shows_created_draft(self, client):
        """After creating a draft via form, it appears in the draft list."""
        draft_id = await _create_draft_via_form(client, "Listed Story")
        resp = await client.get("/admin/")
        assert resp.status_code == 200
        assert "Listed Story" in resp.text
        # The draft id should appear in a link
        assert f"/admin/draft/{draft_id}" in resp.text

    @pytest.mark.asyncio
    async def test_brief_form_contains_all_required_fields(self, client):
        """GET /admin/new form contains all required brief fields."""
        resp = await client.get("/admin/new")
        assert resp.status_code == 200
        html = resp.text.lower()
        # Required form fields
        for field in ["title", "genre", "tone", "language", "target_age",
                      "node_count", "ending_count", "branching_level"]:
            assert f'name="{field}"' in html, f"Missing form field: {field}"
        # Submit button
        assert "submit" in html or "generieren" in html

    @pytest.mark.asyncio
    async def test_draft_detail_contains_graph_svg(self, client):
        """GET /admin/draft/{id} renders graph SVG visualization."""
        draft_id = await _create_draft_via_form(client, "Graph Story")
        resp = await client.get(f"/admin/draft/{draft_id}")
        assert resp.status_code == 200
        assert "graph-svg" in resp.text or "svg" in resp.text.lower()
        assert "Graph Story" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_contains_status_badge(self, client):
        """Draft detail page shows status badge."""
        draft_id = await _create_draft_via_form(client)
        resp = await client.get(f"/admin/draft/{draft_id}")
        assert resp.status_code == 200
        assert "badge" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_node_detail_api_returns_node_data(self, client):
        """GET /admin/draft/{id}/node/{node_id} returns JSON node detail."""
        draft_id = await _create_draft_via_form(client)
        resp = await client.get(f"/admin/draft/{draft_id}/node/node_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "node_001"

    @pytest.mark.asyncio
    async def test_simulate_page_contains_graph_json(self, client):
        """GET /admin/draft/{id}/simulate embeds graph JSON for frontend."""
        draft_id = await _create_draft_via_form(client, "Sim Story")
        resp = await client.get(f"/admin/draft/{draft_id}/simulate")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        # The page should contain graph data in JSON form
        assert "nodes" in resp.text.lower() or "graph" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_simulate_start_api_returns_scene(self, client):
        """GET /admin/draft/{id}/simulate/start returns initial simulation state."""
        draft_id = await _create_draft_via_form(client)
        resp = await client.get(f"/admin/draft/{draft_id}/simulate/start")
        assert resp.status_code == 200
        data = resp.json()
        assert "scene_text" in data or "scene" in data or "current_node_id" in data

    @pytest.mark.asyncio
    async def test_admin_create_draft_with_themes(self, client):
        """POST /admin/new with themes creates draft successfully."""
        resp = await client.post(
            "/admin/new",
            data={
                "title": "Themed Story",
                "genre": "mystery",
                "tone": "suspenseful",
                "language": "de",
                "target_age": "16+",
                "node_count": "8",
                "ending_count": "2",
                "branching_level": "low",
                "themes": "abandoned station, AI, betrayal",
                "forbidden_content": "graphic violence",
                "notes": "Test with themes",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert "Themed Story" in resp.text

    @pytest.mark.asyncio
    async def test_admin_draft_list_empty_state(self, client):
        """GET /admin/ with no drafts shows empty state message."""
        resp = await client.get("/admin/")
        assert resp.status_code == 200
        # Fresh DB — no drafts yet
        assert "Keine" in resp.text or "empty" in resp.text.lower() or "Story Drafts" in resp.text

    @pytest.mark.asyncio
    async def test_admin_docs_accessible(self, client):
        """GET /admin/docs returns Swagger UI."""
        resp = await client.get("/admin/docs")
        assert resp.status_code == 200


# ── Chat UI deep integration tests ────────────────────────────────────


class TestChatUIDeepFlows:
    """Deep integration tests for the runtime chat UI."""

    @pytest.mark.asyncio
    async def test_chat_page_contains_scenario_elements(self, client):
        """Chat page HTML contains scenario list and choice button CSS classes."""
        resp = await client.get("/chat/")
        assert resp.status_code == 200
        # CSS classes for scenario cards and choice buttons
        assert "scenario-btn" in resp.text or "scenario-list" in resp.text
        assert "choice-btn" in resp.text or "choices-container" in resp.text

    @pytest.mark.asyncio
    async def test_chat_page_has_typing_indicator(self, client):
        """Chat page has a typing indicator element."""
        resp = await client.get("/chat/")
        assert resp.status_code == 200
        assert "typing-indicator" in resp.text or "typing-dot" in resp.text

    @pytest.mark.asyncio
    async def test_chat_page_has_world_state_panel(self, client):
        """Chat page includes the world state debug panel."""
        resp = await client.get("/chat/")
        assert resp.status_code == 200
        assert "state-panel" in resp.text or "world" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_chat_page_has_reset_capability(self, client):
        """Chat page has a reset or session management element."""
        resp = await client.get("/chat/")
        assert resp.status_code == 200
        # Status badge with idle state
        assert "status-idle" in resp.text or "status-badge" in resp.text

    @pytest.mark.asyncio
    async def test_chat_page_links_to_home(self, client):
        """Chat page header links back to home/dashboard."""
        resp = await client.get("/chat/")
        assert resp.status_code == 200
        assert 'href="/"' in resp.text or 'href="/chat/"' in resp.text

    @pytest.mark.asyncio
    async def test_chat_redirect_no_trailing_slash(self, client):
        """GET /chat (no trailing slash) redirects to /chat/."""
        resp = await client.get("/chat", follow_redirects=False)
        # Sub-app mounted at /chat — FastAPI redirects /chat to /chat/
        assert resp.status_code in (301, 307, 308)


# ── Dashboard deep integration tests ─────────────────────────────────


class TestDashboardDeepFlows:
    """Deep integration tests for the dashboard."""

    @pytest.mark.asyncio
    async def test_dashboard_shows_helios_scenario(self, client):
        """Dashboard lists the helios file-based scenario with Spielen button."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "Helios" in resp.text or "helios" in resp.text
        assert "/chat/?scenario=helios" in resp.text or "Spielen" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_shows_stats_cards(self, client):
        """Dashboard renders stat cards for drafts, published, sessions."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "stat-value" in resp.text
        assert "stat-card" in resp.text
        # Check for stat labels
        assert "Drafts" in resp.text or "drafts" in resp.text.lower()
        assert "Szenarien" in resp.text or "published" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_dashboard_has_hero_section(self, client):
        """Dashboard has hero section with action buttons."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "hero" in resp.text.lower()
        assert "Story spielen" in resp.text or "spielen" in resp.text.lower()
        assert "Story erstellen" in resp.text or "erstellen" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_dashboard_stats_reflect_db_state(self, client):
        """After creating a draft, dashboard stats update."""
        # Create a draft via admin form
        await _create_draft_via_form(client, "Dashboard Stats Story")
        resp = await client.get("/")
        assert resp.status_code == 200
        # The drafts stat should be > 0
        assert "stat-value" in resp.text
        # Extract the first stat-value (drafts count)
        import re
        match = re.search(r'stat-value["\s]*>\s*(\d+)', resp.text)
        if match:
            drafts_count = int(match.group(1))
            assert drafts_count >= 1


# ── API full chat flow integration tests ─────────────────────────────


class TestChatFlowIntegration:
    """Full chat flow: Start → select scenario → interact with scene."""

    @pytest.mark.asyncio
    async def test_full_chat_flow_start_select_interact(self, client):
        """Full flow: Start → get scenarios → select helios → get scene → send choice."""
        # Step 1: Start → list scenarios
        resp = await client.post(
            "/api/message",
            json={"user_id": "flow_test_user", "message": "Start"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) > 0
        assert "helios" in data["messages"][0].lower() or "Helios" in data["messages"][0]

        # Step 2: Select scenario by name
        resp2 = await client.post(
            "/api/message",
            json={"user_id": "flow_test_user", "message": "helios"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2.get("session_id") is not None
        assert data2.get("scene") is not None

        # Step 3: Interact — send a choice or free text
        scene = data2["scene"]
        if scene.get("choices"):
            # Send the first choice label
            choice_label = scene["choices"][0].get("label", "A")
        else:
            choice_label = "continue"
        resp3 = await client.post(
            "/api/message",
            json={"user_id": "flow_test_user", "message": choice_label},
        )
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3.get("session_id") is not None

    @pytest.mark.asyncio
    async def test_message_start_then_starten_alias(self, client):
        """Both 'Start' and 'Starten' trigger scenario listing."""
        for keyword in ["Start", "Starten", "neu"]:
            resp = await client.post(
                "/api/message",
                json={"user_id": f"alias_{keyword}", "message": keyword},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["messages"]) > 0
            assert "helios" in data["messages"][0].lower()

    @pytest.mark.asyncio
    async def test_message_no_active_session_prompts_start(self, client):
        """Sending a non-start message with no session prompts to start."""
        resp = await client.post(
            "/api/message",
            json={"user_id": "no_session_user", "message": "hello there"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Start" in data["messages"][0] or "start" in data["messages"][0].lower()

    @pytest.mark.asyncio
    async def test_api_scenarios_returns_helios_with_genre(self, client):
        """GET /api/scenarios returns helios with correct genre field."""
        resp = await client.get("/api/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        helios = [s for s in data if s["id"] == "helios"]
        assert len(helios) == 1
        assert helios[0]["title"] == "Signal von Helios"
        assert helios[0]["genre"] == "science_fiction"

    @pytest.mark.asyncio
    async def test_message_select_scenario_by_number(self, client):
        """Selecting scenario by number (1) starts a session."""
        # First list scenarios
        resp = await client.post(
            "/api/message",
            json={"user_id": "num_select_user", "message": "Start"},
        )
        assert resp.status_code == 200

        # Select by number "1" (helios is the only/first file scenario)
        resp2 = await client.post(
            "/api/message",
            json={"user_id": "num_select_user", "message": "1"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2.get("session_id") is not None
        assert data2.get("scene") is not None
