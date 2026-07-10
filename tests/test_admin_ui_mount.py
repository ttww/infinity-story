"""Tests for Admin UI mounting in main app (Spec §8).

Verifies that the admin_app sub-application is properly mounted
under the /admin prefix in the main FastAPI app, so all admin UI
routes are accessible at /admin/, /admin/new, /admin/draft/{id}, etc.
"""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def admin_mounted_client():
    """Client that hits the main app (with admin mounted at /admin)."""
    from app.main import app
    from app.persistence.database import init_db, close_db

    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await close_db()


async def _create_draft_via_db() -> str:
    """Create a draft directly in the DB and return its ID."""
    from app.services.story_authoring_agent import get_authoring_agent
    from app.persistence.authoring_repositories import (
        StoryDraftRepository,
        StoryDraftVersionRepository,
    )
    from app.persistence.database import get_session_factory
    from app.models.enums import DraftStatus

    agent = get_authoring_agent(dummy=True)
    brief = {
        "title": "Mount Test",
        "genre": "science_fiction",
        "tone": "dark_mystery",
        "language": "de",
        "target_age": "16+",
    }
    outline = await agent.generate_outline(brief)
    graph = await agent.generate_graph(outline)

    async with get_session_factory()() as session:
        dr = StoryDraftRepository(session)
        vr = StoryDraftVersionRepository(session)
        draft = await dr.create(
            title="Mount Test",
            genre="science_fiction",
            tone="dark_mystery",
            language="de",
            target_age="16+",
            brief=brief,
        )
        await vr.create(
            draft_id=draft.id,
            graph=graph,
            outline=outline,
            created_by="dummy_agent",
            notes="Mount test",
        )
        await dr.update_status(draft.id, DraftStatus.GENERATING)
        await dr.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
        await session.commit()
        return draft.id


class TestAdminMounting:
    """Test that admin UI is accessible under /admin prefix via main app."""

    @pytest.mark.asyncio
    async def test_admin_draft_list_via_main_app(self, admin_mounted_client):
        """GET /admin/ should return the draft list HTML."""
        resp = await admin_mounted_client.get("/admin/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Infinity Story" in resp.text

    @pytest.mark.asyncio
    async def test_admin_brief_form_via_main_app(self, admin_mounted_client):
        """GET /admin/new should return the brief form HTML."""
        resp = await admin_mounted_client.get("/admin/new")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Neue Story" in resp.text

    @pytest.mark.asyncio
    async def test_admin_draft_detail_via_main_app(self, admin_mounted_client):
        """GET /admin/draft/{id} should return the draft detail with graph."""
        draft_id = await _create_draft_via_db()
        resp = await admin_mounted_client.get(f"/admin/draft/{draft_id}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "graph-svg" in resp.text
        assert "Mount Test" in resp.text

    @pytest.mark.asyncio
    async def test_admin_node_detail_api_via_main_app(self, admin_mounted_client):
        """GET /admin/draft/{id}/node/{node_id} should return JSON."""
        draft_id = await _create_draft_via_db()
        resp = await admin_mounted_client.get(
            f"/admin/draft/{draft_id}/node/node_001"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "node_001"

    @pytest.mark.asyncio
    async def test_admin_simulate_page_via_main_app(self, admin_mounted_client):
        """GET /admin/draft/{id}/simulate should return the simulation page."""
        draft_id = await _create_draft_via_db()
        resp = await admin_mounted_client.get(f"/admin/draft/{draft_id}/simulate")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_admin_links_have_prefix(self, admin_mounted_client):
        """Draft list page should contain links with /admin/ prefix."""
        resp = await admin_mounted_client.get("/admin/")
        assert "/admin/new" in resp.text

    @pytest.mark.asyncio
    async def test_admin_create_draft_form_redirect(self, admin_mounted_client):
        """POST /admin/new should create a draft and redirect to /admin/draft/{id}."""
        resp = await admin_mounted_client.post(
            "/admin/new",
            data={
                "title": "Form Submit Test",
                "genre": "science_fiction",
                "tone": "dark_mystery",
                "language": "de",
                "target_age": "16+",
                "node_count": "5",
                "ending_count": "1",
                "branching_level": "low",
                "themes": "",
                "forbidden_content": "",
                "notes": "",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        location = resp.headers.get("location", "")
        assert location.startswith("/admin/draft/")

    @pytest.mark.asyncio
    async def test_admin_create_and_follow_redirect(self, admin_mounted_client):
        """POST /admin/new then follow redirect to /admin/draft/{id}."""
        resp = await admin_mounted_client.post(
            "/admin/new",
            data={
                "title": "Redirect Follow Test",
                "genre": "mystery",
                "tone": "suspenseful",
                "language": "de",
                "target_age": "16+",
                "node_count": "5",
                "ending_count": "1",
                "branching_level": "low",
                "themes": "",
                "forbidden_content": "",
                "notes": "",
            },
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert "Redirect Follow Test" in resp.text
        assert "graph-svg" in resp.text

    @pytest.mark.asyncio
    async def test_admin_docs_via_main_app(self, admin_mounted_client):
        """GET /admin/docs should return the Swagger UI."""
        resp = await admin_mounted_client.get("/admin/docs")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_main_app_root_still_works(self, admin_mounted_client):
        """Main app root / should still work after mounting admin (now returns dashboard HTML)."""
        resp = await admin_mounted_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Infinity Story" in resp.text

    @pytest.mark.asyncio
    async def test_main_app_health_still_works(self, admin_mounted_client):
        """Main app /health should still work after mounting admin."""
        resp = await admin_mounted_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
