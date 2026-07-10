"""Integration tests for the story authoring workflow.

Spec Kapitel 19 — Akzeptanzkriterien Authoring.
Tests: Brief → Generate → Review → Repair → Validate → Approve → Publish.

Tests exercise both the direct service APIs and the HTTP admin API.
"""

import pytest
import json
import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("MIN_NODE_COUNT", "3")
os.environ.setdefault("MIN_ENDING_COUNT", "1")

from httpx import AsyncClient, ASGITransport


# ── HTTP API integration: full authoring pipeline ──────────────────

class TestAuthoringAPI:
    """End-to-end authoring through the HTTP admin API."""

    @pytest.fixture
    async def client(self):
        from app.core.config import get_settings
        get_settings.cache_clear()
        from app.persistence.database import init_db, close_db
        await init_db()
        from app.main import app as application
        transport = ASGITransport(app=application)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        await close_db()

    @pytest.mark.asyncio
    async def test_create_draft(self, client):
        """POST /api/admin/story-drafts creates a draft with a Helios graph."""
        resp = await client.post(
            "/api/admin/story-drafts",
            json={
                "title": "Test Story",
                "genre": "science_fiction",
                "tone": "dark_mystery",
                "language": "de",
                "target_age": "16+",
                "node_count": 10,
                "ending_count": 2,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "draft_id" in data
        assert data["status"] == "needs_review"

    @pytest.mark.asyncio
    async def test_full_pipeline(self, client):
        """Complete workflow: create → review → repair → validate → approve → publish."""
        # 1. Create
        resp = await client.post(
            "/api/admin/story-drafts",
            json={"title": "Pipeline Test", "genre": "mystery", "tone": "dark"},
        )
        assert resp.status_code == 201
        draft_id = resp.json()["draft_id"]

        # 2. Review
        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/review")
        assert resp.status_code == 200
        review = resp.json()
        assert review["score"] > 0
        assert isinstance(review["issues"], list)
        assert isinstance(review["summary"], str)

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

    @pytest.mark.asyncio
    async def test_list_drafts(self, client):
        """GET /api/admin/story-drafts lists created drafts."""
        await client.post(
            "/api/admin/story-drafts",
            json={"title": "List Test", "genre": "mystery", "tone": "dark"},
        )
        resp = await client.get("/api/admin/story-drafts")
        assert resp.status_code == 200
        drafts = resp.json()
        assert len(drafts) >= 1
        assert any(d["title"] == "List Test" for d in drafts)

    @pytest.mark.asyncio
    async def test_get_draft_detail(self, client):
        """GET /api/admin/story-drafts/{id} returns full draft detail."""
        resp = await client.post(
            "/api/admin/story-drafts",
            json={"title": "Detail Test", "genre": "mystery", "tone": "dark"},
        )
        draft_id = resp.json()["draft_id"]

        resp = await client.get(f"/api/admin/story-drafts/{draft_id}")
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["id"] == draft_id
        assert detail["title"] == "Detail Test"
        assert "versions" in detail

    @pytest.mark.asyncio
    async def test_get_draft_graph(self, client):
        """GET /api/admin/story-drafts/{id}/graph returns the story graph."""
        resp = await client.post(
            "/api/admin/story-drafts",
            json={"title": "Graph Test", "genre": "mystery", "tone": "dark"},
        )
        draft_id = resp.json()["draft_id"]

        resp = await client.get(f"/api/admin/story-drafts/{draft_id}/graph")
        assert resp.status_code == 200
        body = resp.json()
        graph = body.get("graph", body)
        assert "nodes" in graph
        assert len(graph["nodes"]) >= 5
        assert graph.get("start_node_id") is not None

    @pytest.mark.asyncio
    async def test_publish_without_approval_rejected(self, client):
        """Publish should fail if draft is not approved."""
        resp = await client.post(
            "/api/admin/story-drafts",
            json={"title": "No Approve", "genre": "mystery", "tone": "dark"},
        )
        draft_id = resp.json()["draft_id"]

        resp = await client.post(f"/api/admin/story-drafts/{draft_id}/publish")
        assert resp.status_code in (400, 409, 422)


# ── Direct service tests ────────────────────────────────────────────

class TestAuthoringServices:
    """Test the authoring services directly (no HTTP)."""

    @pytest.fixture
    async def session(self):
        from app.core.config import get_settings
        get_settings.cache_clear()
        from app.persistence.database import init_db, close_db, get_session_factory
        await init_db()
        factory = get_session_factory()
        async with factory() as s:
            yield s
        await close_db()

    @pytest.mark.asyncio
    async def test_dummy_authoring_agent(self):
        """DummyStoryAuthoringAgent produces a valid Helios graph."""
        from app.services.story_authoring_agent import DummyStoryAuthoringAgent

        agent = DummyStoryAuthoringAgent()
        brief = {"title": "Test", "genre": "science_fiction"}
        outline, graph = await agent.generate_all(brief)

        assert outline["premise"]
        assert outline["main_conflict"]
        assert "nodes" in graph
        assert graph["start_node_id"] == "node_001"
        assert len(graph["nodes"]) >= 5

    @pytest.mark.asyncio
    async def test_critic_agent(self):
        """StoryCriticAgent.review() returns a structured report."""
        from app.services.story_critic_agent import StoryCriticAgent
        from app.services.llm_service import MockLLMService
        from app.services.story_authoring_agent import DummyStoryAuthoringAgent

        agent = DummyStoryAuthoringAgent()
        outline, graph = await agent.generate_all({})

        critic = StoryCriticAgent(MockLLMService())
        report = await critic.review(outline, graph)

        assert report["score"] > 0
        assert isinstance(report["issues"], list)
        assert isinstance(report["summary"], str)
        assert isinstance(report.get("repair_suggestions"), list)

    @pytest.mark.asyncio
    async def test_repair_agent(self):
        """StoryRepairAgent.repair() returns an improved graph."""
        from app.services.story_repair_agent import StoryRepairAgent
        from app.services.llm_service import MockLLMService
        from app.services.story_authoring_agent import DummyStoryAuthoringAgent

        dummy = DummyStoryAuthoringAgent()
        _, graph = await dummy.generate_all({})
        review_report = {
            "score": 7.5,
            "issues": [],
            "repair_suggestions": [],
            "summary": "Test",
        }

        repair = StoryRepairAgent(MockLLMService())
        result = await repair.repair(graph, review_report)

        assert "graph" in result
        assert "nodes" in result["graph"]
        assert isinstance(result.get("changes"), list)

    @pytest.mark.asyncio
    async def test_validation_service_valid_graph(self):
        """StoryValidationService validates a correct graph."""
        from app.services.story_validation_service import StoryValidationService
        from app.services.story_authoring_agent import DummyStoryAuthoringAgent

        dummy = DummyStoryAuthoringAgent()
        _, graph = await dummy.generate_all({})

        service = StoryValidationService()
        report = await service.validate(graph)

        assert isinstance(report["is_valid"], bool)
        assert isinstance(report["errors"], list)
        assert isinstance(report["warnings"], list)
        assert isinstance(report.get("checks"), dict)

    @pytest.mark.asyncio
    async def test_validation_service_broken_graph(self):
        """StoryValidationService catches broken references."""
        from app.services.story_validation_service import StoryValidationService

        broken_graph = {
            "nodes": {
                "n1": {
                    "id": "n1", "title": "Start", "type": "start",
                    "scene_goal": "begin",
                    "choices": [
                        {"id": "c1", "label": "Go", "next_node_id": "missing"},
                    ],
                    "quality_notes": ["start"],
                },
            },
            "start_node_id": "n1",
        }

        service = StoryValidationService()
        report = await service.validate(broken_graph)

        assert report["is_valid"] is False
        assert len(report["errors"]) > 0


# ── Graph serialization tests ───────────────────────────────────────

class TestGraphSerialization:
    """Test graph load/dump round-trips."""

    def test_graph_roundtrip(self):
        from app.models import StoryGraph, StoryNode, Choice
        from app.story.graph import graph_to_dict, load_graph_from_dict

        graph = StoryGraph(
            title="Test",
            genre="test",
            start_node_id="n1",
            nodes={
                "n1": StoryNode(
                    id="n1", title="Start", type="start", scene_goal="begin",
                    choices=[Choice(id="c1", label="Go", next_node_id="n2")],
                ),
                "n2": StoryNode(
                    id="n2", title="End", type="end", scene_goal="end",
                ),
            },
        )

        graph_dict = graph_to_dict(graph)
        assert graph_dict["title"] == "Test"

        restored = load_graph_from_dict(graph_dict)
        assert restored.title == "Test"
        assert restored.start_node_id == "n1"
        assert "n1" in restored.nodes
        assert restored.nodes["n1"].type == "start"

    def test_helios_scenario_loads(self):
        """The helios.json scenario loads and has expected structure."""
        from app.story.scenario_loader import load_scenario, get_start_node

        scenario = load_scenario("helios")
        assert scenario is not None
        assert scenario["start_node_id"] == "node_001"
        assert len(scenario["nodes"]) >= 5
        assert scenario["nodes"]["node_001"]["type"] == "start"

        # Check endings exist
        endings = [n for n in scenario["nodes"].values() if n.get("type") == "end" or n.get("is_end")]
        assert len(endings) >= 1

    def test_build_initial_world_state(self):
        """build_initial_world_state returns a usable world state."""
        from app.story.scenario_loader import load_scenario, build_initial_world_state

        scenario = load_scenario("helios")
        ws = build_initial_world_state(scenario)
        assert isinstance(ws, dict)
        assert "genre" in ws or "current_location" in ws or len(ws) > 0
