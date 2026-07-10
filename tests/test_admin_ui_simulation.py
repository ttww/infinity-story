"""Tests for Admin UI Simulation View (Spec §8.1.5).

Tests:
- SimulationEngine: start, choose, state-diff, path tracking
- Simulation endpoints: GET /simulate, GET /simulate/start,
  POST /simulate/choose, GET /simulate/state
- HTML rendering of simulation page
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.admin_ui.simulation import (
    SimulationEngine,
    StateDiff,
    _build_scene_data,
    _get_choices,
    _is_ending,
    _apply_state_updates,
    _compute_diff,
)


# ── Test fixtures ─────────────────────────────────────────────────────


def _helios_graph() -> dict:
    """Load the helios graph for testing."""
    from app.services.story_authoring_agent import _load_helios_graph
    return _load_helios_graph()


def _graph_with_state_updates() -> dict:
    """A graph with state_updates on choices for diff testing."""
    return {
        "start_node_id": "s1",
        "nodes": {
            "s1": {
                "id": "s1", "title": "Start", "type": "start", "act": 1,
                "scene_goal": "Begin",
                "location": "Town",
                "characters": ["Hero"],
                "reveals": ["The king is dead"],
                "mood": "tense",
                "choices": [
                    {
                        "id": "c1", "label": "Go north", "next_node_id": "n2",
                        "state_updates": {"flags.north": True, "inventory": ["map"]},
                    },
                    {
                        "id": "c2", "label": "Go south", "next_node_id": "n3",
                        "state_updates": {"flags.south": True},
                    },
                ],
                "quality_notes": ["Establish setting"],
            },
            "n2": {
                "id": "n2", "title": "Forest", "type": "scene", "act": 1,
                "scene_goal": "Explore the forest",
                "location": "Dark Forest",
                "characters": ["Hero", "Elf"],
                "reveals": ["A hidden path exists."],
                "choices": [
                    {
                        "id": "c3", "label": "Follow path", "next_node_id": "n3",
                        "state_updates": {"flags.path_found": True},
                    },
                ],
                "quality_notes": ["Build atmosphere"],
            },
            "n3": {
                "id": "n3", "title": "Castle", "type": "end", "act": 2,
                "scene_goal": "Final confrontation",
                "location": "Castle",
                "characters": ["Hero", "Villain"],
                "reveals": ["The villain was the elf."],
                "choices": [],
                "quality_notes": ["Satisfying ending"],
                "is_end": True,
            },
        },
    }


# ── SimulationEngine.start tests ─────────────────────────────────────


class TestSimulationStart:
    """Test SimulationEngine.start (Spec §8.1.5)."""

    def test_start_returns_initial_state(self):
        result = SimulationEngine.start(_helios_graph())
        assert "current_node_id" in result
        assert "current_node" in result
        assert "world_state" in result
        assert "path" in result
        assert "available_choices" in result
        assert "is_ended" in result
        assert "state_diff" in result
        assert "steps" in result

    def test_start_at_start_node(self):
        result = SimulationEngine.start(_helios_graph())
        assert result["current_node_id"] == "node_001"

    def test_start_path_has_one_node(self):
        result = SimulationEngine.start(_helios_graph())
        assert result["path"] == ["node_001"]

    def test_start_scene_has_title(self):
        result = SimulationEngine.start(_helios_graph())
        assert result["current_node"]["title"] == "Notrufsignal"

    def test_start_scene_has_location(self):
        result = SimulationEngine.start(_helios_graph())
        assert "Orbitalstation Helios" in result["current_node"]["location"]

    def test_start_scene_has_characters(self):
        result = SimulationEngine.start(_helios_graph())
        assert "Mira" in result["current_node"]["characters"]

    def test_start_scene_has_reveals(self):
        result = SimulationEngine.start(_helios_graph())
        assert len(result["current_node"]["reveals"]) == 1

    def test_start_has_choices(self):
        result = SimulationEngine.start(_helios_graph())
        assert len(result["available_choices"]) == 2

    def test_start_not_ended(self):
        result = SimulationEngine.start(_helios_graph())
        assert result["is_ended"] is False

    def test_start_step_count(self):
        result = SimulationEngine.start(_helios_graph())
        assert result["step_count"] == 1

    def test_start_has_steps(self):
        result = SimulationEngine.start(_helios_graph())
        assert len(result["steps"]) == 1
        assert result["steps"][0]["node_id"] == "node_001"

    def test_start_with_empty_graph(self):
        result = SimulationEngine.start({"nodes": {}})
        assert "error" in result

    def test_start_with_no_start_node_id(self):
        graph = {
            "nodes": {
                "n1": {"id": "n1", "title": "N1", "type": "start", "scene_goal": "G", "choices": [], "is_start": True},
            }
        }
        result = SimulationEngine.start(graph)
        assert result["current_node_id"] == "n1"

    def test_start_applies_node_state_updates(self):
        graph = {
            "start_node_id": "s1",
            "nodes": {
                "s1": {
                    "id": "s1", "title": "Start", "type": "start",
                    "scene_goal": "Begin",
                    "choices": [{"id": "c1", "label": "Go", "next_node_id": "s2"}],
                    "state_updates": {"flags.initialized": True},
                },
                "s2": {
                    "id": "s2", "title": "End", "type": "end",
                    "scene_goal": "Done", "choices": [], "is_end": True,
                },
            },
        }
        result = SimulationEngine.start(graph)
        assert result["world_state"].get("flags", {}).get("initialized") is True


# ── SimulationEngine.choose tests ─────────────────────────────────────


class TestSimulationChoose:
    """Test SimulationEngine.choose (Spec §8.1.5)."""

    def test_choose_advances_to_next_node(self):
        graph = _helios_graph()
        start = SimulationEngine.start(graph)
        result = SimulationEngine.choose(
            graph, start["current_node_id"], "answer_signal",
            start["world_state"], start["path"], start["step_count"],
        )
        assert result["current_node_id"] == "node_002"
        assert result["current_node"]["title"] == "Die Stimme im Rauschen"

    def test_choose_extends_path(self):
        graph = _helios_graph()
        start = SimulationEngine.start(graph)
        result = SimulationEngine.choose(
            graph, start["current_node_id"], "answer_signal",
            start["world_state"], start["path"], start["step_count"],
        )
        assert result["path"] == ["node_001", "node_002"]

    def test_choose_increments_step_count(self):
        graph = _helios_graph()
        start = SimulationEngine.start(graph)
        result = SimulationEngine.choose(
            graph, start["current_node_id"], "answer_signal",
            start["world_state"], start["path"], start["step_count"],
        )
        assert result["step_count"] == 2

    def test_choose_returns_choices_for_next_node(self):
        graph = _helios_graph()
        start = SimulationEngine.start(graph)
        result = SimulationEngine.choose(
            graph, start["current_node_id"], "answer_signal",
            start["world_state"], start["path"], start["step_count"],
        )
        assert len(result["available_choices"]) == 2

    def test_choose_invalid_choice_returns_error(self):
        graph = _helios_graph()
        start = SimulationEngine.start(graph)
        result = SimulationEngine.choose(
            graph, start["current_node_id"], "nonexistent_choice",
            start["world_state"], start["path"], start["step_count"],
        )
        assert "error" in result

    def test_choose_invalid_node_returns_error(self):
        graph = _helios_graph()
        result = SimulationEngine.choose(
            graph, "NONEXISTENT", "answer_signal", {}, [], 0,
        )
        assert "error" in result

    def test_choice_with_no_next_node_returns_error(self):
        graph = {
            "start_node_id": "s1",
            "nodes": {
                "s1": {
                    "id": "s1", "title": "Start", "type": "start",
                    "scene_goal": "G",
                    "choices": [{"id": "c1", "label": "Go", "next_node_id": None}],
                },
            },
        }
        result = SimulationEngine.choose(graph, "s1", "c1", {}, ["s1"], 1)
        assert "error" in result

    def test_choice_to_missing_node_returns_error(self):
        graph = {
            "start_node_id": "s1",
            "nodes": {
                "s1": {
                    "id": "s1", "title": "Start", "type": "start",
                    "scene_goal": "G",
                    "choices": [{"id": "c1", "label": "Go", "next_node_id": "MISSING"}],
                },
            },
        }
        result = SimulationEngine.choose(graph, "s1", "c1", {}, ["s1"], 1)
        assert "error" in result

    def test_choose_to_ending_node(self):
        graph = _helios_graph()
        start = SimulationEngine.start(graph)
        # node_001 -> node_002 (via answer_signal)
        r1 = SimulationEngine.choose(
            graph, "node_001", "answer_signal", {}, ["node_001"], 1)
        # node_002 -> node_005 (via keep_secret)
        r2 = SimulationEngine.choose(
            graph, "node_002", "keep_secret",
            r1["world_state"], r1["path"], r1["step_count"])
        assert r2["current_node_id"] == "node_005"
        assert r2["is_ended"] is True
        assert len(r2["available_choices"]) == 0

    def test_choose_records_selected_choice(self):
        graph = _helios_graph()
        start = SimulationEngine.start(graph)
        result = SimulationEngine.choose(
            graph, start["current_node_id"], "answer_signal",
            start["world_state"], start["path"], start["step_count"],
        )
        assert result["selected_choice"]["id"] == "answer_signal"

    def test_choose_applies_state_updates(self):
        graph = _graph_with_state_updates()
        start = SimulationEngine.start(graph)
        result = SimulationEngine.choose(
            graph, start["current_node_id"], "c1",
            start["world_state"], start["path"], start["step_count"],
        )
        assert result["world_state"].get("flags", {}).get("north") is True
        assert "map" in result["world_state"].get("inventory", [])

    def test_choose_computes_state_diff(self):
        graph = _graph_with_state_updates()
        start = SimulationEngine.start(graph)
        result = SimulationEngine.choose(
            graph, start["current_node_id"], "c1",
            start["world_state"], start["path"], start["step_count"],
        )
        diff = result["state_diff"]
        assert "added" in diff
        assert len(diff["added"]) > 0

    def test_choose_dotted_path_state_update(self):
        graph = _graph_with_state_updates()
        start = SimulationEngine.start(graph)
        result = SimulationEngine.choose(
            graph, start["current_node_id"], "c1",
            start["world_state"], start["path"], start["step_count"],
        )
        assert "flags" in result["world_state"]
        assert result["world_state"]["flags"]["north"] is True

    def test_max_steps_exceeded(self):
        graph = _helios_graph()
        result = SimulationEngine.choose(
            graph, "node_001", "answer_signal", {}, ["node_001"], 200)
        assert "error" in result


# ── State diff tests ──────────────────────────────────────────────────


class TestStateDiff:
    """Test state diff computation (Spec §8.1.5)."""

    def test_empty_diff(self):
        diff = _compute_diff({"a": 1}, {"a": 1})
        assert diff.is_empty is True

    def test_added_key(self):
        diff = _compute_diff({"a": 1}, {"a": 1, "b": 2})
        assert "b" in diff.added
        assert diff.added["b"] == 2

    def test_removed_key(self):
        diff = _compute_diff({"a": 1, "b": 2}, {"a": 1})
        assert "b" in diff.removed
        assert diff.removed["b"] == 2

    def test_changed_value(self):
        diff = _compute_diff({"a": 1}, {"a": 2})
        assert "a" in diff.changed
        assert diff.changed["a"] == (1, 2)

    def test_to_dict_format(self):
        diff = _compute_diff({"a": 1}, {"a": 2, "b": 3})
        d = diff.to_dict()
        assert "added" in d
        assert "changed" in d
        assert "removed" in d

    def test_apply_state_updates_basic(self):
        result = _apply_state_updates({"x": 1}, {"y": 2})
        assert result == {"x": 1, "y": 2}

    def test_apply_dotted_path(self):
        result = _apply_state_updates({}, {"flags.test": True})
        assert result["flags"]["test"] is True

    def test_apply_overwrites_value(self):
        result = _apply_state_updates({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_apply_does_not_mutate_original(self):
        original = {"a": 1}
        _apply_state_updates(original, {"a": 2})
        assert original["a"] == 1


# ── Helper function tests ─────────────────────────────────────────────


class TestHelpers:
    """Test helper functions."""

    def test_build_scene_data_basic(self):
        node = _helios_graph()["nodes"]["node_001"]
        scene = _build_scene_data("node_001", node)
        assert scene["id"] == "node_001"
        assert scene["title"] == "Notrufsignal"
        assert scene["type"] == "start"
        assert scene["is_start"] is True
        assert scene["is_end"] is False

    def test_build_scene_data_empty_node(self):
        scene = _build_scene_data("x", {})
        assert scene["id"] == "x"
        assert scene["title"] == "x"
        assert scene["type"] == "scene"
        assert scene["act"] == 1

    def test_get_choices(self):
        node = _helios_graph()["nodes"]["node_001"]
        choices = _get_choices(node)
        assert len(choices) == 2
        assert choices[0]["id"] == "answer_signal"
        assert choices[0]["label"] == "Den Funkspruch beantworten"

    def test_get_choices_empty(self):
        node = _helios_graph()["nodes"]["node_005"]
        choices = _get_choices(node)
        assert choices == []

    def test_is_ending_end_node(self):
        node = _helios_graph()["nodes"]["node_005"]
        assert _is_ending(node) is True

    def test_is_ending_start_node(self):
        node = _helios_graph()["nodes"]["node_001"]
        assert _is_ending(node) is False

    def test_is_ending_no_choices(self):
        node = {"type": "scene", "choices": []}
        assert _is_ending(node) is True


# ── SimulationEngine.get_full_state tests ─────────────────────────────


class TestGetFullState:
    """Test SimulationEngine.get_full_state."""

    def test_get_full_state_empty_path_starts(self):
        result = SimulationEngine.get_full_state(_helios_graph(), [], {}, 0)
        assert result["current_node_id"] == "node_001"

    def test_get_full_state_with_path(self):
        result = SimulationEngine.get_full_state(
            _helios_graph(), ["node_001", "node_002"], {}, 2)
        assert result["current_node_id"] == "node_002"
        assert result["path"] == ["node_001", "node_002"]
        assert result["step_count"] == 2

    def test_get_full_state_at_ending(self):
        result = SimulationEngine.get_full_state(
            _helios_graph(), ["node_001", "node_002", "node_005"], {}, 3)
        assert result["is_ended"] is True


# ── Admin UI endpoint tests ────────────────────────────────────────────


class TestSimulationEndpoints:
    """Test the simulation admin UI endpoints (Spec §8.1.5)."""

    @pytest.fixture
    async def admin_client(self):
        from app.admin_ui.app import admin_app
        from app.persistence.database import init_db, close_db
        await init_db()
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        await close_db()

    async def _create_draft(self, client) -> str:
        from app.services.story_authoring_agent import get_authoring_agent
        from app.persistence.authoring_repositories import (
            StoryDraftRepository, StoryDraftVersionRepository,
        )
        from app.persistence.database import get_session_factory
        from app.models.enums import DraftStatus
        agent = get_authoring_agent(dummy=True)
        brief = {"title": "Test", "genre": "science_fiction", "tone": "dark_mystery"}
        outline = await agent.generate_outline(brief)
        graph = await agent.generate_graph(outline)
        async with get_session_factory()() as session:
            dr = StoryDraftRepository(session)
            vr = StoryDraftVersionRepository(session)
            draft = await dr.create(
                title="Test", genre="science_fiction", tone="dark_mystery",
                language="de", target_age="16+", brief=brief,
            )
            await vr.create(
                draft_id=draft.id, graph=graph, outline=outline,
                created_by="dummy_agent", notes="Test",
            )
            await dr.update_status(draft.id, DraftStatus.GENERATING)
            await dr.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
            await session.commit()
            return draft.id

    @pytest.mark.asyncio
    async def test_simulate_page_renders(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/simulate")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_simulate_page_contains_title(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/simulate")
        assert "Test" in resp.text
        assert "Simulation" in resp.text

    @pytest.mark.asyncio
    async def test_simulate_page_contains_scene(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/simulate")
        assert "Notrufsignal" in resp.text

    @pytest.mark.asyncio
    async def test_simulate_page_contains_choices(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/simulate")
        assert "choice-btn" in resp.text

    @pytest.mark.asyncio
    async def test_simulate_page_contains_path_bar(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/simulate")
        assert "path-bar" in resp.text

    @pytest.mark.asyncio
    async def test_simulate_page_contains_state_panel(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/simulate")
        assert "state-panel" in resp.text

    @pytest.mark.asyncio
    async def test_simulate_page_contains_sim_start_data(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/simulate")
        assert "sim-app" in resp.text
        assert "data-sim-start" in resp.text

    @pytest.mark.asyncio
    async def test_simulate_page_404_missing_draft(self, admin_client):
        resp = await admin_client.get("/draft/nonexistent/simulate")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_simulate_start_api(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/simulate/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_node_id"] == "node_001"
        assert data["current_node"]["title"] == "Notrufsignal"
        assert len(data["available_choices"]) == 2
        assert data["is_ended"] is False

    @pytest.mark.asyncio
    async def test_simulate_choose_api(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        start_resp = await admin_client.get(f"/draft/{draft_id}/simulate/start")
        start_data = start_resp.json()
        resp = await admin_client.post(
            f"/draft/{draft_id}/simulate/choose",
            json={
                "current_node_id": start_data["current_node_id"],
                "choice_id": "answer_signal",
                "world_state": start_data["world_state"],
                "path": start_data["path"],
                "step_count": start_data["step_count"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_node_id"] == "node_002"
        assert data["current_node"]["title"] == "Die Stimme im Rauschen"
        assert data["path"] == ["node_001", "node_002"]
        assert data["step_count"] == 2
        assert data["is_ended"] is False

    @pytest.mark.asyncio
    async def test_simulate_choose_to_ending(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        # Start
        s = (await admin_client.get(f"/draft/{draft_id}/simulate/start")).json()
        # Choose answer_signal -> node_002
        r1 = (await admin_client.post(
            f"/draft/{draft_id}/simulate/choose",
            json={"current_node_id": s["current_node_id"], "choice_id": "answer_signal",
                  "world_state": s["world_state"], "path": s["path"], "step_count": s["step_count"]},
        )).json()
        # Choose keep_secret -> node_005 (ending)
        r2 = (await admin_client.post(
            f"/draft/{draft_id}/simulate/choose",
            json={"current_node_id": r1["current_node_id"], "choice_id": "keep_secret",
                  "world_state": r1["world_state"], "path": r1["path"], "step_count": r1["step_count"]},
        )).json()
        assert r2["current_node_id"] == "node_005"
        assert r2["is_ended"] is True

    @pytest.mark.asyncio
    async def test_simulate_choose_invalid_choice(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        s = (await admin_client.get(f"/draft/{draft_id}/simulate/start")).json()
        resp = await admin_client.post(
            f"/draft/{draft_id}/simulate/choose",
            json={"current_node_id": s["current_node_id"], "choice_id": "BAD",
                  "world_state": {}, "path": s["path"], "step_count": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_simulate_state_api(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(
            "/draft/" + draft_id + "/simulate/state?path=node_001,node_002&world_state={}&step_count=2"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_node_id"] == "node_002"

    @pytest.mark.asyncio
    async def test_simulate_state_api_empty_path(self, admin_client):
        draft_id = await self._create_draft(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/simulate/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_node_id"] == "node_001"

    @pytest.mark.asyncio
    async def test_simulate_choose_404_missing_draft(self, admin_client):
        resp = await admin_client.post(
            "/draft/nonexistent/simulate/choose",
            json={"current_node_id": "x", "choice_id": "y", "world_state": {}, "path": [], "step_count": 0},
        )
        assert resp.status_code == 404
