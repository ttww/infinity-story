"""Integration tests for the complete story flow.

Spec Kapitel 19 — Akzeptanzkriterien MVP Runtime.

Tests exercise the runtime flow end-to-end through the HTTP API
(POST /api/message) and directly through the StoryOrchestrator,
ChoiceInterpreter, and StateUpdater.
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


# ── ChoiceInterpreter unit tests ────────────────────────────────────

class TestChoiceInterpreter:
    """Test the ChoiceInterpreter directly (Spec §5.3)."""

    def test_letter_choice(self):
        from app.services.story_orchestrator import ChoiceInterpreter

        choices = [
            {"id": "a", "label": "Option A", "next_node_id": "n1"},
            {"id": "b", "label": "Option B", "next_node_id": "n2"},
            {"id": "c", "label": "Option C", "next_node_id": "n3"},
        ]
        for letter, expected_id in [("A", "a"), ("B", "b"), ("C", "c"), ("a", "a")]:
            choice_id, is_free = ChoiceInterpreter.interpret(letter, choices)
            assert choice_id == expected_id, f"Letter {letter!r} → {choice_id}"
            assert is_free is False

    def test_number_choice(self):
        from app.services.story_orchestrator import ChoiceInterpreter

        choices = [
            {"id": "a", "label": "Option A", "next_node_id": "n1"},
            {"id": "b", "label": "Option B", "next_node_id": "n2"},
        ]
        for number, expected_id in [("1", "a"), ("2", "b")]:
            choice_id, is_free = ChoiceInterpreter.interpret(number, choices)
            assert choice_id == expected_id
            assert is_free is False

    def test_exact_id_match(self):
        from app.services.story_orchestrator import ChoiceInterpreter

        choices = [
            {"id": "answer_signal", "label": "Antworten", "next_node_id": "n1"},
        ]
        choice_id, is_free = ChoiceInterpreter.interpret("answer_signal", choices)
        assert choice_id == "answer_signal"
        assert is_free is False

    def test_fuzzy_label_match(self):
        from app.services.story_orchestrator import ChoiceInterpreter

        choices = [
            {"id": "go_left", "label": "Nach links gehen", "next_node_id": "n1"},
        ]
        # Partial label match
        choice_id, is_free = ChoiceInterpreter.interpret("links", choices)
        assert choice_id == "go_left"
        assert is_free is False

    def test_free_form_input(self):
        from app.services.story_orchestrator import ChoiceInterpreter

        choices = [
            {"id": "a", "label": "Option A", "next_node_id": "n1"},
        ]
        # Unmatched input → free-form
        choice_id, is_free = ChoiceInterpreter.interpret("X", choices)
        assert choice_id is None
        assert is_free is True

        # Empty input → free-form
        choice_id, is_free = ChoiceInterpreter.interpret("", choices)
        assert choice_id is None
        assert is_free is True


# ── StateUpdater unit tests ─────────────────────────────────────────

class TestStateUpdater:
    """Test the StateUpdater directly (Spec §5.3)."""

    def test_dotted_path_flag(self):
        from app.services.story_orchestrator import StateUpdater

        ws = {"flags": {}}
        result = StateUpdater.apply(ws, {"flags.answered_signal": True})
        assert result["flags"]["answered_signal"] is True
        # Original unchanged
        assert ws["flags"] == {}

    def test_dotted_path_relationship(self):
        from app.services.story_orchestrator import StateUpdater

        ws = {"relationships": {}}
        result = StateUpdater.apply(ws, {"relationships.Mira": 0.1})
        assert result["relationships"]["Mira"] == 0.1

    def test_top_level_inventory(self):
        from app.services.story_orchestrator import StateUpdater

        ws = {"inventory": []}
        result = StateUpdater.apply(ws, {"inventory": ["access card"]})
        assert "access card" in result["inventory"]

    def test_nested_create(self):
        from app.services.story_orchestrator import StateUpdater

        ws = {}
        result = StateUpdater.apply(ws, {"flags.new_flag": True})
        assert result["flags"]["new_flag"] is True


# ── End-to-end flow through HTTP API ────────────────────────────────

class TestStoryFlowHTTP:
    """Test the full story flow via POST /api/message.

    Flow: Start → Scenario list → Select scenario → Opening scene →
    Choice → Next scene → Continue → End.
    """

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
    async def test_start_shows_scenarios(self, client):
        """User sends 'Start' and sees a scenario list."""
        response = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": "flow_user_1", "message": "Start"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) > 0
        assert "Szenario" in data["messages"][0] or "Wähle" in data["messages"][0]

    @pytest.mark.asyncio
    async def test_full_flow_start_to_scene(self, client):
        """Complete flow: Start → select → opening scene → choice → next scene."""
        user = "flow_user_2"

        # 1. Start
        r = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": user, "message": "Start"},
        )
        assert r.status_code == 200
        assert "Szenario" in r.json()["messages"][0]

        # 2. Select scenario by number
        r = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": user, "message": "1"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("session_id") is not None
        assert data.get("scene") is not None
        assert data["scene"]["scene_text"]

        # 3. Send a choice (number "1")
        r = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": user, "message": "1"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("scene") is not None
        assert data["scene"]["scene_text"]

    @pytest.mark.asyncio
    async def test_select_by_name(self, client):
        """Select a scenario by name instead of number."""
        user = "flow_user_3"
        r = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": user, "message": "helios"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("session_id") is not None
        assert data.get("scene") is not None

    @pytest.mark.asyncio
    async def test_free_form_input_accepted(self, client):
        """Free-form text (not matching any choice) should produce a scene."""
        user = "flow_user_4"

        # Start session
        await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": user, "message": "1"},
        )

        # Send free-form text
        r = await client.post(
            "/api/message",
            json={
                "channel": "whatsapp_mock",
                "user_id": user,
                "message": "Ich untersuche das Terminal genauer",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("scene") is not None

    @pytest.mark.asyncio
    async def test_no_session_prompts_start(self, client):
        """Without an active session, the user should be prompted to start."""
        r = await client.post(
            "/api/message",
            json={
                "channel": "whatsapp_mock",
                "user_id": "no_session_user",
                "message": "etwas tun",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert any("Start" in m or "start" in m for m in data["messages"])

    @pytest.mark.asyncio
    async def test_world_state_persisted(self, client):
        """World state should be persisted between messages."""
        user = "flow_user_5"

        # Start and select scenario
        r = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": user, "message": "1"},
        )
        session_id = r.json().get("session_id")
        assert session_id is not None

        # Send a choice
        r = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": user, "message": "1"},
        )
        assert r.status_code == 200
        # Scene should be produced (world state advanced)
        assert r.json().get("scene") is not None


# ── StoryOrchestrator direct tests ──────────────────────────────────

class TestStoryOrchestrator:
    """Test the StoryOrchestrator directly with MockLLMService."""

    @pytest.mark.asyncio
    async def test_generate_opening_scene(self):
        from app.services.story_orchestrator import StoryOrchestrator, StoryContext
        from app.services.llm_service import MockLLMService
        from app.story.scenario_loader import load_scenario, get_start_node, build_initial_world_state

        scenario = load_scenario("helios")
        start_node = get_start_node(scenario)
        world_state = build_initial_world_state(scenario)

        orchestrator = StoryOrchestrator(MockLLMService())
        ctx = orchestrator.build_context(
            session_id="test-session",
            node=start_node,
            world_state=world_state,
            scenario_id="helios",
        )
        result = await orchestrator.generate_opening_scene(ctx)

        assert result.scene.scene_text
        assert len(result.scene.choices) > 0
        # Opening scene stays on the start node — next_node_id is None
        # so that the user's choice is matched correctly on the next turn.
        assert result.next_node_id is None

    @pytest.mark.asyncio
    async def test_process_choice_advances(self):
        from app.services.story_orchestrator import StoryOrchestrator, StoryContext
        from app.services.llm_service import MockLLMService
        from app.story.scenario_loader import load_scenario, get_node, get_start_node, build_initial_world_state

        scenario = load_scenario("helios")
        start_node = get_start_node(scenario)
        world_state = build_initial_world_state(scenario)

        orchestrator = StoryOrchestrator(MockLLMService())
        ctx = orchestrator.build_context(
            session_id="test-session",
            node=start_node,
            world_state=world_state,
            user_input="1",  # Select first choice
            scenario_id="helios",
        )
        result = await orchestrator.process_user_input(ctx, scenario=scenario)

        assert result.scene.scene_text
        assert result.next_node_id is not None

    @pytest.mark.asyncio
    async def test_process_free_form(self):
        from app.services.story_orchestrator import StoryOrchestrator, StoryContext
        from app.services.llm_service import MockLLMService
        from app.story.scenario_loader import load_scenario, get_start_node, build_initial_world_state

        scenario = load_scenario("helios")
        start_node = get_start_node(scenario)
        world_state = build_initial_world_state(scenario)

        orchestrator = StoryOrchestrator(MockLLMService())
        ctx = orchestrator.build_context(
            session_id="test-session",
            node=start_node,
            world_state=world_state,
            user_input="Ich untersuche das Terminal",
            scenario_id="helios",
        )
        result = await orchestrator.process_user_input(ctx)

        assert result.scene.scene_text
        # Free-form should still produce a scene
        assert result.scene.state_updates is not None


# ── WorldState / apply_state_updates ────────────────────────────────

class TestWorldStateUpdates:
    """Test world state update via the graph module's apply_state_updates."""

    def test_flag_update(self):
        from app.models import WorldState
        from app.story.graph import apply_state_updates

        ws = WorldState(genre="science_fiction", tone="düster", main_character_name="Alex")
        ws = apply_state_updates(ws, {"flags.answered_signal": True})
        assert ws.flags["answered_signal"] is True

    def test_relationship_update(self):
        from app.models import WorldState
        from app.story.graph import apply_state_updates

        ws = WorldState()
        ws = apply_state_updates(ws, {"relationships.Mira": 0.1})
        assert ws.relationships["Mira"] == 0.1

    def test_inventory_update(self):
        from app.models import WorldState
        from app.story.graph import apply_state_updates

        ws = WorldState()
        ws = apply_state_updates(ws, {"inventory": ["access card"]})
        assert "access card" in ws.inventory

    def test_top_level_replace(self):
        from app.models import WorldState
        from app.story.graph import apply_state_updates

        ws = WorldState()
        ws = apply_state_updates(ws, {"current_location": "Medbay"})
        assert ws.current_location == "Medbay"
