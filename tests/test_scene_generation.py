"""Tests for the Prompting Runtime & LLM Scene Generation (Spec §5.7, §18 Schritt 9).

Tests cover:
  - Enhanced narrator prompt (SCENE_SYSTEM_PROMPT) content requirements
  - build_scene_user_prompt with full context
  - ChoiceInterpreter: exact id, numeric, letter, fuzzy label, free-form
  - StateUpdater: dotted-path updates, nested dicts, list replacement
  - StoryOrchestrator.process_user_input with mock LLM
  - StoryOrchestrator.generate_opening_scene with mock LLM
  - OrchestratorResult: scene, next_node, updated state, is_ending
  - Scenario loader: load_scenario, get_node, get_start_node
  - Message route: start, scenario selection, choice processing
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.story_orchestrator import (
    ChoiceInterpreter,
    GeneratedScene,
    OrchestratorResult,
    StateUpdater,
    StoryContext,
    StoryOrchestrator,
)
from app.services.llm_service import MockLLMService
from app.story.prompts import SCENE_SYSTEM_PROMPT, build_scene_user_prompt
from app.story.scenario_loader import (
    build_initial_world_state,
    get_node,
    get_start_node,
    get_start_node_id,
    list_scenarios,
    load_scenario,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def helios_scenario():
    return load_scenario("helios")


@pytest.fixture
def start_node(helios_scenario):
    return get_start_node(helios_scenario)


@pytest.fixture
def world_state(helios_scenario):
    ws = build_initial_world_state(helios_scenario)
    ws["main_character_name"] = "Alex"
    ws["current_location"] = "Orbital Station Helios"
    return ws


@pytest.fixture
def mock_llm():
    return MockLLMService()


@pytest.fixture
def orchestrator(mock_llm):
    return StoryOrchestrator(mock_llm)


@pytest.fixture
def choices():
    """Standard choice list for interpreter tests."""
    return [
        {"id": "answer_signal", "label": "Den Funkspruch beantworten", "next_node_id": "node_002"},
        {"id": "ignore_signal", "label": "Es ignorieren und zum Captain gehen", "next_node_id": "node_003"},
    ]


# ── Prompt content tests (Spec §5.7) ────────────────────────────

class TestSceneSystemPrompt:
    """Verify the system prompt enforces all §5.7 requirements."""

    def test_prompt_mentions_narrator_role(self):
        assert "narrator" in SCENE_SYSTEM_PROMPT.lower()

    def test_prompt_requires_concise_immersive(self):
        prompt_lower = SCENE_SYSTEM_PROMPT.lower()
        assert "concise" in prompt_lower
        assert "immersive" in prompt_lower

    def test_prompt_requires_world_state_consistency(self):
        prompt_lower = SCENE_SYSTEM_PROMPT.lower()
        assert "world state" in prompt_lower
        assert "contradict" in prompt_lower

    def test_prompt_requires_decision_point(self):
        assert "decision" in SCENE_SYSTEM_PROMPT.lower()

    def test_prompt_limits_choices_to_3_4(self):
        # The prompt now allows 0-4 choices; verify it mentions the limit
        assert "2-4 choices" in SCENE_SYSTEM_PROMPT or "never more than 4" in SCENE_SYSTEM_PROMPT.lower()

    def test_prompt_allows_free_response(self):
        prompt_lower = SCENE_SYSTEM_PROMPT.lower()
        assert "free" in prompt_lower

    def test_prompt_forbids_fact_changes(self):
        prompt_lower = SCENE_SYSTEM_PROMPT.lower()
        assert "not change" in prompt_lower or "may not" in prompt_lower

    def test_prompt_specifies_output_fields(self):
        for field in ["scene_text", "choices", "state_updates", "suggested_next_node"]:
            assert field in SCENE_SYSTEM_PROMPT

    def test_prompt_no_endless_monologues(self):
        assert "monologue" in SCENE_SYSTEM_PROMPT.lower()


# ── build_scene_user_prompt tests ──────────────────────────────

class TestBuildSceneUserPrompt:

    def test_includes_node_id(self, start_node, world_state):
        prompt = build_scene_user_prompt(
            node_id=start_node["id"],
            scene_goal=start_node["scene_goal"],
            location=start_node["location"],
            characters=start_node["characters"],
            world_state=world_state,
        )
        assert "node_001" in prompt

    def test_includes_scene_goal(self, start_node, world_state):
        prompt = build_scene_user_prompt(
            node_id=start_node["id"],
            scene_goal=start_node["scene_goal"],
            location=start_node["location"],
            characters=start_node["characters"],
            world_state=world_state,
        )
        assert start_node["scene_goal"] in prompt

    def test_includes_world_state_json(self, start_node, world_state):
        prompt = build_scene_user_prompt(
            node_id=start_node["id"],
            scene_goal=start_node["scene_goal"],
            location=start_node["location"],
            characters=start_node["characters"],
            world_state=world_state,
        )
        assert "genre" in prompt
        assert world_state["genre"] in prompt

    def test_includes_user_input(self, start_node, world_state):
        prompt = build_scene_user_prompt(
            node_id=start_node["id"],
            scene_goal=start_node["scene_goal"],
            location=start_node["location"],
            characters=start_node["characters"],
            world_state=world_state,
            user_input="Ich gehe zur Tür",
        )
        assert "Ich gehe zur Tür" in prompt

    def test_includes_predefined_choices(self, start_node, world_state):
        prompt = build_scene_user_prompt(
            node_id=start_node["id"],
            scene_goal=start_node["scene_goal"],
            location=start_node["location"],
            characters=start_node["characters"],
            world_state=world_state,
            predefined_choices=start_node["choices"],
        )
        assert "answer_signal" in prompt
        assert "Den Funkspruch beantworten" in prompt

    def test_includes_reveals(self, start_node, world_state):
        prompt = build_scene_user_prompt(
            node_id=start_node["id"],
            scene_goal=start_node["scene_goal"],
            location=start_node["location"],
            characters=start_node["characters"],
            world_state=world_state,
            reveals=start_node["reveals"],
        )
        assert start_node["reveals"][0] in prompt

    def test_includes_history(self, start_node, world_state):
        history = [
            {"role": "narrator", "text": "Die Station war still."},
            {"role": "user", "text": "Ich höre genauer hin."},
        ]
        prompt = build_scene_user_prompt(
            node_id=start_node["id"],
            scene_goal=start_node["scene_goal"],
            location=start_node["location"],
            characters=start_node["characters"],
            world_state=world_state,
            history=history,
        )
        assert "Die Station war still." in prompt
        assert "Ich höre genauer hin." in prompt

    def test_no_user_input_shows_opening(self, start_node, world_state):
        prompt = build_scene_user_prompt(
            node_id=start_node["id"],
            scene_goal=start_node["scene_goal"],
            location=start_node["location"],
            characters=start_node["characters"],
            world_state=world_state,
            user_input=None,
        )
        assert "opening scene" in prompt.lower()

    def test_no_choices_shows_none(self, start_node, world_state):
        prompt = build_scene_user_prompt(
            node_id=start_node["id"],
            scene_goal=start_node["scene_goal"],
            location=start_node["location"],
            characters=start_node["characters"],
            world_state=world_state,
            predefined_choices=[],
        )
        assert "none" in prompt.lower()


# ── ChoiceInterpreter tests ────────────────────────────────────

class TestChoiceInterpreter:

    def test_exact_id_match(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("answer_signal", choices)
        assert cid == "answer_signal"
        assert is_free is False

    def test_numeric_selection_1(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("1", choices)
        assert cid == "answer_signal"
        assert is_free is False

    def test_numeric_selection_2(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("2", choices)
        assert cid == "ignore_signal"
        assert is_free is False

    def test_letter_selection_a(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("a", choices)
        assert cid == "answer_signal"
        assert is_free is False

    def test_letter_selection_b(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("b", choices)
        assert cid == "ignore_signal"
        assert is_free is False

    def test_letter_uppercase(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("A", choices)
        assert cid == "answer_signal"
        assert is_free is False

    def test_exact_label_match(self, choices):
        cid, is_free = ChoiceInterpreter.interpret(
            "Den Funkspruch beantworten", choices
        )
        assert cid == "answer_signal"
        assert is_free is False

    def test_fuzzy_label_contains(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("Funkspruch", choices)
        assert cid == "answer_signal"
        assert is_free is False

    def test_fuzzy_label_partial_in_input(self, choices):
        """Input containing a key fragment of the label should match."""
        cid, is_free = ChoiceInterpreter.interpret(
            "zum Captain gehen", choices
        )
        assert cid == "ignore_signal"
        assert is_free is False

    def test_free_form_input(self, choices):
        cid, is_free = ChoiceInterpreter.interpret(
            "Ich untersuche das Terminal genauer", choices
        )
        assert cid is None
        assert is_free is True

    def test_empty_input_is_free(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("", choices)
        assert cid is None
        assert is_free is True

    def test_whitespace_input_is_free(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("   ", choices)
        assert cid is None
        assert is_free is True

    def test_no_choices_is_free(self):
        cid, is_free = ChoiceInterpreter.interpret("anything", [])
        assert cid is None
        assert is_free is True

    def test_invalid_number_is_free(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("99", choices)
        assert cid is None
        assert is_free is True

    def test_invalid_letter_is_free(self, choices):
        cid, is_free = ChoiceInterpreter.interpret("z", choices)
        assert cid is None
        assert is_free is True


# ── StateUpdater tests ─────────────────────────────────────────

class TestStateUpdater:

    def test_top_level_update(self):
        ws = {"genre": "sf", "tone": "dark"}
        result = StateUpdater.apply(ws, {"tone": "light"})
        assert result["tone"] == "light"
        assert result["genre"] == "sf"

    def test_dotted_path_update(self):
        ws = {"flags": {"answered_signal": False}}
        result = StateUpdater.apply(ws, {"flags.answered_signal": True})
        assert result["flags"]["answered_signal"] is True

    def test_nested_dotted_path(self):
        ws = {"relationships": {"Mira": 0.7}}
        result = StateUpdater.apply(ws, {"relationships.Mira": 0.3})
        assert result["relationships"]["Mira"] == 0.3

    def test_dotted_path_creates_missing(self):
        ws = {"flags": {}}
        result = StateUpdater.apply(ws, {"flags.new_flag": True})
        assert result["flags"]["new_flag"] is True

    def test_dotted_path_deep_create(self):
        ws = {}
        result = StateUpdater.apply(ws, {"a.b.c": "deep"})
        assert result["a"]["b"]["c"] == "deep"

    def test_does_not_mutate_original(self):
        ws = {"flags": {"answered_signal": False}}
        StateUpdater.apply(ws, {"flags.answered_signal": True})
        assert ws["flags"]["answered_signal"] is False

    def test_list_replacement(self):
        ws = {"inventory": ["card"]}
        result = StateUpdater.apply(ws, {"inventory": ["card", "key"]})
        assert result["inventory"] == ["card", "key"]

    def test_multiple_updates(self):
        ws = {"flags": {}, "inventory": []}
        result = StateUpdater.apply(ws, {
            "flags.answered_signal": True,
            "flags.trusts_mira": True,
            "inventory": ["key"],
        })
        assert result["flags"]["answered_signal"] is True
        assert result["flags"]["trusts_mira"] is True
        assert result["inventory"] == ["key"]


# ── StoryOrchestrator tests ────────────────────────────────────

class TestStoryOrchestrator:

    @pytest.mark.asyncio
    async def test_process_user_input_returns_result(self, orchestrator, start_node, world_state):
        ctx = orchestrator.build_context(
            session_id="s1",
            node=start_node,
            world_state=world_state,
            user_input="1",
        )
        result = await orchestrator.process_user_input(ctx)
        assert isinstance(result, OrchestratorResult)
        assert isinstance(result.scene, GeneratedScene)
        assert result.scene.scene_text
        assert len(result.scene.choices) >= 2

    @pytest.mark.asyncio
    async def test_process_with_choice_selection(self, orchestrator, start_node, world_state):
        ctx = orchestrator.build_context(
            session_id="s1",
            node=start_node,
            world_state=world_state,
            user_input="answer_signal",
        )
        result = await orchestrator.process_user_input(ctx)
        assert result.scene is not None
        # Mock LLM returns suggested_next_node="node_002"
        assert result.next_node_id is not None

    @pytest.mark.asyncio
    async def test_process_with_free_form_input(self, orchestrator, start_node, world_state):
        ctx = orchestrator.build_context(
            session_id="s1",
            node=start_node,
            world_state=world_state,
            user_input="Ich untersuche das Terminal auf weitere Daten",
        )
        result = await orchestrator.process_user_input(ctx)
        assert result.scene is not None
        assert result.scene.scene_text

    @pytest.mark.asyncio
    async def test_process_applies_state_updates(self, orchestrator, start_node, world_state):
        ctx = orchestrator.build_context(
            session_id="s1",
            node=start_node,
            world_state=world_state,
            user_input="1",
        )
        result = await orchestrator.process_user_input(ctx)
        # Mock LLM returns state_updates={"flags.answered_signal": False}
        assert "flags" in result.updated_world_state
        # The original world_state should not be mutated
        assert "answered_signal" not in world_state.get("flags", {})

    @pytest.mark.asyncio
    async def test_generate_opening_scene(self, orchestrator, start_node, world_state):
        ctx = orchestrator.build_context(
            session_id="s1",
            node=start_node,
            world_state=world_state,
        )
        result = await orchestrator.generate_opening_scene(ctx)
        assert isinstance(result, OrchestratorResult)
        assert result.scene.scene_text
        assert len(result.scene.choices) >= 2

    @pytest.mark.asyncio
    async def test_ending_node_detected(self, orchestrator, helios_scenario, world_state):
        end_node = get_node(helios_scenario, "node_005")
        ctx = orchestrator.build_context(
            session_id="s1",
            node=end_node,
            world_state=world_state,
            user_input="was passiert jetzt",
        )
        result = await orchestrator.process_user_input(ctx)
        assert result.is_ending is True

    @pytest.mark.asyncio
    async def test_non_ending_node_not_ending(self, orchestrator, start_node, world_state):
        ctx = orchestrator.build_context(
            session_id="s1",
            node=start_node,
            world_state=world_state,
            user_input="1",
        )
        result = await orchestrator.process_user_input(ctx)
        assert result.is_ending is False

    @pytest.mark.asyncio
    async def test_build_context_sets_available_choices(self, orchestrator, start_node, world_state):
        ctx = orchestrator.build_context(
            session_id="s1",
            node=start_node,
            world_state=world_state,
        )
        assert len(ctx.available_choices) == 2
        assert ctx.available_choices[0]["id"] == "answer_signal"

    @pytest.mark.asyncio
    async def test_process_with_no_user_input(self, orchestrator, start_node, world_state):
        ctx = orchestrator.build_context(
            session_id="s1",
            node=start_node,
            world_state=world_state,
        )
        result = await orchestrator.process_user_input(ctx)
        assert result.scene is not None

    @pytest.mark.asyncio
    async def test_llm_service_injection(self, mock_llm):
        orch = StoryOrchestrator(mock_llm)
        assert orch.llm_service is mock_llm

    @pytest.mark.asyncio
    async def test_lazy_llm_service_creation(self):
        orch = StoryOrchestrator()
        # Accessing the property should create a service (mock by test config)
        svc = orch.llm_service
        assert svc is not None
        assert svc.provider_name == "mock"


# ── Scenario loader tests ──────────────────────────────────────

class TestScenarioLoader:

    def test_load_helios(self):
        scenario = load_scenario("helios")
        assert scenario["id"] == "helios"
        assert len(scenario["nodes"]) >= 5

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_scenario("nonexistent")

    def test_list_scenarios_includes_helios(self):
        scenarios = list_scenarios()
        ids = [s["id"] for s in scenarios]
        assert "helios" in ids

    def test_get_start_node(self, helios_scenario):
        node = get_start_node(helios_scenario)
        assert node is not None
        assert node["id"] == "node_001"

    def test_get_start_node_id(self, helios_scenario):
        assert get_start_node_id(helios_scenario) == "node_001"

    def test_get_node(self, helios_scenario):
        node = get_node(helios_scenario, "node_002")
        assert node is not None
        assert node["title"] == "Die Stimme im Rauschen"

    def test_get_nonexistent_node(self, helios_scenario):
        assert get_node(helios_scenario, "nonexistent") is None

    def test_build_initial_world_state(self, helios_scenario):
        ws = build_initial_world_state(helios_scenario)
        assert ws["genre"] == "science_fiction"
        assert ws["tone"] == "dark_mystery"
        assert ws["language"] == "de"
        assert ws["inventory"] == []
        assert ws["flags"] == {}


# ── Integration: LLM generate_scene with enriched prompt ───────

class TestLLMSceneGeneration:

    @pytest.mark.asyncio
    async def test_generate_scene_with_full_context(self, mock_llm, start_node, world_state):
        ctx = StoryContext(
            session_id="s1",
            current_node=start_node,
            world_state=world_state,
            user_input="Ich antworte dem Signal",
            history=[{"role": "narrator", "text": "Die Station war still."}],
        )
        scene = await mock_llm.generate_scene(ctx)
        assert isinstance(scene, GeneratedScene)
        assert scene.scene_text
        assert len(scene.choices) >= 2
        # Mock no longer hardcodes suggested_next_node — it returns None
        # so the orchestrator uses the user's choice next_node_id instead.
        assert scene.suggested_next_node is None

    @pytest.mark.asyncio
    async def test_generate_scene_user_input_override(self, mock_llm, start_node, world_state):
        ctx = StoryContext(
            session_id="s1",
            current_node=start_node,
            world_state=world_state,
        )
        scene = await mock_llm.generate_scene(ctx, user_input="Gehe zum Captain")
        assert isinstance(scene, GeneratedScene)
        assert scene.scene_text

    @pytest.mark.asyncio
    async def test_generate_scene_no_user_input(self, mock_llm, start_node, world_state):
        ctx = StoryContext(
            session_id="s1",
            current_node=start_node,
            world_state=world_state,
        )
        scene = await mock_llm.generate_scene(ctx)
        assert isinstance(scene, GeneratedScene)
        assert scene.scene_text


# ── Message route integration tests ────────────────────────────

class TestMessageRouteIntegration:

    @pytest.mark.asyncio
    async def test_start_returns_scenarios(self, client):
        """POST /api/message with 'Start' should list scenarios."""
        response = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": "u1", "message": "Start"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert any("Helios" in m for m in data["messages"])

    @pytest.mark.asyncio
    async def test_select_scenario_creates_session(self, client):
        """POST /api/message with '1' should start a story and return a scene."""
        response = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": "u2", "message": "Start"},
        )
        assert response.status_code == 200

        response = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": "u2", "message": "1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("session_id") is not None
        assert data.get("scene") is not None
        assert data["scene"]["scene_text"]

    @pytest.mark.asyncio
    async def test_select_scenario_by_name(self, client):
        """POST /api/message with scenario name should start a story."""
        response = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": "u3", "message": "helios"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("session_id") is not None
        assert data.get("scene") is not None

    @pytest.mark.asyncio
    async def test_choice_in_active_session(self, client):
        """After starting a session, sending a choice should advance the story."""
        # Start session
        await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": "u4", "message": "1"},
        )
        # Send a choice
        response = await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": "u4", "message": "1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("scene") is not None
        assert data["scene"]["scene_text"]

    @pytest.mark.asyncio
    async def test_free_form_input_in_session(self, client):
        """Free-form text should be accepted and produce a scene."""
        await client.post(
            "/api/message",
            json={"channel": "whatsapp_mock", "user_id": "u5", "message": "1"},
        )
        response = await client.post(
            "/api/message",
            json={
                "channel": "whatsapp_mock",
                "user_id": "u5",
                "message": "Ich untersuche das Terminal genauer",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("scene") is not None

    @pytest.mark.asyncio
    async def test_no_session_prompts_start(self, client):
        """Without an active session, the user should be prompted to start."""
        response = await client.post(
            "/api/message",
            json={
                "channel": "whatsapp_mock",
                "user_id": "no_session_user",
                "message": "etwas tun",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert any("Start" in m or "start" in m for m in data["messages"])

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
