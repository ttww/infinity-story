"""Story orchestrator (Spec §5.3, §5.7).

Central runtime logic: decides what happens next based on the current
node, world state, and user input. Delegates text generation to the
LLM service.

Responsibilities (Spec §5.3):
  - Decide what happens next
  - Load current story node
  - Interpret user answer (choice or free action)
  - Map decision to choice or free action
  - Update story state
  - Build LLM prompt
  - Generate new scene
  - Generate decision options
  - Create new dynamic nodes for free deviations (future)

The orchestrator works primarily with structured world state, the
current node, and the story graph — NOT the full chat history (§5.3).
"""

from __future__ import annotations

import json
import logging
import string
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────

@dataclass
class StoryContext:
    """Structured context passed to the LLM for scene generation.

    Contains everything the LLM needs to generate a scene without
    the full chat history (Spec §5.3).
    """
    session_id: str
    current_node: dict[str, Any]
    world_state: dict[str, Any]
    user_input: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    scenario_id: str | None = None
    available_choices: list[dict[str, Any]] = field(default_factory=list)
    # ── protagonist / narrative voice ───────────────────────────────
    protagonist_name: str = ""
    protagonist_pronouns: str = "er"


@dataclass
class GeneratedScene:
    """Result of an LLM scene generation call (Spec §5.7).

    Fields match the spec output: scene_text, choices, state_updates,
    suggested_next_node.
    """
    scene_text: str
    choices: list[dict[str, Any]] = field(default_factory=list)
    state_updates: dict[str, Any] = field(default_factory=dict)
    suggested_next_node: str | None = None


@dataclass
class OrchestratorResult:
    """Full result from processing user input.

    Contains the generated scene plus the updated world state and
    the next node id to advance to.
    """
    scene: GeneratedScene
    next_node_id: str | None = None
    updated_world_state: dict[str, Any] = field(default_factory=dict)
    is_ending: bool = False
    mode: str = "multi_choice"
    auto_advance_delay_ms: int | None = None


# ── Choice interpretation ────────────────────────────────────────

class ChoiceInterpreter:
    """Maps user input to a predefined choice or flags it as free-form."""

    @staticmethod
    def interpret(
        user_input: str,
        choices: list[dict[str, Any]],
    ) -> tuple[str | None, bool]:
        """Interpret user input against available choices.

        Returns (choice_id, is_free_form).
        - If the input matches a choice (by id, number, or fuzzy label),
          returns (choice_id, False).
        - If the input doesn't match any choice, returns (None, True)
          to indicate a free-form action.
        """
        if not user_input or not user_input.strip():
            return None, True

        text = user_input.strip().lower()

        # Exact choice id match
        for ch in choices:
            if ch.get("id", "").lower() == text:
                return ch["id"], False

        # Numeric selection (1-based)
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < len(choices):
                return choices[idx].get("id"), False

        # Letter selection (a, b, c, d, e, f, ...)
        if len(text) == 1 and text in string.ascii_lowercase:
            idx = ord(text) - ord("a")
            if 0 <= idx < len(choices):
                return choices[idx].get("id"), False

        # Fuzzy label match (contains or starts-with)
        for ch in choices:
            label = ch.get("label", "").lower()
            if not label:
                continue
            # Exact label match
            if text == label:
                return ch.get("id"), False
            # Label contains the input (user typed part of it)
            if len(text) >= 3 and text in label:
                return ch.get("id"), False
            # Input contains the label
            if len(label) >= 3 and label in text:
                return ch.get("id"), False

        # No match — free-form input
        return None, True


# ── State update application ─────────────────────────────────────

class StateUpdater:
    """Applies state_updates from GeneratedScene to the world state.

    Supports dotted-path updates (e.g. "flags.answered_signal": true)
    so the LLM can update nested fields without replacing the entire
    world state.
    """

    @staticmethod
    def apply(
        world_state: dict[str, Any],
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply updates to a copy of world_state and return the result.

        Dotted paths like "flags.answered_signal" set nested values.
        Non-dotted keys set top-level values.
        Lists are replaced, not merged.
        """
        import copy
        result = copy.deepcopy(world_state)

        for key, value in updates.items():
            if "." in key:
                parts = key.split(".")
                target = result
                for part in parts[:-1]:
                    if part not in target or not isinstance(target[part], dict):
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value
            else:
                result[key] = value

        return result


# ── Orchestrator ─────────────────────────────────────────────────

class StoryOrchestrator:
    """Coordinates story progression in the runtime (Spec §5.3).

    The orchestrator holds a reference to an LLM service and processes
    user input by:
      1. Interpreting the input against the current node's choices
      2. Building a structured prompt from node + world state
      3. Calling the LLM to generate the next scene
      4. Applying state updates
      5. Determining the next node
    """

    def __init__(self, llm_service: Any | None = None) -> None:
        """Initialize with an optional LLM service.

        If no service is provided, one is created via get_llm_service()
        on first use (deferred import to avoid circular dependencies).
        """
        self._llm_service = llm_service
        self._interpreter = ChoiceInterpreter()
        self._state_updater = StateUpdater()

    @property
    def llm_service(self) -> Any:
        """Lazily create the LLM service if not injected."""
        if self._llm_service is None:
            from app.services.llm_service import get_llm_service
            self._llm_service = get_llm_service()
        return self._llm_service

    async def process_user_input(
        self,
        context: StoryContext,
        scenario: dict[str, Any] | None = None,
    ) -> OrchestratorResult:
        """Process user input and return the next scene.

        Steps:
        1. Interpret the user input (choice or free-form)
        2. Find the next node (if a predefined choice was selected)
        3. Generate scene for the NEXT node (so the user sees the new
           scene and the next node's choices, not the old node's choices)
        4. Apply state updates
        5. Return the next node id so the caller can update the session

        For free-form input (no matching choice), the scene is generated
        for the current node as before.

        Auto-advance: when the current node has 0 choices and a
        next_node_id, the orchestrator reads next_node_id and advances
        without requiring user input.
        """
        node = context.current_node
        choices = context.available_choices or node.get("choices", [])

        # ── Auto-advance: 0 choices with next_node_id ──────────────
        if not choices:
            auto_next = node.get("next_node_id")
            is_end = node.get("is_end", False) or node.get("type") == "end"
            if is_end or not auto_next:
                # Ending node — generate final scene
                scene = await self.llm_service.generate_scene(context)
                updated_state = self._state_updater.apply(
                    context.world_state,
                    scene.state_updates,
                )
                return OrchestratorResult(
                    scene=scene,
                    next_node_id=None,
                    updated_world_state=updated_state,
                    is_ending=True,
                    mode="ending",
                )
            # Auto-advance to the next node
            if scenario:
                from app.story.scenario_loader import get_node
                next_node = get_node(scenario, auto_next)
                if next_node:
                    next_ctx = StoryContext(
                        session_id=context.session_id,
                        current_node=next_node,
                        world_state=context.world_state,
                        user_input=None,
                        scenario_id=context.scenario_id,
                        available_choices=next_node.get("choices", []),
                        history=context.history,
                    )
                    scene = await self.llm_service.generate_scene(next_ctx)
                    updated_state = self._state_updater.apply(
                        next_ctx.world_state,
                        scene.state_updates,
                    )
                    next_is_end = next_node.get("is_end", False) or next_node.get("type") == "end"
                    next_choices = next_node.get("choices", [])
                    next_mode = self._derive_mode(next_choices, next_is_end, next_node.get("next_node_id"))
                    return OrchestratorResult(
                        scene=scene,
                        next_node_id=auto_next,
                        updated_world_state=updated_state,
                        is_ending=next_is_end,
                        mode=next_mode,
                        auto_advance_delay_ms=next_node.get("auto_advance_delay_ms"),
                    )
            # No scenario — fall through to generate for current node

        # Step 1: Interpret input
        choice_id, is_free = self._interpreter.interpret(
            context.user_input or "",
            choices,
        )

        # If a predefined choice was selected, find its next_node_id
        selected_next_node: str | None = None
        if choice_id:
            for ch in choices:
                if ch.get("id") == choice_id:
                    selected_next_node = ch.get("next_node_id")
                    break

        # Step 2: If a choice was selected and we have the scenario,
        # generate the scene for the NEXT node — so the response shows
        # the new scene and the next node's choices.
        if selected_next_node and scenario:
            from app.story.scenario_loader import get_node
            next_node = get_node(scenario, selected_next_node)
            if next_node:
                next_ctx = StoryContext(
                    session_id=context.session_id,
                    current_node=next_node,
                    world_state=context.world_state,
                    user_input=None,  # no user input for the auto-advance
                    scenario_id=context.scenario_id,
                    available_choices=next_node.get("choices", []),
                    history=context.history,
                )
                scene = await self.llm_service.generate_scene(next_ctx)
                updated_state = self._state_updater.apply(
                    next_ctx.world_state,
                    scene.state_updates,
                )
                is_ending = next_node.get("is_end", False) or next_node.get("type") == "end"
                next_choices = next_node.get("choices", [])
                next_mode = self._derive_mode(next_choices, is_ending, next_node.get("next_node_id"))
                return OrchestratorResult(
                    scene=scene,
                    next_node_id=selected_next_node,
                    updated_world_state=updated_state,
                    is_ending=is_ending,
                    mode=next_mode,
                    auto_advance_delay_ms=next_node.get("auto_advance_delay_ms"),
                )

        # Step 3: Free-form input or no scenario — generate for current node
        scene = await self.llm_service.generate_scene(context)

        # Step 4: Apply state updates
        updated_state = self._state_updater.apply(
            context.world_state,
            scene.state_updates,
        )

        # Step 5: Determine next node
        next_node = selected_next_node or scene.suggested_next_node

        # Check if we've reached an ending
        is_ending = node.get("is_end", False) or node.get("type") == "end"

        # Derive mode for the current response
        mode = self._derive_mode(choices, is_ending, node.get("next_node_id"))

        return OrchestratorResult(
            scene=scene,
            next_node_id=next_node,
            updated_world_state=updated_state,
            is_ending=is_ending,
            mode=mode,
            auto_advance_delay_ms=node.get("auto_advance_delay_ms"),
        )

    async def generate_opening_scene(
        self,
        context: StoryContext,
    ) -> OrchestratorResult:
        """Generate the opening scene for a new story session.

        Similar to process_user_input but with no prior user input —
        the scene is generated from the start node.

        The session stays on the start node — the user picks a choice,
        and process_user_input advances the graph on the next turn.
        """
        # No user input for the opening
        context.user_input = None

        scene = await self.llm_service.generate_scene(context)

        updated_state = self._state_updater.apply(
            context.world_state,
            scene.state_updates,
        )

        node = context.current_node
        is_ending = node.get("is_end", False) or node.get("type") == "end"
        choices = node.get("choices", [])
        mode = self._derive_mode(choices, is_ending, node.get("next_node_id"))

        # The opening scene stays on the start node so that the
        # user's choice (from the start node's choices) is matched
        # correctly in process_user_input on the next turn.
        # Do NOT pre-advance to the next node.
        return OrchestratorResult(
            scene=scene,
            next_node_id=None,
            updated_world_state=updated_state,
            is_ending=is_ending,
            mode=mode,
            auto_advance_delay_ms=node.get("auto_advance_delay_ms"),
        )

    @staticmethod
    def _derive_mode(
        choices: list[dict[str, Any]],
        is_end: bool,
        next_node_id: str | None,
    ) -> str:
        """Derive the node mode from choice count and end status."""
        from app.story.graph import derive_node_mode
        return derive_node_mode(choices, is_end, next_node_id)

    def build_context(
        self,
        session_id: str,
        node: dict[str, Any],
        world_state: dict[str, Any],
        user_input: str | None = None,
        history: list[dict[str, str]] | None = None,
        scenario_id: str | None = None,
    ) -> StoryContext:
        """Build a StoryContext from runtime data."""
        return StoryContext(
            session_id=session_id,
            current_node=node,
            world_state=world_state,
            user_input=user_input,
            history=history or [],
            scenario_id=scenario_id,
            available_choices=node.get("choices", []),
        )
