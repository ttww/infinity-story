"""Simulation engine for the Admin UI Simulation View (Spec §8.1.5).

Provides a deterministic, LLM-free test run through a story draft graph.
The simulation:
  - Starts at the start node
  - Displays the scene (goal, location, characters, reveals, mood)
  - Shows available choices
  - Applies choice state_updates to the world state
  - Computes a state-diff (what changed)
  - Tracks the path (sequence of visited nodes)
  - Detects endings

This is used for Pre-Freigabe-Pruefung (pre-approval review) —
an editor walks through the story graph to verify flow and
state changes before approving a draft for publication.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any


# ── Data structures ─────────────────────────────────────────────────


@dataclass
class StateDiff:
    """Difference between two world states.

    - added: keys that appeared (key → new_value)
    - changed: keys that changed value (key → (old_value, new_value))
    - removed: keys that were removed (key → old_value)
    """

    added: dict[str, Any] = field(default_factory=dict)
    changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    removed: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.changed or self.removed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "changed": [
                {"key": k, "old": v[0], "new": v[1]} for k, v in self.changed.items()
            ],
            "removed": self.removed,
        }


@dataclass
class SimulationStep:
    """One step in the simulation — a visited node with its context."""

    node_id: str
    node_title: str
    scene: dict[str, Any]  # scene data: goal, location, characters, etc.
    choices: list[dict[str, Any]]
    is_ending: bool
    world_state_before: dict[str, Any]
    world_state_after: dict[str, Any]
    state_diff: StateDiff
    selected_choice_id: str | None = None
    selected_choice_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_title": self.node_title,
            "scene": self.scene,
            "choices": self.choices,
            "is_ending": self.is_ending,
            "world_state_before": self.world_state_before,
            "world_state_after": self.world_state_after,
            "state_diff": self.state_diff.to_dict(),
            "selected_choice_id": self.selected_choice_id,
            "selected_choice_label": self.selected_choice_label,
        }


@dataclass
class SimulationState:
    """Full state of a simulation run."""

    draft_id: str
    graph: dict[str, Any]
    current_node_id: str
    world_state: dict[str, Any] = field(default_factory=dict)
    path: list[str] = field(default_factory=list)
    steps: list[SimulationStep] = field(default_factory=list)
    is_ended: bool = False
    step_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        node = self.graph.get("nodes", {}).get(self.current_node_id, {})
        return {
            "draft_id": self.draft_id,
            "current_node_id": self.current_node_id,
            "current_node": _build_scene_data(self.current_node_id, node),
            "world_state": self.world_state,
            "path": self.path,
            "steps": [s.to_dict() for s in self.steps],
            "is_ended": self.is_ended,
            "step_count": self.step_count,
            "available_choices": _get_choices(node),
        }


# ── Engine ───────────────────────────────────────────────────────────


class SimulationEngine:
    """Runs a deterministic simulation through a story graph.

    The engine is stateless between requests — the frontend tracks the
    path and world state, and sends them with each request. The engine
    processes one step at a time:
      - start: initializes from the start node
      - choose: applies a choice, advances to the next node
    """

    MAX_STEPS = 200  # safety limit to prevent infinite loops

    @staticmethod
    def start(graph: dict[str, Any]) -> dict[str, Any]:
        """Initialize a simulation at the start node.

        Returns the initial simulation state as a dict.
        """
        start_id = graph.get("start_node_id")
        if not start_id:
            # Find by type
            for nid, node in graph.get("nodes", {}).items():
                if isinstance(node, dict) and (
                    node.get("type") == "start" or node.get("is_start")
                ):
                    start_id = nid
                    break
        if not start_id:
            nodes = graph.get("nodes", {})
            if nodes:
                start_id = next(iter(nodes))
        if not start_id:
            return {"error": "No start node found in graph"}

        nodes = graph.get("nodes", {})
        node = nodes.get(start_id, {})

        # Build initial scene
        scene = _build_scene_data(start_id, node)
        choices = _get_choices(node)
        is_ending = _is_ending(node)

        # Apply any node-level state_updates as initial state
        world_state = {}
        node_updates = node.get("state_updates", {})
        if node_updates:
            world_state = _apply_state_updates(world_state, node_updates)

        state_diff = _compute_diff({}, world_state)

        step = SimulationStep(
            node_id=start_id,
            node_title=node.get("title", start_id),
            scene=scene,
            choices=choices,
            is_ending=is_ending,
            world_state_before={},
            world_state_after=world_state,
            state_diff=state_diff,
        )

        return {
            "current_node_id": start_id,
            "current_node": scene,
            "world_state": world_state,
            "path": [start_id],
            "steps": [step.to_dict()],
            "is_ended": is_ending,
            "step_count": 1,
            "available_choices": choices,
            "state_diff": state_diff.to_dict(),
        }

    @staticmethod
    def choose(
        graph: dict[str, Any],
        current_node_id: str,
        choice_id: str,
        world_state: dict[str, Any],
        path: list[str],
        step_count: int = 0,
    ) -> dict[str, Any]:
        """Process a choice and advance to the next node.

        Returns the updated simulation state.
        """
        if step_count >= SimulationEngine.MAX_STEPS:
            return {"error": "Maximum simulation steps exceeded (possible loop)"}

        nodes = graph.get("nodes", {})
        current_node = nodes.get(current_node_id)
        if not current_node:
            return {"error": f"Node '{current_node_id}' not found"}

        choices = _get_choices(current_node)
        selected = None
        for ch in choices:
            if ch.get("id") == choice_id:
                selected = ch
                break

        if selected is None:
            return {"error": f"Choice '{choice_id}' not found in node '{current_node_id}'"}

        next_node_id = selected.get("next_node_id")
        if not next_node_id:
            return {"error": f"Choice '{choice_id}' has no next_node_id"}

        if next_node_id not in nodes:
            return {"error": f"Next node '{next_node_id}' not found in graph"}

        # Apply state updates from the choice
        choice_updates = selected.get("state_updates", {})
        world_state_before = copy.deepcopy(world_state)
        new_world_state = _apply_state_updates(world_state, choice_updates)

        # Apply node-level state_updates from the target node
        target_node = nodes[next_node_id]
        node_updates = target_node.get("state_updates", {})
        if node_updates:
            new_world_state = _apply_state_updates(new_world_state, node_updates)

        state_diff = _compute_diff(world_state_before, new_world_state)

        # Build scene for the next node
        scene = _build_scene_data(next_node_id, target_node)
        next_choices = _get_choices(target_node)
        is_ending = _is_ending(target_node)

        new_path = path + [next_node_id]

        step = SimulationStep(
            node_id=next_node_id,
            node_title=target_node.get("title", next_node_id),
            scene=scene,
            choices=next_choices,
            is_ending=is_ending,
            world_state_before=world_state_before,
            world_state_after=new_world_state,
            state_diff=state_diff,
            selected_choice_id=choice_id,
            selected_choice_label=selected.get("label", ""),
        )

        return {
            "current_node_id": next_node_id,
            "current_node": scene,
            "world_state": new_world_state,
            "path": new_path,
            "steps": [step.to_dict()],
            "is_ended": is_ending,
            "step_count": step_count + 1,
            "available_choices": next_choices,
            "state_diff": state_diff.to_dict(),
            "selected_choice": {
                "id": choice_id,
                "label": selected.get("label", ""),
                "next_node_id": next_node_id,
            },
        }

    @staticmethod
    def get_full_state(
        graph: dict[str, Any],
        path: list[str],
        world_state: dict[str, Any],
        step_count: int = 0,
    ) -> dict[str, Any]:
        """Reconstruct the full simulation state from a path.

        Used when the page reloads and needs to restore the simulation
        from stored path + world state.
        """
        if not path:
            return SimulationEngine.start(graph)

        current_node_id = path[-1]
        nodes = graph.get("nodes", {})
        node = nodes.get(current_node_id, {})

        scene = _build_scene_data(current_node_id, node)
        choices = _get_choices(node)
        is_ending = _is_ending(node)

        return {
            "current_node_id": current_node_id,
            "current_node": scene,
            "world_state": world_state,
            "path": path,
            "steps": [],
            "is_ended": is_ending,
            "step_count": step_count,
            "available_choices": choices,
        }


# ── Helpers ──────────────────────────────────────────────────────────


def _build_scene_data(node_id: str, node: dict[str, Any]) -> dict[str, Any]:
    """Build the scene display data from a node (Spec §8.1.5)."""
    return {
        "id": node.get("id", node_id),
        "title": node.get("title", node_id),
        "type": node.get("type", "scene"),
        "act": node.get("act", 1),
        "location": node.get("location", ""),
        "scene_goal": node.get("scene_goal", ""),
        "characters": node.get("characters", []),
        "mood": node.get("mood", ""),
        "reveals": node.get("reveals", []),
        "known_facts": node.get("known_facts", []),
        "is_start": node.get("is_start", False) or node.get("type") == "start",
        "is_end": _is_ending(node),
        "quality_notes": node.get("quality_notes", []),
    }


def _get_choices(node: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract choices from a node, ensuring each has id/label/next_node_id."""
    raw = node.get("choices", []) or []
    result = []
    for ch in raw:
        if not isinstance(ch, dict):
            continue
        result.append({
            "id": ch.get("id", ""),
            "label": ch.get("label", ""),
            "next_node_id": ch.get("next_node_id"),
            "state_updates": ch.get("state_updates", {}),
        })
    return result


def _is_ending(node: dict[str, Any]) -> bool:
    """Check if a node is an ending node."""
    return (
        node.get("is_end", False)
        or node.get("type") in ("end", "ending")
        or not node.get("choices")
    )


def _apply_state_updates(
    world_state: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Apply state updates to a copy of world_state.

    Supports dotted-path updates (e.g. "flags.answered_signal": true).
    """
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


def _compute_diff(
    old: dict[str, Any],
    new: dict[str, Any],
) -> StateDiff:
    """Compute the difference between two world states."""
    diff = StateDiff()

    all_keys = set(old.keys()) | set(new.keys())
    for key in all_keys:
        in_old = key in old
        in_new = key in new

        if in_new and not in_old:
            diff.added[key] = new[key]
        elif in_old and not in_new:
            diff.removed[key] = old[key]
        elif old[key] != new[key]:
            diff.changed[key] = (old[key], new[key])

    return diff
