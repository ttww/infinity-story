
import json
from pathlib import Path
from typing import Any
from app.models import StoryNode, StoryGraph, Choice, WorldState
from app.core.config import SCENARIOS_DIR


def derive_node_mode(
    choices: list,
    is_end: bool = False,
    next_node_id: str | None = None,
) -> str:
    """Derive the interaction mode of a node from its data.

    Returns one of:
    - "ending":       the story ends here (is_end, or 0 choices + no next_node_id)
    - "auto_advance": 0 choices, next_node_id set — auto-advance after a delay
    - "single_path":  exactly 1 choice — implicit continuation
    - "multi_choice":  2+ choices — explicit user decision
    """
    if is_end:
        return "ending"
    n = len(choices)
    if n == 0:
        if next_node_id is None:
            return "ending"
        return "auto_advance"
    if n == 1:
        return "single_path"
    return "multi_choice"


def load_graph_from_dict(data: dict) -> StoryGraph:
    nodes = {}
    raw_nodes = data.get("nodes", {})
    if isinstance(raw_nodes, dict):
        for nid, ndata in raw_nodes.items():
            choices = [Choice(**c) for c in ndata.get("choices", [])]
            nodes[nid] = StoryNode(
                id=nid, title=ndata.get("title", ""), type=ndata.get("type", "scene"),
                act=ndata.get("act", 1), scene_goal=ndata.get("scene_goal", ""),
                scene_text=ndata.get("scene_text", ""),
                location=ndata.get("location", ""), characters=ndata.get("characters", []),
                mood=ndata.get("mood", ""), known_facts=ndata.get("known_facts", []),
                reveals=ndata.get("reveals", []), choices=choices,
                next_node_id=ndata.get("next_node_id"),
                auto_advance_delay_ms=ndata.get("auto_advance_delay_ms"),
                quality_notes=ndata.get("quality_notes", []), state_updates=ndata.get("state_updates", {}),
                is_start=ndata.get("is_start", False), is_end=ndata.get("is_end", False),
            )
    return StoryGraph(nodes=nodes, start_node_id=data.get("start_node_id"), title=data.get("title", ""), genre=data.get("genre", ""), tone=data.get("tone", ""))


def load_graph_from_file(filepath: Path) -> StoryGraph:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return load_graph_from_dict(data)


def graph_to_dict(graph: StoryGraph) -> dict:
    nodes = {}
    for nid, node in graph.nodes.items():
        nodes[nid] = {
            "id": node.id, "title": node.title, "type": node.type, "act": node.act,
            "scene_goal": node.scene_goal, "scene_text": node.scene_text,
            "location": node.location,
            "characters": node.characters, "mood": node.mood,
            "known_facts": node.known_facts, "reveals": node.reveals,
            "choices": [c.model_dump() for c in node.choices],
            "next_node_id": node.next_node_id,
            "auto_advance_delay_ms": node.auto_advance_delay_ms,
            "quality_notes": node.quality_notes, "state_updates": node.state_updates,
            "is_start": node.is_start, "is_end": node.is_end,
        }
    return {"title": graph.title, "genre": graph.genre, "tone": graph.tone, "start_node_id": graph.start_node_id, "nodes": nodes}


class PublishedScenarioStore:
    @staticmethod
    async def list_scenarios() -> list[dict]:
        from app.persistence.repositories import PublishedScenarioRepository
        return await PublishedScenarioRepository.list_all()

    @staticmethod
    async def get_scenario(scenario_id: str) -> dict | None:
        from app.persistence.repositories import PublishedScenarioRepository
        return await PublishedScenarioRepository.get(scenario_id)

    @staticmethod
    async def get_graph(scenario_id: str) -> StoryGraph | None:
        scenario = await PublishedScenarioStore.get_scenario(scenario_id)
        if not scenario: return None
        graph_json = scenario.get("graph_json", "{}")
        data = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
        return load_graph_from_dict(data)


def apply_state_updates(world_state: WorldState, updates: dict[str, Any]) -> WorldState:
    ws_dict = world_state.model_dump()
    for key, value in updates.items():
        parts = key.split(".")
        target = ws_dict
        for part in parts[:-1]:
            if part not in target: target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return WorldState(**ws_dict)
