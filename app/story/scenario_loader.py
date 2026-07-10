"""Scenario loading utilities.

Loads published scenario JSON files from the scenarios directory
and provides access to nodes, start nodes, and graph navigation.

Also supports DB-published scenarios from the ``published_scenarios``
table, merged with the file-based scenarios so the runtime can list
and load both sources transparently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.story.graph import StoryGraph, load_graph_from_dict

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def load_scenario(scenario_id: str) -> dict[str, Any]:
    """Load a scenario by id from the scenarios directory.

    Returns the raw JSON dict.
    Raises FileNotFoundError if the scenario file does not exist.
    """
    path = _SCENARIOS_DIR / f"{scenario_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_scenarios() -> list[dict[str, str]]:
    """List all file-based scenarios with id, title, and genre."""
    results = []
    for path in sorted(_SCENARIOS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append({
            "id": data.get("id", path.stem),
            "title": data.get("title", path.stem),
            "genre": data.get("genre", ""),
        })
    return results


# -- Unified (file + DB) scenario access -------------------------------


async def list_all_scenarios(session: Any) -> list[dict[str, str]]:
    """Merge file-based and DB-published scenarios into one list.

    Returns a list of ``{id, title, genre}`` dicts.  File-based
    scenarios come first (sorted by filename), then DB-published
    scenarios (sorted by ``published_at`` descending).  Duplicates
    (same id) are deduplicated -- file-based takes precedence.
    """
    from app.models.published_scenario import PublishedScenario
    from sqlalchemy import select

    file_scenarios = list_scenarios()
    seen_ids = {s["id"] for s in file_scenarios}

    result = await session.execute(
        select(PublishedScenario).order_by(PublishedScenario.published_at.desc())
    )
    for row in result.scalars().all():
        if row.id not in seen_ids:
            file_scenarios.append({
                "id": row.id,
                "title": row.title,
                "genre": row.genre or "",
            })
            seen_ids.add(row.id)

    return file_scenarios


async def load_scenario_unified(
    scenario_id: str, session: Any
) -> dict[str, Any]:
    """Load a scenario by id from file or DB.

    Tries the file-based scenario directory first; if not found,
    falls back to the ``published_scenarios`` table.
    """
    path = _SCENARIOS_DIR / f"{scenario_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    from app.models.published_scenario import PublishedScenario
    from sqlalchemy import select

    result = await session.execute(
        select(PublishedScenario).where(PublishedScenario.id == scenario_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise FileNotFoundError(f"Scenario not found: {scenario_id}")
    data = json.loads(row.graph_json)
    # Ensure id/title/genre are set from the DB row
    data.setdefault("id", row.id)
    data.setdefault("title", row.title)
    data.setdefault("genre", row.genre or "")
    return data


def get_node(scenario: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    """Get a single node dict from a scenario by id."""
    return scenario.get("nodes", {}).get(node_id)


def get_start_node_id(scenario: dict[str, Any]) -> str | None:
    """Get the start node id for a scenario."""
    return scenario.get("start_node_id")


def get_start_node(scenario: dict[str, Any]) -> dict[str, Any] | None:
    """Get the start node dict for a scenario."""
    start_id = get_start_node_id(scenario)
    if start_id is None:
        # Fallback: scan for is_start
        for nid, node in scenario.get("nodes", {}).items():
            if node.get("is_start") or node.get("type") == "start":
                return node
        return None
    return get_node(scenario, start_id)


def scenario_to_graph(scenario: dict[str, Any]) -> StoryGraph:
    """Convert a raw scenario dict into a StoryGraph model."""
    return load_graph_from_dict(scenario)


def build_initial_world_state(scenario: dict[str, Any]) -> dict[str, Any]:
    """Build the initial world state from a scenario's metadata."""
    return {
        "genre": scenario.get("genre", ""),
        "tone": scenario.get("tone", ""),
        "language": scenario.get("language", "de"),
        "current_location": "",
        "main_character_name": "",
        "inventory": [],
        "relationships": {},
        "open_mysteries": [],
        "flags": {},
    }
