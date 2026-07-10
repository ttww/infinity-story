"""Integration test: DB-published scenarios appear in the Start flow.

Verifies that POST /api/message with "Start" lists both file-based
and DB-published scenarios, and that a DB-published scenario can be
selected and started.
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.persistence.database import close_db, get_session_factory, init_db


@pytest.fixture
async def client():
    from app.core.config import get_settings
    get_settings.cache_clear()
    await init_db()
    from app.main import app as application
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await close_db()


def _make_graph():
    """Create a minimal valid scenario graph for publishing."""
    return {
        "id": "test_published",
        "title": "Test Published Story",
        "genre": "mystery",
        "start_node_id": "start",
        "nodes": {
            "start": {
                "id": "start",
                "title": "The Beginning",
                "type": "start",
                "act": 1,
                "scene_goal": "The adventure begins.",
                "location": "A dark room",
                "characters": ["Hero"],
                "reveals": ["Something is wrong."],
                "choices": [
                    {"id": "go_north", "label": "Go north", "next_node_id": "end_1"},
                    {"id": "go_south", "label": "Go south", "next_node_id": "end_2"},
                ],
                "is_start": True,
                "is_end": False,
            },
            "end_1": {
                "id": "end_1",
                "title": "Ending 1",
                "type": "end",
                "act": 3,
                "scene_goal": "You found the exit.",
                "location": "Exit",
                "characters": ["Hero"],
                "reveals": ["The mystery is solved."],
                "choices": [],
                "is_start": False,
                "is_end": True,
            },
            "end_2": {
                "id": "end_2",
                "title": "Ending 2",
                "type": "end",
                "act": 3,
                "scene_goal": "You are lost forever.",
                "location": "Darkness",
                "characters": ["Hero"],
                "reveals": ["The mystery remains."],
                "choices": [],
                "is_start": False,
                "is_end": True,
            },
        },
    }


async def _insert_published_scenario():
    """Insert a published scenario via ORM (creates prerequisite draft row)."""
    from app.models.story_draft import StoryDraft
    from app.models.published_scenario import PublishedScenario

    factory = get_session_factory()
    graph = _make_graph()
    async with factory() as session:
        # Create prerequisite draft row (FK target)
        draft = StoryDraft(
            id="draft_test_001",
            title="Test Published Story",
            genre="mystery",
            tone="dark",
            language="de",
            target_age="16+",
            brief_json=json.dumps({"title": "Test Published Story"}),
            status="published",
        )
        session.add(draft)
        await session.flush()

        # Create published scenario
        scenario = PublishedScenario(
            id="test_published",
            draft_id="draft_test_001",
            title="Test Published Story",
            genre="mystery",
            graph_json=json.dumps(graph, ensure_ascii=False),
        )
        session.add(scenario)
        await session.commit()


@pytest.mark.asyncio
async def test_start_lists_db_published_scenarios(client):
    """POST /api/message 'Start' must include DB-published scenarios."""
    await _insert_published_scenario()

    response = await client.post(
        "/api/message",
        json={
            "channel": "whatsapp_mock",
            "user_id": "test_user_pub_1",
            "message": "Start",
        },
    )
    assert response.status_code == 200
    data = response.json()
    msg = data["messages"][0]
    # The DB-published scenario must appear in the list
    assert "Test Published Story" in msg


@pytest.mark.asyncio
async def test_select_db_published_scenario_by_name(client):
    """Selecting a DB-published scenario by name starts a session."""
    await _insert_published_scenario()

    response = await client.post(
        "/api/message",
        json={
            "channel": "whatsapp_mock",
            "user_id": "test_user_pub_2",
            "message": "Test Published Story",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("session_id") is not None
    assert data.get("scene") is not None
    assert data["scene"]["scene_text"]


@pytest.mark.asyncio
async def test_file_based_still_listed(client):
    """File-based scenarios (helios) must still appear alongside DB-published."""
    await _insert_published_scenario()

    response = await client.post(
        "/api/message",
        json={
            "channel": "whatsapp_mock",
            "user_id": "test_user_pub_3",
            "message": "Start",
        },
    )
    assert response.status_code == 200
    msg = response.json()["messages"][0]
    # File-based scenario must still be there
    assert "Helios" in msg or "helios" in msg.lower()
    # DB-published scenario must also be there
    assert "Test Published Story" in msg


@pytest.mark.asyncio
async def test_get_api_scenarios_merges_sources(client):
    """GET /api/scenarios must return both file-based and DB-published."""
    await _insert_published_scenario()

    response = await client.get("/api/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    ids = [s["id"] for s in scenarios]
    # File-based
    assert "helios" in ids
    # DB-published
    assert "test_published" in ids
