"""Tests for Admin UI Graph Visualization & Node Details (Spec §8.1.3-8.1.4).

Tests:
- Graph layout computation (positions, layers, edges)
- Problematic node identification
- Node detail enrichment (all fields, review issues, tech warnings)
- Admin UI endpoints (draft detail page, node detail API)
- SVG structure in rendered HTML
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.admin_ui.graph_layout import (
    compute_layout,
    enrich_node_detail,
    NODE_WIDTH,
    NODE_HEIGHT,
    LAYER_GAP,
    VERTICAL_GAP,
    LEFT_PADDING,
    TOP_PADDING,
)


# ── Test fixtures ─────────────────────────────────────────────────────


def _helios_graph() -> dict:
    """Load the helios graph for testing."""
    from app.services.story_authoring_agent import _load_helios_graph
    return _load_helios_graph()


def _broken_graph() -> dict:
    """A graph with various problems for testing."""
    return {
        "start_node_id": "n1",
        "nodes": {
            "n1": {
                "id": "n1", "title": "Start", "type": "start", "act": 1,
                "scene_goal": "Begin the adventure",
                "location": "Town",
                "characters": ["Hero"],
                "reveals": ["The king is dead"],
                "choices": [
                    {"id": "c1", "label": "Go north", "next_node_id": "n2"},
                    {"id": "c2", "label": "", "next_node_id": "n3"},  # empty label
                ],
                "quality_notes": ["Establish setting"],
            },
            "n2": {
                "id": "n2", "title": "Mid", "type": "scene", "act": 1,
                "scene_goal": "",  # missing goal
                "location": "Forest",
                "characters": ["Hero", "Wizard"],
                "choices": [{"id": "c3", "label": "Continue", "next_node_id": "n3"}],
                "quality_notes": [],
            },
            "n3": {
                "id": "n3", "title": "End", "type": "end", "act": 2,
                "scene_goal": "Final confrontation",
                "location": "Castle",
                "characters": ["Hero", "Villain"],
                "reveals": ["The villain was the wizard"],
                "choices": [],
                "quality_notes": ["Satisfying conclusion"],
            },
        },
    }


def _dangling_graph() -> dict:
    """A graph with dangling references."""
    return {
        "start_node_id": "n1",
        "nodes": {
            "n1": {
                "id": "n1", "title": "Start", "type": "start", "act": 1,
                "scene_goal": "Begin",
                "choices": [
                    {"id": "c1", "label": "Go", "next_node_id": "MISSING"},
                ],
                "quality_notes": ["ok"],
            },
        },
    }


# ── Layout computation tests ──────────────────────────────────────────


class TestComputeLayout:
    """Test graph layout computation (Spec §8.1.3)."""

    def test_layout_returns_required_keys(self):
        layout = compute_layout(_helios_graph())
        assert "nodes" in layout
        assert "edges" in layout
        assert "svg_width" in layout
        assert "svg_height" in layout
        assert "start_node_id" in layout
        assert "end_node_ids" in layout
        assert "problematic_node_ids" in layout

    def test_layout_node_count(self):
        layout = compute_layout(_helios_graph())
        assert len(layout["nodes"]) == 5

    def test_layout_edge_count(self):
        layout = compute_layout(_helios_graph())
        # node_001: 2 choices, node_002: 2, node_003: 2, node_004: 2, node_005: 0
        assert len(layout["edges"]) == 8

    def test_layout_start_node_identified(self):
        layout = compute_layout(_helios_graph())
        assert layout["start_node_id"] == "node_001"

    def test_layout_end_nodes_identified(self):
        layout = compute_layout(_helios_graph())
        assert "node_005" in layout["end_node_ids"]

    def test_layout_node_positions(self):
        layout = compute_layout(_helios_graph())
        node_map = {n["id"]: n for n in layout["nodes"]}

        # Start node should be at the leftmost layer
        start = node_map["node_001"]
        assert start["x"] == LEFT_PADDING
        assert start["y"] == TOP_PADDING

        # All nodes should have correct dimensions
        for node in layout["nodes"]:
            assert node["width"] == NODE_WIDTH
            assert node["height"] == NODE_HEIGHT

    def test_layout_layers_are_progressive(self):
        """Nodes deeper in the graph should be further right."""
        layout = compute_layout(_helios_graph())
        node_map = {n["id"]: n for n in layout["nodes"]}

        # Start node is layer 0 (leftmost)
        assert node_map["node_001"]["x"] < node_map["node_002"]["x"]
        assert node_map["node_001"]["x"] < node_map["node_003"]["x"]

        # node_005 (end) should be at the rightmost layer
        assert node_map["node_005"]["x"] > node_map["node_001"]["x"]

    def test_layout_edge_coordinates(self):
        """Edges should connect node boundaries, not centers."""
        layout = compute_layout(_helios_graph())
        node_map = {n["id"]: n for n in layout["nodes"]}

        for edge in layout["edges"]:
            from_node = node_map[edge["from"]]
            to_node = node_map[edge["to"]]

            # Edge starts at right edge of source node
            assert edge["x1"] == from_node["x"] + NODE_WIDTH
            # Edge ends at left edge of target node
            assert edge["x2"] == to_node["x"]

    def test_layout_edge_bezier_control_points(self):
        """Edges should have bezier control points for smooth curves."""
        layout = compute_layout(_helios_graph())
        for edge in layout["edges"]:
            assert "cx1" in edge
            assert "cy1" in edge
            assert "cx2" in edge
            assert "cy2" in edge

    def test_layout_empty_graph(self):
        layout = compute_layout({})
        assert layout["nodes"] == []
        assert layout["edges"] == []

    def test_layout_node_has_type_and_act(self):
        layout = compute_layout(_helios_graph())
        for node in layout["nodes"]:
            assert "type" in node
            assert "act" in node
            assert "title" in node

    def test_layout_node_flags(self):
        layout = compute_layout(_helios_graph())
        node_map = {n["id"]: n for n in layout["nodes"]}

        assert node_map["node_001"]["is_start"] is True
        assert node_map["node_001"]["is_end"] is False
        assert node_map["node_005"]["is_start"] is False
        assert node_map["node_005"]["is_end"] is True

    def test_layout_svg_dimensions(self):
        layout = compute_layout(_helios_graph())
        assert layout["svg_width"] >= 800
        assert layout["svg_height"] >= 400


class TestProblematicNodeDetection:
    """Test identification of problematic nodes (Spec §8.1.3)."""

    def test_clean_graph_no_problematic(self):
        layout = compute_layout(_helios_graph())
        assert layout["problematic_node_ids"] == []

    def test_empty_scene_goal_flagged(self):
        layout = compute_layout(_broken_graph())
        assert "n2" in layout["problematic_node_ids"]

    def test_empty_choice_label_flagged(self):
        layout = compute_layout(_broken_graph())
        # n1 has a choice with empty label
        assert "n1" in layout["problematic_node_ids"]

    def test_dangling_reference_flagged(self):
        layout = compute_layout(_dangling_graph())
        assert "n1" in layout["problematic_node_ids"]

    def test_problematic_flag_on_layout_nodes(self):
        layout = compute_layout(_broken_graph())
        node_map = {n["id"]: n for n in layout["nodes"]}
        assert node_map["n1"]["is_problematic"] is True
        assert node_map["n2"]["is_problematic"] is True
        assert node_map["n3"]["is_problematic"] is False


# ── Node detail enrichment tests ──────────────────────────────────────


class TestEnrichNodeDetail:
    """Test node detail enrichment (Spec §8.1.4)."""

    def test_basic_fields(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node)

        assert detail["id"] == "node_001"
        assert detail["title"] == "Notrufsignal"
        assert detail["type"] == "start"
        assert detail["act"] == 1
        assert detail["location"] == "Orbitalstation Helios"
        assert detail["scene_goal"].startswith("Der Spieler entdeckt")

    def test_characters(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node)
        assert "Mira" in detail["characters"]
        assert "Captain Rao" in detail["characters"]

    def test_reveals(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node)
        assert len(detail["reveals"]) == 1
        assert "Notruf" in detail["reveals"][0]

    def test_choices_with_details(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node)

        assert len(detail["choices"]) == 2
        choice = detail["choices"][0]
        assert choice["id"] == "answer_signal"
        assert choice["label"] == "Den Funkspruch beantworten"
        assert choice["next_node_id"] == "node_002"
        assert "state_updates" in choice

    def test_state_updates(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node)
        assert "state_updates" in detail
        assert isinstance(detail["state_updates"], dict)

    def test_quality_notes(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node)
        assert len(detail["quality_notes"]) == 1

    def test_is_start_flag(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node)
        assert detail["is_start"] is True
        assert detail["is_end"] is False

    def test_is_end_flag(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_005"]
        detail = enrich_node_detail("node_005", node)
        assert detail["is_end"] is True
        assert detail["is_start"] is False

    def test_review_issues_filtered_by_node(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_002"]
        review_issues = [
            {"severity": "high", "node_id": "node_002", "problem": "Pacing issue", "suggestion": "Add more tension"},
            {"severity": "medium", "node_id": "node_003", "problem": "Other node", "suggestion": ""},
        ]
        detail = enrich_node_detail("node_002", node, review_issues, [], [])

        assert len(detail["review_issues"]) == 1
        assert detail["review_issues"][0]["problem"] == "Pacing issue"

    def test_tech_warnings_missing_scene_goal(self):
        node = {"id": "n1", "title": "Test", "type": "scene", "scene_goal": "", "choices": [], "quality_notes": []}
        detail = enrich_node_detail("n1", node)
        warning_msgs = [w["message"] for w in detail["tech_warnings"]]
        assert any("scene_goal" in m for m in warning_msgs)

    def test_tech_warnings_missing_quality_notes(self):
        node = {"id": "n1", "title": "Test", "type": "scene", "scene_goal": "Goal", "choices": [], "quality_notes": []}
        detail = enrich_node_detail("n1", node)
        warning_msgs = [w["message"] for w in detail["tech_warnings"]]
        assert any("quality_notes" in m for m in warning_msgs)

    def test_tech_warnings_empty_label(self):
        node = {
            "id": "n1", "title": "Test", "type": "scene",
            "scene_goal": "Goal", "quality_notes": ["ok"],
            "choices": [{"id": "c1", "label": "", "next_node_id": None}],
        }
        detail = enrich_node_detail("n1", node)
        warning_msgs = [w["message"] for w in detail["tech_warnings"]]
        assert any("empty label" in m for m in warning_msgs)

    def test_validation_errors_matched_to_node(self):
        node = {"id": "n1", "title": "Test", "type": "scene", "scene_goal": "Goal", "choices": [], "quality_notes": ["ok"]}
        val_errors = ["Node 'n1' has a broken reference.", "Node 'n2' has another issue."]
        detail = enrich_node_detail("n1", node, [], val_errors, [])
        assert len(detail["tech_warnings"]) >= 1
        error_msgs = [w["message"] for w in detail["tech_warnings"] if w["severity"] == "error"]
        assert any("n1" in m for m in error_msgs)

    def test_no_tech_warnings_for_clean_node(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node, [], [], [])
        assert detail["tech_warnings"] == []

    def test_mood_field(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node)
        assert "mood" in detail

    def test_known_facts_field(self):
        graph = _helios_graph()
        node = graph["nodes"]["node_001"]
        detail = enrich_node_detail("node_001", node)
        assert "known_facts" in detail
        assert isinstance(detail["known_facts"], list)


# ── Admin UI endpoint tests ───────────────────────────────────────────


class TestAdminUIEndpoints:
    """Test the admin UI FastAPI endpoints."""

    @pytest.fixture
    async def admin_client(self):
        """Create admin UI test client with in-memory DB."""
        from app.admin_ui.app import admin_app
        from app.persistence.database import init_db, close_db

        await init_db()
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        await close_db()

    async def _create_draft_via_api(self, client) -> str:
        """Create a draft via the API and return its ID."""
        from app.services.story_authoring_agent import get_authoring_agent
        from app.persistence.authoring_repositories import StoryDraftRepository, StoryDraftVersionRepository
        from app.persistence.database import get_session_factory
        from app.models.enums import DraftStatus

        agent = get_authoring_agent(dummy=True)
        brief = {
            "title": "Test Story",
            "genre": "science_fiction",
            "tone": "dark_mystery",
            "language": "de",
            "target_age": "16+",
            "node_count": 25,
            "ending_count": 3,
            "branching_level": "medium",
        }
        outline = await agent.generate_outline(brief)
        graph = await agent.generate_graph(outline)

        async with get_session_factory()() as session:
            draft_repo = StoryDraftRepository(session)
            version_repo = StoryDraftVersionRepository(session)

            draft = await draft_repo.create(
                title="Test Story",
                genre="science_fiction",
                tone="dark_mystery",
                language="de",
                target_age="16+",
                brief=brief,
            )
            await version_repo.create(
                draft_id=draft.id,
                graph=graph,
                outline=outline,
                created_by="dummy_agent",
                notes="Test version",
            )
            await draft_repo.update_status(draft.id, DraftStatus.GENERATING)
            await draft_repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
            await session.commit()
            return draft.id

    @pytest.mark.asyncio
    async def test_draft_detail_page_renders(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_draft_detail_page_contains_graph(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}")
        body = resp.text

        # Should contain SVG graph
        assert "<svg" in body
        assert "graph-svg" in body
        assert "node-rect" in body

    @pytest.mark.asyncio
    async def test_draft_detail_page_contains_node_detail_panel(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}")
        body = resp.text

        # Should contain node detail panel
        assert "node-detail-panel" in body
        assert "showNodeDetail" in body

    @pytest.mark.asyncio
    async def test_draft_detail_page_contains_zoom_pan(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}")
        body = resp.text

        # Should contain zoom/pan controls
        assert "zoomGraph" in body
        assert "resetGraphView" in body
        assert "fitGraphView" in body
        assert "graph-viewport" in body

    @pytest.mark.asyncio
    async def test_draft_detail_page_contains_legend(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}")
        body = resp.text

        assert "graph-legend" in body
        assert "Start" in body
        assert "Ende" in body
        assert "Problematic" in body

    @pytest.mark.asyncio
    async def test_draft_detail_page_contains_node_data(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}")
        body = resp.text

        # Should contain nodes_detail_json for JavaScript
        assert "nodesDetail" in body
        assert "Notrufsignal" in body

    @pytest.mark.asyncio
    async def test_node_detail_api_returns_json(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/node/node_001")
        assert resp.status_code == 200
        data = resp.json()

        assert data["id"] == "node_001"
        assert data["title"] == "Notrufsignal"
        assert data["type"] == "start"
        assert data["is_start"] is True
        assert "characters" in data
        assert "reveals" in data
        assert "choices" in data
        assert "state_updates" in data
        assert "quality_notes" in data
        assert "review_issues" in data
        assert "tech_warnings" in data

    @pytest.mark.asyncio
    async def test_node_detail_api_404_for_missing_node(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}/node/NONEXISTENT")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_node_detail_api_404_for_missing_draft(self, admin_client):
        resp = await admin_client.get("/draft/nonexistent/node/node_001")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_draft_detail_404_for_missing_draft(self, admin_client):
        resp = await admin_client.get("/draft/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_draft_list_page_renders(self, admin_client):
        await self._create_draft_via_api(admin_client)
        resp = await admin_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Test Story" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_contains_edge_paths(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}")
        body = resp.text

        # Should contain edge paths (bezier curves)
        assert "edge-path" in body
        assert "arrowhead" in body

    @pytest.mark.asyncio
    async def test_draft_detail_contains_actions_bar(self, admin_client):
        draft_id = await self._create_draft_via_api(admin_client)
        resp = await admin_client.get(f"/draft/{draft_id}")
        body = resp.text

        assert "actions-bar" in body
        assert "Review" in body
        assert "Repair" in body
        assert "Validate" in body
        assert "Approve" in body
        assert "Publish" in body

    @pytest.mark.asyncio
    async def test_brief_form_renders(self, admin_client):
        resp = await admin_client.get("/new")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_create_draft_via_form(self, admin_client):
        resp = await admin_client.post("/new", data={
            "title": "Form Test Story",
            "genre": "fantasy",
            "tone": "adventure",
            "language": "de",
            "target_age": "12+",
            "node_count": "10",
            "ending_count": "2",
            "branching_level": "low",
            "themes": "magic, friendship",
            "forbidden_content": "",
            "notes": "Test via form",
        }, follow_redirects=False)
        assert resp.status_code == 303
        assert "/draft/" in resp.headers.get("location", "")
