"""Tests for the node enrichment feature (Spec §8.1.6 — Knoten ausschmücken/erweitern).

Tests:
- scene_text field on nodes (stored narrative text)
- choice.rationale field (why a choice matters)
- Node split API + admin UI
- Story templates API + admin UI
- Graph diff with new fields
- Model/graph serialization with new fields
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.story.graph_diff import compute_graph_diff
from app.story.story_templates import (
    list_templates,
    get_template,
    apply_template_to_node as _apply,
    list_categories,
)


# ── Model / Serialization Tests ─────────────────────────────────────


class TestStoryNodeNewFields:
    """Test that StoryNode and Choice models handle the new fields."""

    def test_story_node_has_scene_text(self):
        from app.models import StoryNode
        node = StoryNode(id="n1", title="Test", scene_text="This is the narrative.")
        assert node.scene_text == "This is the narrative."

    def test_story_node_scene_text_defaults_empty(self):
        from app.models import StoryNode
        node = StoryNode(id="n1", title="Test")
        assert node.scene_text == ""

    def test_choice_has_rationale(self):
        from app.models import Choice
        choice = Choice(id="c1", label="Go", rationale="This reveals the character's loyalty")
        assert choice.rationale == "This reveals the character's loyalty"

    def test_choice_rationale_defaults_empty(self):
        from app.models import Choice
        choice = Choice(id="c1", label="Go")
        assert choice.rationale == ""

    def test_story_node_with_all_new_fields(self):
        from app.models import StoryNode, Choice
        choices = [Choice(id="c1", label="Stay", rationale="Safety")]
        node = StoryNode(
            id="n1", title="Test", scene_text="Long text", choices=choices,
        )
        assert node.scene_text == "Long text"
        assert node.choices[0].rationale == "Safety"

    def test_graph_round_trip_with_new_fields(self):
        from app.story.graph import load_graph_from_dict, graph_to_dict
        data = {
            "title": "Test",
            "start_node_id": "n1",
            "nodes": {
                "n1": {
                    "id": "n1", "title": "Node 1", "type": "scene", "act": 1,
                    "scene_goal": "Goal", "scene_text": "Long narrative text",
                    "location": "Room", "characters": ["Hero"],
                    "mood": "tense", "known_facts": [], "reveals": [],
                    "choices": [{"id": "c1", "label": "Go", "next_node_id": None, "rationale": "Important"}],
                    "quality_notes": [], "state_updates": {},
                }
            }
        }
        graph = load_graph_from_dict(data)
        assert graph.nodes["n1"].scene_text == "Long narrative text"
        assert graph.nodes["n1"].choices[0].rationale == "Important"

        # Round-trip back to dict
        out = graph_to_dict(graph)
        assert out["nodes"]["n1"]["scene_text"] == "Long narrative text"
        assert out["nodes"]["n1"]["choices"][0]["rationale"] == "Important"


# ── Graph Diff Tests with New Fields ────────────────────────────────


class TestGraphDiffNewFields:
    """Test that graph diff captures scene_text and rationale changes."""

    def test_scene_text_change_detected(self):
        old = {"nodes": {"n1": {"title": "A", "act": 1, "scene_text": "old text"}}}
        new = {"nodes": {"n1": {"title": "A", "act": 1, "scene_text": "new text"}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["modified_nodes"]) == 1
        changes = diff["modified_nodes"][0]["changes"]
        assert any(c["field"] == "scene_text" for c in changes)

    def test_rationale_change_in_choice_detected(self):
        old = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "Go", "rationale": "old"}]}}}
        new = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "Go", "rationale": "new"}]}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["modified_choices"]) == 1
        changes = diff["modified_choices"][0]["changes"]
        assert any(c["field"] == "rationale" for c in changes)

    def test_scene_text_no_change_no_diff(self):
        old = {"nodes": {"n1": {"title": "A", "act": 1, "scene_text": "same"}}}
        new = {"nodes": {"n1": {"title": "A", "act": 1, "scene_text": "same"}}}
        diff = compute_graph_diff(old, new)
        assert diff["modified_nodes"] == []


# ── Story Template Tests ────────────────────────────────────────────


class TestStoryTemplates:
    """Test the story template system."""

    def test_list_templates_returns_list(self):
        templates = list_templates()
        assert len(templates) > 0
        for t in templates:
            assert "id" in t
            assert "name" in t
            assert "description" in t
            assert "category" in t

    def test_list_categories(self):
        cats = list_categories()
        assert len(cats) > 0
        assert "atmosphere" in cats
        assert "character" in cats
        assert "plot" in cats

    def test_get_template_valid_id(self):
        tpl = get_template("arrival_at_location")
        assert tpl is not None
        assert tpl["id"] == "arrival_at_location"
        assert tpl["name"] == "Ankunft am Ort"
        assert "fields" in tpl

    def test_get_template_invalid_id(self):
        tpl = get_template("nonexistent_template")
        assert tpl is None

    def test_apply_template_returns_fields(self):
        fields = _apply("tension_rising")
        assert fields is not None
        assert "scene_text" in fields
        assert "mood" in fields

    def test_apply_template_invalid_returns_none(self):
        fields = _apply("nonexistent")
        assert fields is None

    def test_apply_template_does_not_mutate_original(self):
        fields1 = _apply("confrontation")
        fields1["scene_text"] = "MODIFIED"
        fields2 = _apply("confrontation")
        assert fields2["scene_text"] != "MODIFIED"


# ── API Endpoint Tests ──────────────────────────────────────────────


class TestNodeEnrichmentAPI:
    """Test the REST API endpoints for node enrichment."""

    @pytest.fixture
    async def client(self):
        from app.main import app
        from app.persistence.database import init_db, close_db
        await init_db()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        await close_db()

    async def _create_draft(self, client) -> str:
        from app.services.story_authoring_agent import get_authoring_agent
        from app.persistence.authoring_repositories import StoryDraftRepository, StoryDraftVersionRepository
        from app.persistence.database import get_session_factory
        from app.models.enums import DraftStatus

        agent = get_authoring_agent(dummy=True)
        brief = {"title": "Test", "genre": "science_fiction", "tone": "dark_mystery"}
        outline = await agent.generate_outline(brief)
        graph = await agent.generate_graph(outline)

        async with get_session_factory()() as session:
            draft_repo = StoryDraftRepository(session)
            version_repo = StoryDraftVersionRepository(session)
            draft = await draft_repo.create(
                title="Test", genre="science_fiction", tone="dark_mystery",
                language="de", target_age="16+", brief=brief,
            )
            await version_repo.create(
                draft_id=draft.id, graph=graph, outline=outline,
                created_by="dummy_agent", notes="Initial",
            )
            await draft_repo.update_status(draft.id, DraftStatus.GENERATING)
            await draft_repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
            await session.commit()
            return draft.id

    # ── scene_text + rationale via update_node ──

    @pytest.mark.asyncio
    async def test_update_node_scene_text(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
            json={"scene_text": "Es war dunkel und kalt in der Nacht."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "node_001"
        assert data["version_number"] == 2

    @pytest.mark.asyncio
    async def test_update_node_scene_text_persists(self, client):
        draft_id = await self._create_draft(client)
        await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
            json={"scene_text": "Ein langes Narrative."},
        )
        # Verify it persisted by getting the graph
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}/diff")
        assert resp.status_code == 200
        diff = resp.json()["diff"]
        # scene_text should show in modified_nodes
        modified = diff.get("modified_nodes", [])
        assert any(
            any(c["field"] == "scene_text" for c in m["changes"])
            for m in modified
        )

    @pytest.mark.asyncio
    async def test_update_choice_rationale(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/choices/answer_signal",
            json={"rationale": "Diese Wahl bestimmt die Loyalität zum Captain."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "node_001"

    @pytest.mark.asyncio
    async def test_update_choice_rationale_persists(self, client):
        draft_id = await self._create_draft(client)
        await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/choices/answer_signal",
            json={"rationale": "Wichtig für die Geschichte."},
        )
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}/diff")
        assert resp.status_code == 200
        diff = resp.json()["diff"]
        modified_choices = diff.get("modified_choices", [])
        assert any(
            any(c["field"] == "rationale" for c in m["changes"])
            for m in modified_choices
        )

    # ── Node Split API ──

    @pytest.mark.asyncio
    async def test_split_node_api(self, client):
        draft_id = await self._create_draft(client)
        # First, set scene_text on a node
        text = "Erster Teil des Textes. " + "Zweiter Teil des Textes. " + "Dritter Teil."
        await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
            json={"scene_text": text},
        )
        # Split at position 23 and 45 (roughly between parts)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/split",
            json={"split_points": [23, 45]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["original_node_id"] == "node_001"
        assert len(data["new_node_ids"]) == 3  # 2 split points = 3 segments
        # The new nodes should be in the diff's added_nodes
        assert len(data["diff"]["added_nodes"]) == 2

    @pytest.mark.asyncio
    async def test_split_node_no_text_api(self, client):
        draft_id = await self._create_draft(client)
        # node_001 may have scene_goal but let's ensure it has no scene_text
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/split",
            json={"split_points": [10]},
        )
        # Should either work (falling back to scene_goal) or fail with 400
        assert resp.status_code in (200, 400)

    @pytest.mark.asyncio
    async def test_split_node_invalid_split_points(self, client):
        draft_id = await self._create_draft(client)
        text = "Some text here."
        await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
            json={"scene_text": text},
        )
        # Split point beyond text length
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/split",
            json={"split_points": [9999]},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_split_node_with_titles(self, client):
        draft_id = await self._create_draft(client)
        text = "Part one. Part two. Part three."
        await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
            json={"scene_text": text},
        )
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/split",
            json={
                "split_points": [9, 19],
                "titles": ["Beginning", "Middle", "End"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["new_node_ids"]) == 3

    @pytest.mark.asyncio
    async def test_split_node_wrong_title_count(self, client):
        draft_id = await self._create_draft(client)
        text = "Part one. Part two. Part three."
        await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
            json={"scene_text": text},
        )
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/split",
            json={
                "split_points": [9, 19],
                "titles": ["Only one title"],
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_split_node_not_found(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/nonexistent/split",
            json={"split_points": [10]},
        )
        assert resp.status_code == 404

    # ── Story Templates API ──

    @pytest.mark.asyncio
    async def test_list_templates_api(self, client):
        resp = await client.get("/api/admin/story-templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert "categories" in data
        assert len(data["templates"]) > 0

    @pytest.mark.asyncio
    async def test_get_template_api(self, client):
        resp = await client.get("/api/admin/story-templates/tension_rising")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "tension_rising"
        assert "fields" in data

    @pytest.mark.asyncio
    async def test_get_template_not_found_api(self, client):
        resp = await client.get("/api/admin/story-templates/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_apply_template_api(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/apply-template",
            json={"template_id": "tension_rising"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "node_001"
        assert data["template_id"] == "tension_rising"
        assert "applied_fields" in data
        assert "scene_text" in data["applied_fields"]

    @pytest.mark.asyncio
    async def test_apply_template_with_override(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/apply-template",
            json={
                "template_id": "tension_rising",
                "fields_override": {"mood": "extreme_tension"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "mood" in data["applied_fields"]

    @pytest.mark.asyncio
    async def test_apply_template_not_found(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/apply-template",
            json={"template_id": "nonexistent"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_apply_template_node_not_found(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/nonexistent/apply-template",
            json={"template_id": "tension_rising"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_apply_template_with_reveals(self, client):
        draft_id = await self._create_draft(client)
        # Apply betrayal_reveal which has reveals
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_002/apply-template",
            json={"template_id": "betrayal_reveal"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "reveals" in data["applied_fields"]


# ── Admin UI Route Tests ─────────────────────────────────────────────


class TestAdminUINodeEnrichment:
    """Test the admin UI form-based routes for node enrichment."""

    @pytest.fixture
    async def admin_client(self):
        from app.admin_ui.app import admin_app
        from app.persistence.database import init_db, close_db
        await init_db()
        transport = ASGITransport(app=admin_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        await close_db()

    async def _create_draft(self) -> str:
        from app.services.story_authoring_agent import get_authoring_agent
        from app.persistence.authoring_repositories import StoryDraftRepository, StoryDraftVersionRepository
        from app.persistence.database import get_session_factory
        from app.models.enums import DraftStatus

        agent = get_authoring_agent(dummy=True)
        brief = {"title": "Test", "genre": "science_fiction", "tone": "dark_mystery"}
        outline = await agent.generate_outline(brief)
        graph = await agent.generate_graph(outline)

        async with get_session_factory()() as session:
            draft_repo = StoryDraftRepository(session)
            version_repo = StoryDraftVersionRepository(session)
            draft = await draft_repo.create(
                title="Test", genre="science_fiction", tone="dark_mystery",
                language="de", target_age="16+", brief=brief,
            )
            await version_repo.create(
                draft_id=draft.id, graph=graph, outline=outline,
                created_by="dummy_agent", notes="Initial",
            )
            await draft_repo.update_status(draft.id, DraftStatus.GENERATING)
            await draft_repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)
            await session.commit()
            return draft.id

    @pytest.mark.asyncio
    async def test_edit_node_with_scene_text(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/node_001/edit",
            data={
                "title": "Updated",
                "type": "start",
                "act": "1",
                "scene_goal": "Goal",
                "scene_text": "Eine lange narrative Beschreibung.",
                "location": "Ship",
                "mood": "tense",
                "characters": "Hero",
                "reveals": "",
                "quality_notes": "",
            },
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_edit_choice_with_rationale(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/node_001/choices/answer_signal/edit",
            data={
                "label": "Signal beantworten",
                "next_node_id": "node_003",
                "rationale": "Diese Wahl zeigt Vertrauen und riskiert Entdeckung.",
            },
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_add_node_with_scene_text(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/add",
            data={
                "title": "New Scene",
                "type": "scene",
                "act": "2",
                "scene_goal": "Goal",
                "scene_text": "Langer narrativer Text für die Szene.",
                "location": "Station",
                "characters": "Hero",
                "mood": "dark",
            },
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_split_node_form(self, admin_client):
        draft_id = await self._create_draft()
        text = "Erster Abschnitt. Zweiter Abschnitt. Dritter Abschnitt."
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/node_001/split",
            data={
                "split_text": text,
                "split_positions": "16,33",
            },
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_split_node_form_with_titles(self, admin_client):
        draft_id = await self._create_draft()
        text = "Opening scene text. Middle scene text. End scene text."
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/node_001/split",
            data={
                "split_text": text,
                "split_positions": "18,36",
                "titles": "Opening,Middle,End",
            },
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_apply_template_form(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/node_001/apply-template",
            data={"template_id": "calm_before_storm"},
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_list_templates_admin_api(self, admin_client):
        resp = await admin_client.get("/story-templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert len(data["templates"]) > 0

    @pytest.mark.asyncio
    async def test_get_template_admin_api(self, admin_client):
        resp = await admin_client.get("/story-templates/confrontation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "confrontation"

    @pytest.mark.asyncio
    async def test_get_template_not_found_admin(self, admin_client):
        resp = await admin_client.get("/story-templates/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_draft_detail_has_split_button(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "showSplitModal" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_has_template_button(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "showTemplateModal" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_has_split_modal(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "split-modal" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_has_template_modal(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "template-modal" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_has_scene_text_field(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "scene_text" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_has_rationale_field(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "rationale" in resp.text
