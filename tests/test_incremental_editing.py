"""Tests for incremental story graph editing (Spec §8.1.6).

Tests:
- Graph diff computation
- API endpoints: PATCH/POST/DELETE nodes, PATCH choices, regenerate, diff, versions
- Admin UI routes: edit/add/delete nodes, edit choices, regenerate, diff, versions
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.story.graph_diff import compute_graph_diff


# ── Graph Diff Tests ─────────────────────────────────────────────────────


class TestGraphDiff:
    """Test graph diff computation."""

    def test_identical_graphs_no_changes(self):
        graph = {"nodes": {"n1": {"title": "A"}}, "start_node_id": "n1"}
        diff = compute_graph_diff(graph, graph)
        assert diff["summary"] == "no changes"
        assert diff["added_nodes"] == []
        assert diff["removed_nodes"] == []
        assert diff["modified_nodes"] == []

    def test_added_node(self):
        old = {"nodes": {"n1": {"title": "A", "act": 1}}}
        new = {"nodes": {"n1": {"title": "A", "act": 1}, "n2": {"title": "B", "act": 1}}}
        diff = compute_graph_diff(old, new)
        assert diff["added_nodes"] == ["n2"]
        assert diff["removed_nodes"] == []

    def test_removed_node(self):
        old = {"nodes": {"n1": {"title": "A", "act": 1}, "n2": {"title": "B", "act": 1}}}
        new = {"nodes": {"n1": {"title": "A", "act": 1}}}
        diff = compute_graph_diff(old, new)
        assert diff["removed_nodes"] == ["n2"]
        assert diff["added_nodes"] == []

    def test_modified_node_title(self):
        old = {"nodes": {"n1": {"title": "Old", "act": 1}}}
        new = {"nodes": {"n1": {"title": "New", "act": 1}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["modified_nodes"]) == 1
        assert diff["modified_nodes"][0]["node_id"] == "n1"
        changes = diff["modified_nodes"][0]["changes"]
        assert any(c["field"] == "title" and c["old"] == "Old" and c["new"] == "New" for c in changes)

    def test_modified_node_act(self):
        old = {"nodes": {"n1": {"title": "A", "act": 1, "scene_goal": "g"}}}
        new = {"nodes": {"n1": {"title": "A", "act": 2, "scene_goal": "g"}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["modified_nodes"]) == 1
        changes = diff["modified_nodes"][0]["changes"]
        assert any(c["field"] == "act" and c["old"] == 1 and c["new"] == 2 for c in changes)

    def test_added_choice(self):
        old = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "Go"}]}}}
        new = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "Go"}, {"id": "c2", "label": "Stay"}]}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["added_choices"]) == 1
        assert diff["added_choices"][0]["node_id"] == "n1"

    def test_removed_choice(self):
        old = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "Go"}, {"id": "c2", "label": "Stay"}]}}}
        new = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "Go"}]}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["removed_choices"]) == 1
        assert diff["removed_choices"][0]["choice_id"] == "c2"

    def test_modified_choice_label(self):
        old = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "Old", "next_node_id": "n2"}]}}}
        new = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "New", "next_node_id": "n2"}]}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["modified_choices"]) == 1
        assert diff["modified_choices"][0]["choice_id"] == "c1"

    def test_modified_choice_next_node(self):
        old = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "Go", "next_node_id": "n2"}]}}}
        new = {"nodes": {"n1": {"title": "A", "choices": [{"id": "c1", "label": "Go", "next_node_id": "n3"}]}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["modified_choices"]) == 1
        changes = diff["modified_choices"][0]["changes"]
        assert any(c["field"] == "next_node_id" for c in changes)

    def test_graph_level_changes(self):
        old = {"title": "Old", "start_node_id": "n1", "nodes": {"n1": {"title": "A", "act": 1}}}
        new = {"title": "New", "start_node_id": "n1", "nodes": {"n1": {"title": "A", "act": 1}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["graph_changes"]) == 1
        assert diff["graph_changes"][0]["field"] == "title"

    def test_summary_string(self):
        old = {"nodes": {"n1": {"title": "A", "act": 1}}}
        new = {"nodes": {"n1": {"title": "B", "act": 1}, "n2": {"title": "C", "act": 1}}}
        diff = compute_graph_diff(old, new)
        assert "+1 nodes" in diff["summary"]
        assert "~1 nodes" in diff["summary"]

    def test_empty_graphs(self):
        diff = compute_graph_diff({}, {})
        assert diff["summary"] == "no changes"

    def test_multiple_field_changes(self):
        old = {"nodes": {"n1": {"title": "A", "act": 1, "scene_goal": "g1", "location": "L1"}}}
        new = {"nodes": {"n1": {"title": "B", "act": 2, "scene_goal": "g2", "location": "L2"}}}
        diff = compute_graph_diff(old, new)
        assert len(diff["modified_nodes"]) == 1
        changes = diff["modified_nodes"][0]["changes"]
        assert len(changes) == 4  # title, act, scene_goal, location
        fields = {c["field"] for c in changes}
        assert "title" in fields
        assert "act" in fields
        assert "scene_goal" in fields
        assert "location" in fields


# ── API Endpoint Tests ──────────────────────────────────────────────────


class TestIncrementalEditAPI:
    """Test the incremental editing REST API endpoints."""

    @pytest.fixture
    async def client(self):
        """Create test client with in-memory DB."""
        from app.main import app
        from app.persistence.database import init_db, close_db

        await init_db()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        await close_db()

    async def _create_draft(self, client) -> str:
        """Create a draft and return its ID."""
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
    async def test_update_node(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
            json={"title": "Updated Title"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "node_001"
        assert data["version_number"] == 2
        assert "diff" in data

    @pytest.mark.asyncio
    async def test_update_node_not_found(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/nonexistent",
            json={"title": "X"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_node(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes",
            json={"title": "New Scene", "type": "scene", "act": 2, "scene_goal": "Test goal"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["node_id"] is not None
        assert data["version_number"] == 2
        assert data["node_id"] in data["diff"]["added_nodes"]

    @pytest.mark.asyncio
    async def test_add_node_with_custom_id(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes",
            json={"id": "custom_node", "title": "Custom", "type": "scene", "act": 1},
        )
        assert resp.status_code == 201
        assert resp.json()["node_id"] == "custom_node"

    @pytest.mark.asyncio
    async def test_add_duplicate_node(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes",
            json={"id": "node_001", "title": "Dup", "type": "scene", "act": 1},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_delete_node(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.delete(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_002",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "node_002"
        assert "node_002" in data["diff"]["removed_nodes"]

    @pytest.mark.asyncio
    async def test_delete_start_node_fails(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.delete(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_delete_nonexistent_node(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.delete(
            f"/api/admin/story-drafts/{draft_id}/nodes/nonexistent",
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_choice(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/choices/answer_signal",
            json={"label": "New Label"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "node_001"

    @pytest.mark.asyncio
    async def test_update_choice_next_node(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/choices/answer_signal",
            json={"next_node_id": "node_003"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["diff"]["modified_choices"]) >= 1

    @pytest.mark.asyncio
    async def test_update_choice_not_found(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/choices/nonexistent",
            json={"label": "X"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_regenerate_node(self, client):
        draft_id = await self._create_draft(client)
        resp = await client.post(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001/regenerate",
            json={"instruction": "make it more dramatic"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_id"] == "node_001"
        assert data["version_number"] == 2

    @pytest.mark.asyncio
    async def test_get_version_diff(self, client):
        draft_id = await self._create_draft(client)
        # Make a change first
        await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
            json={"title": "Changed"},
        )
        # Get diff
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}/diff")
        assert resp.status_code == 200
        data = resp.json()
        assert data["new_version_number"] == 2
        assert data["old_version_number"] == 1
        assert "diff" in data

    @pytest.mark.asyncio
    async def test_list_versions_with_diffs(self, client):
        draft_id = await self._create_draft(client)
        # Make a change
        await client.patch(
            f"/api/admin/story-drafts/{draft_id}/nodes/node_001",
            json={"title": "Changed"},
        )
        resp = await client.get(f"/api/admin/story-drafts/{draft_id}/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["versions"]) == 2
        assert data["versions"][0]["version_number"] == 1
        assert data["versions"][1]["version_number"] == 2


# ── Admin UI Route Tests ─────────────────────────────────────────────────


class TestAdminUIIncrementalEditing:
    """Test the admin UI form-based incremental editing routes."""

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
    async def test_edit_node_form(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/node_001/edit",
            data={
                "title": "Updated via Form",
                "type": "start",
                "act": "1",
                "scene_goal": "New goal",
                "location": "New location",
                "mood": "tense",
                "characters": "Hero, Villain",
                "reveals": "Secret 1\nSecret 2",
                "quality_notes": "Note 1",
            },
        )
        assert resp.status_code == 303
        assert f"/admin/draft/{draft_id}" in resp.headers.get("location", "")

    @pytest.mark.asyncio
    async def test_add_node_form(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/add",
            data={
                "title": "New Scene",
                "type": "scene",
                "act": "2",
                "scene_goal": "Test goal",
                "location": "Forest",
                "characters": "Hero",
                "mood": "mysterious",
            },
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_delete_node_form(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/node_003/delete",
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_edit_choice_form(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/node_001/choices/answer_signal/edit",
            data={
                "label": "New Choice Label",
                "next_node_id": "node_003",
            },
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_regenerate_node_form(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.post(
            f"/draft/{draft_id}/nodes/node_001/regenerate",
            data={"instruction": "make it more dramatic"},
        )
        assert resp.status_code == 303

    @pytest.mark.asyncio
    async def test_get_diff_endpoint(self, admin_client):
        draft_id = await self._create_draft()
        # Make a change
        await admin_client.post(
            f"/draft/{draft_id}/nodes/node_001/edit",
            data={"title": "Changed", "type": "start", "act": "1"},
        )
        resp = await admin_client.get(f"/draft/{draft_id}/diff")
        assert resp.status_code == 200
        data = resp.json()
        assert "diff" in data
        assert data["new_version"] == 2

    @pytest.mark.asyncio
    async def test_get_versions_endpoint(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["versions"]) == 1
        assert data["versions"][0]["version_number"] == 1

    @pytest.mark.asyncio
    async def test_draft_detail_page_has_add_node_button(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "Add Node" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_page_has_version_history(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "Version History" in resp.text
        assert "version-history" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_page_has_edit_buttons(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "toggleEditForm" in resp.text
        assert "regenerateNode" in resp.text
        assert "deleteNode" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_page_has_add_node_modal(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "add-node-modal" in resp.text
        assert "showAddNodeModal" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_page_has_diff_display(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "loadVersionDiff" in resp.text
        assert "diff-display" in resp.text

    @pytest.mark.asyncio
    async def test_draft_detail_page_has_choice_edit(self, admin_client):
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        assert resp.status_code == 200
        assert "toggleChoiceEdit" in resp.text

    @pytest.mark.asyncio
    async def test_draft_id_declared_before_version_history_call(self, admin_client):
        """Bug fix: draftId was declared AFTER loadVersionHistory() call,
        causing a TDZ ReferenceError that crashed the entire script."""
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        text = resp.text
        # draftId must appear before loadVersionHistory() call
        draftid_pos = text.find("const draftId =")
        loadvh_pos = text.find("loadVersionHistory()")
        assert draftid_pos != -1, "const draftId declaration not found"
        assert loadvh_pos != -1, "loadVersionHistory() call not found"
        assert draftid_pos < loadvh_pos, (
            "const draftId must be declared before loadVersionHistory() is called"
        )

    @pytest.mark.asyncio
    async def test_ajax_submit_functions_exist(self, admin_client):
        """Verify AJAX helper functions are present in the template."""
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        text = resp.text
        assert "submitEditForm" in text, "submitEditForm function not found"
        assert "submitChoiceEdit" in text, "submitChoiceEdit function not found"
        assert "submitAddNode" in text, "submitAddNode function not found"
        assert "refreshNodeDetail" in text, "refreshNodeDetail function not found"
        assert "showToast" in text, "showToast function not found"

    @pytest.mark.asyncio
    async def test_no_duplicate_draftid_declaration(self, admin_client):
        """Ensure there is only one const draftId declaration (no duplicate)."""
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        text = resp.text
        count = text.count("const draftId =")
        assert count == 1, f"Expected exactly 1 'const draftId =' declaration, found {count}"

    @pytest.mark.asyncio
    async def test_edit_form_uses_ajax_not_native_submit(self, admin_client):
        """The edit form's save button should call submitEditForm, not type=submit."""
        draft_id = await self._create_draft()
        resp = await admin_client.get(f"/draft/{draft_id}")
        text = resp.text
        assert "onclick=\"submitEditForm(" in text, "Edit save button should use onclick=submitEditForm"
        assert "submitChoiceEdit(" in text, "Choice save button should use onclick=submitChoiceEdit"
        assert "onclick=\"submitAddNode()\"" in text, "Add node button should use onclick=submitAddNode"
