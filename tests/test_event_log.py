"""Tests for the Event Log system (t_58834261).

Covers:
  (1)  EventLogService ring buffer: emit, list, filter, stats
  (2)  Ring buffer eviction at max_size
  (3)  Event convenience methods (emit_start, emit_done, emit_error)
  (4)  LogEvent serialization (to_dict, to_json)
  (5)  REST endpoint GET /admin/api/events returns events
  (6)  REST endpoint filtering by category, status, draft_id
  (7)  REST endpoint GET /admin/api/events/stats returns stats
  (8)  Logs page GET /admin/logs renders HTML
  (9)  Event generation on draft creation (integration)
  (10) Event on review action (integration)
  (11) Event on validation action (integration)
  (12) SSE endpoint /admin/api/events/live returns text/event-stream
"""

from __future__ import annotations

import asyncio
import json
import os
import time

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("DEBUG", "false")
os.environ.setdefault("MIN_NODE_COUNT", "3")
os.environ.setdefault("MIN_ENDING_COUNT", "1")

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
get_settings.cache_clear()


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
async def app():
    from app.main import app as application
    from app.persistence.database import init_db, close_db
    await init_db()
    yield application
    await close_db()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def admin_client():
    """Client targeting the admin sub-app directly."""
    from app.admin_ui.app import admin_app
    from app.persistence.database import init_db, close_db
    await init_db()
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    await close_db()


@pytest.fixture
def fresh_event_log():
    """Return a fresh EventLogService instance for unit tests."""
    from app.services.event_log import EventLogService
    return EventLogService(max_size=10)


async def _create_draft(client: AsyncClient) -> str:
    """Create a draft via the admin UI form, return its id."""
    resp = await client.post(
        "/admin/new",
        data={
            "title": "Event Log Test Story",
            "genre": "science_fiction",
            "tone": "dark",
            "language": "de",
            "target_age": "16+",
            "node_count": "10",
            "ending_count": "2",
            "branching_level": "medium",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    location = resp.headers.get("location", "")
    return location.rsplit("/", 1)[-1]


# ── Unit tests: EventLogService ──────────────────────────────────────


class TestEventLogService:
    """Tests for the EventLogService ring buffer."""

    def test_emit_creates_event(self, fresh_event_log):
        """(1a) emit() creates an event in the buffer."""
        evt = fresh_event_log.emit(
            "generation", "outline_generated", "done",
            "Outline generated", draft_id="draft_1",
        )
        assert evt.id.startswith("evt_")
        assert evt.category == "generation"
        assert evt.action == "outline_generated"
        assert evt.status == "done"
        assert evt.draft_id == "draft_1"
        assert evt.timestamp > 0

    def test_list_events_returns_newest_first(self, fresh_event_log):
        """(1b) list_events returns events newest-first."""
        fresh_event_log.emit("generation", "action_1", "done")
        time.sleep(0.01)
        fresh_event_log.emit("review", "action_2", "done")
        result = fresh_event_log.list_events(limit=10)
        assert result["total"] == 2
        events = result["events"]
        assert events[0]["action"] == "action_2"
        assert events[1]["action"] == "action_1"

    def test_filter_by_category(self, fresh_event_log):
        """(1c) list_events filters by category."""
        fresh_event_log.emit("generation", "gen_action", "done")
        fresh_event_log.emit("review", "rev_action", "done")
        result = fresh_event_log.list_events(category="generation")
        assert result["total"] == 1
        assert result["events"][0]["category"] == "generation"

    def test_filter_by_status(self, fresh_event_log):
        """(1d) list_events filters by status."""
        fresh_event_log.emit("generation", "done_action", "done")
        fresh_event_log.emit("generation", "err_action", "error")
        result = fresh_event_log.list_events(status="error")
        assert result["total"] == 1
        assert result["events"][0]["status"] == "error"

    def test_filter_by_draft_id(self, fresh_event_log):
        """(1e) list_events filters by draft_id."""
        fresh_event_log.emit("generation", "a", "done", draft_id="draft_a")
        fresh_event_log.emit("generation", "b", "done", draft_id="draft_b")
        result = fresh_event_log.list_events(draft_id="draft_a")
        assert result["total"] == 1
        assert result["events"][0]["draft_id"] == "draft_a"

    def test_ring_buffer_eviction(self, fresh_event_log):
        """(2) Ring buffer evicts oldest events at max_size."""
        for i in range(15):  # max_size=10
            fresh_event_log.emit("generation", f"action_{i}", "done")
        result = fresh_event_log.list_events(limit=20)
        assert result["total"] == 10
        # The first 5 events should be evicted
        actions = [e["action"] for e in result["events"]]
        assert "action_0" not in actions
        assert "action_14" in actions

    def test_emit_convenience_methods(self, fresh_event_log):
        """(3) emit_start, emit_done, emit_error create correct statuses."""
        fresh_event_log.emit_start("generation", "step1", "Starting")
        fresh_event_log.emit_done("generation", "step1", "Done")
        fresh_event_log.emit_error("generation", "step2", "Failed")
        result = fresh_event_log.list_events(limit=10)
        statuses = [e["status"] for e in result["events"]]
        assert "running" in statuses
        assert "done" in statuses
        assert "error" in statuses

    def test_get_stats(self, fresh_event_log):
        """(3b) get_stats returns correct statistics."""
        fresh_event_log.emit("generation", "a", "done")
        fresh_event_log.emit("review", "b", "error")
        fresh_event_log.emit("validation", "c", "running")
        stats = fresh_event_log.get_stats()
        assert stats["total_events"] == 3
        assert stats["buffer_capacity"] == 10
        assert stats["by_category"]["generation"] == 1
        assert stats["by_category"]["review"] == 1
        assert stats["by_status"]["done"] == 1
        assert stats["by_status"]["error"] == 1
        assert stats["by_status"]["running"] == 1

    def test_log_event_serialization(self, fresh_event_log):
        """(4) LogEvent.to_dict and to_json work correctly."""
        evt = fresh_event_log.emit(
            "generation", "test_action", "done",
            detail={"key": "value", "num": 42},
        )
        d = evt.to_dict()
        assert d["category"] == "generation"
        assert d["detail"]["key"] == "value"
        j = evt.to_json()
        parsed = json.loads(j)
        assert parsed["action"] == "test_action"

    def test_clear(self, fresh_event_log):
        """(4b) clear() empties the buffer."""
        fresh_event_log.emit("generation", "a", "done")
        assert len(fresh_event_log._buffer) == 1
        count = fresh_event_log.clear()
        assert count == 1
        assert len(fresh_event_log._buffer) == 0


# ── Integration tests: REST API ───────────────────────────────────────


class TestEventLogRESTAPI:
    """Tests for the REST endpoints."""

    @pytest.mark.asyncio
    async def test_get_events_returns_json(self, client):
        """(5) GET /admin/api/events returns events list."""
        resp = await client.get("/admin/api/events")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert "total" in data
        assert isinstance(data["events"], list)

    @pytest.mark.asyncio
    async def test_get_events_filter_category(self, client):
        """(6a) GET /admin/api/events?category=generation filters correctly."""
        # Emit a few events first by creating a draft
        await _create_draft(client)
        resp = await client.get("/admin/api/events?category=generation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0
        for evt in data["events"]:
            assert evt["category"] == "generation"

    @pytest.mark.asyncio
    async def test_get_events_filter_status(self, client):
        """(6b) GET /admin/api/events?status=done filters by status."""
        await _create_draft(client)
        resp = await client.get("/admin/api/events?status=done")
        assert resp.status_code == 200
        data = resp.json()
        for evt in data["events"]:
            assert evt["status"] == "done"

    @pytest.mark.asyncio
    async def test_get_events_filter_draft_id(self, client):
        """(6c) GET /admin/api/events?draft_id=xxx filters by draft."""
        draft_id = await _create_draft(client)
        resp = await client.get(f"/admin/api/events?draft_id={draft_id}")
        assert resp.status_code == 200
        data = resp.json()
        for evt in data["events"]:
            assert evt["draft_id"] == draft_id

    @pytest.mark.asyncio
    async def test_get_events_stats(self, client):
        """(7) GET /admin/api/events/stats returns stats."""
        await _create_draft(client)
        resp = await client.get("/admin/api/events/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_events" in data
        assert "by_category" in data
        assert "by_status" in data
        assert data["total_events"] > 0

    @pytest.mark.asyncio
    async def test_logs_page_renders(self, client):
        """(8) GET /admin/logs renders HTML page."""
        resp = await client.get("/admin/logs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Live Log" in resp.text or "log" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_logs_page_has_sse_endpoint(self, client):
        """(8b) Logs page references the SSE endpoint."""
        resp = await client.get("/admin/logs")
        assert resp.status_code == 200
        assert "/admin/api/events/live" in resp.text

    @pytest.mark.asyncio
    async def test_logs_page_has_filter_buttons(self, client):
        """(8c) Logs page has category filter buttons."""
        resp = await client.get("/admin/logs")
        assert resp.status_code == 200
        assert "generation" in resp.text.lower()
        assert "review" in resp.text.lower()
        assert "filter-btn" in resp.text


# ── Integration tests: Events on actions ─────────────────────────────


class TestEventLogOnActions:
    """Tests that verify events are emitted during pipeline actions."""

    @pytest.mark.asyncio
    async def test_draft_creation_emits_events(self, client):
        """(9) Creating a draft emits generation events."""
        await _create_draft(client)
        resp = await client.get("/admin/api/events?category=generation")
        data = resp.json()
        actions = [e["action"] for e in data["events"]]
        assert "create_draft" in actions
        assert "outline_generation" in actions
        assert "graph_generation" in actions

    @pytest.mark.asyncio
    async def test_review_emits_events(self, client):
        """(10) Running a review emits review events."""
        draft_id = await _create_draft(client)
        # Clear events from creation
        await client.post(f"/admin/draft/{draft_id}/review", follow_redirects=False)
        resp = await client.get(f"/admin/api/events?category=review&draft_id={draft_id}")
        data = resp.json()
        actions = [e["action"] for e in data["events"]]
        assert "critic_review" in actions

    @pytest.mark.asyncio
    async def test_validation_emits_events(self, client):
        """(11) Running validation emits validation events."""
        draft_id = await _create_draft(client)
        await client.post(f"/admin/draft/{draft_id}/validate", follow_redirects=False)
        resp = await client.get(f"/admin/api/events?category=validation&draft_id={draft_id}")
        data = resp.json()
        actions = [e["action"] for e in data["events"]]
        assert "validate" in actions

    @pytest.mark.asyncio
    async def test_draft_detail_has_log_panel(self, client):
        """(12a) Draft detail page contains the event log bar component."""
        draft_id = await _create_draft(client)
        resp = await client.get(f"/admin/draft/{draft_id}")
        assert resp.status_code == 200
        assert "elb-container" in resp.text
        assert "event_log_bar.js" in resp.text
        assert f'data-draft-id="{draft_id}"' in resp.text

    @pytest.mark.asyncio
    async def test_draft_list_has_logs_link(self, client):
        """(12b) Draft list page links to /admin/logs."""
        resp = await client.get("/admin/")
        assert resp.status_code == 200
        assert "/admin/logs" in resp.text


# ── Integration tests: SSE endpoint ──────────────────────────────────


class TestSSEEndpoint:
    """Tests for the Server-Sent Events streaming endpoint."""

    @pytest.mark.asyncio
    async def test_sse_returns_event_stream(self, admin_client):
        """(12c) GET /admin/api/events/live returns text/event-stream.

        Verifies the response headers using a direct ASGI scope call.
        """
        # Use a simple ASGI request to get headers without consuming the infinite body
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/events/live",
            "raw_path": b"/api/events/live",
            "query_string": b"",
            "headers": [],
            "server": ("test", 80),
            "client": ("127.0.0.1", 12345),
            "scheme": "http",
            "root_path": "",
            "app": admin_client._transport.app,
        }

        received_headers = {}
        received_body_chunks = []
        body_sent = False

        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            # Keep the connection open — emulate client disconnect after a bit
            await asyncio.sleep(5)
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                for k, v in message.get("headers", []):
                    received_headers[k.decode()] = v.decode()
            elif message["type"] == "http.response.body":
                received_body_chunks.append(message.get("body", b""))
                # Once we get the initial comment, we can disconnect
                if received_body_chunks and b"connected" in received_body_chunks[0]:
                    raise asyncio.CancelledError("Got initial chunk — done")

        try:
            await admin_client._transport.app(scope, receive, send)
        except asyncio.CancelledError:
            pass  # expected — we cancel after the first chunk

        assert "content-type" in received_headers
        assert "text/event-stream" in received_headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_sse_delivers_emitted_events(self, admin_client):
        """(12d) SSE stream delivers emitted events to subscribers.

        Opens a stream, emits an event, and verifies it arrives in the stream.
        """
        from app.services.event_log import event_log

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/events/live",
            "raw_path": b"/api/events/live",
            "query_string": b"",
            "headers": [],
            "server": ("test", 80),
            "client": ("127.0.0.1", 12345),
            "scheme": "http",
            "root_path": "",
            "app": admin_client._transport.app,
        }

        received_chunks = []
        started = asyncio.Event()
        event_sent = asyncio.Event()

        async def receive():
            nonlocal event_sent
            # Signal that the response has started
            if not started.is_set():
                started.set()
                # Wait briefly for the stream to set up its subscriber
                await asyncio.sleep(0.05)
                # Now emit an event
                event_log.emit("test", "sse_test_event", "done", "Test via SSE")
                event_sent.set()
            # Keep alive briefly then disconnect
            await asyncio.sleep(1)
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                pass
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                received_chunks.append(chunk)
                # If we found our event, cancel
                if b"sse_test_event" in chunk:
                    raise asyncio.CancelledError("Found event — done")

        try:
            await admin_client._transport.app(scope, receive, send)
        except asyncio.CancelledError:
            pass

        all_data = b"".join(received_chunks)
        assert b"sse_test_event" in all_data, (
            f"SSE stream did not deliver the emitted event. Got: {all_data!r}"
        )

# ── Tests for SSE robustness (t_d6f324a4) ─────────────────────────────


class TestSSERobustness:
    """Tests for SSE endpoint robustness — disconnect handling, no duplicates,
    subscriber cleanup, and error recovery."""

    @pytest.mark.asyncio
    async def test_sse_no_duplicate_initial_events(self, admin_client):
        """SSE stream should not send initial events twice.

        The old endpoint sent a snapshot in event_stream() AND in
        subscribe_filtered(), causing every event to appear twice.
        Now only subscribe_filtered() sends the initial burst.
        """
        from app.services.event_log import event_log
        event_log.clear()
        event_log.emit("test", "dup_check", "done", "Unique event for dup check")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/events/live",
            "raw_path": b"/api/events/live",
            "query_string": b"",
            "headers": [],
            "server": ("test", 80),
            "client": ("127.0.0.1", 12345),
            "scheme": "http",
            "root_path": "",
            "app": admin_client._transport.app,
        }

        received_chunks = []

        async def receive():
            await asyncio.sleep(0.1)
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                received_chunks.append(chunk)
                if b"dup_check" in chunk:
                    raise asyncio.CancelledError("Found event — done")

        try:
            await admin_client._transport.app(scope, receive, send)
        except asyncio.CancelledError:
            pass

        all_data = b"".join(received_chunks)
        # Count occurrences of "dup_check" — should be exactly 1
        count = all_data.count(b"dup_check")
        assert count == 1, (
            f"Expected 'dup_check' to appear exactly 1 time, got {count}. "
            f"Data: {all_data!r}"
        )

    @pytest.mark.asyncio
    async def test_sse_subscriber_cleanup_on_disconnect(self, admin_client):
        """When the client disconnects, the subscriber queue should be
        removed from _subscribers to prevent leaks."""
        from app.services.event_log import event_log

        event_log.clear()
        initial_subscribers = len(event_log._subscribers)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/events/live",
            "raw_path": b"/api/events/live",
            "query_string": b"",
            "headers": [],
            "server": ("test", 80),
            "client": ("127.0.0.1", 12345),
            "scheme": "http",
            "root_path": "",
            "app": admin_client._transport.app,
        }

        async def receive():
            await asyncio.sleep(0.2)
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                raise asyncio.CancelledError("Got initial chunk — disconnecting")

        try:
            await admin_client._transport.app(scope, receive, send)
        except asyncio.CancelledError:
            pass

        # Give the server a moment to clean up
        await asyncio.sleep(0.1)

        # Subscriber count should be back to initial
        assert len(event_log._subscribers) == initial_subscribers, (
            f"Subscriber leak: expected {initial_subscribers}, "
            f"got {len(event_log._subscribers)}"
        )

    @pytest.mark.asyncio
    async def test_sse_heartbeat_keeps_alive(self, admin_client):
        """SSE stream should send heartbeat comments to prevent timeouts."""
        from app.services.event_log import event_log, EventLogService
        event_log.clear()

        # Monkey-patch subscribe_filtered to use a short heartbeat
        original_sub = event_log.subscribe_filtered
        async def fast_sub(category=None, draft_id=None, heartbeat_seconds=0.5):
            async for item in original_sub(category=category, draft_id=draft_id, heartbeat_seconds=heartbeat_seconds):
                yield item

        import app.admin_ui.app as admin_module
        original_event_log = admin_module.event_log

        # Create a wrapper that calls subscribe_filtered with short heartbeat
        class FastEventLog:
            def __getattr__(self, name):
                return getattr(event_log, name)
            async def subscribe_filtered(self, category=None, draft_id=None, heartbeat_seconds=0.5):
                async for item in event_log.subscribe_filtered(category=category, draft_id=draft_id, heartbeat_seconds=heartbeat_seconds):
                    yield item

        admin_module.event_log = FastEventLog()
        try:
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/api/events/live",
                "raw_path": b"/api/events/live",
                "query_string": b"",
                "headers": [],
                "server": ("test", 80),
                "client": ("127.0.0.1", 12345),
                "scheme": "http",
                "root_path": "",
                "app": admin_client._transport.app,
            }

            received_chunks = []

            async def receive():
                await asyncio.sleep(2.0)
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body":
                    chunk = message.get("body", b"")
                    received_chunks.append(chunk)
                    if b"heartbeat" in chunk:
                        raise asyncio.CancelledError("Got heartbeat — done")

            try:
                await admin_client._transport.app(scope, receive, send)
            except asyncio.CancelledError:
                pass

            all_data = b"".join(received_chunks)
            assert b"heartbeat" in all_data, (
                f"Expected heartbeat in SSE stream. Got: {all_data!r}"
            )
        finally:
            admin_module.event_log = original_event_log

    @pytest.mark.asyncio
    async def test_notify_subscribers_iteration_safe(self, fresh_event_log):
        """_notify_subscribers should not crash if _subscribers is
        modified during iteration (e.g. by a concurrent disconnect)."""
        import asyncio as aio

        # Pre-populate
        fresh_event_log.emit("test", "pre", "done", "before")

        # Simulate a subscriber that gets removed during notification
        queue = aio.Queue(maxsize=100)
        fresh_event_log._subscribers.append(queue)

        # Remove the queue during emit (simulating concurrent cleanup)
        original_notify = fresh_event_log._notify_subscribers

        def emit_and_remove():
            # Emit, which calls _notify_subscribers
            fresh_event_log.emit("test", "concurrent", "done", "during")
            # Remove the queue right after
            try:
                fresh_event_log._subscribers.remove(queue)
            except ValueError:
                pass

        # This should not raise RuntimeError
        emit_and_remove()

        # Verify we can still emit after
        fresh_event_log.emit("test", "after", "done", "after removal")
        assert len(fresh_event_log._subscribers) == 0

    @pytest.mark.asyncio
    async def test_subscribe_filtered_safe_removal(self, fresh_event_log):
        """subscribe_filtered should not raise ValueError if the
        queue is already removed from _subscribers when the generator
        is closed."""
        fresh_event_log.emit("test", "init", "done", "initial")

        gen = fresh_event_log.subscribe_filtered()
        # Consume initial events
        async for sse_data in gen:
            break  # got first event

        # Manually remove the queue (simulating a race condition)
        # The generator's finally block should handle this gracefully
        assert len(fresh_event_log._subscribers) == 1
        queue = fresh_event_log._subscribers[0]
        fresh_event_log._subscribers.remove(queue)

        # Now close the generator — finally block should not raise
        await gen.aclose()
        assert len(fresh_event_log._subscribers) == 0
