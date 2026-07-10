"""In-memory event log for the Admin UI.

Records events from background processes (story generation, review,
validation, publishing, etc.) so the Admin UI can display what's
happening behind the scenes.

Events are stored in memory (no persistence) and capped at
``max_size`` entries — oldest entries are evicted when full.

Supports SSE (Server-Sent Events) for real-time streaming.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from typing import Any, AsyncGenerator

DEFAULT_MAX_SIZE = 500


class LogEvent:
    """A single event in the background process log."""

    def __init__(
        self,
        category: str,
        action: str,
        status: str,
        message: str = "",
        draft_id: str | None = None,
        detail: dict[str, Any] | None = None,
        timestamp: float | None = None,
        event_id: str | None = None,
    ) -> None:
        self.id = event_id or f"evt_{uuid.uuid4().hex[:12]}"
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.category = category
        self.action = action
        self.status = status
        self.message = message
        self.draft_id = draft_id
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "category": self.category,
            "action": self.action,
            "status": self.status,
            "message": self.message,
            "draft_id": self.draft_id,
            "detail": self.detail,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class EventLogService:
    """In-memory event log with ring buffer and SSE support."""

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        self._buffer: deque[LogEvent] = deque(maxlen=max_size)
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._subscribers: list[asyncio.Queue] = []

    # ── Emit methods ────────────────────────────────────────────────

    def emit(
        self,
        category: str,
        action: str,
        status: str,
        message: str = "",
        draft_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> LogEvent:
        """Create an event and add it to the buffer."""
        event = LogEvent(
            category=category,
            action=action,
            status=status,
            message=message,
            draft_id=draft_id,
            detail=detail,
        )
        self._buffer.append(event)
        self._notify_subscribers(event)
        return event

    def emit_start(
        self,
        category: str,
        action: str,
        message: str = "",
        draft_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> LogEvent:
        """Record a 'running' event."""
        return self.emit(category, action, "running", message, draft_id, detail)

    def emit_done(
        self,
        category: str,
        action: str,
        message: str = "",
        draft_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> LogEvent:
        """Record a 'done' event."""
        return self.emit(category, action, "done", message, draft_id, detail)

    def emit_error(
        self,
        category: str,
        action: str,
        message: str = "",
        draft_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> LogEvent:
        """Record an 'error' event."""
        return self.emit(category, action, "error", message, draft_id, detail)

    def emit_info(
        self,
        category: str,
        action: str,
        message: str = "",
        draft_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> LogEvent:
        """Record an 'info' event."""
        return self.emit(category, action, "info", message, draft_id, detail)

    def _notify_subscribers(self, event: LogEvent) -> None:
        # Iterate over a copy to avoid "list changed size during iteration"
        # if a subscriber is removed while we're notifying.
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # ── Query methods ──────────────────────────────────────────────

    def list_events(
        self,
        limit: int = 50,
        offset: int = 0,
        category: str | None = None,
        status: str | None = None,
        draft_id: str | None = None,
    ) -> dict[str, Any]:
        """Return events as dicts, newest first."""
        # Treat "all" as None
        if category == "all":
            category = None
        if status == "all":
            status = None

        events = list(self._buffer)
        if category:
            events = [e for e in events if e.category == category]
        if status:
            events = [e for e in events if e.status == status]
        if draft_id:
            events = [e for e in events if e.draft_id == draft_id]
        total = len(events)
        # newest first, then apply offset and limit
        events = list(reversed(events))[offset:offset + limit]
        return {"total": total, "events": [e.to_dict() for e in events]}

    async def recent(self, limit: int = 50, category: str | None = None) -> list[dict]:
        """Return recent events as a list of dicts (compatibility API)."""
        return self.list_events(limit=limit, category=category)["events"]

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        events = list(self._buffer)
        total = len(events)
        by_category: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for e in events:
            by_category[e.category] = by_category.get(e.category, 0) + 1
            by_status[e.status] = by_status.get(e.status, 0) + 1
        return {
            "total_events": total,
            "buffer_capacity": self._max_size,
            "by_category": by_category,
            "by_status": by_status,
        }

    def clear(self) -> int:
        """Clear all events. Return the number removed."""
        count = len(self._buffer)
        self._buffer.clear()
        return count

    # ── SSE streaming ──────────────────────────────────────────────

    async def subscribe_filtered(
        self,
        category: str | None = None,
        draft_id: str | None = None,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE-formatted event strings as they occur.

        Sends a heartbeat comment every ``heartbeat_seconds`` to keep
        the connection alive and prevent browser/proxy timeouts.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        try:
            # Send recent events as a burst
            existing = list(self._buffer)
            if category:
                existing = [e for e in existing if e.category == category]
            if draft_id:
                existing = [e for e in existing if e.draft_id == draft_id]
            for e in existing[-20:]:
                yield self._format_sse(e)

            # Then stream new events with heartbeat
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=heartbeat_seconds
                    )
                    if category and event.category != category:
                        continue
                    if draft_id and event.draft_id != draft_id:
                        continue
                    yield self._format_sse(event)
                except asyncio.TimeoutError:
                    # Heartbeat — keep connection alive
                    yield ": heartbeat\n\n"
        finally:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass  # already removed

    @staticmethod
    def _format_sse(event: LogEvent) -> str:
        data = event.to_json()
        return f"data: {data}\n\n"


# ── Compatibility wrapper ────────────────────────────────────────────


class EventLog(EventLogService):
    """Compatibility wrapper for older API (add/add_sync)."""

    async def add(
        self,
        category: str,
        status: str,
        message: str,
        detail: dict[str, Any] | None = None,
    ) -> LogEvent:
        return self.emit(category, "generic", status, message, None, detail)


# ── Global singletons ───────────────────────────────────────────────

event_log = EventLogService()

_event_log_compat = EventLog()


def get_event_log() -> EventLog:
    """Return the global event log singleton (compatibility API)."""
    return _event_log_compat
