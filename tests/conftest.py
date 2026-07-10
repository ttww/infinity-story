"""Shared test fixtures."""

import asyncio
import os

# Use in-memory SQLite for tests (honored by database._get_database_url)
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["DEBUG"] = "false"
# Lower quality thresholds for testing (dummy graph has 5–9 nodes)
os.environ["MIN_NODE_COUNT"] = "3"
os.environ["MIN_ENDING_COUNT"] = "1"

import pytest
from httpx import AsyncClient, ASGITransport

# Clear any cached settings so our env vars are picked up
from app.core.config import get_settings
get_settings.cache_clear()

from app.persistence.database import init_db, close_db


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def app():
    """Create the FastAPI app with in-memory DB.

    Each test function gets a fresh in-memory database for isolation.
    """
    from app.main import app as application
    await init_db()
    yield application
    await close_db()


@pytest.fixture
async def client(app):
    """Async HTTP client for API testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
