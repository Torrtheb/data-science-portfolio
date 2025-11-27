# tests/conftest.py
import os, sys, asyncio
import contextlib
import pytest
from typing import AsyncIterator

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Minimal, safe env so app boots fast in tests
os.environ.setdefault("ENV", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("RAG_ENABLED", "true")
os.environ.setdefault("VECTORSTORE", "qdrant")
os.environ.setdefault("QDRANT_URL", "http://dummy-qdrant:6333")  # won't be called (we mock)
os.environ.setdefault("QDRANT_COLLECTION", "finance_docs_test")
os.environ.setdefault("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("MCP_PROXY_ENABLE", "false")  # avoid MCP startup in tests
os.environ.setdefault("SESSION_TOKEN_SECRET", "test-session-secret")
# Shared on-disk SQLite for test session so tables persist across connections.
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/finassist_test.db")

# Secrets that your code expects to exist
os.environ.setdefault("OPENAI_API_KEY", "test-openai")
os.environ.setdefault("FINNHUB_API_KEY", "test-finnhub")
os.environ.setdefault("TWELVEDATA_API_KEY", "test-twelvedata")

# Import the FastAPI app
from back_app.main import app
from back_app.core.db import init_db

@pytest.fixture(scope="session")
def anyio_backend():
    # httpx AsyncClient uses anyio under the hood
    return "asyncio"

@pytest.fixture(scope="session")
def fastapi_app():
    init_db()
    return app
