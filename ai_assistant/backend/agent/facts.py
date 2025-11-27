from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_postgres.vectorstores import PGVector
except Exception:
    OpenAIEmbeddings = None
    PGVector = None

log = logging.getLogger(__name__)


def _conn_string() -> str:
    """Resolve the PGVector connection URL from environment.

    Reads, in order: 'PGVECTOR_URL', 'PG_DSN', 'DATABASE_URL',
    'BACKEND_DATABASE_URL'.

    Returns:
        A database connection string.

    Raises:
        RuntimeError: If no environment variable is set.
    """
    url = (
        os.getenv("PGVECTOR_URL")
        or os.getenv("PG_DSN")
        or os.getenv("DATABASE_URL")
        or os.getenv("BACKEND_DATABASE_URL")
    )
    if not url:
        raise RuntimeError("Set PGVECTOR_URL, PG_DSN, or DATABASE_URL for fact storage")
    return url


@lru_cache(maxsize=1)
def _fact_store():
    """Return a cached PGVector store configured with embeddings.

    Returns:
        A PGVector vector store instance bound to an OpenAI embedding model.

    Raises:
        RuntimeError: If the required optional packages are not installed.
    """
    if PGVector is None or OpenAIEmbeddings is None:
        raise RuntimeError("Missing langchain-postgres or langchain-openai packages")

    conn = _conn_string()
    collection = os.getenv("AGENT_FACT_COLLECTION", "owner_memory")

    embed_model = os.getenv(
        "AGENT_FACT_EMBED_MODEL",
        os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    # Reuse the same retry knob as the chat model where possible. OpenAIEmbeddings
    # also honours 'max_retries' via the underlying openai-python client.
    max_retries = int(os.getenv("AGENT_OPENAI_MAX_RETRIES", "3") or "3")
    embeddings = OpenAIEmbeddings(model=embed_model, max_retries=max_retries)

    store = PGVector(
        connection=conn,
        collection_name=collection,
        embeddings=embeddings,
        use_jsonb=True,
    )
    return store


def store_fact(
    owner_id: str,
    thread_id: str,
    fact_text: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a short fact snippet for later recall.

    Args:
        owner_id: Owner/user identifier used for scoping.
        thread_id: Conversation/thread identifier for grouping facts.
        fact_text: The plain‑text fact or summary chunk to store.
        metadata: Optional extra fields to attach to the stored document.

    Returns:
        None. Errors are swallowed with a log entry so chat flow is unaffected.
    """
    if not fact_text:
        return
    try:
        store = _fact_store()
    except Exception as exc:
        log.debug("Fact store unavailable: %s", exc)
        return

    payload = {"owner_id": owner_id, "thread_id": thread_id}
    if metadata:
        payload.update(metadata)

    try:
        store.add_documents([Document(page_content=fact_text, metadata=payload)])
    except Exception as exc:
        log.warning("Failed to store fact for owner %s: %s", owner_id, exc)


def fetch_facts(owner_id: str, query: str, *, limit: int = 3) -> list[str]:
    """Retrieve the most relevant stored facts for an owner.

    Args:
        owner_id: Owner/user identifier used to filter results.
        query: Free‑text query (e.g., last user message) to match against facts.
        limit: Maximum number of snippets to return (default 3).

    Returns:
        A list of fact text snippets ordered by similarity.
    """
    if not query:
        return []
    try:
        store = _fact_store()
    except Exception as exc:
        log.debug("Fact store unavailable: %s", exc)
        return []

    try:
        docs = store.similarity_search(query, k=limit, filter={"owner_id": owner_id})
    except Exception as exc:
        log.warning("Fact retrieval failed for owner %s: %s", owner_id, exc)
        return []

    return [doc.page_content for doc in docs]
