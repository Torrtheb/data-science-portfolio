from __future__ import annotations
from typing import List
from langchain.schema import Document
from ..core.settings import settings

MIN_SCORE: float = float(getattr(settings, "rag_min_score", 0.25))


def rag_filter_confident(docs: List[Document]) -> List[Document]:
    """
    Confidence filter tailored to the project’s hybrid retriever (see rag.py).

    It aggregates multiple relevance signals placed in 'Document.metadata':
        - "dense_relevance"   (0..1)
        - "dense_similarity"  (0..1)
        - "bm25_score"        (BM25 scale; normalized by /10.0 here)

    The final score = max of the available signals (with BM25 normalized).
    A document is kept iff final_score >= MIN_SCORE.

    Args:
        docs: List of LangChain 'Document' objects.

    Returns:
        Filtered list of 'Document' meeting the MIN_SCORE threshold.
    """
    kept: list[Document] = []
    saw_signal = False  # did we ever see a numeric relevance/similarity signal?
    for d in docs or []:
        md = getattr(d, "metadata", {}) or {}

        s = 0.0
        # Prefer explicit relevance/similarity signals; fall back to generic scores.
        for k in ("dense_relevance", "dense_similarity", "score", "relevance_score"):
            try:
                val = float(md.get(k) or 0.0)
                if val:
                    saw_signal = True
                s = max(s, val)
            except Exception:
                pass
        try:
            bm25 = md.get("bm25_score")
            if bm25 is not None:
                val = float(bm25) / 10.0
                if val:
                    saw_signal = True
                s = max(s, val)
        except Exception:
            pass

        if s >= MIN_SCORE:
            kept.append(d)

    # If no docs passed the threshold and we never saw any scoring signals,
    # fall back to returning the top docs as-is (already ranked by the retriever).
    if not kept and not saw_signal and docs:
        return list(docs)

    return kept
