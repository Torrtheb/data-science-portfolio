# tests/test_rag_utils.py
from __future__ import annotations
import pytest
from types import SimpleNamespace
from langchain.schema import Document

from back_app.utils.rag_utils import rag_filter_confident, MIN_SCORE


def make_doc(**metadata) -> Document:
    """Helper to make a Document with arbitrary metadata."""
    return Document(page_content="dummy", metadata=metadata)


def test_empty_list_returns_empty():
    assert rag_filter_confident([]) == []


@pytest.mark.parametrize(
    "metadata, keep",
    [
        # dense_relevance meets threshold
        ({"dense_relevance": MIN_SCORE}, True),
        ({"dense_relevance": MIN_SCORE - 0.01}, False),
        # dense_similarity meets threshold
        ({"dense_similarity": MIN_SCORE + 0.1}, True),
        ({"dense_similarity": 0.0}, False),
        # bm25_score normalized
        ({"bm25_score": MIN_SCORE * 10}, True),
        ({"bm25_score": (MIN_SCORE * 10) - 1}, False),
    ],
)
def test_individual_signals(metadata, keep):
    doc = make_doc(**metadata)
    out = rag_filter_confident([doc])
    assert (doc in out) == keep


def test_combines_signals_takes_max():
    # dense_relevance is low, but bm25_score is high
    doc = make_doc(dense_relevance=0.05, bm25_score=(MIN_SCORE * 10) + 1)
    out = rag_filter_confident([doc])
    assert doc in out

    # both signals low
    doc2 = make_doc(dense_relevance=0.05, bm25_score=1)
    out2 = rag_filter_confident([doc2])
    assert doc2 not in out2


def test_multiple_docs_mixed():
    good = make_doc(dense_similarity=1.0)
    bad = make_doc(dense_relevance=0.0, bm25_score=1)
    result = rag_filter_confident([good, bad])
    assert good in result
    assert bad not in result


def test_handles_missing_metadata_and_none():
    # Document with metadata omitted (defaults to {}), should be filtered out.
    doc1 = Document(page_content="a")

    # Non-Document object with metadata=None: exercises the getattr(..., {}) path.
    doc2 = SimpleNamespace(page_content="b", metadata=None)

    # Non-Document with empty dict metadata.
    doc3 = SimpleNamespace(page_content="c", metadata={})

    out = rag_filter_confident([doc1, doc2, doc3])
    assert out == []



def test_non_numeric_metadata_ignored():
    doc = make_doc(dense_relevance="not-a-number", bm25_score="oops")
    # Should not raise, and doc should be filtered out
    assert rag_filter_confident([doc]) == []
