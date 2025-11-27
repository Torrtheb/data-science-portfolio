from __future__ import annotations

from typing import List

from back_app.llm.rag import HybridRetriever, Document


def _doc(text: str, keywords: List[str]) -> Document:
    return Document(page_content=text, metadata={"keywords": keywords})


def test_hybrid_retriever_prefers_keyword_matching_docs():
    """
    After dense+BM25 fusion, HybridRetriever should prefer documents whose
    metadata['keywords'] overlap the user query, while preserving original order
    among the filtered set.
    """

    class DummyVS:
        pass

    hr = HybridRetriever(vs=DummyVS(), k=3)

    # Stub out dense/sparse legs to avoid real vector calls.
    hr._dense_topk = lambda q, k: [
        _doc("about diversification", ["diversification", "etf"]),
        _doc("about cooking", ["recipes", "cooking"]),
    ]
    hr._bm25_topk = lambda q, k: []

    out = hr._get_relevant_documents("How does diversification in an ETF work?")
    assert len(out) == 1
    assert "diversification" in [kw.lower() for kw in out[0].metadata.get("keywords", [])]


def test_hybrid_retriever_falls_back_when_no_keyword_match():
    """
    If no documents have matching keywords, HybridRetriever should return the
    fused ranking unchanged (up to k).
    """

    class DummyVS:
        pass

    hr = HybridRetriever(vs=DummyVS(), k=2)

    d1 = _doc("about cooking", ["recipes", "cooking"])
    d2 = _doc("about gardening", ["plants"])

    hr._dense_topk = lambda q, k: [d1, d2]
    hr._bm25_topk = lambda q, k: []

    out = hr._get_relevant_documents("What is portfolio beta?")
    # No overlap between query and keywords → original fused ordering preserved.
    assert out == [d1, d2]

