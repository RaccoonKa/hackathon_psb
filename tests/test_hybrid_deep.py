from unittest.mock import patch

from rag_core.retrieval import hybrid as hybrid_mod
from tests._common import make_bm25_stats, make_chunk_dict


class _FakeCollection:
    def __init__(self, ids):
        self._ids = ids
        self.last_n = None

    def query(self, **kwargs):
        self.last_n = kwargs.get("n_results")
        return {
            "ids": [self._ids],
            "documents": [[""] * len(self._ids)],
            "metadatas": [[{} for _ in self._ids]],
        }


def _patch_hybrid(chunks_map, bm25_results, dense_ids):
    coll = _FakeCollection(dense_ids)
    return (
        patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map),
        patch.object(
            hybrid_mod.bm25_store,
            "search_with_stats",
            return_value=(bm25_results, make_bm25_stats(bm25_results)),
        ),
        patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024),
        patch.object(hybrid_mod, "get_chroma_collection", return_value=coll),
        coll,
    )


def test_rrf_exact_score_single_source():
    chunks_map = {"a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01")}
    p1, p2, p3, p4, _ = _patch_hybrid(chunks_map, [("a", 5.0)], [])
    with p1, p2, p3, p4:
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=False, rrf_k=60,
        )
    expected = 0.5 / (60 + 0 + 1)
    assert abs(results[0]["score"] - expected) < 1e-6


def test_rrf_custom_rrf_k():
    chunks_map = {"a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01")}
    p1, p2, p3, p4, _ = _patch_hybrid(chunks_map, [("a", 5.0)], [])
    with p1, p2, p3, p4:
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=False, rrf_k=10,
        )
    expected = 0.5 / (10 + 0 + 1)
    assert abs(results[0]["score"] - expected) < 1e-6


def test_rrf_rank_penalty_applied():
    chunks_map = {
        "a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01"),
        "b": make_chunk_dict(chunk_id="b", effective_from="2020-01-01"),
    }
    p1, p2, p3, p4, _ = _patch_hybrid(
        chunks_map, [("a", 10.0), ("b", 5.0)], [],
    )
    with p1, p2, p3, p4:
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=False, rrf_k=60,
        )
    scores = {r["chunk_id"]: r["score"] for r in results}
    assert abs(scores["a"] - 0.5 / 61) < 1e-6
    assert abs(scores["b"] - 0.5 / 62) < 1e-6


def test_rrf_dense_and_sparse_combined():
    chunks_map = {"a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01")}
    p1, p2, p3, p4, _ = _patch_hybrid(chunks_map, [("a", 5.0)], ["a"])
    with p1, p2, p3, p4:
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=False, rrf_k=60,
        )
    expected = 0.5 / 61 + 0.5 / 61
    assert abs(results[0]["score"] - expected) < 1e-6


def test_amendment_bonus_factor_1_15():
    chunks_map = {
        "base": make_chunk_dict(chunk_id="base", document_id="contract_x",
                                effective_from="2020-01-01"),
        "amend": make_chunk_dict(chunk_id="amend", document_id="amend_1",
                                 amends_document_id="contract_x",
                                 effective_from="2020-01-01"),
    }
    p1, p2, p3, p4, _ = _patch_hybrid(
        chunks_map, [("base", 5.0), ("amend", 5.0)], [],
    )
    with p1, p2, p3, p4:
        # H-2C: boost только при явной релевантности запроса допсоглашению.
        results = hybrid_mod.hybrid_search(
            query="дополнительное соглашение редакция договора",
            top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=False, rrf_k=60,
        )
    scores = {r["chunk_id"]: r["score"] for r in results}
    base_expected = 0.5 / 61
    amend_expected = (0.5 / 62) * 1.15
    assert abs(scores["base"] - base_expected) < 1e-6
    assert abs(scores["amend"] - amend_expected) < 1e-6


def test_no_amendment_bonus_when_query_not_version_related():
    chunks_map = {
        "base": make_chunk_dict(chunk_id="base", document_id="contract_x",
                                effective_from="2020-01-01"),
        "amend": make_chunk_dict(chunk_id="amend", document_id="amend_1",
                                 amends_document_id="contract_x",
                                 effective_from="2020-01-01"),
    }
    p1, p2, p3, p4, _ = _patch_hybrid(
        chunks_map, [("base", 5.0), ("amend", 5.0)], [],
    )
    with p1, p2, p3, p4:
        results = hybrid_mod.hybrid_search(
            query="какие условия поставки товара",
            top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=False, rrf_k=60,
        )
    scores = {r["chunk_id"]: r["score"] for r in results}
    # Без boost допсоглашение сохраняет свой обычный RRF-скор.
    assert abs(scores["base"] - 0.5 / 61) < 1e-6
    assert abs(scores["amend"] - 0.5 / 62) < 1e-6


def test_intent_exact_lookup_uses_bm25_top_40():
    chunks_map = {"c1": make_chunk_dict(chunk_id="c1", effective_from="2020-01-01")}
    coll = _FakeCollection([])
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=([], make_bm25_stats([]))) as mock_search, \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=coll):
        hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="exact_lookup", rerank=False,
        )
    assert mock_search.call_args.kwargs["top_k"] == 40


def test_intent_exact_lookup_uses_dense_top_20():
    chunks_map = {"c1": make_chunk_dict(chunk_id="c1", effective_from="2020-01-01")}
    coll = _FakeCollection([])
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=([], make_bm25_stats([]))), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=coll):
        hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="exact_lookup", rerank=False,
        )
    assert coll.last_n == 20


def test_intent_semantic_uses_dense_top_40():
    chunks_map = {"c1": make_chunk_dict(chunk_id="c1", effective_from="2020-01-01")}
    coll = _FakeCollection([])
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=([], make_bm25_stats([]))), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=coll):
        hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="semantic_query", rerank=False,
        )
    assert coll.last_n == 40


def test_intent_semantic_uses_bm25_top_20():
    chunks_map = {"c1": make_chunk_dict(chunk_id="c1", effective_from="2020-01-01")}
    coll = _FakeCollection([])
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=([], make_bm25_stats([]))) as mock_search, \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=coll):
        hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="semantic_query", rerank=False,
        )
    assert mock_search.call_args.kwargs["top_k"] == 20


def test_intent_mixed_uses_bm25_top_30():
    chunks_map = {"c1": make_chunk_dict(chunk_id="c1", effective_from="2020-01-01")}
    coll = _FakeCollection([])
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=([], make_bm25_stats([]))) as mock_search, \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=coll):
        hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=False,
        )
    assert mock_search.call_args.kwargs["top_k"] == 30


def test_dense_chunk_not_in_allowed_excluded():
    chunks_map = {
        "allowed": make_chunk_dict(chunk_id="allowed", effective_from="2025-06-01"),
        "forbidden": make_chunk_dict(chunk_id="forbidden", effective_from="2020-01-01",
                                      effective_to="2024-01-01"),
    }
    coll = _FakeCollection(["allowed", "forbidden"])
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=([], make_bm25_stats([]))), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=coll):
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-06-01",
            intent="mixed_query", rerank=False,
        )
    ids = [r["chunk_id"] for r in results]
    assert "allowed" in ids
    assert "forbidden" not in ids


def test_rerank_receives_candidates():
    chunks_map = {"a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01")}
    coll = _FakeCollection([])
    bm25_results = [("a", 5.0)]
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=(bm25_results, make_bm25_stats(bm25_results))), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=coll), \
         patch.object(hybrid_mod, "rerank_candidates",
                      return_value=[{"chunk_id": "a", "score": 0.99}]) as mock_rr:
        hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=True,
        )
    mock_rr.assert_called_once()
    passed_candidates = mock_rr.call_args.args[1]
    assert any(c["chunk_id"] == "a" for c in passed_candidates)


def test_rerank_false_skips_reranker():
    chunks_map = {"a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01")}
    coll = _FakeCollection([])
    bm25_results = [("a", 5.0)]
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=(bm25_results, make_bm25_stats(bm25_results))), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=coll), \
         patch.object(hybrid_mod, "rerank_candidates") as mock_rr:
        hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=False,
        )
    mock_rr.assert_not_called()


# --- M-1: один вызов BM25 на поисковую задачу ------------------------------


def test_hybrid_search_calls_search_with_stats_once():
    chunks_map = {"a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01")}
    coll = _FakeCollection([])
    bm25_results = [("a", 5.0)]
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=(bm25_results, make_bm25_stats(bm25_results))) as mock_sws, \
         patch.object(hybrid_mod.bm25_store, "search",
                      side_effect=AssertionError("повторный search() запрещён")), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=coll):
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01",
            intent="mixed_query", rerank=False,
        )

    mock_sws.assert_called_once()
    assert results


def test_bm25_search_calls_search_with_stats_once():
    chunks_map = {"a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01")}
    bm25_results = [("a", 5.0)]
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=(bm25_results, make_bm25_stats(bm25_results))) as mock_sws, \
         patch.object(hybrid_mod.bm25_store, "search",
                      side_effect=AssertionError("повторный search() запрещён")):
        results = hybrid_mod.bm25_search(
            query="q", top_k=5, event_date="2025-01-01", rerank=False,
        )

    mock_sws.assert_called_once()
    assert results


def test_bm25_diagnostics_are_forwarded_unchanged():
    chunks_map = {"a": make_chunk_dict(chunk_id="a", effective_from="2020-01-01")}
    stats = {
        "candidates_before_filters": 10,
        "candidates_after_date_filter": 7,
        "candidates_after_document_filter": 2,
        "returned_count": 1,
    }
    diagnostics: dict = {}
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=chunks_map), \
         patch.object(hybrid_mod.bm25_store, "search_with_stats",
                      return_value=([("a", 5.0)], stats)):
        hybrid_mod.bm25_search(
            query="q", top_k=5, event_date="2025-01-01",
            rerank=False, diagnostics=diagnostics,
        )

    assert diagnostics["bm25_candidates_count"] == 1
    assert diagnostics["candidates_before_filters"] == 10
    assert diagnostics["candidates_after_date_filter"] == 7
    assert diagnostics["candidates_after_document_filter"] == 2
    assert diagnostics["dense_candidates_count"] == 0
    assert diagnostics["final_chunks_count"] == 1