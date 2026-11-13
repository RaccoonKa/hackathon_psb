from unittest.mock import patch

from rag_core.retrieval import hybrid as hybrid_mod
from rag_core.retrieval.hybrid import (
    detect_query_intent,
    load_canonical_chunks_map,
    resolve_allowed_versions_by_date,
)
from tests._common import make_chunk_dict


def test_detect_intent_exact_article():
    cfg = detect_query_intent("Статья 112 НК РФ")
    assert cfg["intent"] == "exact_lookup"
    assert cfg["w_bm25"] > cfg["w_dense"]


def test_detect_intent_exact_contract_number():
    cfg = detect_query_intent("Договор поставки № П-17/2024")
    assert cfg["intent"] == "exact_lookup"


def test_detect_intent_semantic():
    cfg = detect_query_intent("Что означает понятие сверхурочной работы?")
    assert cfg["intent"] == "semantic_query"
    assert cfg["w_dense"] > cfg["w_bm25"]


def test_detect_intent_mixed():
    cfg = detect_query_intent(
        "Статья 112 НК РФ: что означает понятие смягчающих обстоятельств?"
    )
    assert cfg["intent"] == "mixed_query"
    assert cfg["w_bm25"] == cfg["w_dense"]


def test_detect_intent_precomputed_exact():
    cfg = detect_query_intent("что угодно", precomputed_intent="exact_lookup")
    assert cfg["intent"] == "exact_lookup"


def test_detect_intent_precomputed_semantic():
    cfg = detect_query_intent("Статья 112", precomputed_intent="semantic_query")
    assert cfg["intent"] == "semantic_query"


def test_detect_intent_empty_query_defaults_semantic():
    cfg = detect_query_intent("")
    assert cfg["intent"] == "semantic_query"


def test_detect_intent_weights_sum_to_one():
    for q in ("Статья 112 НК", "Что означает порядок", "Статья 112: что означает порядок"):
        cfg = detect_query_intent(q)
        assert abs(cfg["w_bm25"] + cfg["w_dense"] - 1.0) < 1e-9


def test_load_canonical_chunks_map_missing_file():
    from pathlib import Path
    with patch.object(hybrid_mod, "PROCESSED_DATA_DIR", Path("/tmp/nonexistent_xyz_12345")):
        result = load_canonical_chunks_map()
        assert result == {}

def _build_date_map():
    return {
        "c_law": make_chunk_dict(
            chunk_id="c_law", document_id="doc_law", document_type="law",
            effective_from="2025-01-01", effective_to=None,
        ),
        "c_old": make_chunk_dict(
            chunk_id="c_old", document_id="doc_old", document_type="internal",
            effective_from="2023-01-01", effective_to="2025-03-01",
        ),
        "c_new": make_chunk_dict(
            chunk_id="c_new", document_id="doc_new", document_type="internal",
            effective_from="2025-03-01", effective_to=None,
        ),
        "c_amend": make_chunk_dict(
            chunk_id="c_amend", document_id="amend_1_contract_x",
            document_type="contract",
            effective_from="2024-09-03", effective_to=None,
            amends_document_id="contract_x",
        ),
        "c_unknown": make_chunk_dict(
            chunk_id="c_unknown", document_id="doc_unknown",
            effective_from=None, effective_to=None,
            effective_dates_known=False,
        ),
    }


def test_resolve_2024():
    unknown = set()
    allowed = resolve_allowed_versions_by_date(
        "2024-06-01", _build_date_map(), validity_unknown=unknown,
    )
    assert "c_law" in allowed
    assert "c_old" in allowed
    assert "c_new" not in allowed
    assert "c_amend" not in allowed
    # Третье состояние: неизвестные даты не отсекаются, но помечаются.
    assert "c_unknown" in allowed
    assert "c_unknown" in unknown


def test_resolve_2025():
    allowed = resolve_allowed_versions_by_date("2025-06-01", _build_date_map())
    assert "c_law" in allowed
    assert "c_old" not in allowed
    assert "c_new" in allowed
    assert "c_amend" in allowed


def test_resolve_uses_year_from_query():
    allowed = resolve_allowed_versions_by_date(None, _build_date_map(), query="2023 год")
    assert "c_old" in allowed
    assert "c_new" not in allowed


def test_resolve_no_year_defaults_latest():
    allowed = resolve_allowed_versions_by_date(None, _build_date_map())
    assert "c_law" in allowed
    assert "c_new" in allowed
    assert "c_old" not in allowed


def test_resolve_keeps_laws_before_eff_from():
    allowed = resolve_allowed_versions_by_date("2024-01-01", _build_date_map())
    assert "c_law" in allowed


def test_resolve_boundary_from_inclusive():
    cm = {"c": make_chunk_dict(
        chunk_id="c", document_type="internal",
        effective_from="2025-01-01", effective_to=None,
    )}
    assert "c" in resolve_allowed_versions_by_date("2025-01-01", cm)
    assert "c" not in resolve_allowed_versions_by_date("2024-12-31", cm)


def test_resolve_boundary_to_exclusive():
    cm = {"c": make_chunk_dict(
        chunk_id="c", document_type="internal",
        effective_from="2020-01-01", effective_to="2025-03-01",
    )}
    assert "c" not in resolve_allowed_versions_by_date("2025-03-01", cm)
    assert "c" in resolve_allowed_versions_by_date("2025-02-28", cm)


def test_resolve_selected_doc_ids_filters():
    allowed = resolve_allowed_versions_by_date(
        "2025-01-01", _build_date_map(), selected_doc_ids={"doc_law"},
    )
    assert "c_law" in allowed
    assert "c_new" not in allowed
    assert "c_amend" not in allowed


def test_resolve_selected_expands_to_amendments():
    allowed = resolve_allowed_versions_by_date(
        "2025-01-01", _build_date_map(), selected_doc_ids={"contract_x"},
    )
    assert "c_amend" in allowed
    assert "c_law" not in allowed


def test_resolve_empty_set_equals_none():
    cm = {"c": make_chunk_dict(chunk_id="c", document_type="internal",
                               effective_from="2020-01-01")}
    a = resolve_allowed_versions_by_date("2025-01-01", cm, selected_doc_ids=set())
    b = resolve_allowed_versions_by_date("2025-01-01", cm, selected_doc_ids=None)
    assert a == b


def test_resolve_returns_set():
    allowed = resolve_allowed_versions_by_date("2025-01-01", _build_date_map())
    assert isinstance(allowed, set)


def test_hybrid_search_empty_allowed_returns_empty():
    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value={}):
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01", rerank=False,
        )
    assert results == []


def test_hybrid_search_returns_candidates():
    fake_chunks = {
        "c1": make_chunk_dict(
            chunk_id="c1", document_id="doc_p17", document_type="contract",
            effective_from="2024-02-12", effective_to="2026-12-31",
        ),
        "c2": make_chunk_dict(
            chunk_id="c2", document_id="doc_a08", document_type="contract",
            effective_from="2024-08-01", effective_to=None,
        ),
    }

    class FakeCollection:
        def query(self, **kwargs):
            return {"ids": [["c1"]], "documents": [["x"]], "metadatas": [[{}]]}

    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=fake_chunks), \
         patch.object(hybrid_mod.bm25_store, "search", return_value=[("c1", 5.0), ("c2", 2.0)]), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=FakeCollection()):
        results = hybrid_mod.hybrid_search(
            query="срок", top_k=5, event_date="2025-01-01", rerank=False,
        )

    assert len(results) >= 1
    ids = [r["chunk_id"] for r in results]
    assert "c1" in ids


def test_hybrid_search_respects_date_filter():
    fake_chunks = {
        "old": make_chunk_dict(
            chunk_id="old", document_id="doc_old", document_type="internal",
            effective_from="2023-01-01", effective_to="2025-03-01",
        ),
        "new": make_chunk_dict(
            chunk_id="new", document_id="doc_new", document_type="internal",
            effective_from="2025-03-01", effective_to=None,
        ),
    }

    class FakeCollection:
        def query(self, **kwargs):
            return {"ids": [["old", "new"]], "documents": [["", ""]], "metadatas": [[{}, {}]]}

    def fake_bm25_search(query, top_k, allowed_chunk_ids=None):
        results = [("old", 5.0), ("new", 3.0)]
        if allowed_chunk_ids is not None:
            results = [(cid, sc) for cid, sc in results if cid in allowed_chunk_ids]
        return results[:top_k]

    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=fake_chunks), \
         patch.object(hybrid_mod.bm25_store, "search", side_effect=fake_bm25_search), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=FakeCollection()):
        results = hybrid_mod.hybrid_search(
            query="порог", top_k=5, event_date="2026-01-01", rerank=False,
        )

    ids = [r["chunk_id"] for r in results]
    assert "new" in ids
    assert "old" not in ids


def test_hybrid_search_scores_in_range():
    fake_chunks = {
        "c1": make_chunk_dict(chunk_id="c1", effective_from="2020-01-01"),
    }

    class FakeCollection:
        def query(self, **kwargs):
            return {"ids": [["c1"]], "documents": [["x"]], "metadatas": [[{}]]}

    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=fake_chunks), \
         patch.object(hybrid_mod.bm25_store, "search", return_value=[("c1", 5.0)]), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=FakeCollection()):
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01", rerank=False,
        )

    for r in results:
        assert 0 < r["score"] < 1


def test_hybrid_search_amendment_bonus():
    fake_chunks = {
        "amend": make_chunk_dict(
            chunk_id="amend", document_id="amend_1",
            amends_document_id="contract_x",
            effective_from="2024-01-01",
        ),
        "base": make_chunk_dict(
            chunk_id="base", document_id="contract_x",
            effective_from="2024-01-01",
        ),
    }

    class FakeCollection:
        def query(self, **kwargs):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}

    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=fake_chunks), \
         patch.object(hybrid_mod.bm25_store, "search", return_value=[("base", 5.0), ("amend", 4.9)]), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=FakeCollection()):
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01", rerank=False,
        )

    assert results[0]["chunk_id"] == "amend"


def test_hybrid_search_with_rerank_calls_reranker():
    fake_chunks = {"c1": make_chunk_dict(chunk_id="c1", effective_from="2020-01-01")}

    class FakeCollection:
        def query(self, **kwargs):
            return {"ids": [["c1"]], "documents": [["x"]], "metadatas": [[{}]]}

    with patch.object(hybrid_mod, "load_canonical_chunks_map", return_value=fake_chunks), \
         patch.object(hybrid_mod.bm25_store, "search", return_value=[("c1", 5.0)]), \
         patch.object(hybrid_mod, "get_query_embedding", return_value=[0.0] * 1024), \
         patch.object(hybrid_mod, "get_chroma_collection", return_value=FakeCollection()), \
         patch.object(hybrid_mod, "rerank_candidates",
                      return_value=[{**fake_chunks["c1"], "score": 0.99}]) as mock_rr:
        results = hybrid_mod.hybrid_search(
            query="q", top_k=5, event_date="2025-01-01", rerank=True,
        )

    mock_rr.assert_called_once()
    assert results[0]["score"] == 0.99