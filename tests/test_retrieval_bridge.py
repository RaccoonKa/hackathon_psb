from unittest.mock import patch

from src import config
from src.generation.retrieval import bridge as rb_mod
from src.generation.retrieval.bridge import (
    _dict_to_chunk,
    filter_chunks_by_threshold,
    rerank_chunks,
    retrieve_chunks,
)
from src.generation.types import Intent, RetrievedChunk
from tests._common import make_chunk_dict, make_retrieved_chunk


def test_dict_to_chunk_full():
    raw = make_chunk_dict(chunk_id="x", score=0.9)
    ch = _dict_to_chunk(raw)
    assert isinstance(ch, RetrievedChunk)
    assert ch.chunk_id == "x"
    assert ch.score == 0.9
    assert ch.provision_id == "1.1"
    assert ch.structure_path == ["Раздел 1"]


def test_dict_to_chunk_missing_score_uses_default():
    raw = {
        "chunk_id": "x", "text": "t", "document_id": "d", "version_id": "v",
        "document_title": "T",
    }
    ch = _dict_to_chunk(raw)
    assert ch.score == config.REAL_RAG_DEFAULT_SCORE


def test_dict_to_chunk_missing_structure():
    raw = {
        "chunk_id": "x", "text": "t", "document_id": "d", "version_id": "v",
        "document_title": "T", "score": 0.5,
    }
    ch = _dict_to_chunk(raw)
    assert ch.provision_id is None
    assert ch.structure_path == []


def test_dict_to_chunk_missing_path():
    raw = make_chunk_dict()
    raw["structure"] = {"provision_id": "1.1"}
    ch = _dict_to_chunk(raw)
    assert ch.structure_path == []


def test_dict_to_chunk_amends():
    raw = make_chunk_dict(amends_document_id="parent_1")
    ch = _dict_to_chunk(raw)
    assert ch.amends_document_id == "parent_1"


def test_dict_to_chunk_effective_dates():
    raw = make_chunk_dict(effective_from="2024-01-01", effective_to="2025-01-01")
    ch = _dict_to_chunk(raw)
    assert ch.effective_from == "2024-01-01"
    assert ch.effective_to == "2025-01-01"


def test_retrieve_chunks_real_mode():
    fake = [make_chunk_dict(chunk_id="c1", score=0.8)]
    with patch.object(rb_mod.config, "USE_STUB_RAG", False), \
         patch.object(rb_mod, "hybrid_search", return_value=fake) as mock_hs:
        chunks, step = retrieve_chunks(query="q", top_k=5)
    mock_hs.assert_called_once()
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    assert step.step == config.STEP_RAG_RETRIEVAL
    assert step.input["mode"] == "real"


def test_retrieve_chunks_passes_event_date():
    with patch.object(rb_mod.config, "USE_STUB_RAG", False), \
         patch.object(rb_mod, "hybrid_search", return_value=[]) as mock_hs:
        retrieve_chunks(query="q", top_k=5, event_date="2025-01-01")
    assert mock_hs.call_args.kwargs["event_date"] == "2025-01-01"


def test_retrieve_chunks_passes_selected_doc_ids():
    with patch.object(rb_mod.config, "USE_STUB_RAG", False), \
         patch.object(rb_mod, "hybrid_search", return_value=[]) as mock_hs:
        retrieve_chunks(query="q", top_k=5, selected_doc_ids={"d1", "d2"})
    passed = mock_hs.call_args.kwargs["selected_doc_ids"]
    assert passed == {"d1", "d2"}


def test_retrieve_chunks_passes_intent():
    with patch.object(rb_mod.config, "USE_STUB_RAG", False), \
         patch.object(rb_mod, "hybrid_search", return_value=[]) as mock_hs:
        retrieve_chunks(query="q", top_k=5, intent=Intent.EXACT_LOOKUP)
    assert mock_hs.call_args.kwargs["intent"] == "exact_lookup"


def test_retrieve_chunks_disables_inner_rerank():
    # Реранкинг выполняется отдельным этапом rerank_chunks (локальный Qwen),
    # поэтому внутри hybrid_search он должен быть выключен — без двойного реранка.
    with patch.object(rb_mod.config, "USE_STUB_RAG", False), \
         patch.object(rb_mod, "hybrid_search", return_value=[]) as mock_hs:
        retrieve_chunks(query="q", top_k=5)
    assert mock_hs.call_args.kwargs["rerank"] is False


def test_retrieve_chunks_passes_top_k_unchanged():
    # Размер пула (RAG_TOP_K) задаётся планировщиком; мост не масштабирует top_k.
    with patch.object(rb_mod.config, "USE_STUB_RAG", False), \
         patch.object(rb_mod, "hybrid_search", return_value=[]) as mock_hs:
        retrieve_chunks(query="q", top_k=5)
    assert mock_hs.call_args.kwargs["top_k"] == 5


def test_retrieve_chunks_stub_mode():
    with patch.object(rb_mod.config, "USE_STUB_RAG", True), \
         patch.object(rb_mod, "stub_rag_retrieve",
                      return_value=[make_retrieved_chunk(chunk_id="stub")]):
        chunks, step = retrieve_chunks(query="q", top_k=5)
    assert step.input["mode"] == "stub"
    assert chunks[0].chunk_id == "stub"


def test_retrieve_chunks_trace_output():
    fake = [make_chunk_dict(chunk_id="c1", score=0.8)]
    with patch.object(rb_mod.config, "USE_STUB_RAG", False), \
         patch.object(rb_mod, "hybrid_search", return_value=fake):
        _, step = retrieve_chunks(query="q", top_k=5)
    assert step.output["chunk_ids"] == ["c1"]
    assert step.output["count"] == 1


def test_rerank_chunks_real_mode():
    chunks = [
        make_retrieved_chunk(chunk_id="c1"),
        make_retrieved_chunk(chunk_id="c2"),
    ]
    fake_reranked = [{**make_chunk_dict(chunk_id="c1", score=0.99)}]
    with patch.object(rb_mod.config, "USE_STUB_RERANKER", False), \
         patch.object(rb_mod, "rerank_candidates",
                      return_value=fake_reranked) as mock_rr:
        reranked, step = rerank_chunks(chunks=chunks, query="q", top_n=3)
    mock_rr.assert_called_once()
    assert len(reranked) == 1
    assert step.input["mode"] == "real"


def test_rerank_chunks_builds_candidate_dicts():
    chunks = [
        make_retrieved_chunk(chunk_id="c1", provision_id="3.2"),
        make_retrieved_chunk(chunk_id="c2"),
    ]
    with patch.object(rb_mod.config, "USE_STUB_RERANKER", False), \
         patch.object(rb_mod, "rerank_candidates", return_value=[]) as mock_rr:
        rerank_chunks(chunks=chunks, query="q", top_n=3)
    passed = mock_rr.call_args.kwargs["candidates"]
    assert passed[0]["chunk_id"] == "c1"
    assert passed[0]["structure"]["provision_id"] == "3.2"


def test_rerank_chunks_uses_embedding_text():
    chunks = [
        make_retrieved_chunk(
            chunk_id="c1", document_title="Договор",
            structure_path=["Раздел 1"], text="тело",
        ),
        make_retrieved_chunk(chunk_id="c2"),
    ]
    with patch.object(rb_mod.config, "USE_STUB_RERANKER", False), \
         patch.object(rb_mod, "rerank_candidates", return_value=[]) as mock_rr:
        rerank_chunks(chunks=chunks, query="q", top_n=3)
    passed = mock_rr.call_args.kwargs["candidates"]
    assert "Договор" in passed[0]["embedding_text"]


def test_rerank_chunks_single_candidate_passthrough():
    chunks = [make_retrieved_chunk(chunk_id="c1")]
    with patch.object(rb_mod.config, "USE_STUB_RERANKER", False), \
         patch.object(rb_mod, "rerank_candidates") as mock_rr:
        reranked, step = rerank_chunks(chunks=chunks, query="q", top_n=3)
    mock_rr.assert_not_called()
    assert step.input["mode"] == "passthrough"
    assert reranked == chunks


def test_rerank_chunks_empty_result_passthrough():
    chunks = [
        make_retrieved_chunk(chunk_id="c1"),
        make_retrieved_chunk(chunk_id="c2"),
    ]
    with patch.object(rb_mod.config, "USE_STUB_RERANKER", False), \
         patch.object(rb_mod, "rerank_candidates", return_value=[]):
        reranked, step = rerank_chunks(chunks=chunks, query="q", top_n=3)
    assert step.input["mode"] == "passthrough"
    assert [c.chunk_id for c in reranked] == ["c1", "c2"]


def test_rerank_chunks_stub_mode():
    chunks = [make_retrieved_chunk(chunk_id="c1")]
    with patch.object(rb_mod.config, "USE_STUB_RERANKER", True), \
         patch.object(rb_mod, "stub_reranker", return_value=chunks):
        result, step = rerank_chunks(chunks=chunks, query="q", top_n=3)
    assert step.input["mode"] == "stub"
    assert result == chunks


def test_filter_empty_fails():
    passed, out, step = filter_chunks_by_threshold([], threshold=0.0)
    assert not passed
    assert out == []
    assert step.step == config.STEP_THRESHOLD_FILTER


def test_filter_above_threshold():
    chunks = [make_retrieved_chunk(chunk_id="a", score=0.9)]
    passed, out, _ = filter_chunks_by_threshold(chunks, threshold=0.5)
    assert passed
    assert len(out) == 1


def test_filter_at_threshold_passes():
    chunks = [make_retrieved_chunk(chunk_id="a", score=0.5)]
    passed, out, _ = filter_chunks_by_threshold(chunks, threshold=0.5)
    assert passed
    assert len(out) == 1


def test_filter_below_threshold_fails():
    chunks = [make_retrieved_chunk(chunk_id="a", score=0.4)]
    passed, out, _ = filter_chunks_by_threshold(chunks, threshold=0.5)
    assert not passed
    assert out == []


def test_filter_adaptive_rrf_low_scores():
    chunks = [make_retrieved_chunk(chunk_id="a", score=0.03)]
    passed, out, step = filter_chunks_by_threshold(chunks, threshold=0.1)
    assert passed
    assert step.input["effective_threshold"] <= 0.01


def test_filter_adaptive_not_triggered_above_0_1():
    chunks = [make_retrieved_chunk(chunk_id="a", score=0.15)]
    passed, _, step = filter_chunks_by_threshold(chunks, threshold=0.1)
    assert passed
    assert step.input["effective_threshold"] == 0.1


def test_filter_trace_input_output():
    chunks = [make_retrieved_chunk(chunk_id="a", score=0.8)]
    _, _, step = filter_chunks_by_threshold(chunks, threshold=0.5)
    assert step.input["threshold"] == 0.5
    assert step.input["input_count"] == 1
    assert step.input["max_score"] == 0.8
    assert step.output["passed"] is True


def test_filter_zero_threshold_always_passes():
    chunks = [make_retrieved_chunk(chunk_id="a", score=0.001)]
    passed, _, _ = filter_chunks_by_threshold(chunks, threshold=0.0)
    assert passed
