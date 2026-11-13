from unittest.mock import MagicMock, patch

from rag_core.retrieval import reranker as rr_mod
from rag_core.retrieval.reranker import rerank_candidates


def _cands(n=3):
    return [
        {"chunk_id": f"c{i}", "text": f"текст {i}",
         "embedding_text": f"документ {i} текст {i}"}
        for i in range(n)
    ]


def test_rerank_empty_returns_empty():
    assert rerank_candidates("q", [], top_k=5) == []


def test_rerank_single_candidate_returns_as_is():
    c = [{"chunk_id": "c1", "text": "t", "embedding_text": "d"}]
    result = rerank_candidates("q", c, top_k=5)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "c1"


def test_rerank_success_reorders():
    cands = _cands(3)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.1},
        ]
    }
    with patch.object(rr_mod.requests, "post", return_value=mock_resp):
        result = rerank_candidates("q", cands, top_k=3)
    assert [r["chunk_id"] for r in result] == ["c2", "c0", "c1"]
    assert result[0]["score"] == 0.9


def test_rerank_limits_to_top_k():
    cands = _cands(5)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [{"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(5)]
    }
    with patch.object(rr_mod.requests, "post", return_value=mock_resp):
        result = rerank_candidates("q", cands, top_k=2)
    assert len(result) == 2


def test_rerank_fallback_on_http_error():
    cands = _cands(3)
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    with patch.object(rr_mod.requests, "post", return_value=mock_resp):
        result = rerank_candidates("q", cands, top_k=3)
    assert [r["chunk_id"] for r in result] == ["c0", "c1", "c2"]


def test_rerank_fallback_on_timeout():
    cands = _cands(3)
    with patch.object(rr_mod.requests, "post", side_effect=Exception("timeout")):
        result = rerank_candidates("q", cands, top_k=3)
    assert [r["chunk_id"] for r in result] == ["c0", "c1", "c2"]


def test_rerank_fallback_on_empty_results():
    cands = _cands(3)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": []}
    with patch.object(rr_mod.requests, "post", return_value=mock_resp):
        result = rerank_candidates("q", cands, top_k=3)
    assert [r["chunk_id"] for r in result] == ["c0", "c1", "c2"]


def test_rerank_ignores_out_of_range_index():
    cands = _cands(2)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [{"index": 99, "relevance_score": 0.9}]
    }
    with patch.object(rr_mod.requests, "post", return_value=mock_resp):
        result = rerank_candidates("q", cands, top_k=2)
    assert [r["chunk_id"] for r in result] == ["c0", "c1"]


def test_rerank_uses_embedding_text():
    cands = _cands(2)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
    with patch.object(rr_mod.requests, "post", return_value=mock_resp) as mock_post:
        rerank_candidates("q", cands, top_k=2)
    call_kwargs = mock_post.call_args.kwargs
    docs = call_kwargs["json"]["documents"]
    assert "документ 0" in docs[0]


def test_rerank_score_fallback_to_score_field():
    cands = _cands(2)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"index": 0, "score": 0.7}]}
    with patch.object(rr_mod.requests, "post", return_value=mock_resp):
        result = rerank_candidates("q", cands, top_k=2)
    assert result[0]["score"] == 0.7


def test_rerank_preserves_metadata():
    cands = [
        {"chunk_id": "c1", "text": "t", "embedding_text": "e",
         "document_id": "d1", "custom": "x"}
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.8}]}
    with patch.object(rr_mod.requests, "post", return_value=mock_resp):
        result = rerank_candidates("q", cands, top_k=1)
    assert result[0]["document_id"] == "d1"
    assert result[0]["custom"] == "x"


def test_rerank_top_n_not_exceed_candidates():
    cands = _cands(2)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"results": [{"index": 0, "relevance_score": 0.5}]}
    with patch.object(rr_mod.requests, "post", return_value=mock_resp) as mock_post:
        rerank_candidates("q", cands, top_k=10)
    assert mock_post.call_args.kwargs["json"]["top_n"] == 2