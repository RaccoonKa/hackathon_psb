"""Глубокие тесты HTTP-слоя эмбеддингов (llama.cpp ``/v1/embeddings``).

Полностью офлайн: локальный сервер не нужен, HTTP-слой проверяется через
``httpx.MockTransport`` либо патч ``_request_embeddings``. Тестов с фейковыми
``torch``/``transformers`` больше нет — локальный инференс удалён.
"""

from unittest.mock import patch

import httpx

from rag_core.retrieval import vector_store as vs_mod
from rag_core.retrieval.vector_store import (
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingDimensionError,
    _api_embeddings,
    _build_headers,
    _embeddings_endpoint,
    _load_local_model,
    _local_model_enabled,
    _normalize_vector,
    _parse_embeddings_response,
    compute_embeddings,
    get_local_model_status,
    prewarm_retrieval_backends,
    reset_local_model_cache,
    warmup_local_embedding_model,
)

_DIM = 1024


def _vector(first: float = 0.1, dim: int = _DIM):
    return [first] + [0.0] * (dim - 1)


def _vector_payload(vectors, indices=None):
    if indices is None:
        indices = list(range(len(vectors)))
    return {
        "object": "list",
        "model": "bge-m3",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in zip(indices, vectors)
        ],
    }


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- совместимые заглушки локальной модели ----------------------------------


def test_local_model_stubs_are_disabled():
    assert _load_local_model() == (None, None, None)
    assert _local_model_enabled() is False


def test_local_model_status_reports_disabled():
    status = get_local_model_status()
    assert status["enabled"] is False
    assert status["available"] is False
    assert status["device"] is None
    assert status["error_type"] is None
    assert status["provider"] == "api"


# --- endpoint и заголовки ---------------------------------------------------


def test_embeddings_endpoint_appends_path():
    with patch.object(
        vs_mod, "EMBEDDING_API_BASE_URL", "http://127.0.0.1:8082/v1/"
    ):
        assert (
            _embeddings_endpoint()
            == "http://127.0.0.1:8082/v1/embeddings"
        )


def test_embeddings_endpoint_rejects_empty_url():
    with patch.object(vs_mod, "EMBEDDING_API_BASE_URL", "   "):
        try:
            _embeddings_endpoint()
        except EmbeddingConfigError:
            pass
        else:
            raise AssertionError("Ожидалась EmbeddingConfigError")


def test_embeddings_endpoint_rejects_non_http_url():
    with patch.object(
        vs_mod, "EMBEDDING_API_BASE_URL", "ftp://127.0.0.1:8082/v1"
    ):
        try:
            _embeddings_endpoint()
        except EmbeddingConfigError:
            pass
        else:
            raise AssertionError("Ожидалась EmbeddingConfigError")


def test_build_headers_includes_bearer_key():
    with patch.object(vs_mod, "EMBEDDING_API_KEY", "local"):
        headers = _build_headers()
    assert headers["Authorization"] == "Bearer local"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"


def test_build_headers_skips_empty_key():
    with patch.object(vs_mod, "EMBEDDING_API_KEY", ""):
        headers = _build_headers()
    assert "Authorization" not in headers


# --- парсинг ответа и нормализация ------------------------------------------


def test_parse_embeddings_response_sorts_by_index():
    payload = _vector_payload(
        [_vector(first=0.2), _vector(first=0.1)], indices=[1, 0]
    )
    vectors = _parse_embeddings_response(payload, 2)
    assert vectors[0][0] == 0.1
    assert vectors[1][0] == 0.2


def test_parse_embeddings_response_rejects_different_dimensions():
    payload = _vector_payload([_vector(), [0.1, 0.2, 0.3]])
    try:
        _parse_embeddings_response(payload, 2)
    except EmbeddingError:
        pass
    else:
        raise AssertionError("Ожидалась EmbeddingError")


def test_parse_embeddings_response_rejects_wrong_count():
    payload = _vector_payload([_vector()])
    try:
        _parse_embeddings_response(payload, 2)
    except EmbeddingError as exc:
        assert "неверное число векторов" in str(exc)
    else:
        raise AssertionError("Ожидалась EmbeddingError")


def test_parse_embeddings_response_rejects_dimension_mismatch():
    payload = _vector_payload([_vector(dim=8)])
    with patch.object(vs_mod, "EMBEDDING_DIMENSION", 1024):
        try:
            _parse_embeddings_response(payload, 1)
        except EmbeddingDimensionError:
            pass
        else:
            raise AssertionError("Ожидалась EmbeddingDimensionError")


def test_parse_embeddings_response_error_field():
    try:
        _parse_embeddings_response(
            {"error": {"message": "model not loaded"}}, 1
        )
    except EmbeddingError as exc:
        assert "model not loaded" in str(exc)
    else:
        raise AssertionError("Ожидалась EmbeddingError")


def test_normalize_vector_unit_length():
    normalized = _normalize_vector([3.0, 4.0])
    assert abs(sum(value * value for value in normalized) - 1.0) < 1e-9


def test_normalize_vector_zero_vector_unchanged():
    assert _normalize_vector([0.0, 0.0]) == [0.0, 0.0]


# --- батчинг и compute_embeddings -------------------------------------------


def test_api_embeddings_batches_requests():
    calls = []

    def fake_request(texts):
        calls.append(list(texts))
        return [_vector() for _ in texts]

    with patch.object(vs_mod, "_request_embeddings", side_effect=fake_request):
        result = _api_embeddings(["a", "b", "c", "d", "e"], batch_size=2)

    assert len(result) == 5
    assert [len(batch) for batch in calls] == [2, 2, 1]


def test_api_embeddings_empty_input_no_requests():
    with patch.object(vs_mod, "_request_embeddings") as mock_request:
        assert _api_embeddings([]) == []
    assert mock_request.call_count == 0


def test_api_embeddings_replaces_blank_texts():
    captured = {}

    def fake_request(texts):
        captured["texts"] = list(texts)
        return [_vector(dim=4) for _ in texts]

    with patch.object(
        vs_mod, "_request_embeddings", side_effect=fake_request
    ), patch.object(vs_mod, "EMBEDDING_NORMALIZE", False), patch.object(
        vs_mod, "EMBEDDING_DIMENSION", 0
    ):
        _api_embeddings(["ok", "   ", ""])

    assert captured["texts"] == ["ok", " ", " "]


def test_compute_embeddings_empty_returns_empty():
    with patch.object(vs_mod, "_request_embeddings") as mock_request:
        assert compute_embeddings([]) == []
    assert mock_request.call_count == 0


def test_compute_embeddings_batches_by_batch_size():
    with patch.object(
        vs_mod,
        "_request_embeddings",
        side_effect=lambda texts: [_vector() for _ in texts],
    ) as mock_request:
        result = compute_embeddings(["a", "b", "c", "d", "e"], batch_size=2)

    assert len(result) == 5
    assert mock_request.call_count == 3


def test_compute_embeddings_uses_http_server():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _vector_payload([_vector()])
        return httpx.Response(200, json=payload)

    client = _client(handler)
    try:
        with patch.object(vs_mod, "_get_http_client", return_value=client):
            result = compute_embeddings(["запрос"])
    finally:
        client.close()

    assert len(result) == 1
    assert len(result[0]) == _DIM
    diagnostics = vs_mod.get_last_embedding_diagnostics()
    assert diagnostics["provider"] == "api"
    assert diagnostics["success"] is True
    assert diagnostics["fallback_used"] is False


# --- warmup / prewarm -------------------------------------------------------


def test_warmup_uses_http_layer():
    with patch.object(
        vs_mod,
        "_request_embeddings",
        side_effect=lambda texts: [_vector() for _ in texts],
    ) as mock_request:
        report = warmup_local_embedding_model()

    assert report["status"] == "ok"
    assert report["texts"] == 4
    assert report["dimension"] == _DIM
    assert report["warmup_ms"] is not None
    assert mock_request.call_count == 1


def test_warmup_error_does_not_raise():
    with patch.object(
        vs_mod,
        "_request_embeddings",
        side_effect=EmbeddingError("server is down"),
    ):
        report = warmup_local_embedding_model()

    assert report["status"] == "error"
    assert report["error_type"] == "EmbeddingError"
    assert report["dimension"] is None


def test_prewarm_report_keeps_contract_keys():
    reset_local_model_cache()
    try:
        with patch.object(
            vs_mod,
            "_request_embeddings",
            side_effect=lambda texts: [_vector() for _ in texts],
        ), patch.object(
            vs_mod,
            "get_chroma_collection",
            side_effect=RuntimeError("chroma недоступна"),
        ):
            report = prewarm_retrieval_backends()
    finally:
        reset_local_model_cache()

    for key in (
        "local_model",
        "collection",
        "collection_size",
        "dimension",
        "warmup",
        "warmup_ms",
        "warmup_dimension",
        "errors",
    ):
        assert key in report

    assert report["local_model"] == "disabled"
    assert report["collection"] == "error"
    assert report["warmup"] == "ok"
    assert report["embedding_provider"] == "api"
    assert report["embedding_endpoint"].endswith("/embeddings")
