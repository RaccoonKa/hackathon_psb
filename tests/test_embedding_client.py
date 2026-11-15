"""Офлайн-тесты HTTP-клиента эмбеддингов llama.cpp (POST /v1/embeddings).

Живой сервер не нужен: транспорт подменяется ``httpx.MockTransport``, поэтому
проверяются формат запроса/ответа, батчинг, ретраи, заголовки и таймаут.
"""

import json
from unittest.mock import patch

import httpx

from rag_core.retrieval import vector_store as vs_mod
from rag_core.retrieval.vector_store import (
    EmbeddingDimensionError,
    EmbeddingError,
    _api_embeddings,
    _request_embeddings,
    compute_embeddings,
)

_DIM = 1024


def _vector(first: float = 0.1, dim: int = _DIM):
    return [first] + [0.0] * (dim - 1)


def _payload(vectors, indices=None):
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


class _Transport:
    """Транспорт с очередью действий; последнее действие повторяется."""

    def __init__(self, *actions):
        self.actions = list(actions)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        action = (
            self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]
        )
        if isinstance(action, httpx.Response):
            return action
        return action(request)

    @property
    def count(self) -> int:
        return len(self.requests)

    def last_input(self):
        return json.loads(
            self.requests[-1].content.decode("utf-8")
        ).get("input")


def _echo_ok(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content.decode("utf-8")).get("input") or []
    return httpx.Response(
        200, json=_payload([_vector() for _ in texts])
    )


def _run(transport: _Transport):
    """Патч клиента и запрет реальных пауз между повторами."""
    client = httpx.Client(transport=httpx.MockTransport(transport))
    patcher_client = patch.object(
        vs_mod, "_get_http_client", return_value=client
    )
    patcher_sleep = patch.object(vs_mod, "_sleep_before_retry")
    patcher_client.start()
    patcher_sleep.start()
    return client, patcher_client, patcher_sleep


def _stop(client, *patchers):
    for patcher in patchers:
        patcher.stop()
    client.close()


def test_parses_data_with_shuffled_index():
    transport = _Transport(
        httpx.Response(
            200,
            json=_payload(
                [_vector(first=0.2), _vector(first=0.1)], indices=[1, 0]
            ),
        )
    )
    client, pc, ps = _run(transport)
    try:
        vectors = _request_embeddings(["a", "b"])
    finally:
        _stop(client, pc, ps)

    assert vectors[0][0] == 0.1
    assert vectors[1][0] == 0.2
    assert transport.count == 1


def test_one_post_per_batch():
    transport = _Transport(_echo_ok)
    client, pc, ps = _run(transport)
    try:
        result = _api_embeddings(["a", "b"], batch_size=2)
    finally:
        _stop(client, pc, ps)

    assert len(result) == 2
    assert transport.count == 1
    assert transport.last_input() == ["a", "b"]


def test_three_posts_for_five_texts_in_batches_of_two():
    transport = _Transport(_echo_ok)
    client, pc, ps = _run(transport)
    try:
        result = _api_embeddings(["a", "b", "c", "d", "e"], batch_size=2)
    finally:
        _stop(client, pc, ps)

    assert len(result) == 5
    assert transport.count == 3


def test_empty_input_makes_no_requests():
    transport = _Transport(_echo_ok)
    client, pc, ps = _run(transport)
    try:
        assert _api_embeddings([]) == []
        assert compute_embeddings([]) == []
    finally:
        _stop(client, pc, ps)

    assert transport.count == 0


def test_retries_on_503_then_succeeds():
    transport = _Transport(
        httpx.Response(503, json={"error": {"message": "busy"}}),
        _echo_ok,
    )
    client, pc, ps = _run(transport)
    try:
        with patch.object(vs_mod, "EMBEDDING_API_MAX_RETRIES", 1):
            vectors = _request_embeddings(["a"])
    finally:
        _stop(client, pc, ps)

    assert len(vectors) == 1
    assert transport.count == 2


def test_no_retry_on_400():
    transport = _Transport(
        httpx.Response(400, json={"error": {"message": "bad input"}})
    )
    client, pc, ps = _run(transport)
    try:
        with patch.object(vs_mod, "EMBEDDING_API_MAX_RETRIES", 3):
            try:
                _request_embeddings(["a"])
            except EmbeddingError:
                pass
            else:
                raise AssertionError("Ожидалась EmbeddingError на HTTP 400")
    finally:
        _stop(client, pc, ps)

    assert transport.count == 1


def test_authorization_header_uses_configured_key():
    transport = _Transport(_echo_ok)
    client, pc, ps = _run(transport)
    try:
        with patch.object(vs_mod, "EMBEDDING_API_KEY", "local"):
            _request_embeddings(["a"])
    finally:
        _stop(client, pc, ps)

    assert (
        transport.requests[0].headers.get("authorization")
        == "Bearer local"
    )


def test_client_timeout_taken_from_config():
    vs_mod.reset_embedding_http_client()
    try:
        with patch.object(vs_mod, "EMBEDDING_TIMEOUT", 42.0):
            client = vs_mod._get_http_client()
        assert client.timeout.connect == 42.0
        assert client.timeout.read == 42.0
        assert client.timeout.write == 42.0
        assert client.timeout.pool == 42.0
    finally:
        vs_mod.reset_embedding_http_client()


def test_rejects_vectors_with_different_dimensions():
    transport = _Transport(
        httpx.Response(
            200,
            json=_payload([_vector(), [0.1, 0.2, 0.3]]),
        )
    )
    client, pc, ps = _run(transport)
    try:
        try:
            _request_embeddings(["a", "b"])
        except EmbeddingError:
            pass
        else:
            raise AssertionError(
                "Ожидалась EmbeddingError при разной размерности векторов"
            )
    finally:
        _stop(client, pc, ps)


def test_rejects_vector_with_unexpected_dimension():
    transport = _Transport(
        httpx.Response(200, json=_payload([_vector(dim=8)]))
    )
    client, pc, ps = _run(transport)
    try:
        with patch.object(vs_mod, "EMBEDDING_DIMENSION", 1024):
            try:
                _request_embeddings(["a"])
            except EmbeddingDimensionError:
                pass
            else:
                raise AssertionError(
                    "Ожидалась EmbeddingDimensionError при несовпадении "
                    "с EMBEDDING_DIMENSION"
                )
    finally:
        _stop(client, pc, ps)


def test_error_field_in_body_raises_embedding_error():
    transport = _Transport(
        httpx.Response(
            200, json={"error": {"message": "model 'bge-m3' not loaded"}}
        )
    )
    client, pc, ps = _run(transport)
    try:
        try:
            _request_embeddings(["a"])
        except EmbeddingError as exc:
            assert "not loaded" in str(exc)
        else:
            raise AssertionError(
                "Ожидалась EmbeddingError при error в теле ответа"
            )
    finally:
        _stop(client, pc, ps)


def test_model_and_input_sent_in_payload():
    transport = _Transport(_echo_ok)
    client, pc, ps = _run(transport)
    try:
        with patch.object(vs_mod, "EMBEDDING_MODEL", "bge-m3"):
            _request_embeddings(["текст"])
    finally:
        _stop(client, pc, ps)

    body = json.loads(transport.requests[0].content.decode("utf-8"))
    assert body["model"] == "bge-m3"
    assert body["input"] == ["текст"]
