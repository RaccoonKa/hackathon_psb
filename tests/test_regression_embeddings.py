"""Регрессионные тесты HTTP-провайдера эмбеддингов (пункты 9–10).

Полностью офлайн: живой llama.cpp не нужен, HTTP-слой замокан через
``httpx.MockTransport``. Проверяются hash-fallback при недоступном сервере,
явная ошибка при запрещённом fallback, ``EmbeddingConfigError`` при пустом
``EMBEDDING_API_BASE_URL`` и совместимость заглушек локальной модели.
"""

import json
from unittest.mock import patch

import httpx

from rag_core.retrieval import vector_store as vs_mod
from rag_core.retrieval.vector_store import (
    EmbeddingConfigError,
    EmbeddingError,
    compute_embeddings,
    get_last_embedding_diagnostics,
    reset_embedding_diagnostics,
    reset_local_model_cache,
)

_DIM = 1024


def _vector(dim: int = _DIM, first: float = 0.1):
    return [first] + [0.0] * (dim - 1)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _down_handler(request: httpx.Request) -> httpx.Response:
    """Сервер эмбеддингов не запущен: соединение отклонено."""
    raise httpx.ConnectError("connection refused", request=request)


def _ok_handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content.decode("utf-8"))
    texts = payload.get("input") or []
    return httpx.Response(
        200,
        json={
            "data": [
                {"index": index, "embedding": _vector()}
                for index in range(len(texts))
            ]
        },
    )


def _reset() -> None:
    reset_local_model_cache()
    reset_embedding_diagnostics()


# --- 9. Hash-fallback при недоступном HTTP-сервере --------------------------


def test_api_unavailable_uses_hash_fallback():
    _reset()
    client = _client(_down_handler)
    try:
        with patch.object(vs_mod, "_get_http_client", return_value=client), \
             patch.object(vs_mod, "USE_SUBMIT", False), \
             patch.object(vs_mod, "ALLOW_HASH_FALLBACK", True), \
             patch.object(vs_mod, "EMBEDDING_API_MAX_RETRIES", 0):
            first = compute_embeddings(["первый текст"])
            second = compute_embeddings(["второй текст"])
            diagnostics = get_last_embedding_diagnostics()
    finally:
        client.close()
        _reset()

    assert len(first[0]) == _DIM
    assert len(second[0]) == _DIM
    assert diagnostics["provider"] == "hash_fallback"
    assert diagnostics["success"] is True
    assert diagnostics["fallback_used"] is True
    assert diagnostics["dimension"] == _DIM
    assert diagnostics["error_type"] == "EmbeddingError"


def test_api_available_uses_api_provider():
    _reset()
    client = _client(_ok_handler)
    try:
        with patch.object(vs_mod, "_get_http_client", return_value=client), \
             patch.object(vs_mod, "USE_SUBMIT", False), \
             patch.object(vs_mod, "ALLOW_HASH_FALLBACK", True):
            vectors = compute_embeddings(["текст"])
            diagnostics = get_last_embedding_diagnostics()
    finally:
        client.close()
        _reset()

    assert len(vectors[0]) == _DIM
    assert diagnostics["provider"] == "api"
    assert diagnostics["success"] is True
    assert diagnostics["fallback_used"] is False
    assert diagnostics["error_type"] is None


# --- 10. Оба провайдера недоступны ------------------------------------------


def test_api_unavailable_without_fallback_raises_embedding_error():
    _reset()
    client = _client(_down_handler)
    try:
        with patch.object(vs_mod, "_get_http_client", return_value=client), \
             patch.object(vs_mod, "USE_SUBMIT", False), \
             patch.object(vs_mod, "ALLOW_HASH_FALLBACK", False), \
             patch.object(vs_mod, "EMBEDDING_API_MAX_RETRIES", 0), \
             patch.object(
                 vs_mod,
                 "_fallback_embedding",
                 side_effect=AssertionError(
                     "hash-fallback отключён и не должен вызываться"
                 ),
             ):
            try:
                compute_embeddings(["текст без провайдера"])
            except EmbeddingError:
                pass
            else:
                raise AssertionError(
                    "Ожидалась EmbeddingError при недоступном сервере "
                    "и запрещённом hash-fallback"
                )

            diagnostics = get_last_embedding_diagnostics()
    finally:
        client.close()
        _reset()

    assert diagnostics["provider"] == "api"
    assert diagnostics["success"] is False
    assert diagnostics["fallback_used"] is False
    assert diagnostics["error_type"] == "EmbeddingError"


def test_empty_embedding_base_url_raises_config_error():
    _reset()
    try:
        with patch.object(vs_mod, "USE_SUBMIT", False), \
             patch.object(vs_mod, "ALLOW_HASH_FALLBACK", False), \
             patch.object(vs_mod, "EMBEDDING_API_BASE_URL", ""), \
             patch.object(
                 vs_mod,
                 "_get_http_client",
                 side_effect=AssertionError("HTTP не должен вызываться"),
             ):
            try:
                compute_embeddings(["текст"])
            except EmbeddingConfigError:
                pass
            else:
                raise AssertionError(
                    "Ожидалась EmbeddingConfigError при пустом "
                    "EMBEDDING_API_BASE_URL"
                )

            diagnostics = get_last_embedding_diagnostics()
    finally:
        _reset()

    assert diagnostics["provider"] == "api"
    assert diagnostics["success"] is False
    assert diagnostics["fallback_used"] is False
    assert diagnostics["error_type"] == "EmbeddingConfigError"


def test_submit_mode_forbids_hash_fallback():
    _reset()
    client = _client(_down_handler)
    try:
        with patch.object(vs_mod, "_get_http_client", return_value=client), \
             patch.object(vs_mod, "USE_SUBMIT", True), \
             patch.object(vs_mod, "ALLOW_HASH_FALLBACK", True), \
             patch.object(vs_mod, "EMBEDDING_API_MAX_RETRIES", 0):
            try:
                compute_embeddings(["текст"])
            except EmbeddingError:
                pass
            else:
                raise AssertionError(
                    "Ожидалась EmbeddingError: в SUBMIT hash-fallback запрещён"
                )
    finally:
        client.close()
        _reset()


# --- совместимость заглушек локальной модели --------------------------------


def test_local_model_status_is_disabled():
    status = vs_mod.get_local_model_status()
    assert status["enabled"] is False
    assert status["available"] is False
    assert vs_mod._load_local_model() == (None, None, None)
