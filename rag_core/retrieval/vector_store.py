"""Векторное хранилище и HTTP-провайдер эмбеддингов (LOCAL/SUBMIT).

Эмбеддинги считает llama.cpp, запущенный как OpenAI-совместимый сервер:
``POST {EMBEDDING_API_BASE_URL}/embeddings``. Локальный инференс через
``transformers``/``torch`` в проекте больше не используется.

Ленивые инициализации (HTTP-клиент ``httpx``, коллекция Chroma) защищены
модульными блокировками с double-checked locking: параллельные
retrieval-задачи не должны одновременно создавать ``httpx.Client`` или
``PersistentClient``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from rag_core.config import (
    ALLOW_HASH_FALLBACK,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_API_MAX_RETRIES,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_TIMEOUT,
    USE_SUBMIT,
)

logger = logging.getLogger(__name__)

# Размерность hash-fallback (совпадает с bge-m3).
HASH_FALLBACK_DIMENSION = 1024
# Статусы, при которых запрос имеет смысл повторить.
_RETRYABLE_STATUSES: frozenset = frozenset({429, 500, 502, 503, 504})
# Верхняя граница паузы между повторами (секунды).
_MAX_RETRY_BACKOFF_S: float = 5.0

_http_client: Optional[httpx.Client] = None
_http_client_signature: Optional[tuple] = None
_chroma_collection = None
_collection_dimension: Optional[int] = None

_API_LOCK = threading.Lock()
_CHROMA_LOCK = threading.Lock()
_DIMENSION_LOCK = threading.Lock()
_PREWARM_LOCK = threading.Lock()

_prewarm_done = False
_prewarm_report: Dict[str, Any] = {}

# Диагностика последнего вызова эмбеддингов в текущем потоке (для bridge).
_thread_state = threading.local()


class EmbeddingError(RuntimeError):
    """Ошибка получения эмбеддингов без раскрытия секретов."""


class EmbeddingConfigError(EmbeddingError):
    """Невалидная конфигурация провайдера эмбеддингов (URL/ключ)."""


class EmbeddingDimensionError(EmbeddingError):
    """Размерность эмбеддингов не совпадает с ожидаемой/сохранённой."""


# --- совместимые заглушки локального инференса ------------------------------
# Локальная torch-модель удалена; функции оставлены, потому что их ожидают
# тесты, verify_submit.py и внешние импорты.


def _local_model_enabled() -> bool:
    """Локальный torch-инференс отключён навсегда (no-op совместимость)."""
    return False


def _load_local_model():
    """Совместимая заглушка: (None, None, None) — локальной модели нет."""
    return None, None, None


def reset_local_model_cache() -> None:
    """Сброс служебных кэшей провайдера эмбеддингов и prewarm-отчёта.

    Сохранён как публичный контракт (тесты, verify_submit.py).
    """
    global _prewarm_done, _prewarm_report

    with _PREWARM_LOCK:
        _prewarm_done = False
        _prewarm_report = {}
    reset_embedding_http_client()


def get_local_model_status() -> Dict[str, Any]:
    """Диагностика провайдера эмбеддингов без раскрытия путей/секретов."""
    return {
        "enabled": False,
        "available": False,
        "device": None,
        "model": EMBEDDING_MODEL,
        "provider": "api",
        "error_type": None,
        "error_message": None,
    }


def reset_embedding_http_client() -> None:
    """Закрывает и сбрасывает ленивый HTTP-клиент эмбеддингов."""
    global _http_client, _http_client_signature

    with _API_LOCK:
        client = _http_client
        _http_client = None
        _http_client_signature = None
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001 - закрытие кэша не фатально
            logger.debug("Не удалось закрыть HTTP-клиент эмбеддингов")


# --- HTTP-клиент llama.cpp --------------------------------------------------


def _http_client_config_signature() -> tuple:
    """Снимок конфига, при изменении которого клиент пересоздаётся."""
    return (
        str(EMBEDDING_API_BASE_URL or ""),
        str(EMBEDDING_API_KEY or ""),
        float(EMBEDDING_TIMEOUT),
    )


def _get_http_client() -> httpx.Client:
    """Ленивый потокобезопасный httpx-клиент (пересоздаётся при смене конфига)."""
    global _http_client, _http_client_signature

    signature = _http_client_config_signature()
    client = _http_client
    if client is not None and _http_client_signature == signature:
        return client

    with _API_LOCK:
        # double-checked locking: параллельные задачи не создают клиенты.
        if _http_client is not None and _http_client_signature == signature:
            return _http_client

        previous = _http_client
        _http_client = None
        if previous is not None:
            try:
                previous.close()
            except Exception:  # noqa: BLE001
                logger.debug("Не удалось закрыть прежний HTTP-клиент эмбеддингов")

        timeout_value = float(EMBEDDING_TIMEOUT)
        timeout = httpx.Timeout(
            timeout_value,
            connect=timeout_value,
            read=timeout_value,
            write=timeout_value,
            pool=timeout_value,
        )
        _http_client = httpx.Client(timeout=timeout)
        _http_client_signature = signature
        return _http_client


def _embeddings_endpoint() -> str:
    """URL эндпоинта эмбеддингов; невалидный/пустой base -> ConfigError."""
    base = (EMBEDDING_API_BASE_URL or "").strip()
    if not base:
        raise EmbeddingConfigError(
            "Не задан EMBEDDING_API_BASE_URL. Укажите URL локального "
            "llama.cpp эмбеддинг-сервера, например http://127.0.0.1:8082/v1"
        )

    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise EmbeddingConfigError(
            "EMBEDDING_API_BASE_URL должен быть абсолютным http(s) URL "
            "(значение не логируется)."
        )
    return f"{base.rstrip('/')}/embeddings"


def _build_headers() -> Dict[str, str]:
    """Заголовки запроса; ключ передаётся только в Authorization."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    api_key = (EMBEDDING_API_KEY or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _validate_embeddings(
    vectors: List[List[float]], expected_count: int
) -> None:
    if len(vectors) != expected_count:
        raise EmbeddingError(
            "API эмбеддингов вернул неверное число векторов: "
            f"ожидалось {expected_count}, получено {len(vectors)}"
        )

    dimensions = set()
    for vector in vectors:
        if not vector:
            raise EmbeddingError("API эмбеддингов вернул пустой вектор")
        dimensions.add(len(vector))

    if len(dimensions) != 1:
        raise EmbeddingError(
            "API эмбеддингов вернул векторы разной размерности: "
            f"{sorted(dimensions)}"
        )

    actual_dimension = dimensions.pop()
    if EMBEDDING_DIMENSION and actual_dimension != EMBEDDING_DIMENSION:
        raise EmbeddingDimensionError(
            "Размерность эмбеддингов не соответствует ожидаемой: "
            f"получено {actual_dimension}, ожидалось {EMBEDDING_DIMENSION}. "
            "Проверьте EMBEDDING_MODEL/EMBEDDING_DIMENSION."
        )


def _parse_embeddings_response(
    payload: Any, expected_count: int
) -> List[List[float]]:
    """Разбирает ответ ``/embeddings``: сортировка по index и валидация."""
    if not isinstance(payload, dict):
        raise EmbeddingError(
            "Ответ сервера эмбеддингов имеет неожиданный формат"
        )

    error = payload.get("error")
    if error:
        if isinstance(error, dict):
            message = str(
                error.get("message") or error.get("type") or error
            )[:300]
        else:
            message = str(error)[:300]
        raise EmbeddingError(f"Сервер эмбеддингов вернул ошибку: {message}")

    raw_items = payload.get("data")
    if not isinstance(raw_items, list):
        raise EmbeddingError("В ответе сервера эмбеддингов нет поля data")

    indexed: List[tuple] = []
    for position, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise EmbeddingError(
                "Элемент data в ответе эмбеддингов не является объектом"
            )
        raw_embedding = item.get("embedding")
        if not isinstance(raw_embedding, (list, tuple)):
            raise EmbeddingError(
                "Элемент data в ответе эмбеддингов не содержит embedding"
            )
        try:
            vector = [float(value) for value in raw_embedding]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError(
                "Эмбеддинг содержит нечисловые значения"
            ) from exc

        index = item.get("index")
        indexed.append((index if isinstance(index, int) else position, vector))

    # Некоторые серверы/прокси возвращают элементы не по порядку.
    indexed.sort(key=lambda pair: pair[0])
    vectors = [vector for _, vector in indexed]
    _validate_embeddings(vectors, expected_count)
    return vectors


def _normalize_vector(vector: List[float]) -> List[float]:
    """L2-нормализация; нулевой вектор возвращается как есть."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


def _sanitize_batch(texts: List[str]) -> List[str]:
    """Пустые/пробельные строки заменяются пробелом (llama.cpp даёт 400)."""
    cleaned: List[str] = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            cleaned.append(text)
        else:
            logger.debug(
                "Пустой текст в батче эмбеддингов заменён на пробел"
            )
            cleaned.append(" ")
    return cleaned


def _sleep_before_retry(attempt: int) -> None:
    """Экспоненциальный backoff: 0.5, 1, 2, 4 ... с ограничением 5 с."""
    time.sleep(min(_MAX_RETRY_BACKOFF_S, 0.5 * (2 ** max(0, attempt))))


def _request_embeddings(texts: List[str]) -> List[List[float]]:
    """Один HTTP-запрос ``POST {BASE}/embeddings`` с ретраями."""
    endpoint = _embeddings_endpoint()
    headers = _build_headers()
    payload = {
        "model": EMBEDDING_MODEL,
        "input": list(texts),
    }
    attempts = max(0, int(EMBEDDING_API_MAX_RETRIES)) + 1
    last_error: Optional[BaseException] = None

    for attempt in range(attempts):
        is_last = attempt + 1 >= attempts
        try:
            response = _get_http_client().post(
                endpoint, json=payload, headers=headers
            )
        except httpx.TimeoutException as exc:
            last_error = exc
            if is_last:
                raise EmbeddingError(
                    "Таймаут запроса к серверу эмбеддингов "
                    f"({type(exc).__name__}); проверьте EMBEDDING_TIMEOUT"
                ) from exc
            _sleep_before_retry(attempt)
            continue
        except httpx.TransportError as exc:
            last_error = exc
            if is_last:
                raise EmbeddingError(
                    "Сервер эмбеддингов недоступен "
                    f"({type(exc).__name__}); проверьте EMBEDDING_API_BASE_URL"
                ) from exc
            _sleep_before_retry(attempt)
            continue

        status = int(response.status_code)
        if status in _RETRYABLE_STATUSES:
            last_error = EmbeddingError(
                f"Сервер эмбеддингов вернул HTTP {status}"
            )
            if is_last:
                raise last_error
            _sleep_before_retry(attempt)
            continue

        if status in {401, 403, 404}:
            raise EmbeddingConfigError(
                "Сервер эмбеддингов отклонил запрос "
                f"(HTTP {status}). Проверьте EMBEDDING_API_BASE_URL/"
                "EMBEDDING_MODEL/EMBEDDING_API_KEY "
                "(значения ключей не логируются)."
            )

        if status == 400:
            raise EmbeddingError(
                "Сервер эмбеддингов отклонил запрос (HTTP 400): "
                "проверьте EMBEDDING_MODEL и формат поля input"
            )

        if status >= 400:
            raise EmbeddingError(
                f"Сервер эмбеддингов вернул HTTP {status}"
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise EmbeddingError(
                "Сервер эмбеддингов вернул не-JSON ответ"
            ) from exc

        return _parse_embeddings_response(response_payload, len(texts))

    raise EmbeddingError(
        "Не удалось получить эмбеддинги"
        + (f" ({type(last_error).__name__})" if last_error else "")
    )


def _api_embeddings(
    texts: List[str], batch_size: Optional[int] = None
) -> List[List[float]]:
    """Батчевый вызов HTTP-сервера эмбеддингов с L2-нормализацией."""
    if not texts:
        return []

    size = int(batch_size or EMBED_BATCH_SIZE) or len(texts)
    size = max(1, size)

    vectors: List[List[float]] = []
    for start in range(0, len(texts), size):
        batch = _sanitize_batch(texts[start : start + size])
        vectors.extend(_request_embeddings(batch))

    if EMBEDDING_NORMALIZE:
        return [_normalize_vector(vector) for vector in vectors]
    return vectors


def get_chroma_collection():
    """Потокобезопасная инициализация персистентного ChromaDB."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    with _CHROMA_LOCK:
        # double-checked locking: 4 потока не создают PersistentClient
        # одновременно (иначе ValueError: Could not connect to tenant ...).
        if _chroma_collection is not None:
            return _chroma_collection

        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )
        return _chroma_collection


def reset_chroma_cache() -> None:
    """Сброс кэша коллекции Chroma (например, после смены каталога)."""
    global _chroma_collection, _collection_dimension
    with _CHROMA_LOCK:
        _chroma_collection = None
    with _DIMENSION_LOCK:
        _collection_dimension = None


def clear_chroma_collection() -> None:
    """Удаляет и пересоздаёт коллекцию ChromaDB для чистой переиндексации."""
    global _chroma_collection, _collection_dimension
    with _CHROMA_LOCK:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        _chroma_collection = client.create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )
    with _DIMENSION_LOCK:
        _collection_dimension = None


def build_chroma_where_filter(
    document_ids: Optional[List[str]] = None,
    version_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Формирует metadata-фильтр ChromaDB по разрешённым document_id и version_id."""
    clauses: List[Dict[str, Any]] = []

    if document_ids:
        unique_docs = sorted({str(d) for d in document_ids if d})
        if len(unique_docs) == 1:
            clauses.append({"document_id": unique_docs[0]})
        elif len(unique_docs) > 1:
            clauses.append({"document_id": {"$in": unique_docs}})

    if version_ids:
        unique_vers = sorted({str(v) for v in version_ids if v})
        if len(unique_vers) == 1:
            clauses.append({"version_id": unique_vers[0]})
        elif len(unique_vers) > 1:
            clauses.append({"version_id": {"$in": unique_vers}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def query_chroma_collection(
    collection: Any,
    query_vector: List[float],
    n_results: int,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Выполняет поиск в ChromaDB с пре-фильтрацией по метаданным where."""
    total_count: Optional[int] = None
    try:
        total_count = int(collection.count())
    except Exception:  # noqa: BLE001
        total_count = None

    effective_n = max(1, int(n_results))
    if total_count is not None:
        if total_count <= 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        effective_n = min(effective_n, total_count)

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": effective_n,
        "include": ["documents", "metadatas"],
    }
    if where:
        query_kwargs["where"] = where

    try:
        return collection.query(**query_kwargs)
    except Exception as exc:  # noqa: BLE001
        if where:
            logger.warning(
                "Chroma query с where=%s завершился ошибкой (%s), "
                "выполняю запрос с пост-фильтрацией по allowed_chunk_ids",
                where,
                exc,
            )
            query_kwargs.pop("where", None)
            return collection.query(**query_kwargs)
        raise


def expected_embedding_dimension(provider: Optional[str] = None) -> Optional[int]:
    """Ожидаемая размерность векторов для текущего провайдера (или None)."""
    if EMBEDDING_DIMENSION:
        return int(EMBEDDING_DIMENSION)
    if provider == "hash_fallback":
        return HASH_FALLBACK_DIMENSION
    return None


def get_collection_embedding_dimension(
    collection: Optional[Any] = None, refresh: bool = False
) -> Optional[int]:
    """Читает размерность сохранённых эмбеддингов (без записи в коллекцию)."""
    global _collection_dimension
    if not refresh:
        with _DIMENSION_LOCK:
            if _collection_dimension is not None:
                return _collection_dimension

    try:
        coll = collection if collection is not None else get_chroma_collection()
    except Exception as exc:  # noqa: BLE001 - диагностика не должна падать
        logger.debug("Chroma недоступна для проверки размерности: %s", exc)
        return None

    raw_embeddings = None
    try:
        result = coll.get(limit=1, include=["embeddings"])
        raw_embeddings = (result or {}).get("embeddings")
    except Exception:  # noqa: BLE001 - старые версии Chroma
        try:
            peeked = coll.peek(limit=1)
            raw_embeddings = (peeked or {}).get("embeddings")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Не удалось прочитать размерность Chroma: %s", exc)
            return None

    if raw_embeddings is None:
        return None
    try:
        first = raw_embeddings[0]
        dimension = len(first) if first is not None else None
    except (TypeError, IndexError):
        return None
    if not dimension:
        return None

    with _DIMENSION_LOCK:
        _collection_dimension = int(dimension)
    return _collection_dimension


# Служебные тексты warm-up. Не содержат пользовательских данных, секретов и
# путей; нужны только чтобы прогреть HTTP-сервер эмбеддингов (загрузка весов,
# первые аллокации в llama.cpp) до реального поиска.
_WARMUP_TEXTS: tuple[str, ...] = (
    "retrieval warmup probe one",
    "retrieval warmup probe two",
    "retrieval warmup probe three",
    "retrieval warmup probe four",
)


def warmup_local_embedding_model() -> Dict[str, Any]:
    """Warm-up HTTP-запроса к серверу эмбеддингов.

    Вызывается prewarm-ом до первого реального поиска. Не бросает исключений:
    при ошибке фиксирует тип и возвращает статус ``error``, чтобы prewarm
    продолжил работу по существующему fallback-пути.
    """
    started = time.perf_counter()
    try:
        vectors = compute_embeddings(
            list(_WARMUP_TEXTS), batch_size=len(_WARMUP_TEXTS)
        )
    except Exception as exc:  # noqa: BLE001 - warm-up не фатален
        return {
            "status": "error",
            "warmup_ms": int(
                round((time.perf_counter() - started) * 1000)
            ),
            "error_type": type(exc).__name__,
            "dimension": None,
            "texts": len(_WARMUP_TEXTS),
        }
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    return {
        "status": "ok",
        "warmup_ms": elapsed_ms,
        "error_type": None,
        "dimension": len(vectors[0]) if vectors else None,
        "texts": len(_WARMUP_TEXTS),
    }


def validate_query_embedding_dimension(
    collection: Optional[Any], vector: List[float]
) -> None:
    """Явная ошибка при несовпадении размерности запроса и индекса."""
    stored = get_collection_embedding_dimension(collection)
    actual = len(vector) if vector is not None else 0
    if stored is not None and actual != stored:
        raise EmbeddingDimensionError(
            "Размерность эмбеддингов запроса не совпадает с индексом Chroma: "
            f"ожидалось {stored}, получено {actual}. "
            "Проверьте EMBEDDING_MODEL/EMBEDDING_DIMENSION и переиндексацию."
        )
    expected = expected_embedding_dimension()
    if expected is not None and actual and actual != expected:
        raise EmbeddingDimensionError(
            "Размерность эмбеддингов не соответствует ожидаемой: "
            f"получено {actual}, ожидалось {expected}. "
            "Проверьте EMBEDDING_MODEL/EMBEDDING_DIMENSION."
        )


def prewarm_retrieval_backends() -> Dict[str, Any]:
    """Однократный warm-up HTTP-провайдера эмбеддингов и коллекции Chroma.

    Вызывается до создания ThreadPoolExecutor, чтобы исключить гонки
    инициализации даже теоретически. Возвращает отчёт о состоянии; исключений
    не бросает.
    """
    global _prewarm_done, _prewarm_report

    if _prewarm_done:
        return dict(_prewarm_report)

    with _PREWARM_LOCK:
        if _prewarm_done:
            return dict(_prewarm_report)

        report: Dict[str, Any] = {
            # Локальной torch-модели больше нет — ключ сохранён для контракта.
            "local_model": "disabled",
            "collection": "unknown",
            "collection_size": None,
            "dimension": None,
            "warmup": "skipped",
            "warmup_ms": None,
            "warmup_dimension": None,
            "embedding_provider": "api",
            "embedding_endpoint": None,
            "errors": [],
        }

        try:
            report["embedding_endpoint"] = _embeddings_endpoint()
        except EmbeddingConfigError:
            report["embedding_endpoint"] = None

        try:
            collection = get_chroma_collection()
        except Exception as exc:  # noqa: BLE001 - prewarm не фатален
            report["collection"] = "error"
            report["errors"].append(
                {"type": type(exc).__name__, "message": str(exc)[:300]}
            )
        else:
            report["collection"] = "ok"
            try:
                report["collection_size"] = int(collection.count())
            except Exception:  # noqa: BLE001
                report["collection_size"] = None
            dimension = get_collection_embedding_dimension(
                collection, refresh=True
            )
            report["dimension"] = dimension
            expected = expected_embedding_dimension()
            if dimension and expected and dimension != expected:
                report["errors"].append(
                    {
                        "type": "EmbeddingDimensionError",
                        "message": (
                            "Размерность индекса Chroma не совпадает с "
                            f"ожидаемой: индекс {dimension}, ожидалось "
                            f"{expected}"
                        ),
                    }
                )

        # Warm-up HTTP-запроса выполняется до создания пула потоков:
        # первый реальный запрос станет «горячим».
        warmup = warmup_local_embedding_model()
        report["warmup"] = warmup.get("status", "error")
        report["warmup_ms"] = warmup.get("warmup_ms")
        report["warmup_dimension"] = warmup.get("dimension")
        if report["warmup"] != "ok":
            report["errors"].append(
                {
                    "type": warmup.get("error_type") or "WarmupError",
                    "message": (
                        "Warm-up HTTP-запроса к серверу эмбеддингов не "
                        "завершился; первый реальный вызов может быть "
                        "холодным"
                    ),
                }
            )
        logger.info(
            "Retrieval prewarm: warmup=%s warmup_ms=%s texts=%s",
            report["warmup"],
            report["warmup_ms"],
            warmup.get("texts"),
        )

        _prewarm_report = report
        _prewarm_done = True
        if report["errors"]:
            logger.warning("Prewarm retrieval backends: %s", report["errors"])
        return dict(report)


def _record_embedding_state(**state: Any) -> None:
    """Сохраняет диагностику эмбеддингов для текущего потока."""
    _thread_state.embedding = state


def reset_embedding_diagnostics() -> None:
    """Сбрасывает диагностику эмбеддингов текущего потока.

    Нужен перед каждой retrieval-задачей: пул переиспользует потоки, и
    состояние предыдущей задачи не должно протекать в следующую.
    """
    if getattr(_thread_state, "embedding", None) is not None:
        delattr(_thread_state, "embedding")


def get_last_embedding_diagnostics() -> Dict[str, Any]:
    """Диагностика последнего вычисления эмбеддингов в текущем потоке.

    Если в текущем потоке эмбеддинги ещё не вычислялись, ``success`` и
    ``measured`` честно сообщают об отсутствии измерения: ``provider:
    none`` не должен сочетаться с ``success: true``.
    """
    recorded = getattr(_thread_state, "embedding", None)
    if recorded is None:
        return {
            "provider": "none",
            "model": None,
            "dimension": None,
            "success": None,
            "fallback_used": False,
            "error_type": None,
            "measured": False,
        }
    diagnostics = dict(recorded)
    diagnostics.setdefault("measured", True)
    return diagnostics


def compute_embeddings(
    texts: List[str], batch_size: int = 16
) -> List[List[float]]:
    """Вычисляет эмбеддинги через llama.cpp (``POST {BASE}/embeddings``).

    При недоступности сервера и ``ALLOW_HASH_FALLBACK`` используется
    hash-fallback (провайдер ``hash_fallback`` в диагностике). В SUBMIT-режиме
    и при выключенном fallback бросается ``EmbeddingError``.
    """
    if not texts:
        return []

    try:
        vectors = _api_embeddings(texts, batch_size=batch_size)
    except Exception as exc:  # noqa: BLE001 - сеть/прокси/конфиг/валидация
        config_error = isinstance(exc, EmbeddingConfigError)
        if USE_SUBMIT or not ALLOW_HASH_FALLBACK:
            reason = (
                "конфигурация API эмбеддингов невалидна"
                if config_error
                else "сервер эмбеддингов недоступен"
            )
            error_type = (
                EmbeddingConfigError
                if config_error
                else EmbeddingError
            )
            _record_embedding_state(
                provider="api",
                model=EMBEDDING_MODEL,
                dimension=None,
                success=False,
                fallback_used=False,
                error_type=error_type.__name__,
            )
            raise error_type(
                f"{reason}; hash-fallback "
                f"{'запрещён в SUBMIT-режиме' if USE_SUBMIT else 'отключён'} "
                f"({type(exc).__name__}). Значения ключей не логируются."
            ) from exc

        logger.warning(
            "Сервер эмбеддингов недоступен (%s). Использован hash-fallback.",
            type(exc).__name__,
        )
        print(
            "[Предупреждение] Сервер эмбеддингов недоступен "
            f"({type(exc).__name__}). Использован локальный hash-fallback."
        )
        vectors = [_fallback_embedding(text) for text in texts]
        _record_embedding_state(
            provider="hash_fallback",
            model=None,
            dimension=len(vectors[0]) if vectors else None,
            success=True,
            fallback_used=True,
            error_type=type(exc).__name__,
        )
        return vectors

    _record_embedding_state(
        provider="api",
        model=EMBEDDING_MODEL,
        dimension=len(vectors[0]) if vectors else None,
        success=True,
        fallback_used=False,
        error_type=None,
    )
    return vectors


def get_query_embedding(query: str) -> List[float]:
    """Возвращает нормализованный вектор для поискового запроса."""
    return compute_embeddings([query])[0]


def _fallback_embedding(
    text: str, dim: int = HASH_FALLBACK_DIMENSION
) -> List[float]:
    digest = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return [((digest >> (i % 64)) & 1) * 0.1 for i in range(dim)]


def prepare_flat_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Строгие скалярные метаданные под ChromaDB без None и вложенностей."""
    allowed = [
        "document_id",
        "version_id",
        "chunk_index",
        "document_title",
        "document_type",
        "scope",
        "chat_id",
        "effective_from",
        "effective_to",
        "effective_dates_known",
        "source_hash",
        "amends_document_id",
        "chunker_version",
    ]
    metadata: Dict[str, Any] = {}
    for key in allowed:
        value = chunk.get(key)
        if value is not None:
            metadata[key] = (
                value if isinstance(value, (int, float, bool)) else str(value)
            )

    provision_id = chunk.get("structure", {}).get("provision_id")
    if provision_id is not None:
        metadata["provision_id"] = str(provision_id)

    return metadata


def save_embeddings_to_chroma(
    chunks: List[Dict[str, Any]], batch_size: int = 16
) -> None:
    """Индексация чанков в ChromaDB с расчётом эмбеддингов по embedding_text."""
    collection = get_chroma_collection()
    logger.info(
        "Индексация в ChromaDB через HTTP-сервер эмбеддингов (model=%s)",
        EMBEDDING_MODEL,
    )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts_to_embed = [item["embedding_text"] for item in batch]
        embeddings = compute_embeddings(
            texts_to_embed, batch_size=batch_size
        )
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                "Число эмбеддингов не совпадает с числом чанков в батче"
            )

        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            embeddings=embeddings,
            documents=[item["text"] for item in batch],
            metadatas=[prepare_flat_metadata(item) for item in batch],
        )
