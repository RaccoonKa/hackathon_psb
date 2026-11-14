"""Векторное хранилище и провайдеры эмбеддингов (LOCAL/SUBMIT).

Все ленивые инициализации (модель bge-m3, клиент API, коллекция Chroma)
защищены модульными блокировками с double-checked locking: параллельные
retrieval-задачи не должны одновременно выполнять ``import torch`` /
``from transformers import ...`` или создавать ``PersistentClient``.
"""

import hashlib
import logging
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from rag_core.config import (
    ALLOW_HASH_FALLBACK,
    BASE_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_API_MAX_RETRIES,
    EMBEDDING_API_TIMEOUT_S,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    USE_LOCAL_EMBEDDING_MODEL,
    USE_SUBMIT,
)

logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH = BASE_DIR / "models" / "bge-m3"
# Размерность bge-m3; используется как ожидаемая, если config не задал явно.
LOCAL_MODEL_FALLBACK_DIMENSION = 1024
HASH_FALLBACK_DIMENSION = 1024

_local_tokenizer = None
_local_model = None
_local_device = None
# Кэш неудачи загрузки: модель не пытаемся грузить на каждый запрос.
_local_model_error: Optional[BaseException] = None
_local_model_error_type: Optional[str] = None
_local_model_error_path: Optional[str] = None
_local_model_error_logged = False
_api_client = None
_chroma_collection = None
_collection_dimension: Optional[int] = None

_MODEL_LOCK = threading.Lock()
_API_LOCK = threading.Lock()
_CHROMA_LOCK = threading.Lock()
_DIMENSION_LOCK = threading.Lock()
_PREWARM_LOCK = threading.Lock()
# Локальный инференс сериализуем: torch/transformers не гарантируют
# потокобезопасность модели при конкурентных вызовах.
_LOCAL_INFERENCE_LOCK = threading.Lock()

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


def _local_model_enabled() -> bool:
    """Локальная модель допустима только в LOCAL-режиме и по флагу config."""
    return bool(USE_LOCAL_EMBEDDING_MODEL) and not USE_SUBMIT


def _record_local_model_failure(exc: BaseException, path: str) -> None:
    """Кэширует неудачу и печатает предупреждение ровно один раз."""
    global _local_model_error, _local_model_error_type
    global _local_model_error_path, _local_model_error_logged

    _local_model_error = exc
    _local_model_error_type = type(exc).__name__
    _local_model_error_path = path
    if not _local_model_error_logged:
        _local_model_error_logged = True
        message = (
            "[Предупреждение] Не удалось загрузить локальную модель "
            f"bge-m3 ({_local_model_error_type}: {exc}). "
            "Используются внешние эмбеддинги/hash-fallback."
        )
        print(message)
        logger.warning("%s", message)


def _load_local_model():
    """Ленивая потокобезопасная загрузка bge-m3 (только LOCAL-режим).

    Неудача загрузки кэшируется в ``_local_model_error`` и повторно не
    логируется; контролируемый сброс — :func:`reset_local_model_cache`.
    """
    global _local_tokenizer, _local_model, _local_device

    if not _local_model_enabled():
        return None, None, None

    if _local_model is not None:
        return _local_tokenizer, _local_model, _local_device

    path_str = str(LOCAL_MODEL_PATH)
    if _local_model_error is not None and _local_model_error_path == path_str:
        return None, None, None

    with _MODEL_LOCK:
        # double-checked locking
        if _local_model is not None:
            return _local_tokenizer, _local_model, _local_device
        if _local_model_error is not None and _local_model_error_path == path_str:
            return None, None, None

        if not LOCAL_MODEL_PATH.exists():
            _record_local_model_failure(
                FileNotFoundError(
                    "Локальная модель bge-m3 не найдена в models/bge-m3"
                ),
                path_str,
            )
            return None, None, None

        try:
            # Импорты под lock: параллельный lazy-import transformers
            # детерминированно падал с cannot import name 'AutoModel'.
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(path_str)
            model = AutoModel.from_pretrained(path_str)
            model.to(device)
            model.eval()
        except Exception as exc:  # noqa: BLE001 - локальная модель опциональна
            _record_local_model_failure(exc, path_str)
            return None, None, None

        _local_tokenizer = tokenizer
        _local_model = model
        _local_device = device
        return _local_tokenizer, _local_model, _local_device


def reset_local_model_cache() -> None:
    """Контролируемый сброс кэша модели (успеха и неудачи).

    Нужен после смены конфигурации/появления модели; предзагрузка после
    сброса выполняется заново.
    """
    global _local_tokenizer, _local_model, _local_device
    global _local_model_error, _local_model_error_type
    global _local_model_error_path, _local_model_error_logged
    global _prewarm_done, _prewarm_report

    with _MODEL_LOCK:
        _local_tokenizer = None
        _local_model = None
        _local_device = None
        _local_model_error = None
        _local_model_error_type = None
        _local_model_error_path = None
        _local_model_error_logged = False
    with _PREWARM_LOCK:
        _prewarm_done = False
        _prewarm_report = {}


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


def get_local_model_status() -> Dict[str, Any]:
    """Диагностика локальной модели без раскрытия путей/секретов."""
    return {
        "enabled": _local_model_enabled(),
        "available": _local_model is not None,
        "device": _local_device,
        "model": EMBEDDING_MODEL,
        "error_type": _local_model_error_type,
        "error_message": (
            str(_local_model_error)[:300] if _local_model_error else None
        ),
    }


def _get_api_client():
    """Создаёт OpenAI-совместимый клиент эмбеддингов по настройкам config."""
    global _api_client
    if _api_client is not None:
        return _api_client

    with _API_LOCK:
        if _api_client is not None:
            return _api_client

        if not EMBEDDING_API_BASE_URL or not EMBEDDING_API_KEY:
            raise EmbeddingConfigError(
                "Не заданы EMBEDDING_API_BASE_URL/EMBEDDING_API_KEY. "
                "В SUBMIT-режиме они должны указывать на платформенный прокси."
            )

        parsed = urlparse(EMBEDDING_API_BASE_URL)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise EmbeddingConfigError(
                "EMBEDDING_API_BASE_URL должен быть абсолютным http(s) URL "
                "(значение не логируется)."
            )

        from openai import OpenAI

        _api_client = OpenAI(
            base_url=EMBEDDING_API_BASE_URL,
            api_key=EMBEDDING_API_KEY,
            max_retries=int(EMBEDDING_API_MAX_RETRIES),
            timeout=float(EMBEDDING_API_TIMEOUT_S),
        )
    return _api_client


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
    if provider == "local" or _local_model is not None:
        return LOCAL_MODEL_FALLBACK_DIMENSION
    if provider == "hash_fallback":
        return HASH_FALLBACK_DIMENSION
    # Локальный режим с установленной bge-m3: ожидаем её размерность даже
    # до фактической загрузки модели (проверка согласованности индекса).
    if _local_model_enabled() and LOCAL_MODEL_PATH.exists():
        return LOCAL_MODEL_FALLBACK_DIMENSION
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


# Служебные тексты warm-up forward. Не содержат пользовательских данных,
# секретов и путей; нужны только чтобы прогреть первый forward bge-m3
# (ленивый импорт CUDA/ядер, аллокации, JIT) до реального поиска.
_WARMUP_TEXTS: tuple[str, ...] = (
    "retrieval warmup probe one",
    "retrieval warmup probe two",
    "retrieval warmup probe three",
    "retrieval warmup probe four",
)


def warmup_local_embedding_model() -> Dict[str, Any]:
    """Один warm-up forward локальной модели тем же путём, что и поиск.

    Вызывается только когда локальная модель уже загружена. Не бросает
    исключений: при ошибке фиксирует тип и возвращает статус ``error``,
    чтобы prewarm продолжил работу по существующему fallback-пути.
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
    """Однократная предзагрузка модели и коллекции (без исключений).

    Вызывается до создания ThreadPoolExecutor, чтобы исключить гонки
    инициализации даже теоретически. Возвращает отчёт о состоянии.
    """
    global _prewarm_done, _prewarm_report

    if _prewarm_done:
        return dict(_prewarm_report)

    with _PREWARM_LOCK:
        if _prewarm_done:
            return dict(_prewarm_report)

        report: Dict[str, Any] = {
            "local_model": "disabled",
            "collection": "unknown",
            "collection_size": None,
            "dimension": None,
            "warmup": "skipped",
            "warmup_ms": None,
            "errors": [],
        }

        tokenizer, model, _device = _load_local_model()
        if not _local_model_enabled():
            report["local_model"] = "disabled"
        elif model is not None:
            report["local_model"] = "ok"
        else:
            report["local_model"] = "unavailable"
            status = get_local_model_status()
            report["local_model_error_type"] = status.get("error_type")

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

        # Warm-up forward выполняется после загрузки модели и Chroma, но до
        # создания пула потоков: первый реальный forward станет «горячим».
        if model is not None:
            warmup = warmup_local_embedding_model()
            report["warmup"] = warmup.get("status", "error")
            report["warmup_ms"] = warmup.get("warmup_ms")
            report["warmup_dimension"] = warmup.get("dimension")
            if report["warmup"] != "ok":
                report["errors"].append(
                    {
                        "type": warmup.get("error_type") or "WarmupError",
                        "message": (
                            "Warm-up forward локальной модели эмбеддингов "
                            "не завершился; первый реальный вызов может "
                            "быть холодным"
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


def _api_embeddings(texts: List[str]) -> List[List[float]]:
    client = _get_api_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL, input=texts
    )
    data = list(response.data)
    # Некоторые прокси возвращают элементы не по порядку — сортируем по index.
    data.sort(key=lambda item: getattr(item, "index", 0))
    vectors = [list(item.embedding) for item in data]
    _validate_embeddings(vectors, len(texts))
    return vectors


def _local_embeddings(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
) -> List[List[float]]:
    """Инференс локальной модели под блокировкой (потокобезопасно)."""
    import torch

    all_embeddings: List[List[float]] = []
    with _LOCAL_INFERENCE_LOCK:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                cls_rep = outputs.last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(
                    cls_rep, p=2, dim=1
                )
                all_embeddings.extend(normalized.cpu().tolist())
    return all_embeddings


def compute_embeddings(
    texts: List[str], batch_size: int = 16
) -> List[List[float]]:
    """Вычисляет эмбеддинги: локальная модель (LOCAL) или API платформы."""
    if not texts:
        return []

    tokenizer, model, device = _load_local_model()

    if tokenizer is not None and model is not None:
        try:
            vectors = _local_embeddings(
                texts, tokenizer, model, device, batch_size
            )
        except Exception as exc:  # noqa: BLE001 - локальный инференс упал
            error = EmbeddingError(
                "Локальная модель эмбеддингов недоступна для инференса: "
                f"{type(exc).__name__}"
            )
            _record_embedding_state(
                provider="local",
                model=EMBEDDING_MODEL,
                dimension=None,
                success=False,
                fallback_used=False,
                error_type=type(exc).__name__,
            )
            raise error from exc

        _record_embedding_state(
            provider="local",
            model=EMBEDDING_MODEL,
            dimension=len(vectors[0]) if vectors else None,
            success=True,
            fallback_used=False,
            error_type=None,
        )
        return vectors

    # Локальная модель включена, но недоступна → вызов API является
    # fallback-путём (диагностика должна это отражать).
    local_fallback = tokenizer is None and _local_model_enabled()

    try:
        vectors = _api_embeddings(texts)
    except Exception as exc:  # noqa: BLE001 - сеть/прокси/конфиг/валидация
        config_error = isinstance(exc, EmbeddingConfigError)
        if USE_SUBMIT or not ALLOW_HASH_FALLBACK:
            reason = (
                "конфигурация API эмбеддингов невалидна"
                if config_error
                else "API эмбеддингов недоступен"
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

        print(
            "[Предупреждение] API эмбеддингов недоступен "
            f"({type(exc).__name__}). Использован локальный hash-fallback."
        )
        vectors = [
            _fallback_embedding(text) for text in texts
        ]
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
        fallback_used=local_fallback,
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
    tokenizer, _, device = _load_local_model()
    if tokenizer is not None:
        print(
            "-> Индексация в ChromaDB с использованием локального bge-m3 "
            f"на устройстве: {device}"
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
