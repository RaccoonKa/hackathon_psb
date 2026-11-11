import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import (
    ALLOW_HASH_FALLBACK,
    BASE_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    USE_SUBMIT,
)

LOCAL_MODEL_PATH = BASE_DIR / "models" / "bge-m3"

_local_tokenizer = None
_local_model = None
_local_device = None
_api_client = None
_chroma_collection = None


class EmbeddingError(RuntimeError):
    """Ошибка получения эмбеддингов без раскрытия секретов."""


def _load_local_model():
    """Ленивая загрузка локальной модели bge-m3 только в LOCAL-режиме."""
    global _local_tokenizer, _local_model, _local_device

    if USE_SUBMIT:
        return None, None, None

    if _local_model is not None:
        return _local_tokenizer, _local_model, _local_device

    if not LOCAL_MODEL_PATH.exists():
        return None, None, None

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _local_device = "cuda" if torch.cuda.is_available() else "cpu"
        _local_tokenizer = AutoTokenizer.from_pretrained(
            str(LOCAL_MODEL_PATH)
        )
        _local_model = AutoModel.from_pretrained(str(LOCAL_MODEL_PATH))
        _local_model.to(_local_device)
        _local_model.eval()
        return _local_tokenizer, _local_model, _local_device
    except Exception as exc:  # noqa: BLE001 - локальная модель опциональна
        print(
            "[Предупреждение] Не удалось загрузить локальную модель "
            f"bge-m3 ({exc}). Переход на API."
        )
        return None, None, None


def _get_api_client():
    """Создаёт OpenAI-совместимый клиент эмбеддингов по настройкам config."""
    global _api_client
    if _api_client is not None:
        return _api_client

    if not EMBEDDING_API_BASE_URL or not EMBEDDING_API_KEY:
        raise EmbeddingError(
            "Не заданы EMBEDDING_API_BASE_URL/EMBEDDING_API_KEY. "
            "В SUBMIT-режиме они должны указывать на платформенный прокси."
        )

    from openai import OpenAI

    _api_client = OpenAI(
        base_url=EMBEDDING_API_BASE_URL,
        api_key=EMBEDDING_API_KEY,
        max_retries=1,
        timeout=60.0,
    )
    return _api_client


def get_chroma_collection():
    """Инициализирует персистентное хранилище ChromaDB с косинусной метрикой."""
    global _chroma_collection
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
        raise EmbeddingError(
            "Размерность эмбеддингов не соответствует ожидаемой: "
            f"получено {actual_dimension}, ожидалось {EMBEDDING_DIMENSION}. "
            "Проверьте EMB_MODEL/EMBEDDING_DIMENSION."
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


def compute_embeddings(
    texts: List[str], batch_size: int = 16
) -> List[List[float]]:
    """Вычисляет эмбеддинги: локальная модель (LOCAL) или API платформы."""
    if not texts:
        return []

    tokenizer, model, device = _load_local_model()

    if tokenizer is not None and model is not None:
        import torch

        all_embeddings: List[List[float]] = []
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

    try:
        return _api_embeddings(texts)
    except Exception as exc:  # noqa: BLE001 - в SUBMIT ошибка обязательна
        if USE_SUBMIT or not ALLOW_HASH_FALLBACK:
            raise EmbeddingError(
                "API эмбеддингов недоступен; hash-fallback в SUBMIT-режиме "
                f"запрещён: {exc}"
            ) from exc

        print(
            "[Предупреждение] API эмбеддингов недоступен "
            f"({exc}). Использован локальный hash-fallback."
        )
        return [_fallback_embedding(text) for text in texts]


def get_query_embedding(query: str) -> List[float]:
    """Возвращает нормализованный вектор для поискового запроса."""
    return compute_embeddings([query])[0]


def _fallback_embedding(text: str, dim: int = 1024) -> List[float]:
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
