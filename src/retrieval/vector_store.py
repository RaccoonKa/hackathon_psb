import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from openai import OpenAI
from src.config import (
    BASE_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
)

LOCAL_MODEL_PATH = BASE_DIR / "models" / "bge-m3"

_local_tokenizer = None
_local_model = None
_local_device = None


def _load_local_model():
    """Ленивая загрузка локальной модели bge-m3 из ./models/bge-m3/."""
    global _local_tokenizer, _local_model, _local_device
    if _local_model is not None:
        return _local_tokenizer, _local_model, _local_device

    if not LOCAL_MODEL_PATH.exists():
        return None, None, None

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _local_device = "cuda" if torch.cuda.is_available() else "cpu"
        _local_tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_PATH))
        _local_model = AutoModel.from_pretrained(str(LOCAL_MODEL_PATH))
        _local_model.to(_local_device)
        _local_model.eval()
        return _local_tokenizer, _local_model, _local_device
    except Exception as e:
        print(f"[Предупреждение] Не удалось загрузить локальную модель bge-m3 ({e}). Переход на API.")
        return None, None, None


client_api = OpenAI(base_url=EMBEDDING_API_BASE_URL, api_key=EMBEDDING_API_KEY)


def get_chroma_collection():
    """Инициализирует локальное персистентное хранилище ChromaDB с косинусной метрикой."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


def compute_embeddings(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    Вычисляет 1024-мерные эмбеддинги.
    Приоритет 1: локальный инференс bge-m3 (Transformers + PyTorch).
    Приоритет 2: API платформы соревнований.
    Приоритет 3: детерминированный фоллбэк.
    """
    tok, model, device = _load_local_model()

    # 1. Локальный расчет через transformers
    if tok is not None and model is not None:
        import torch

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tok(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # CLS token dense representation + L2 normalize
                cls_rep = outputs.last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(cls_rep, p=2, dim=1)
                all_embeddings.extend(normalized.cpu().tolist())
        return all_embeddings

    # 2. Удаленный API платформы
    try:
        res = client_api.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in res.data]
    except Exception as e:
        print(f"[Предупреждение] API эмбеддингов недоступен ({e}). Использован fallback.")
        return [_fallback_embedding(t) for t in texts]


def get_query_embedding(query: str) -> List[float]:
    """Возвращает нормализованный вектор для поискового запроса."""
    return compute_embeddings([query])[0]


def _fallback_embedding(text: str, dim: int = 1024) -> List[float]:
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return [((h >> (i % 64)) & 1) * 0.1 for i in range(dim)]


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
    meta: Dict[str, Any] = {}
    for k in allowed:
        v = chunk.get(k)
        if v is not None:
            meta[k] = v if isinstance(v, (int, float, bool)) else str(v)

    prov_id = chunk.get("structure", {}).get("provision_id")
    if prov_id is not None:
        meta["provision_id"] = str(prov_id)

    return meta


def save_embeddings_to_chroma(chunks: List[Dict[str, Any]], batch_size: int = 16):
    """Индексация чанков в ChromaDB с расчетом эмбеддингов по embedding_text."""
    collection = get_chroma_collection()
    tok, _, dev = _load_local_model()
    if tok is not None:
        print(f"-> Индексация в ChromaDB с использованием локального bge-m3 на устройстве: {dev}")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts_to_embed = [item["embedding_text"] for item in batch]
        embeddings = compute_embeddings(texts_to_embed, batch_size=batch_size)

        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            embeddings=embeddings,
            documents=[item["text"] for item in batch],
            metadatas=[prepare_flat_metadata(item) for item in batch],
        )