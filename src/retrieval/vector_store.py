import hashlib
from typing import Any, Dict, List
import chromadb
from openai import OpenAI
from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
)

client_api = OpenAI(base_url=EMBEDDING_API_BASE_URL, api_key=EMBEDDING_API_KEY)


def get_chroma_collection():
    """Инициализирует персистентное подключение к ChromaDB с косинусной метрикой."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


def prepare_flat_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Преобразует метаданные в плоские скалярные типы без None для Chroma."""
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


def _fallback_embedding(text: str, dim: int = 1024) -> List[float]:
    """Детерминированный вектор на случай локального отсутствия API."""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return [((h >> (i % 64)) & 1) * 0.1 for i in range(dim)]


def save_embeddings_to_chroma(
    chunks: List[Dict[str, Any]], batch_size: int = 32
):
    collection = get_chroma_collection()
    api_available = True

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts_to_embed = [item["embedding_text"] for item in batch]
        embeddings = None

        if api_available:
            try:
                res = client_api.embeddings.create(
                    model=EMBEDDING_MODEL, input=texts_to_embed
                )
                embeddings = [d.embedding for d in res.data]
            except Exception as e:
                print(
                    f"[Предупреждение] API эмбеддингов недоступен ({e}). Применяется локальный фоллбэк."
                )
                api_available = False

        if not embeddings:
            embeddings = [_fallback_embedding(t) for t in texts_to_embed]

        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            embeddings=embeddings,
            documents=[item["text"] for item in batch],
            metadatas=[prepare_flat_metadata(item) for item in batch],
        )