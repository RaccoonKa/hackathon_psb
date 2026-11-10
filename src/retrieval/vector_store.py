from typing import List, Dict, Any
import chromadb
from openai import OpenAI
from src.config import CHROMA_DIR, API_BASE_URL, API_KEY, EMBEDDING_MODEL

client_api = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def get_chroma_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    db_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Согласно rag.md: коллекция document_chunks и configuration с cosine
    return db_client.get_or_create_collection(
        name="document_chunks",
        configuration={"hnsw": {"space": "cosine"}}
    )

def prepare_flat_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Удаляет None и вложенные структуры под строгий контракт ChromaDB."""
    allowed_keys = [
        "document_id", "version_id", "chunk_index", "document_title", 
        "document_type", "scope", "effective_from", "effective_to", 
        "effective_dates_known", "chunker_version"
    ]
    meta = {}
    for key in allowed_keys:
        val = chunk.get(key)
        if val is not None:
            meta[key] = val if isinstance(val, (int, float, bool)) else str(val)
    return meta

def index_chunks_in_chroma(chunks: List[Dict[str, Any]], batch_size: int = 32):
    collection = get_chroma_collection()
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts_to_embed = [item["embedding_text"] for item in batch]
        
        # Получаем эмбеддинги по embedding_text
        res = client_api.embeddings.create(model=EMBEDDING_MODEL, input=texts_to_embed)
        embeddings = [item.embedding for item in res.data]
        
        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            embeddings=embeddings,
            documents=[item["text"] for item in batch], # Сохраняем точный текст оригинала
            metadatas=[prepare_flat_metadata(item) for item in batch]
        )