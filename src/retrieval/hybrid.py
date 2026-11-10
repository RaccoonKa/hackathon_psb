import re
import json
from typing import List, Dict, Any
from src.config import PROCESSED_DATA_DIR, EMBEDDING_API_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL
from src.retrieval.vector_store import get_chroma_collection, _fallback_embedding
from src.retrieval.keyword_store import BM25KeywordStore
from openai import OpenAI

client_api = OpenAI(base_url=EMBEDDING_API_BASE_URL, api_key=EMBEDDING_API_KEY)
bm25_store = BM25KeywordStore()

def detect_query_intent(query: str) -> Dict[str, float]:
    """Определяет приоритет: exact_lookup для номеров/статей или semantic_query."""
    q_low = query.lower()
    has_article = bool(re.search(r"(?:стать[яеи]|пункт[ае]?|раздел|№|\bфз\b)", q_low))
    if has_article:
        return {"w_bm25": 0.7, "w_dense": 0.3, "bm25_top": 40, "dense_top": 20}
    return {"w_bm25": 0.3, "w_dense": 0.7, "bm25_top": 20, "dense_top": 40}

def load_canonical_chunks_map() -> Dict[str, Dict[str, Any]]:
    path = PROCESSED_DATA_DIR / "chunks.jsonl"
    store = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ch = json.loads(line)
            store[ch["chunk_id"]] = ch
    return store

def hybrid_search(query: str, top_k: int = 8, rrf_k: int = 60) -> List[Dict[str, Any]]:
    intent = detect_query_intent(query)
    
    # 1. BM25 Search
    bm25_results = bm25_store.search(query, top_k=intent["bm25_top"])
    
    # 2. Dense Vector Search (Chroma)
    try:
        res = client_api.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        query_vector = res.data[0].embedding
    except Exception:
        query_vector = _fallback_embedding(query)
    
    collection = get_chroma_collection()
    chroma_res = collection.query(
        query_embeddings=[query_vector],
        n_results=intent["dense_top"],
        include=["documents", "metadatas"]
    )
    dense_chunk_ids = chroma_res["ids"][0] if chroma_res["ids"] else []

    # 3. Weighted Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    
    for rank, (cid, score) in enumerate(bm25_results):
        if score > 0:
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (intent["w_bm25"] / (rrf_k + rank + 1))
            
    for rank, cid in enumerate(dense_chunk_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (intent["w_dense"] / (rrf_k + rank + 1))

    # Сортировка по финальному скору RRF
    sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    
    chunks_map = load_canonical_chunks_map()
    return [chunks_map[cid] for cid in sorted_cids if cid in chunks_map]