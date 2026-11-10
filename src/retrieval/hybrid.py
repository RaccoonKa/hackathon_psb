import json
import re
from typing import Any, Dict, List, Optional, Set
from openai import OpenAI
from src.config import (
    EMBEDDING_API_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
    PROCESSED_DATA_DIR,
)
from src.retrieval.keyword_store import BM25KeywordStore
from src.retrieval.reranker import rerank_candidates
from src.retrieval.vector_store import (
    _fallback_embedding,
    get_chroma_collection,
)

client_api = OpenAI(base_url=EMBEDDING_API_BASE_URL, api_key=EMBEDDING_API_KEY)
bm25_store = BM25KeywordStore()


def detect_query_intent(query: str) -> Dict[str, Any]:
    """3 класса намерений по разделу 5.3-5.5 rag.md: exact_lookup, semantic_query, mixed_query."""
    q_low = query.lower()

    has_exact = bool(
        re.search(
            r"(?:стать[яеи]\s*\d+|пункт[ае]?\s*[\d\.]+|раздел\s*\d+|№\s*[\w\d\-\/]+|\bфз\b|\bдоговор[ае]?\b)",
            q_low,
        )
    )
    has_semantic = bool(
        re.search(
            r"(?:что\s+означает|какие\s+условия|как\s+регулируется|в\s+каком\s+порядке|порядок|ответственность|срок|расторжен)",
            q_low,
        )
    )

    if has_exact and has_semantic:
        return {
            "intent": "mixed_query",
            "w_bm25": 0.5,
            "w_dense": 0.5,
            "bm25_top": 30,
            "dense_top": 30,
        }
    if has_exact:
        return {
            "intent": "exact_lookup",
            "w_bm25": 0.7,
            "w_dense": 0.3,
            "bm25_top": 40,
            "dense_top": 20,
        }
    return {
        "intent": "semantic_query",
        "w_bm25": 0.3,
        "w_dense": 0.7,
        "bm25_top": 20,
        "dense_top": 40,
    }


def load_canonical_chunks_map() -> Dict[str, Dict[str, Any]]:
    path = PROCESSED_DATA_DIR / "chunks.jsonl"
    store = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ch = json.loads(line)
                store[ch["chunk_id"]] = ch
    return store


def resolve_allowed_versions_by_date(
    event_date: Optional[str],
    chunks_map: Dict[str, Dict[str, Any]],
    selected_doc_ids: Optional[Set[str]] = None,
) -> Set[str]:
    """
    Разрешает допустимые chunk_id с учетом:
    1. Даты события: effective_from <= event_date < effective_to.
    2. Связки договора с допниками: если выбран договор, допники к нему включаются автоматически.
    """
    allowed_cids = set()

    # Расширяем selected_doc_ids связями amends_document_id
    expanded_doc_ids = set(selected_doc_ids) if selected_doc_ids else None
    if expanded_doc_ids:
        for ch in chunks_map.values():
            amends = ch.get("amends_document_id")
            if amends and amends in expanded_doc_ids:
                expanded_doc_ids.add(ch["document_id"])

    for cid, ch in chunks_map.items():
        doc_id = ch["document_id"]
        if expanded_doc_ids and doc_id not in expanded_doc_ids:
            continue

        if event_date:
            eff_from = ch.get("effective_from")
            eff_to = ch.get("effective_to")
            dates_known = ch.get("effective_dates_known", True)

            if not dates_known:
                continue
            if eff_from and event_date < eff_from:
                continue
            if eff_to and event_date >= eff_to:
                continue

        allowed_cids.add(cid)

    return allowed_cids


def hybrid_search(
    query: str,
    top_k: int = 8,
    event_date: Optional[str] = None,
    selected_doc_ids: Optional[Set[str]] = None,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    chunks_map = load_canonical_chunks_map()
    allowed_chunk_ids = resolve_allowed_versions_by_date(
        event_date, chunks_map, selected_doc_ids
    )

    if not allowed_chunk_ids:
        return []

    intent = detect_query_intent(query)

    # 1. BM25 Search с фильтрацией до ранжирования (раздел 4.2 rag.md)
    bm25_results = bm25_store.search(
        query, top_k=intent["bm25_top"], allowed_chunk_ids=allowed_chunk_ids
    )

    # 2. Dense Vector Search в Chroma
    try:
        res = client_api.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        query_vector = res.data[0].embedding
    except Exception:
        query_vector = _fallback_embedding(query)

    collection = get_chroma_collection()
    chroma_res = collection.query(
        query_embeddings=[query_vector],
        n_results=intent["dense_top"],
        include=["documents", "metadatas"],
    )

    dense_chunk_ids = []
    if chroma_res["ids"] and chroma_res["ids"][0]:
        for cid in chroma_res["ids"][0]:
            if cid in allowed_chunk_ids:
                dense_chunk_ids.append(cid)

    # 3. Weighted RRF
    rrf_scores: Dict[str, float] = {}

    for rank, (cid, score) in enumerate(bm25_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
            intent["w_bm25"] / (rrf_k + rank + 1)
        )

    for rank, cid in enumerate(dense_chunk_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
            intent["w_dense"] / (rrf_k + rank + 1)
        )

    # Приоритет доп. соглашений (более поздней редакции) при совпадении условий
    for cid in rrf_scores:
        ch = chunks_map.get(cid)
        if ch and ch.get("amends_document_id"):
            rrf_scores[cid] *= 1.15

    sorted_cids = sorted(
        rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
    )[:40]
    candidate_chunks = [chunks_map[cid] for cid in sorted_cids if cid in chunks_map]

    # 4. Reranking через Qwen3 Reranker 8B (раздел 6 rag.md)
    final_chunks = rerank_candidates(query, candidate_chunks, top_k=top_k)
    return final_chunks