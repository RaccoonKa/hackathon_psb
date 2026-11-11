import json
import re
from typing import Any, Dict, List, Optional, Set
from src.config import PROCESSED_DATA_DIR
from src.retrieval.keyword_store import BM25KeywordStore
from src.retrieval.reranker import rerank_candidates
from src.retrieval.vector_store import (
    get_chroma_collection,
    get_query_embedding,
)

bm25_store = BM25KeywordStore()

_INTENT_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "mixed_query": {
        "intent": "mixed_query",
        "w_bm25": 0.5,
        "w_dense": 0.5,
        "bm25_top": 30,
        "dense_top": 30,
    },
    "exact_lookup": {
        "intent": "exact_lookup",
        "w_bm25": 0.7,
        "w_dense": 0.3,
        "bm25_top": 40,
        "dense_top": 20,
    },
    "semantic_query": {
        "intent": "semantic_query",
        "w_bm25": 0.3,
        "w_dense": 0.7,
        "bm25_top": 20,
        "dense_top": 40,
    },
}


def detect_query_intent(
    query: str, precomputed_intent: Optional[str] = None
) -> Dict[str, Any]:
    """3 класса намерений по разделу 5.3-5.5 rag.md."""
    if precomputed_intent and precomputed_intent in _INTENT_WEIGHTS:
        return dict(_INTENT_WEIGHTS[precomputed_intent])

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
        return dict(_INTENT_WEIGHTS["mixed_query"])
    if has_exact:
        return dict(_INTENT_WEIGHTS["exact_lookup"])
    return dict(_INTENT_WEIGHTS["semantic_query"])


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
    query: str = "",
) -> Set[str]:
    """
    Разрешает допустимые chunk_id с учетом временных интервалов и связок документов.
    - Автоматически отсекает утратившие силу редакции (например, Регламент 2023 при запросах на 2025-2026).
    - Законы РФ не отсекаются при event_date < eff_from, чтобы модель знала о действующей норме
      и могла сформулировать обоснованный отказ о нехватке архивной редакции.
    """
    allowed_cids = set()

    # Если дата события не передана явно, проверяем год в тексте вопроса либо берем дату актуальности базы
    if not event_date:
        m_year = re.search(r"\b(20[12]\d)\b", query)
        check_date = f"{m_year.group(1)}-06-01" if m_year else "2026-10-10"
    else:
        check_date = event_date

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

        eff_from = ch.get("effective_from")
        eff_to = ch.get("effective_to")
        dates_known = ch.get("effective_dates_known", True)
        is_law = ch.get("document_type") == "law"

        if not dates_known:
            continue

        # Внутренние регламенты и договоры отсекаются, если норма еще не вступила в силу
        if eff_from and check_date < eff_from and not is_law:
            continue

        # Документ утратил силу к моменту анализируемого события
        if eff_to and check_date >= eff_to:
            continue

        allowed_cids.add(cid)

    return allowed_cids


def hybrid_search(
    query: str,
    top_k: int = 10,
    event_date: Optional[str] = None,
    selected_doc_ids: Optional[Set[str]] = None,
    rrf_k: int = 60,
    intent: Optional[str] = None,
    rerank: bool = True,
) -> List[Dict[str, Any]]:
    chunks_map = load_canonical_chunks_map()
    allowed_chunk_ids = resolve_allowed_versions_by_date(
        event_date=event_date,
        chunks_map=chunks_map,
        selected_doc_ids=selected_doc_ids,
        query=query,
    )

    if not allowed_chunk_ids:
        return []

    intent_cfg = detect_query_intent(query, precomputed_intent=intent)

    # 1. Поиск BM25
    bm25_results = bm25_store.search(
        query, top_k=intent_cfg["bm25_top"], allowed_chunk_ids=allowed_chunk_ids
    )

    # 2. Поиск Dense Vector
    query_vector = get_query_embedding(query)
    collection = get_chroma_collection()
    chroma_res = collection.query(
        query_embeddings=[query_vector],
        n_results=intent_cfg["dense_top"],
        include=["documents", "metadatas"],
    )

    dense_chunk_ids = []
    if chroma_res["ids"] and chroma_res["ids"][0]:
        for cid in chroma_res["ids"][0]:
            if cid in allowed_chunk_ids:
                dense_chunk_ids.append(cid)

    # 3. Взвешенный RRF
    rrf_scores: Dict[str, float] = {}

    for rank, (cid, _score) in enumerate(bm25_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
            intent_cfg["w_bm25"] / (rrf_k + rank + 1)
        )

    for rank, cid in enumerate(dense_chunk_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
            intent_cfg["w_dense"] / (rrf_k + rank + 1)
        )

    # Приоритет актуальных доп. соглашений
    for cid in rrf_scores:
        ch = chunks_map.get(cid)
        if ch and ch.get("amends_document_id"):
            rrf_scores[cid] *= 1.15

    sorted_cids = sorted(
        rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
    )[:40]

    candidate_chunks: List[Dict[str, Any]] = []
    for cid in sorted_cids:
        if cid in chunks_map:
            chunk_copy = dict(chunks_map[cid])
            chunk_copy["score"] = round(float(rrf_scores[cid]), 6)
            candidate_chunks.append(chunk_copy)

    # 4. Реранкинг
    if rerank:
        return rerank_candidates(query, candidate_chunks, top_k=top_k)
    return candidate_chunks[:top_k]