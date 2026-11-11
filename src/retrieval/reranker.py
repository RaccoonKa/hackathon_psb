from typing import Any, Dict, List
import requests
from src.config import RERANKER_API_BASE_URL, RERANKER_API_KEY, RERANKER_MODEL


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Ранжирует пул кандидатов через Qwen3 Reranker 8B (по разделу 6 rag.md).
    Если API недоступен, возвращает порядок кандидатов из Weighted RRF.
    """
    if not candidates:
        return []

    if len(candidates) == 1:
        return candidates[:top_k]

    try:
        url = f"{RERANKER_API_BASE_URL.rstrip('/')}/rerank"
        headers = {
            "Authorization": f"Bearer {RERANKER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": RERANKER_MODEL,
            "query": query,
            "documents": [c["text"] for c in candidates],
            "top_n": min(top_k, len(candidates)),
        }
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            data = res.json()
            reranked: List[Dict[str, Any]] = []
            for item in data.get("results", []):
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(candidates):
                    cand_copy = dict(candidates[idx])
                    raw_score = item.get("relevance_score", item.get("score"))
                    if raw_score is not None:
                        cand_copy["score"] = float(raw_score)
                    reranked.append(cand_copy)
            if reranked:
                return reranked[:top_k]
    except Exception:
        # Прозрачный фоллбэк на RRF скоринг
        pass

    return candidates[:top_k]