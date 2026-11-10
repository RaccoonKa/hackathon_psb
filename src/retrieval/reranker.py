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

    # Если кандидатов меньше или равно top_k, реранкинг не требуется
    if len(candidates) <= top_k:
        return candidates

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
            "top_n": top_k,
        }
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            data = res.json()
            ranked_indices = [item["index"] for item in data.get("results", [])]
            return [candidates[idx] for idx in ranked_indices if idx < len(candidates)]
    except Exception:
        # Прозрачный фоллбэк на RRF скоринг
        pass

    return candidates[:top_k]