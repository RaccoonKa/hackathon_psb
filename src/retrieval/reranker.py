import logging
from typing import Any, Dict, List

from src.config import (
    RERANKER_API_BASE_URL,
    RERANKER_API_KEY,
    RERANKER_MODEL,
    RERANKER_TIMEOUT,
)

logger = logging.getLogger(__name__)


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Ранжирует пул кандидатов через Qwen3 Reranker 8B.
    Использует embedding_text для сохранения контекста статьи при ранжировании.
    Если API недоступен, возвращает порядок кандидатов из Weighted RRF.
    """
    if not candidates:
        return []

    if len(candidates) <= 1:
        return candidates[:top_k]

    try:
        import requests

        url = f"{RERANKER_API_BASE_URL.rstrip('/')}/rerank"
        headers = {
            "Authorization": f"Bearer {RERANKER_API_KEY}",
            "Content-Type": "application/json",
        }
        doc_texts = [
            c.get("embedding_text") or c["text"] for c in candidates
        ]
        payload = {
            "model": RERANKER_MODEL,
            "query": query,
            "documents": doc_texts,
            "top_n": min(top_k, len(candidates)),
        }
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=RERANKER_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            reranked: List[Dict[str, Any]] = []
            for item in data.get("results", []):
                index = item.get("index")
                if isinstance(index, int) and 0 <= index < len(candidates):
                    candidate_copy = dict(candidates[index])
                    raw_score = item.get(
                        "relevance_score", item.get("score")
                    )
                    if raw_score is not None:
                        candidate_copy["score"] = float(raw_score)
                    reranked.append(candidate_copy)
            if reranked:
                return reranked[:top_k]
        else:
            logger.warning(
                "Реранкер вернул HTTP %s; сохранён исходный порядок",
                response.status_code,
            )
    except Exception as exc:  # noqa: BLE001 - реранкер необязателен
        logger.warning(
            "Реранкер недоступен, сохранён исходный RRF-порядок: %s", exc
        )

    return candidates[:top_k]
