import pickle
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from rank_bm25 import BM25Okapi
from src.config import BM25_DIR


def tokenize_ru(text: str) -> List[str]:
    """Простая токенизация русских юридических терминов и номеров."""
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower())


class BM25KeywordStore:

    def __init__(self):
        self.chunk_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.tokenized_corpus = [tokenize_ru(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def save(self):
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        with open(BM25_DIR / "bm25.pkl", "wb") as f:
            pickle.dump(
                (self.chunk_ids, self.tokenized_corpus, self.bm25), f
            )

    def load(self):
        with open(BM25_DIR / "bm25.pkl", "rb") as f:
            self.chunk_ids, self.tokenized_corpus, self.bm25 = pickle.load(f)

    def search(
        self,
        query: str,
        top_k: int = 40,
        allowed_chunk_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        if not self.bm25:
            self.load()

        tokens = tokenize_ru(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Фильтрация кандидатов до ранжирования по разделу 4.2 rag.md
        scored = []
        for cid, sc in zip(self.chunk_ids, scores):
            if sc > 0:
                if allowed_chunk_ids is None or cid in allowed_chunk_ids:
                    scored.append((cid, float(sc)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]