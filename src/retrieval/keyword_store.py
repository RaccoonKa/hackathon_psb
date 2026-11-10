import re
import pickle
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from src.config import BM25_DIR

def tokenize_ru(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

class BM25KeywordStore:
    def __init__(self):
        self.chunk_ids: List[str] = []
        self.bm25: BM25Okapi = None

    def build_index(self, chunks: List[Dict[str, Any]]):
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        tokenized_corpus = [tokenize_ru(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self):
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        with open(BM25_DIR / "bm25.pkl", "wb") as f:
            pickle.dump((self.chunk_ids, self.bm25), f)

    def load(self):
        with open(BM25_DIR / "bm25.pkl", "rb") as f:
            self.chunk_ids, self.bm25 = pickle.load(f)

    def search(self, query: str, top_k: int = 40) -> List[Tuple[str, float]]:
        if not self.bm25:
            self.load()
        tokens = tokenize_ru(query)
        scores = self.bm25.get_scores(tokens)
        scored_ids = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return scored_ids[:top_k]