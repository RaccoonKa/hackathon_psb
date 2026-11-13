import pickle
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from rag_core.config import BM25_DIR


def tokenize_ru(text: str) -> List[str]:
    """Токенизация русских терминов, латиницы и числовых номеров статей/пунктов."""
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower())


class BM25KeywordStore:

    def __init__(self):
        self.chunk_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Any = None
        self._is_built: bool = False
        self._load_lock = threading.Lock()

    def build_index(self, chunks: List[Dict[str, Any]]):
        from rank_bm25 import BM25Okapi

        self.chunk_ids = [c["chunk_id"] for c in chunks]
        # Индексируем embedding_text (заголовок + путь + текст), чтобы BM25
        # находил фрагменты по названиям кодексов и номерам статей
        self.tokenized_corpus = [
            tokenize_ru(c.get("embedding_text") or c["text"]) for c in chunks
        ]
        self._is_built = True

        if not self.tokenized_corpus:
            self.bm25 = None
            return

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
        self._is_built = True

    def get_chunk_ids(self) -> List[str]:
        """Снимок id индекса (неразрушающий, ленивая загрузка под lock)."""
        if not self._is_built:
            with self._load_lock:
                if not self._is_built:
                    self.load()
        return list(self.chunk_ids)


    def _search_with_stats_impl(
        self,
        query: str,
        top_k: int,
        allowed_chunk_ids: Optional[Set[str]],
        date_allowed_chunk_ids: Optional[Set[str]],
    ) -> Tuple[List[Tuple[str, float]], Dict[str, int]]:
        """Общая реализация поиска: один ``get_scores`` и один проход.

        Возвращает результаты (после фильтра по ``allowed_chunk_ids``) и
        диагностические счётчики, сохраняющие прежний смысл:
        кандидаты до фильтрации, после фильтра по датам, после фильтра по
        документам и фактическое число возвращённых результатов.
        """
        stats: Dict[str, int] = {
            "candidates_before_filters": 0,
            "candidates_after_date_filter": 0,
            "candidates_after_document_filter": 0,
            "returned_count": 0,
        }

        if self.bm25 is None:
            if self._is_built:
                return [], stats
            with self._load_lock:
                if self.bm25 is None and not self._is_built:
                    self.load()
            if self.bm25 is None:
                return [], stats

        tokens = tokenize_ru(query)
        if not tokens:
            return [], stats

        scores = self.bm25.get_scores(tokens)

        candidates_before = 0
        candidates_after_date = 0
        candidates_after_document = 0
        # Фильтрация кандидатов до ранжирования (раздел 4.2 rag.md)
        scored: List[Tuple[str, float]] = []
        for cid, sc in zip(self.chunk_ids, scores):
            if sc <= 0:
                continue
            candidates_before += 1
            if (
                date_allowed_chunk_ids is None
                or cid in date_allowed_chunk_ids
            ):
                candidates_after_date += 1
            if allowed_chunk_ids is None or cid in allowed_chunk_ids:
                candidates_after_document += 1
                scored.append((cid, float(sc)))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        stats["candidates_before_filters"] = candidates_before
        stats["candidates_after_date_filter"] = candidates_after_date
        stats["candidates_after_document_filter"] = candidates_after_document
        stats["returned_count"] = len(results)
        return results, stats

    def search_with_stats(
        self,
        query: str,
        top_k: int = 40,
        allowed_chunk_ids: Optional[Set[str]] = None,
        date_allowed_chunk_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, int]]:
        """Один вызов BM25 с результатами и диагностикой за один проход."""
        return self._search_with_stats_impl(
            query,
            top_k,
            allowed_chunk_ids,
            date_allowed_chunk_ids,
        )
    def search(
        self,
        query: str,
        top_k: int = 40,
        allowed_chunk_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Тонкая обёртка над ``search_with_stats`` (без диагностики дат)."""
        results, _stats = self._search_with_stats_impl(
            query,
            top_k,
            allowed_chunk_ids,
            None,
        )
        return results
