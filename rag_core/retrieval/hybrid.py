import copy
import datetime
import json
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from rag_core.config import PROCESSED_DATA_DIR
from rag_core.retrieval.keyword_store import BM25KeywordStore
from rag_core.retrieval.reranker import rerank_candidates
from rag_core.retrieval.vector_store import (
    build_chroma_where_filter,
    expected_embedding_dimension,
    get_chroma_collection,
    get_collection_embedding_dimension,
    get_query_embedding,
    query_chroma_collection,
    validate_query_embedding_dimension,
)

logger = logging.getLogger(__name__)

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

# Ключи диагностики поиска, которые заполняют все режимы (bm25/dense/hybrid).
SEARCH_DIAGNOSTICS_KEYS: Tuple[str, ...] = (
    "selected_doc_ids",
    "allowed_chunk_ids_count",
    "bm25_candidates_count",
    "dense_candidates_count",
    "candidates_before_filters",
    "candidates_after_date_filter",
    "candidates_after_document_filter",
    "final_chunks_count",
    "validity_unknown_count",
)

# Год внутри номера договора (``А-08/2024``, ``П-17/2024``, ``ПТ-05/2023``)
# не является временной привязкой запроса и не должен влиять на дату проверки.
_CONTRACT_NUMBER_RE = re.compile(
    r"[А-ЯЁа-яёA-Za-z]+[-\u2013\s]?\d+[^\s/]*/\s*20[12]\d"
)
# Полная дата ``dd.mm.yyyy`` / ``dd/mm/yyyy`` — берём из неё только год.
_DATE_WITH_YEAR_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./](20[12]\d)\b")
# Год с явным словом «год/г.»: «2024 год», «2024 году», «2024 г.».
_YEAR_WORD_RE = re.compile(
    r"(20[12]\d)\s*(?:гг?\.?|год(?:у|а|ом|е)?)(?![А-Яа-яЁё0-9])",
    re.IGNORECASE,
)
# Год после временного предлога: «в 2024», «за 2023», «на 2025».
_YEAR_PREPOSITION_RE = re.compile(
    r"\b(?:в|во|за|на|с|до|по|к)\s+(20[12]\d)\b",
    re.IGNORECASE,
)

# H-2C: boost допсоглашений применяется только при явной релевантности
# запроса версии/допсоглашению, а не всем чанкам с ``amends_document_id``.
_AMENDMENT_RELEVANCE_RE = re.compile(
    r"(?:дополнительн\w*\s+соглашени|доп\.?\s*соглашени|допсоглашени|"
    r"редакци|изменени|изменён|изменен|в\s+редакции)",
    re.IGNORECASE,
)
_TEMPORAL_OR_VERSION_RE = re.compile(
    r"(?:\b\d{1,2}[./]\d{1,2}[./]20[12]\d|\b20[12]\d\s*(?:гг?\.?|год)|"
    r"действующ\w*|актуальн\w*|утратил\w*|в\s+силу)",
    re.IGNORECASE,
)
_AMENDMENT_BOOST_FACTOR = 1.15


def should_boost_amendments(
    query: str, intent: Optional[str] = None
) -> bool:
    """Нужно ли повышать допсоглашения для данного запроса.

    Boost допустим только при явной релевантности: маркерах
    «дополнительное соглашение», «редакция», «изменения» и т.п., либо для
    намерения ``date_version`` с временной/версионной формулировкой.
    Один лишь факт связи с выбранным договором (``selected_doc_ids``)
    основанием не является.
    """
    text = (query or "").strip()
    if not text:
        return False
    if _AMENDMENT_RELEVANCE_RE.search(text):
        return True
    normalized_intent = str(intent or "").strip().lower()
    if normalized_intent == "date_version" and _TEMPORAL_OR_VERSION_RE.search(
        text
    ):
        return True
    return False


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


@lru_cache(maxsize=8)
def _read_canonical_chunks_file(
    path_str: str,
    _mtime_ns: int,
    _size: int,
) -> Dict[str, Dict[str, Any]]:
    """Читает и разбирает ``chunks.jsonl``; ключ кэша — путь и признак файла."""
    store: Dict[str, Dict[str, Any]] = {}
    with open(path_str, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Пропущена повреждённая строка chunks.jsonl")
                continue
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            chunk["chunk_id"] = str(chunk_id)
            store[str(chunk_id)] = chunk
    return store


def clear_canonical_chunks_cache() -> None:
    """Сбрасывает кэш ``chunks.jsonl`` (вызывается после пересборки индекса)."""
    _read_canonical_chunks_file.cache_clear()


def load_canonical_chunks_map() -> Dict[str, Dict[str, Any]]:
    """Канонический корпус чанков; ключ и ``chunk_id`` нормализованы в str.

    Файл читается и разбирается не более одного раза для одной пары
    «путь + признак изменения файла (``mtime_ns``, размер)». Наружу
    возвращается безопасная копия, чтобы вызовы не мутировали общий кэш.
    """
    path = PROCESSED_DATA_DIR / "chunks.jsonl"
    try:
        stat = path.stat()
    except OSError:
        return {}

    cached = _read_canonical_chunks_file(
        str(path), stat.st_mtime_ns, stat.st_size
    )
    return copy.deepcopy(cached)


def _extract_year_from_query(query: str) -> Optional[str]:
    """Извлекает год только из явных временных указаний в вопросе.

    Номера договоров (``А-08/2024``, ``П-17/2024``, ``ПТ-05/2023``) перед
    поиском вырезаются, чтобы их год не становился датой проверки.
    Поддерживаются полная дата (``15.08.2024``), «2024 год/году/г.», а также
    год после временного предлога («в 2024»). Возвращается первый по позиции.
    """
    if not query:
        return None

    cleaned = _CONTRACT_NUMBER_RE.sub(" ", query)
    matches: List[Tuple[int, str]] = []
    for pattern in (
        _DATE_WITH_YEAR_RE,
        _YEAR_WORD_RE,
        _YEAR_PREPOSITION_RE,
    ):
        for match in pattern.finditer(cleaned):
            matches.append((match.start(), match.group(1)))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _resolve_check_date(
    event_date: Optional[str], query: str = ""
) -> str:
    """Дата, на которую проверяется актуальность редакции документа.

    Бизнес-правило трёх состояний:
    1. ``event_date`` передан явно → оцениваем документ на эту дату;
    2. иначе берём год из явных временных указаний в вопросе (середина года),
       если он есть; год из номера договора датой не считается;
    3. иначе — сегодняшняя дата (``datetime.date.today()``), т.е. считаем,
       что пользователя интересует актуальная на текущий момент редакция.
    Хардкод «даты актуальности базы» не используется.
    """
    if event_date and str(event_date).strip():
        return str(event_date).strip()[:10]

    year = _extract_year_from_query(query or "")
    if year:
        return f"{year}-06-01"

    return datetime.date.today().isoformat()


def resolve_allowed_versions_by_date(
    event_date: Optional[str],
    chunks_map: Dict[str, Dict[str, Any]],
    selected_doc_ids: Optional[Set[str]] = None,
    query: str = "",
    validity_unknown: Optional[Set[str]] = None,
) -> Set[str]:
    """Разрешает допустимые chunk_id с учетом временных интервалов и связок.

    Три состояния дат:
    1. даты известны и документ актуален на ``check_date`` → включаем;
    2. даты известны и документ неактуален (``effective_to`` пройден,
       ``effective_from`` ещё не наступил, кроме законов) → исключаем;
    3. даты неизвестны (``effective_dates_known=False`` или обе даты пусты)
       → **не исключаем** только по этому признаку; id добавляется в
       ``validity_unknown`` (если передан) для диагностики.

    Законы РФ не отсекаются при ``event_date < effective_from``, чтобы модель
    знала о действующей норме и могла сформулировать обоснованный отказ.
    """
    allowed_cids: Set[str] = set()
    check_date = _resolve_check_date(event_date, query)

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

        # Состояние 3: даты неизвестны — не отсекаем, но помечаем.
        if not dates_known or (not eff_from and not eff_to):
            allowed_cids.add(cid)
            if validity_unknown is not None:
                validity_unknown.add(cid)
            continue

        # Состояние 2: внутренние документы до вступления в силу.
        if eff_from and check_date < eff_from and not is_law:
            continue

        # Состояние 2: документ утратил силу к моменту события.
        if eff_to and check_date >= eff_to:
            continue

        allowed_cids.add(cid)

    return allowed_cids


def _new_stats(diagnostics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Рабочая копия счётчиков; публикуется в ``diagnostics`` в конце."""
    stats: Dict[str, Any] = {key: 0 for key in SEARCH_DIAGNOSTICS_KEYS}
    stats["selected_doc_ids"] = None
    return stats


def _publish_stats(
    stats: Dict[str, Any], diagnostics: Optional[Dict[str, Any]]
) -> None:
    if diagnostics is not None:
        diagnostics.update(stats)


def _prepare_context(
    query: str,
    event_date: Optional[str],
    selected_doc_ids: Optional[Set[str]],
) -> Tuple[
    Dict[str, Dict[str, Any]], Set[str], Set[str], Set[str]
]:
    """Готовит корпус, разрешённые id и множество неизвестных редакций."""
    chunks_map = load_canonical_chunks_map()
    date_allowed = resolve_allowed_versions_by_date(
        event_date=event_date,
        chunks_map=chunks_map,
        selected_doc_ids=None,
        query=query,
    )
    unknown: Set[str] = set()
    allowed = resolve_allowed_versions_by_date(
        event_date=event_date,
        chunks_map=chunks_map,
        selected_doc_ids=selected_doc_ids,
        query=query,
        validity_unknown=unknown,
    )
    return chunks_map, allowed, date_allowed, unknown


def _apply_context_stats(
    stats: Dict[str, Any],
    allowed: Set[str],
    unknown: Set[str],
    selected_doc_ids: Optional[Set[str]],
) -> None:
    stats["allowed_chunk_ids_count"] = len(allowed)
    stats["validity_unknown_count"] = len(unknown)
    stats["selected_doc_ids"] = (
        sorted(selected_doc_ids) if selected_doc_ids else None
    )


def _apply_bm25_stats(
    stats: Dict[str, Any], bm25_stats: Dict[str, int]
) -> None:
    """Переносит диагностику единственного BM25-вызова в общие счётчики."""
    stats["bm25_candidates_count"] = bm25_stats.get("returned_count", 0)
    stats["candidates_before_filters"] += bm25_stats.get(
        "candidates_before_filters", 0
    )
    stats["candidates_after_date_filter"] += bm25_stats.get(
        "candidates_after_date_filter", 0
    )
    stats["candidates_after_document_filter"] += bm25_stats.get(
        "candidates_after_document_filter", 0
    )


def _dense_hits(
    query: str,
    top_k: int,
    allowed: Set[str],
    date_allowed: Set[str],
    stats: Dict[str, Any],
    collect_diagnostics: bool = False,
    chunks_map: Optional[Dict[str, Dict[str, Any]]] = None,
    selected_doc_ids: Optional[Set[str]] = None,
    event_date: Optional[str] = None,
) -> List[str]:
    """Векторные кандидаты с пре-фильтрацией Chroma where и проверкой по allowed."""
    if not allowed:
        stats["dense_candidates_count"] = 0
        return []

    query_vector = get_query_embedding(query)
    collection = get_chroma_collection()
    validate_query_embedding_dimension(collection, query_vector)

    use_where = bool(selected_doc_ids) or bool(event_date and str(event_date).strip())
    where_filter: Optional[Dict[str, Any]] = None
    if use_where and chunks_map:
        allowed_doc_ids = [
            chunks_map[cid]["document_id"]
            for cid in allowed
            if cid in chunks_map and chunks_map[cid].get("document_id")
        ]
        allowed_ver_ids = [
            chunks_map[cid]["version_id"]
            for cid in allowed
            if cid in chunks_map and chunks_map[cid].get("version_id")
        ]
        where_filter = build_chroma_where_filter(
            document_ids=allowed_doc_ids,
            version_ids=allowed_ver_ids,
        )

    chroma_res = query_chroma_collection(
        collection=collection,
        query_vector=query_vector,
        n_results=top_k,
        where=where_filter,
    )

    ids_outer = chroma_res.get("ids") or [[]]
    dense_ids = list(ids_outer[0] or []) if ids_outer else []
    stats["dense_candidates_count"] = len(dense_ids)

    if collect_diagnostics:
        stats["candidates_before_filters"] += len(dense_ids)
        stats["candidates_after_date_filter"] += sum(
            1 for cid in dense_ids if cid in date_allowed
        )
        stats["candidates_after_document_filter"] += sum(
            1 for cid in dense_ids if cid in allowed
        )

    return [cid for cid in dense_ids if cid in allowed]


def _build_candidates(
    rrf_scores: Dict[str, float],
    chunks_map: Dict[str, Dict[str, Any]],
    boost_amendments: bool = False,
) -> List[Dict[str, Any]]:
    """Единый постпроцессинг: условный приоритет соглашений + лимит пула 40."""
    scored = dict(rrf_scores)
    if boost_amendments:
        # H-2C: повышаем только допсоглашения и только когда запрос явно
        # релевантен версии/допсоглашению (решение принимает вызывающий код).
        for cid in scored:
            chunk = chunks_map.get(cid)
            if chunk and chunk.get("amends_document_id"):
                scored[cid] *= _AMENDMENT_BOOST_FACTOR

    sorted_cids = sorted(
        scored.keys(), key=lambda x: scored[x], reverse=True
    )[:40]

    candidate_chunks: List[Dict[str, Any]] = []
    for cid in sorted_cids:
        chunk = chunks_map.get(cid)
        if chunk is not None:
            chunk_copy = dict(chunk)
            chunk_copy["score"] = round(float(scored[cid]), 6)
            candidate_chunks.append(chunk_copy)
    return candidate_chunks


def _weighted_rrf(
    bm25_results: List[Tuple[str, float]],
    dense_ids: List[str],
    intent_cfg: Dict[str, Any],
    rrf_k: int,
) -> Dict[str, float]:
    rrf_scores: Dict[str, float] = {}

    for rank, (cid, _score) in enumerate(bm25_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
            intent_cfg["w_bm25"] / (rrf_k + rank + 1)
        )

    for rank, cid in enumerate(dense_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
            intent_cfg["w_dense"] / (rrf_k + rank + 1)
        )

    return rrf_scores


def _finalize(
    rrf_scores: Dict[str, float],
    chunks_map: Dict[str, Dict[str, Any]],
    query: str,
    top_k: int,
    rerank: bool,
    stats: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]],
    boost_amendments: bool = False,
) -> List[Dict[str, Any]]:
    candidate_chunks = _build_candidates(
        rrf_scores, chunks_map, boost_amendments=boost_amendments
    )
    if rerank:
        result = rerank_candidates(query, candidate_chunks, top_k=top_k)
    else:
        result = candidate_chunks[:top_k]

    stats["final_chunks_count"] = len(result)
    _publish_stats(stats, diagnostics)
    return result


def hybrid_search(
    query: str,
    top_k: int = 10,
    event_date: Optional[str] = None,
    selected_doc_ids: Optional[Set[str]] = None,
    rrf_k: int = 60,
    intent: Optional[str] = None,
    rerank: bool = True,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """BM25 + dense с взвешенным RRF; неизвестный intent → mixed_query."""
    chunks_map, allowed, date_allowed, unknown = _prepare_context(
        query, event_date, selected_doc_ids
    )
    stats = _new_stats(diagnostics)
    _apply_context_stats(stats, allowed, unknown, selected_doc_ids)

    if not allowed:
        _publish_stats(stats, diagnostics)
        return []

    intent_cfg = detect_query_intent(query, precomputed_intent=intent)

    # 1. Единственный вызов BM25: результаты и диагностика за один проход.
    bm25_results, bm25_stats = bm25_store.search_with_stats(
        query,
        top_k=intent_cfg["bm25_top"],
        allowed_chunk_ids=allowed,
        date_allowed_chunk_ids=date_allowed,
    )
    _apply_bm25_stats(stats, bm25_stats)

    # 2. Поиск Dense Vector
    dense_ids = _dense_hits(
        query=query,
        top_k=intent_cfg["dense_top"],
        allowed=allowed,
        date_allowed=date_allowed,
        stats=stats,
        collect_diagnostics=diagnostics is not None,
        chunks_map=chunks_map,
        selected_doc_ids=selected_doc_ids,
        event_date=event_date,
    )

    # 3. Взвешенный RRF
    rrf_scores = _weighted_rrf(bm25_results, dense_ids, intent_cfg, rrf_k)

    return _finalize(
        rrf_scores=rrf_scores,
        chunks_map=chunks_map,
        query=query,
        top_k=top_k,
        rerank=rerank,
        stats=stats,
        diagnostics=diagnostics,
        boost_amendments=should_boost_amendments(query, intent),
    )


def bm25_search(
    query: str,
    top_k: int = 10,
    event_date: Optional[str] = None,
    selected_doc_ids: Optional[Set[str]] = None,
    rrf_k: int = 60,
    intent: Optional[str] = None,
    rerank: bool = True,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Только лексический канал: без вызова эмбеддингов и Chroma."""
    chunks_map, allowed, date_allowed, unknown = _prepare_context(
        query, event_date, selected_doc_ids
    )
    stats = _new_stats(diagnostics)
    _apply_context_stats(stats, allowed, unknown, selected_doc_ids)

    if not allowed:
        _publish_stats(stats, diagnostics)
        return []

    intent_cfg = detect_query_intent(query, precomputed_intent=intent)

    # Единственный вызов BM25: результаты и диагностика за один проход.
    bm25_results, bm25_stats = bm25_store.search_with_stats(
        query,
        top_k=intent_cfg["bm25_top"],
        allowed_chunk_ids=allowed,
        date_allowed_chunk_ids=date_allowed,
    )
    _apply_bm25_stats(stats, bm25_stats)

    rrf_scores: Dict[str, float] = {}
    for rank, (cid, _score) in enumerate(bm25_results):
        rrf_scores[cid] = intent_cfg["w_bm25"] / (rrf_k + rank + 1)

    return _finalize(
        rrf_scores=rrf_scores,
        chunks_map=chunks_map,
        query=query,
        top_k=top_k,
        rerank=rerank,
        stats=stats,
        diagnostics=diagnostics,
        boost_amendments=should_boost_amendments(query, intent),
    )


def dense_search(
    query: str,
    top_k: int = 10,
    event_date: Optional[str] = None,
    selected_doc_ids: Optional[Set[str]] = None,
    rrf_k: int = 60,
    intent: Optional[str] = None,
    rerank: bool = True,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Только векторный канал: без BM25."""
    chunks_map, allowed, date_allowed, unknown = _prepare_context(
        query, event_date, selected_doc_ids
    )
    stats = _new_stats(diagnostics)
    _apply_context_stats(stats, allowed, unknown, selected_doc_ids)

    if not allowed:
        _publish_stats(stats, diagnostics)
        return []

    intent_cfg = detect_query_intent(query, precomputed_intent=intent)

    dense_ids = _dense_hits(
        query=query,
        top_k=intent_cfg["dense_top"],
        allowed=allowed,
        date_allowed=date_allowed,
        stats=stats,
        collect_diagnostics=diagnostics is not None,
        chunks_map=chunks_map,
        selected_doc_ids=selected_doc_ids,
        event_date=event_date,
    )

    rrf_scores: Dict[str, float] = {}
    for rank, cid in enumerate(dense_ids):
        rrf_scores[cid] = intent_cfg["w_dense"] / (rrf_k + rank + 1)

    return _finalize(
        rrf_scores=rrf_scores,
        chunks_map=chunks_map,
        query=query,
        top_k=top_k,
        rerank=rerank,
        stats=stats,
        diagnostics=diagnostics,
        boost_amendments=should_boost_amendments(query, intent),
    )


def check_index_consistency() -> Dict[str, Any]:
    """Неразрушающая сверка chunk_id: chunks.jsonl / Chroma / BM25.

    Ничего не удаляет и не переиндексирует: только читает снимки id и
    размерность сохранённых эмбеддингов. Возвращает отчёт для verify_submit.
    """
    report: Dict[str, Any] = {
        "jsonl_chunks": 0,
        "chroma_chunks": None,
        "bm25_chunks": None,
        "missing_in_chroma": [],
        "stale_in_chroma": [],
        "missing_in_bm25": [],
        "stale_in_bm25": [],
        "chroma_embedding_dimension": None,
        "dimension_mismatch": False,
        "warnings": [],
        "errors": [],
    }

    chunks_map = load_canonical_chunks_map()
    jsonl_ids = set(chunks_map)
    report["jsonl_chunks"] = len(jsonl_ids)

    try:
        bm25_ids = set(bm25_store.get_chunk_ids())
        report["bm25_chunks"] = len(bm25_ids)
        report["missing_in_bm25"] = sorted(jsonl_ids - bm25_ids)[:50]
        report["stale_in_bm25"] = sorted(bm25_ids - jsonl_ids)[:50]
    except Exception as exc:  # noqa: BLE001 - отчёт не должен падать
        report["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)[:300]}
        )

    try:
        collection = get_chroma_collection()
        chroma_ids = set(collection.get(include=[])["ids"])
        report["chroma_chunks"] = len(chroma_ids)
        report["missing_in_chroma"] = sorted(jsonl_ids - chroma_ids)[:50]
        report["stale_in_chroma"] = sorted(chroma_ids - jsonl_ids)[:50]
        dimension = get_collection_embedding_dimension(collection, refresh=True)
        report["chroma_embedding_dimension"] = dimension
        expected = expected_embedding_dimension()
        report["expected_embedding_dimension"] = expected
        if dimension is not None and expected and dimension != expected:
            report["dimension_mismatch"] = True
            report["warnings"].append(
                "Размерность эмбеддингов Chroma "
                f"({dimension}) не совпадает с текущей моделью ({expected})"
            )
    except Exception as exc:  # noqa: BLE001 - отчёт не должен падать
        report["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)[:300]}
        )

    if report["missing_in_chroma"]:
        report["warnings"].append(
            "В Chroma отсутствуют чанки из chunks.jsonl: "
            f"{len(report['missing_in_chroma'])} (показаны первые 50)"
        )
    if report["stale_in_chroma"]:
        report["warnings"].append(
            "В Chroma есть устаревшие chunk_id (нет в chunks.jsonl): "
            f"{len(report['stale_in_chroma'])} (показаны первые 50)"
        )
    if report["missing_in_bm25"]:
        report["warnings"].append(
            "В BM25 отсутствуют чанки из chunks.jsonl: "
            f"{len(report['missing_in_bm25'])}"
        )
    if report["stale_in_bm25"]:
        report["warnings"].append(
            "В BM25 есть устаревшие chunk_id: "
            f"{len(report['stale_in_bm25'])}"
        )

    return report
