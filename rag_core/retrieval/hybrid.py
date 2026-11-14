import copy
import datetime
import json
import logging
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rag_core.config import PROCESSED_DATA_DIR
from rag_core.generation.analysis.source_kind import (
    SOURCE_KIND_UNKNOWN,
    source_kind_for_document_type,
)
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
    "date_mode",
    "date_precision",
    "source_kind_counts",
    "version_status_counts",
    "deduped_versions_count",
)

# Год внутри номера договора (``<префикс>-<номер>/<год>``) не является
# временной привязкой запроса и не должен влиять на дату проверки.
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

# Статусы редакции документа на дату проверки.
EFFECTIVE_STATUS_APPLICABLE = "applicable"
EFFECTIVE_STATUS_NOT_APPLICABLE = "not_applicable"
EFFECTIVE_STATUS_UNKNOWN = "unknown"
EFFECTIVE_STATUS_CURRENT_ONLY = "current_only"

_UNCERTAIN_STATUSES = frozenset(
    {EFFECTIVE_STATUS_UNKNOWN, EFFECTIVE_STATUS_CURRENT_ONLY}
)
_UNCERTAIN_PRECISIONS = frozenset({"year", "month", "none"})
_FALSE_FLAGS = frozenset({"false", "0", ""})
_STATUS_PRIORITY = {
    EFFECTIVE_STATUS_APPLICABLE: 4,
    EFFECTIVE_STATUS_CURRENT_ONLY: 3,
    EFFECTIVE_STATUS_UNKNOWN: 2,
    EFFECTIVE_STATUS_NOT_APPLICABLE: 1,
}


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

    Номера договоров (``<префикс>-<номер>/<год>``) перед поиском вырезаются,
    чтобы их год не становился датой проверки. Поддерживаются полная дата
    (``dd.mm.yyyy``), «<год> год/году/г.», а также год после временного
    предлога («в <год>»). Возвращается первый по позиции.
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


def _is_false_flag(value: Any) -> bool:
    """Приводит флаг ``effective_dates_known`` к bool без исключений."""
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return not bool(value)
    if isinstance(value, str):
        return value.strip().lower() in _FALSE_FLAGS
    return False


def _is_blank(value: Any) -> bool:
    """True для ``None`` и строк из одних пробелов."""
    return value is None or (isinstance(value, str) and not value.strip())


def parse_iso_date(value: Any) -> Optional[datetime.date]:
    """Безопасно разбирает ``YYYY-MM-DD``; некорректные значения → ``None``."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) > 10:
        text = text[:10]
    try:
        return datetime.date.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def effective_status(chunk: Dict[str, Any], check_date: str) -> str:
    """Единый статус редакции чанка на ``check_date``.

    Границы: ``effective_from <= дата < effective_to``; отсутствие
    ``effective_from`` — открытая нижняя граница, отсутствие ``effective_to``
    — открытая верхняя. Для законов ``check_date < effective_from`` даёт
    ``current_only`` (в базе только актуальная редакция), для остальных —
    ``not_applicable``.
    """
    if _is_false_flag(chunk.get("effective_dates_known", True)):
        return EFFECTIVE_STATUS_UNKNOWN

    eff_from = chunk.get("effective_from")
    eff_to = chunk.get("effective_to")
    if _is_blank(eff_from) and _is_blank(eff_to):
        return EFFECTIVE_STATUS_UNKNOWN

    from_date = None
    if not _is_blank(eff_from):
        from_date = parse_iso_date(eff_from)
        if from_date is None:
            return EFFECTIVE_STATUS_UNKNOWN

    to_date = None
    if not _is_blank(eff_to):
        to_date = parse_iso_date(eff_to)
        if to_date is None:
            return EFFECTIVE_STATUS_UNKNOWN

    check = parse_iso_date(check_date)
    if check is None:
        return EFFECTIVE_STATUS_UNKNOWN

    if from_date is not None and check < from_date:
        if chunk.get("document_type") == "law":
            return EFFECTIVE_STATUS_CURRENT_ONLY
        return EFFECTIVE_STATUS_NOT_APPLICABLE
    if to_date is not None and check >= to_date:
        return EFFECTIVE_STATUS_NOT_APPLICABLE
    return EFFECTIVE_STATUS_APPLICABLE


def resolve_check_date_details(
    event_date: Optional[str], query: str = ""
) -> Tuple[str, str, str]:
    """Возвращает ``(check_date, date_mode, date_precision)``.

    1. ``event_date`` передан и парсится → ``historical``/``day``;
    2. иначе год из запроса → ``historical``/``year`` (середина года);
    3. иначе сегодня → ``current``/``day``.
    Невалидный ``event_date`` не подменяется молча: режим ``current``,
    точность ``none`` — это видно в diagnostics.
    """
    today = datetime.date.today().isoformat()
    if event_date and str(event_date).strip():
        parsed = parse_iso_date(str(event_date).strip())
        if parsed is not None:
            return parsed.isoformat(), "historical", "day"
        return today, "current", "none"

    year = _extract_year_from_query(query or "")
    if year:
        return f"{year}-06-01", "historical", "year"
    return today, "current", "day"


def _resolve_check_date(
    event_date: Optional[str], query: str = ""
) -> str:
    """Дата, на которую проверяется актуальность редакции документа.

    Обёртка над :func:`resolve_check_date_details` для обратной совместимости.
    """
    return resolve_check_date_details(event_date, query)[0]


def _normalize_document_types(
    document_types: Iterable[str],
) -> Set[str]:
    """Нормализует набор типов документов для фильтрации."""
    return {
        str(value).strip().lower()
        for value in document_types
        if value is not None and str(value).strip()
    }


def _filter_by_document_types(
    allowed_cids: Set[str],
    chunks_map: Dict[str, Dict[str, Any]],
    document_types: Optional[Set[str]],
) -> Set[str]:
    """Оставляет чанки с ``document_type`` из явного набора.

    Чанки без ``document_type`` при явном фильтре отбрасываются: это
    фильтр, а не классификация.
    """
    if document_types is None:
        return set(allowed_cids)
    normalized = _normalize_document_types(document_types)
    result: Set[str] = set()
    for cid in allowed_cids:
        chunk = chunks_map.get(cid)
        if chunk is None:
            continue
        raw_type = chunk.get("document_type")
        if raw_type is None:
            continue
        if str(raw_type).strip().lower() in normalized:
            result.add(cid)
    return result


def _version_status(statuses: Iterable[str]) -> str:
    """Статус версии — наиболее пригодный статус среди её чанков."""
    best = EFFECTIVE_STATUS_UNKNOWN
    best_priority = -1
    for status in statuses:
        priority = _STATUS_PRIORITY.get(status, 0)
        if priority > best_priority:
            best = status
            best_priority = priority
    return best


def _version_effective_from(
    chunks: Iterable[Dict[str, Any]],
) -> Optional[datetime.date]:
    """Первая разбираемая ``effective_from`` среди чанков версии."""
    for chunk in chunks:
        parsed = parse_iso_date(chunk.get("effective_from"))
        if parsed is not None:
            return parsed
    return None


def _latest_versions(
    version_ids: Iterable[str],
    version_from: Dict[str, Optional[datetime.date]],
) -> Set[str]:
    """Выбирает версии с максимальной ``effective_from <= check_date``.

    Если ни у одной версии нет разбираемой даты — возвращает все
    версии-кандидаты (неизвестная дата не позволяет выбрать одну).
    """
    candidates = list(version_ids)
    dated = {
        vid: version_from.get(vid)
        for vid in candidates
        if version_from.get(vid) is not None
    }
    if not dated:
        return set(candidates)
    latest = max(dated.values())
    return {vid for vid, value in dated.items() if value == latest}


def select_dominant_versions(
    allowed_cids: Set[str],
    chunks_map: Dict[str, Dict[str, Any]],
    check_date: str,
) -> Set[str]:
    """Убирает одновременно взаимоисключающие редакции одного документа.

    Группировка — по ``document_id``; документы с одной версией сохраняются
    целиком. При нескольких версиях приоритет: ``applicable`` (самая свежая
    ``effective_from``), затем ``current_only``, затем ``unknown``.
    ``not_applicable`` не удаляются здесь — это уже сделано на уровне
    ``resolve_allowed_versions_by_date``.
    """
    groups: Dict[str, List[Tuple[str, str]]] = {}
    passthrough: Set[str] = set()
    for cid in allowed_cids:
        chunk = chunks_map.get(cid)
        if chunk is None:
            passthrough.add(cid)
            continue
        document_id = chunk.get("document_id")
        if not document_id:
            passthrough.add(cid)
            continue
        version_id = chunk.get("version_id") or cid
        groups.setdefault(str(document_id), []).append((cid, str(version_id)))

    selected: Set[str] = set(passthrough)
    for members in groups.values():
        versions: Dict[str, List[str]] = {}
        for cid, version_id in members:
            versions.setdefault(version_id, []).append(cid)

        if len(versions) == 1:
            selected.update(next(iter(versions.values())))
            continue

        version_statuses: Dict[str, str] = {}
        version_from: Dict[str, Optional[datetime.date]] = {}
        for version_id, cids in versions.items():
            version_chunks = [chunks_map[cid] for cid in cids if cid in chunks_map]
            version_statuses[version_id] = _version_status(
                effective_status(chunk, check_date)
                for chunk in version_chunks
            )
            version_from[version_id] = _version_effective_from(version_chunks)

        applicable = [
            vid
            for vid, status in version_statuses.items()
            if status == EFFECTIVE_STATUS_APPLICABLE
        ]
        if applicable:
            chosen = _latest_versions(applicable, version_from)
        elif any(
            status == EFFECTIVE_STATUS_CURRENT_ONLY
            for status in version_statuses.values()
        ):
            chosen = {
                vid
                for vid, status in version_statuses.items()
                if status == EFFECTIVE_STATUS_CURRENT_ONLY
            }
        elif any(
            status == EFFECTIVE_STATUS_UNKNOWN
            for status in version_statuses.values()
        ):
            chosen = {
                vid
                for vid, status in version_statuses.items()
                if status == EFFECTIVE_STATUS_UNKNOWN
            }
        else:
            chosen = set(versions)

        for version_id in chosen:
            selected.update(versions[version_id])

    return selected


def resolve_allowed_versions_by_date(
    event_date: Optional[str],
    chunks_map: Dict[str, Dict[str, Any]],
    selected_doc_ids: Optional[Set[str]] = None,
    query: str = "",
    validity_unknown: Optional[Set[str]] = None,
) -> Set[str]:
    """Разрешает допустимые chunk_id с учетом временных интервалов и связок.

    Единая семантика :func:`effective_status`:
    * ``applicable`` → включаем;
    * ``current_only`` (закон до ``effective_from``) → включаем;
    * ``unknown`` → **не исключаем**, id добавляется в ``validity_unknown``;
    * ``not_applicable`` → исключаем.
    """
    allowed_cids: Set[str] = set()
    check_date = _resolve_check_date(event_date, query)

    expanded_doc_ids = set(selected_doc_ids) if selected_doc_ids else None
    if expanded_doc_ids:
        for ch in chunks_map.values():
            amends = ch.get("amends_document_id")
            if not amends or amends not in expanded_doc_ids:
                continue
            document_id = ch.get("document_id")
            if document_id:
                expanded_doc_ids.add(str(document_id))

    for cid, ch in chunks_map.items():
        doc_id = ch.get("document_id")
        if expanded_doc_ids and doc_id not in expanded_doc_ids:
            continue

        status = effective_status(ch, check_date)
        if status == EFFECTIVE_STATUS_NOT_APPLICABLE:
            continue

        allowed_cids.add(cid)
        if (
            status == EFFECTIVE_STATUS_UNKNOWN
            and validity_unknown is not None
        ):
            validity_unknown.add(cid)

    return allowed_cids


def _new_stats(diagnostics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Рабочая копия счётчиков; публикуется в ``diagnostics`` в конце."""
    stats: Dict[str, Any] = {key: 0 for key in SEARCH_DIAGNOSTICS_KEYS}
    stats["selected_doc_ids"] = None
    stats["date_mode"] = None
    stats["date_precision"] = None
    stats["source_kind_counts"] = {}
    stats["version_status_counts"] = {}
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
    document_types: Optional[Set[str]] = None,
) -> Tuple[
    Dict[str, Dict[str, Any]], Set[str], Set[str], Set[str], int
]:
    """Готовит корпус, разрешённые id, неизвестные редакции и счётчик дедупа.

    ``document_types`` — явный фильтр по типу документа (``None`` отключает).
    После фильтрации к обоим множествам применяется выбор доминирующей
    редакции, чтобы не возвращать одновременно взаимоисключающие версии.
    """
    chunks_map = load_canonical_chunks_map()
    check_date = _resolve_check_date(event_date, query)
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

    if document_types is not None:
        date_allowed = _filter_by_document_types(
            date_allowed, chunks_map, document_types
        )
        allowed = _filter_by_document_types(
            allowed, chunks_map, document_types
        )

    before = len(allowed) + len(date_allowed)
    date_allowed = select_dominant_versions(date_allowed, chunks_map, check_date)
    allowed = select_dominant_versions(allowed, chunks_map, check_date)
    unknown &= allowed
    deduped_versions = max(before - (len(allowed) + len(date_allowed)), 0)
    return chunks_map, allowed, date_allowed, unknown, deduped_versions


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


def _merge_where_filter(
    where_filter: Optional[Dict[str, Any]],
    document_types: Optional[Set[str]],
) -> Optional[Dict[str, Any]]:
    """Добавляет к Chroma-фильтру условие ``document_type`` через ``$in``."""
    if not document_types:
        return where_filter
    normalized = sorted(_normalize_document_types(document_types))
    if not normalized:
        return where_filter
    type_clause: Dict[str, Any] = (
        {"document_type": normalized[0]}
        if len(normalized) == 1
        else {"document_type": {"$in": normalized}}
    )
    if where_filter is None:
        return type_clause
    return {"$and": [where_filter, type_clause]}


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
    document_types: Optional[Set[str]] = None,
) -> List[str]:
    """Векторные кандидаты с пре-фильтрацией Chroma where и проверкой по allowed."""
    if not allowed:
        stats["dense_candidates_count"] = 0
        return []

    query_vector = get_query_embedding(query)
    collection = get_chroma_collection()
    validate_query_embedding_dimension(collection, query_vector)

    need_allowed_clause = bool(selected_doc_ids) or bool(
        event_date and str(event_date).strip()
    )
    where_filter: Optional[Dict[str, Any]] = None
    if need_allowed_clause and chunks_map:
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
    where_filter = _merge_where_filter(where_filter, document_types)

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


def _annotate_chunk(
    chunk: Dict[str, Any],
    check_date: Optional[str],
    date_mode: str,
    date_precision: str,
) -> Dict[str, Any]:
    """Проставляет назначение источника, статус версии и признак неопределённости."""
    chunk["document_type"] = chunk.get("document_type")
    chunk["source_kind"] = source_kind_for_document_type(
        chunk.get("document_type")
    )
    status = effective_status(chunk, check_date) if check_date else None
    chunk["version_status"] = status
    chunk["validity_uncertain"] = bool(
        status in _UNCERTAIN_STATUSES
        or date_precision in _UNCERTAIN_PRECISIONS
    )
    chunk["effective_date_mode"] = date_mode
    return chunk


def _set_chunk_field_stats(
    stats: Dict[str, Any], chunks: Iterable[Dict[str, Any]]
) -> None:
    """Считает распределение итоговых чанков по назначению и статусу версии."""
    source_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    for chunk in chunks:
        source_kind = chunk.get("source_kind") or SOURCE_KIND_UNKNOWN
        source_counts[str(source_kind)] = (
            source_counts.get(str(source_kind), 0) + 1
        )
        status = chunk.get("version_status")
        if status:
            status_counts[str(status)] = status_counts.get(str(status), 0) + 1
    stats["source_kind_counts"] = source_counts
    stats["version_status_counts"] = status_counts


def _build_candidates(
    rrf_scores: Dict[str, float],
    chunks_map: Dict[str, Dict[str, Any]],
    boost_amendments: bool = False,
    check_date: Optional[str] = None,
    date_mode: str = "current",
    date_precision: str = "day",
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
            # Копия: исходный chunks_map не мутируется (кэш/повторные вызовы).
            chunk_copy = _annotate_chunk(
                dict(chunk), check_date, date_mode, date_precision
            )
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
    check_date: Optional[str] = None,
    date_mode: str = "current",
    date_precision: str = "day",
) -> List[Dict[str, Any]]:
    candidate_chunks = _build_candidates(
        rrf_scores,
        chunks_map,
        boost_amendments=boost_amendments,
        check_date=check_date,
        date_mode=date_mode,
        date_precision=date_precision,
    )
    if rerank:
        result = rerank_candidates(query, candidate_chunks, top_k=top_k)
    else:
        result = candidate_chunks[:top_k]

    stats["final_chunks_count"] = len(result)
    _set_chunk_field_stats(stats, result)
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
    document_types: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """BM25 + dense с взвешенным RRF; неизвестный intent → mixed_query."""
    check_date, date_mode, date_precision = resolve_check_date_details(
        event_date, query
    )
    (
        chunks_map,
        allowed,
        date_allowed,
        unknown,
        deduped_versions,
    ) = _prepare_context(query, event_date, selected_doc_ids, document_types)
    stats = _new_stats(diagnostics)
    stats["date_mode"] = date_mode
    stats["date_precision"] = date_precision
    stats["deduped_versions_count"] = deduped_versions
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
        document_types=document_types,
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
        check_date=check_date,
        date_mode=date_mode,
        date_precision=date_precision,
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
    document_types: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Только лексический канал: без вызова эмбеддингов и Chroma."""
    check_date, date_mode, date_precision = resolve_check_date_details(
        event_date, query
    )
    (
        chunks_map,
        allowed,
        date_allowed,
        unknown,
        deduped_versions,
    ) = _prepare_context(query, event_date, selected_doc_ids, document_types)
    stats = _new_stats(diagnostics)
    stats["date_mode"] = date_mode
    stats["date_precision"] = date_precision
    stats["deduped_versions_count"] = deduped_versions
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
        check_date=check_date,
        date_mode=date_mode,
        date_precision=date_precision,
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
    document_types: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Только векторный канал: без BM25."""
    check_date, date_mode, date_precision = resolve_check_date_details(
        event_date, query
    )
    (
        chunks_map,
        allowed,
        date_allowed,
        unknown,
        deduped_versions,
    ) = _prepare_context(query, event_date, selected_doc_ids, document_types)
    stats = _new_stats(diagnostics)
    stats["date_mode"] = date_mode
    stats["date_precision"] = date_precision
    stats["deduped_versions_count"] = deduped_versions
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
        document_types=document_types,
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
        check_date=check_date,
        date_mode=date_mode,
        date_precision=date_precision,
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
