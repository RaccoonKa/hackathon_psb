from dataclasses import replace
import datetime
import inspect
import logging
import time

from rag_core import config
from rag_core.generation.analysis.source_kind import (
    SOURCE_KIND_UNKNOWN,
    document_types_for_source_kind,
    source_kind_for_document_type,
)
from rag_core.generation.errors import (
    RetrievalAPIError,
    RetrievalTimeoutError,
)
from rag_core.generation.stubs import stub_rag_retrieve, stub_reranker
from rag_core.generation.types import (
    Intent,
    RetrievalTask,
    RetrievedChunk,
    TraceStep,
)
from rag_core.retrieval.hybrid import (
    bm25_search,
    dense_search,
    hybrid_search,
)
from rag_core.retrieval.reranker import rerank_candidates
from rag_core.retrieval.vector_store import (
    get_last_embedding_diagnostics,
    prewarm_retrieval_backends,
    reset_embedding_diagnostics,
)

logger = logging.getLogger(__name__)

_MODE_TO_HYBRID_INTENT: dict[str, str] = {
    "bm25": "exact_lookup",
    "dense": "semantic_query",
    "hybrid": "mixed_query",
}

_SEARCH_DIAGNOSTICS_DEFAULTS: dict = {
    "selected_doc_ids": None,
    "allowed_chunk_ids_count": 0,
    "bm25_candidates_count": 0,
    "dense_candidates_count": 0,
    "candidates_before_filters": 0,
    "candidates_after_date_filter": 0,
    "candidates_after_document_filter": 0,
    "final_chunks_count": 0,
    "validity_unknown_count": 0,
    "date_mode": None,
    "date_precision": None,
    "source_kinds": [],
    "source_kind_counts": {},
    "source_kind_empty": [],
    "version_status_counts": {},
    "deduped_versions_count": 0,
}

# Для задач, которые вовсе не используют эмбеддинги (bm25): provider none
# означает «канал не задействован», поэтому success не может быть true без
# измерения.
_EMPTY_EMBEDDINGS_DIAGNOSTICS: dict = {
    "provider": "none",
    "model": None,
    "dimension": None,
    "success": None,
    "fallback_used": False,
    "error_type": None,
    "measured": False,
}

# Для задач, которые не завершились (timeout/queue_timeout/cancelled/
# not_started): измерений нет, поэтому provider=unknown, counts=None.
_UNKNOWN_EMBEDDINGS_DIAGNOSTICS: dict = {
    "provider": "unknown",
    "model": None,
    "dimension": None,
    "success": None,
    "fallback_used": None,
    "error_type": None,
    "measured": False,
}

_SEARCH_NOT_EXECUTED_DEFAULTS: dict = {
    key: None for key in _SEARCH_DIAGNOSTICS_DEFAULTS
}

_SIGNATURE_CACHE: dict = {}


def _now_iso() -> str:
    """UTC-время в ISO 8601 для диагностики (без секретов)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="milliseconds"
    )


def _apply_diagnostics_to_state(
    state: dict | None,
    *,
    stats: dict | None = None,
    embeddings: dict | None = None,
) -> None:
    """Переносит измеренные счётчики каналов и эмбеддингов в состояние."""
    if state is None:
        return
    if stats is not None:
        state["search"] = dict(stats)
        state["bm25_candidates_count"] = stats.get("bm25_candidates_count")
        state["dense_candidates_count"] = stats.get("dense_candidates_count")
        state["candidates_before_filters"] = stats.get(
            "candidates_before_filters"
        )
        after_filters = stats.get("candidates_after_document_filter")
        if after_filters is None:
            after_filters = stats.get("candidates_after_date_filter")
        state["candidates_after_filters"] = after_filters
        state["final_chunks_count"] = stats.get("final_chunks_count")
    if embeddings is not None:
        state["embeddings"] = dict(embeddings)
        state["embedding_provider"] = embeddings.get("provider")
        state["embedding_model"] = embeddings.get("model")
        state["model_reused"] = embeddings.get("model_reused")


def _accepts(func, param_name: str) -> bool:
    """Проверяет наличие параметра в сигнатуре (обратная совместимость).

    ``(*args, **kwargs)`` (в том числе unittest.mock) считается
    принимающим любой параметр.
    """
    key = (id(func), param_name)
    cached = _SIGNATURE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        accepted = True
    else:
        accepted = param_name in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    _SIGNATURE_CACHE[key] = accepted
    return accepted


def _dict_to_chunk(raw: dict) -> RetrievedChunk:
    structure = raw.get("structure") or {}
    raw_score = raw.get("score")
    score = (
        float(raw_score)
        if raw_score is not None
        else config.REAL_RAG_DEFAULT_SCORE
    )
    return RetrievedChunk(
        chunk_id=raw["chunk_id"],
        text=raw["text"],
        document_id=raw["document_id"],
        version_id=raw["version_id"],
        document_title=raw["document_title"],
        provision_id=structure.get("provision_id"),
        structure_path=list(structure.get("path") or []),
        score=score,
        effective_from=raw.get("effective_from"),
        effective_to=raw.get("effective_to"),
        effective_dates_known=bool(raw.get("effective_dates_known", True)),
        amends_document_id=raw.get("amends_document_id"),
        document_type=raw.get("document_type"),
        source_kind=raw.get("source_kind"),
        version_status=raw.get("version_status"),
        validity_uncertain=bool(raw.get("validity_uncertain", False)),
        effective_date_mode=raw.get("effective_date_mode"),
    )


def _call_search(
    search_func,
    *,
    query: str,
    top_k: int,
    event_date: str | None,
    selected_doc_ids: set[str] | None,
    intent_str: str | None,
    diagnostics: dict,
    document_types: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Общий вызов поискового канала с прокидыванием диагностики."""
    kwargs: dict = {
        "query": query,
        "top_k": top_k,
        "event_date": event_date,
        "selected_doc_ids": selected_doc_ids,
    }
    if _accepts(search_func, "intent") and intent_str is not None:
        kwargs["intent"] = intent_str
    if _accepts(search_func, "rerank"):
        # Реранкинг выполняется отдельным этапом rerank_chunks (локальный
        # Qwen3-reranker из ветки A), поэтому внутри поиска его отключаем,
        # чтобы не выполнять реранкинг дважды.
        kwargs["rerank"] = False
    if _accepts(search_func, "diagnostics"):
        kwargs["diagnostics"] = diagnostics
    if document_types is not None and _accepts(search_func, "document_types"):
        kwargs["document_types"] = document_types

    raw_chunks = search_func(**kwargs)
    return [_dict_to_chunk(raw) for raw in raw_chunks]


def _call_hybrid_search(
    query: str,
    top_k: int,
    event_date: str | None,
    selected_doc_ids: set[str] | None,
    intent_str: str | None,
    diagnostics: dict | None = None,
    document_types: set[str] | None = None,
) -> list[RetrievedChunk]:
    if config.USE_STUB_RAG:
        return stub_rag_retrieve(query=query, top_k=top_k)
    return _call_search(
        hybrid_search,
        query=query,
        top_k=top_k,
        event_date=event_date,
        selected_doc_ids=selected_doc_ids,
        intent_str=intent_str,
        diagnostics=diagnostics if diagnostics is not None else {},
        document_types=document_types,
    )


def _call_lexical_search(
    query: str,
    top_k: int,
    event_date: str | None,
    selected_doc_ids: set[str] | None,
    intent_str: str | None,
    diagnostics: dict | None = None,
    document_types: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Только BM25: эмбеддинги и Chroma не вызываются."""
    return _call_search(
        bm25_search,
        query=query,
        top_k=top_k,
        event_date=event_date,
        selected_doc_ids=selected_doc_ids,
        intent_str=intent_str,
        diagnostics=diagnostics if diagnostics is not None else {},
        document_types=document_types,
    )


def _call_dense_search(
    query: str,
    top_k: int,
    event_date: str | None,
    selected_doc_ids: set[str] | None,
    intent_str: str | None,
    diagnostics: dict | None = None,
    document_types: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Только векторный поиск: BM25 не вызывается."""
    return _call_search(
        dense_search,
        query=query,
        top_k=top_k,
        event_date=event_date,
        selected_doc_ids=selected_doc_ids,
        intent_str=intent_str,
        diagnostics=diagnostics if diagnostics is not None else {},
        document_types=document_types,
    )


def _postprocess_task_chunks(
    chunks: list[RetrievedChunk], task: RetrievalTask
) -> list[RetrievedChunk]:
    """Дедуп по ``chunk_id`` с сохранением лучшего дубля (по score канала).

    ``task.boost_doc_ids`` здесь намеренно не применяется: score канала
    дальше полностью перезаписывается RRF-агрегацией, поэтому любое
    приращение на этом этапе было бы декоративным. Мягкий doc-boost
    выполняется один раз в :func:`_finalize_single_task_results`.
    """
    best_by_id: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    for chunk in chunks:
        cid = chunk.chunk_id
        if cid not in best_by_id:
            order.append(cid)
            best_by_id[cid] = chunk
        elif chunk.score > best_by_id[cid].score:
            best_by_id[cid] = chunk

    return [best_by_id[cid] for cid in order]


def _tag_chunk_source_kind(
    chunk: RetrievedChunk, kind: str
) -> RetrievedChunk:
    """Проставляет ``source_kind`` чанку, если поиск его не вернул."""
    if chunk.source_kind:
        return chunk
    derived = source_kind_for_document_type(chunk.document_type)
    value = derived if derived != SOURCE_KIND_UNKNOWN else kind
    if not value:
        return chunk
    return replace(chunk, source_kind=value)


def _run_single_task(
    task: RetrievalTask,
    task_state: dict | None = None,
) -> tuple[list[RetrievedChunk], dict, dict, int]:
    """Выполняет одну задачу; возвращает (chunks, search, embeddings, ms).

    Последовательный режим: ``task_state`` передаётся напрямую, thread-local
    состояние и пул потоков больше не используются.

    Если у задачи несколько ``source_kinds``, поиск выполняется раздельно по
    каждому назначению (с фильтром ``document_types``), результаты
    объединяются и дедуплицируются по ``chunk_id``. Конфликт решается в
    пользу чанка с известным ``source_kind``.
    """
    state = task_state
    started = time.perf_counter()
    reset_embedding_diagnostics()

    mode = (task.mode or "hybrid").strip().lower()
    # H-2A: явный intent задачи имеет приоритет; fallback по mode нужен
    # только для обратной совместимости (например, follow-up задачи H-4).
    intent_str = task.intent or _MODE_TO_HYBRID_INTENT.get(mode, "mixed_query")
    stats: dict = dict(_SEARCH_DIAGNOSTICS_DEFAULTS)
    if state is not None:
        state["stage"] = "search"
        state["search_type"] = mode

    kinds = list(task.source_kinds or [])
    kind_counts: dict[str, int] = {}

    def _search(document_types: set[str] | None) -> list[RetrievedChunk]:
        common = {
            "query": task.query,
            "top_k": task.top_k,
            "event_date": task.event_date,
            "selected_doc_ids": task.selected_doc_ids,
            "intent_str": intent_str,
            "diagnostics": stats,
        }
        if mode == "bm25":
            return _call_lexical_search(**common, document_types=document_types)
        if mode == "dense":
            return _call_dense_search(**common, document_types=document_types)
        # hybrid и неизвестный mode — безопасный дефолт.
        return _call_hybrid_search(**common, document_types=document_types)

    if config.USE_STUB_RAG:
        chunks = stub_rag_retrieve(query=task.query, top_k=task.top_k)
    elif not kinds:
        # Старое поведение: один вызов без фильтра по назначению источника.
        chunks = _search(None)
    elif len(kinds) == 1:
        kind = kinds[0]
        part = _search(document_types_for_source_kind(kind))
        kind_counts[kind] = len(part)
        chunks = [_tag_chunk_source_kind(chunk, kind) for chunk in part]
    else:
        merged: dict[str, RetrievedChunk] = {}
        for kind in kinds:
            part = _search(document_types_for_source_kind(kind))
            kind_counts[kind] = len(part)
            for chunk in part:
                tagged = _tag_chunk_source_kind(chunk, kind)
                existing = merged.get(tagged.chunk_id)
                if existing is None:
                    merged[tagged.chunk_id] = tagged
                    continue
                existing_known = existing.source_kind not in (
                    None, SOURCE_KIND_UNKNOWN
                )
                tagged_known = tagged.source_kind not in (
                    None, SOURCE_KIND_UNKNOWN
                )
                if not existing_known and tagged_known:
                    merged[tagged.chunk_id] = tagged
        chunks = list(merged.values())

    if state is not None:
        state["stage"] = "postprocess"
    chunks = _postprocess_task_chunks(chunks, task)

    if mode == "bm25":
        embeddings = dict(_EMPTY_EMBEDDINGS_DIAGNOSTICS)
    else:
        embeddings = get_last_embedding_diagnostics()

    stats["final_chunks_count"] = len(chunks)
    if stats.get("selected_doc_ids") is None and task.selected_doc_ids:
        stats["selected_doc_ids"] = sorted(task.selected_doc_ids)

    final_counts: dict[str, int] = {}
    for chunk in chunks:
        key = chunk.source_kind or SOURCE_KIND_UNKNOWN
        final_counts[key] = final_counts.get(key, 0) + 1
    stats["source_kinds"] = list(kinds)
    stats["source_kind_counts"] = final_counts
    stats["source_kind_empty"] = [
        kind for kind in kinds if kind_counts.get(kind, 0) == 0
    ]

    duration_ms = int(
        round((time.perf_counter() - started) * config.MS_PER_SECOND)
    )
    if state is not None:
        _apply_diagnostics_to_state(
            state, stats=stats, embeddings=embeddings
        )
        state["stage"] = "done"
    return list(chunks), dict(stats), embeddings, duration_ms


def _queue_wait_ms(state: dict) -> int | None:
    queued = state.get("queued_at_mono")
    started = state.get("started_at_mono")
    finished = state.get("finished_at_mono")
    reference = started if started is not None else finished
    if queued is None or reference is None:
        return None
    return int(
        round((reference - queued) * config.MS_PER_SECOND)
    )


def _task_duration_ms(state: dict) -> int | None:
    started = state.get("started_at_mono")
    finished = state.get("finished_at_mono")
    if started is None:
        return 0
    if finished is None:
        return None
    return int(
        round((finished - started) * config.MS_PER_SECOND)
    )


def _task_diag(
    index: int,
    task: RetrievalTask,
    status: str,
    task_state: dict | None = None,
) -> dict:
    """Диагностика задачи без фиктивных ``0``/``success: true``.

    Для завершившихся успешно задач используются реальные счётчики каналов;
    для timeout/queue_timeout/cancelled/not_started — ``None`` и явный
    статус, чтобы отличить «не измерено» от «ноль кандидатов».
    """
    state = task_state or {}
    final_status = state.get("status") or status
    if final_status == "ok":
        search = dict(_SEARCH_DIAGNOSTICS_DEFAULTS)
        search.update(state.get("search") or {})
        embeddings = dict(state.get("embeddings") or _UNKNOWN_EMBEDDINGS_DIAGNOSTICS)
    else:
        search = dict(_SEARCH_NOT_EXECUTED_DEFAULTS)
        embeddings = dict(_UNKNOWN_EMBEDDINGS_DIAGNOSTICS)

    duration_ms = state.get("duration_ms")
    if duration_ms is None:
        duration_ms = _task_duration_ms(state)
    stage = state.get("stage")
    if stage is None:
        stage = "queued" if final_status in {
            "pending", "not_started", "queue_timeout", "cancelled"
        } else final_status

    return {
        "task_id": state.get("task_id") or f"task_{index}",
        "mode": task.mode,
        "purpose": task.purpose,
        "query": task.query,
        "queued_at": state.get("queued_at"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "queue_wait_ms": _queue_wait_ms(state),
        "duration_ms": duration_ms,
        "stage": stage,
        "status": final_status,
        "timeout_reason": state.get("timeout_reason"),
        "search_type": state.get("search_type") or task.mode,
        "embedding_provider": embeddings.get("provider"),
        "embedding_model": embeddings.get("model"),
        "model_reused": state.get("model_reused"),
        "bm25_candidates_count": search.get("bm25_candidates_count"),
        "dense_candidates_count": search.get("dense_candidates_count"),
        "candidates_before_filters": search.get("candidates_before_filters"),
        "candidates_after_filters": (
            search.get("candidates_after_document_filter")
            if search.get("candidates_after_document_filter") is not None
            else search.get("candidates_after_date_filter")
        ),
        "final_chunks_count": search.get("final_chunks_count"),
        "source_kinds": (
            search.get("source_kinds")
            if search.get("source_kinds") is not None
            else (list(task.source_kinds) if task.source_kinds else [])
        ),
        "source_kind_counts": search.get("source_kind_counts"),
        "source_kind_empty": search.get("source_kind_empty"),
        "error_type": state.get("error_type"),
        "error_message": state.get("error_message"),
        "search": search,
        "embeddings": embeddings,
    }


def _publish_diagnostics(
    diagnostics_out: dict | None, payload: dict
) -> None:
    if diagnostics_out is None:
        return
    diagnostics_out.update(payload)


# Мягкий doc-boost неоднозначных кандидатов (``boost_doc_ids``): это
# множитель к уже посчитанному внутри hybrid_search RRF-скору, а не
# дополнительный RRF-вклад рангов. Применяется один раз на чанк.
DOC_BOOST_FACTOR: float = 1.1


def _collect_boost_doc_ids(tasks) -> set[str]:
    """Объединяет ``boost_doc_ids`` задач, безопасно пропуская ``None``."""
    boost_ids: set[str] = set()
    for task in tasks:
        raw = getattr(task, "boost_doc_ids", None)
        if not raw:
            continue
        if isinstance(raw, str):
            raw = [raw]
        try:
            boost_ids.update(doc_id for doc_id in raw if doc_id)
        except TypeError:
            continue
    return boost_ids


def _finalize_single_task_results(
    task_results: list[tuple[RetrievalTask, list[RetrievedChunk]]],
) -> tuple[dict[str, RetrievedChunk], list[str]]:
    """Финализирует результаты последовательных задач БЕЗ второго RRF.

    Единственный RRF уже выполнен внутри ``hybrid_search``. Здесь только:
    * дедуп по ``chunk_id`` с сохранением лучшего score;
    * однократный мягкий doc-boost ``* DOC_BOOST_FACTOR`` по
      ``boost_doc_ids``;
    * защита первых трёх чанков (``protected_top_ids``).

    Scores не пересчитываются, поэтому порядок и значения из hybrid_search
    сохраняются.
    """
    merged_by_id: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    protected_top_ids: list[str] = []
    boost_ids = _collect_boost_doc_ids(
        task for task, _chunks in task_results
    )

    for _task, chunk_list in task_results:
        for rank_idx, chunk in enumerate(chunk_list):
            cid = chunk.chunk_id
            if rank_idx < 3 and cid not in protected_top_ids:
                protected_top_ids.append(cid)
            if cid not in merged_by_id:
                order.append(cid)
                merged_by_id[cid] = chunk
            elif chunk.score > merged_by_id[cid].score:
                merged_by_id[cid] = chunk

    deduped_map: dict[str, RetrievedChunk] = {}
    for cid in order:
        chunk = merged_by_id[cid]
        score = float(chunk.score)
        document_id = getattr(chunk, "document_id", None)
        if document_id is not None and document_id in boost_ids:
            score *= DOC_BOOST_FACTOR
        deduped_map[cid] = replace(chunk, score=round(score, 6))
    return deduped_map, protected_top_ids


def retrieve_tasks(
    tasks: list[RetrievalTask],
    *,
    diagnostics_out: dict | None = None,
) -> tuple[list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    diagnostics: dict = {
        "executor": {
            "submitted": len(tasks),
            "started": 0,
            "completed": 0,
            "failed": 0,
            "queued": 0,
            "cancelled": 0,
            "timed_out": 0,
            "queue_timeout": 0,
            "threads_alive_after_grace": 0,
            "batch_duration_ms": None,
        },
        "tasks": [],
        "degraded": False,
        "errors": [],
        "prewarm": {},
    }

    if not tasks:
        diagnostics["executor"]["batch_duration_ms"] = 0
        step = TraceStep(
            step=config.STEP_RAG_RETRIEVAL,
            input={"tasks_count": 0},
            output={"chunk_ids": [], "scores": [], "count": 0},
            latency_ms=0.0,
            stage_idx=3,
        )
        _publish_diagnostics(diagnostics_out, diagnostics)
        return [], step

    # Предзагрузка (включая warm-up forward) до последовательного поиска:
    # первый реальный forward уже «горячий».
    try:
        prewarm_report = prewarm_retrieval_backends()
    except Exception as exc:  # noqa: BLE001 - предзагрузка не фатальна
        logger.warning("Prewarm retrieval backends failed: %s", exc)
        diagnostics["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)[:300]}
        )
    else:
        diagnostics["prewarm"] = {
            "local_model": prewarm_report.get("local_model"),
            "collection": prewarm_report.get("collection"),
            "warmup": prewarm_report.get("warmup"),
            "warmup_ms": prewarm_report.get("warmup_ms"),
        }
        for error in prewarm_report.get("errors", []):
            diagnostics["errors"].append(
                {
                    "type": str(error.get("type", "PrewarmError")),
                    "message": str(error.get("message", ""))[:300],
                }
            )

    timeout_s = float(
        getattr(
            config,
            "RETRIEVAL_TASK_TIMEOUT_S",
            config.RETRIEVAL_TIMEOUT_S,
        )
    )
    task_states: list[dict] = []
    task_results: list[tuple[RetrievalTask, list[RetrievedChunk]]] = []

    def _new_state(index: int, task: RetrievalTask) -> dict:
        queued_mono = time.monotonic()
        return {
            "index": index,
            "task_id": f"task_{index}",
            "task": task,
            "mode": task.mode,
            "purpose": task.purpose,
            "queued_at": _now_iso(),
            "queued_at_mono": queued_mono,
            "started_at": None,
            "started_at_mono": None,
            "finished_at": None,
            "finished_at_mono": None,
            "stage": "queued",
            "status": "pending",
            "timeout_reason": None,
            "search_type": None,
            "embedding_provider": None,
            "embedding_model": None,
            "model_reused": None,
            "bm25_candidates_count": None,
            "dense_candidates_count": None,
            "candidates_before_filters": None,
            "candidates_after_filters": None,
            "final_chunks_count": None,
            "source_kinds": list(task.source_kinds or []),
            "source_kind_counts": None,
            "source_kind_empty": None,
            "search": None,
            "embeddings": None,
            "error_type": None,
            "error_message": None,
            "duration_ms": None,
        }

    def _finalize(
        state: dict,
        status: str,
        *,
        timeout_reason: str | None = None,
        finished_mono: float | None = None,
        stage: str | None = None,
    ) -> None:
        finished = (
            finished_mono if finished_mono is not None else time.monotonic()
        )
        state["status"] = status
        state["timeout_reason"] = timeout_reason
        state["finished_at_mono"] = finished
        state["finished_at"] = _now_iso()
        started = state.get("started_at_mono")
        if started is not None:
            state["duration_ms"] = int(
                round((finished - started) * config.MS_PER_SECOND)
            )
        else:
            state["duration_ms"] = 0
        if stage is not None:
            state["stage"] = stage
        else:
            default_stage = {
                "ok": "done",
                "error": "error",
                "timeout": "timeout",
                "queue_timeout": "queued",
                "cancelled": state.get("stage") or "queued",
                "not_started": "queued",
            }.get(status)
            if default_stage is not None:
                state["stage"] = default_stage

    # Последовательное исполнение: один hybrid-поиск, без пула потоков.
    # Тайм-аут проверяется post-hoc по фактической длительности задачи;
    # уже выполняющийся код не отменяется (отменять нечего).
    timeout_ms = timeout_s * config.MS_PER_SECOND
    for index, task in enumerate(tasks):
        state = _new_state(index, task)
        task_states.append(state)
        state["started_at_mono"] = time.monotonic()
        state["started_at"] = _now_iso()
        state["stage"] = "started"
        diagnostics["executor"]["started"] += 1
        try:
            chunks, stats, embeddings, duration_ms = _run_single_task(
                task, state
            )
        except Exception as exc:  # noqa: BLE001 - изолируем сбой задачи
            _finalize(state, "error", stage="error")
            state["error_type"] = type(exc).__name__
            state["error_message"] = str(exc)[:300]
            diagnostics["executor"]["failed"] += 1
            diagnostics["errors"].append(
                {
                    "type": state["error_type"],
                    "message": state["error_message"],
                }
            )
            logger.warning(
                "Retrieval task %s (%s) failed: %s: %s",
                state["index"],
                state["purpose"],
                state["error_type"],
                state["error_message"],
            )
            continue

        if duration_ms > timeout_ms:
            _finalize(state, "timeout", timeout_reason="task_timeout")
            state["error_type"] = "TimeoutError"
            state["error_message"] = (
                "Задача не завершилась за "
                f"{timeout_s} с (post-hoc тайм-аут)"
            )
            diagnostics["executor"]["timed_out"] += 1
            diagnostics["errors"].append(
                {
                    "type": state["error_type"],
                    "message": state["error_message"],
                }
            )
            logger.warning(
                "Retrieval task %s (%s): task_timeout после %.2f с",
                state["index"],
                state["purpose"],
                timeout_s,
            )
            # Результат просроченной задачи отбрасывается.
            continue

        _finalize(state, "ok", stage="done")
        state["duration_ms"] = duration_ms
        _apply_diagnostics_to_state(
            state, stats=stats, embeddings=embeddings
        )
        diagnostics["executor"]["completed"] += 1
        task_results.append((task, chunks))

    diagnostics["executor"]["queued"] = 0
    diagnostics["executor"]["cancelled"] = 0
    diagnostics["executor"]["queue_timeout"] = 0
    diagnostics["executor"]["threads_alive_after_grace"] = 0
    diagnostics["tasks"] = [
        _task_diag(
            state["index"],
            state["task"],
            state.get("status", "pending"),
            task_state=state,
        )
        for state in sorted(task_states, key=lambda item: item["index"])
    ]

    completed = diagnostics["executor"]["completed"]
    timed_out = diagnostics["executor"]["timed_out"]
    diagnostics["degraded"] = completed < len(tasks)
    diagnostics["executor"]["batch_duration_ms"] = int(
        round((time.perf_counter() - start_time) * config.MS_PER_SECOND)
    )

    if completed == 0 and tasks:
        _publish_diagnostics(diagnostics_out, diagnostics)
        error_types = sorted(
            {error["type"] for error in diagnostics["errors"]}
        ) or ["UnknownError"]
        summary = ", ".join(error_types)
        statuses = {state.get("status") for state in task_states}
        only_timeouts = statuses <= {
            "timeout", "queue_timeout", "cancelled", "not_started"
        } and bool(statuses & {"timeout", "queue_timeout"})
        if only_timeouts or timed_out >= len(tasks):
            raise RetrievalTimeoutError(
                "Все retrieval-задачи превысили дедлайн "
                f"(task={timeout_s} с; причины: {summary})"
            )
        raise RetrievalAPIError(
            f"Все retrieval-задачи завершились ошибкой ({summary})"
        )

    # Второго RRF нет: scores из hybrid_search только дедуплицируются и
    # получают однократный мягкий doc-boost.
    deduped_map, protected_top_ids = _finalize_single_task_results(task_results)
    boost_ids = _collect_boost_doc_ids(
        task for task, _chunks in task_results
    )
    doc_boost_applied = sum(
        1
        for chunk in deduped_map.values()
        if getattr(chunk, "document_id", None) in boost_ids
    )

    sorted_all = sorted(deduped_map.values(), key=lambda c: c.score, reverse=True)
    max_keep = max(int(config.RAG_TOP_K), int(config.RERANKER_TOP_N), 10)

    final_chunks: list[RetrievedChunk] = []
    seen_final: set[str] = set()
    for ch in sorted_all:
        if len(final_chunks) >= max_keep:
            break
        final_chunks.append(ch)
        seen_final.add(ch.chunk_id)

    for pid in protected_top_ids:
        if pid not in seen_final and pid in deduped_map:
            final_chunks.append(deduped_map[pid])
            seen_final.add(pid)

    final_chunks.sort(key=lambda c: c.score, reverse=True)

    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_RAG_RETRIEVAL,
        input={
            "tasks_count": len(tasks),
            "queries": [t.query for t in tasks],
            "modes": [t.mode for t in tasks],
            "intents": [t.intent for t in tasks],
        },
        output={
            "chunk_ids": [c.chunk_id for c in final_chunks],
            "scores": [c.score for c in final_chunks],
            "count": len(final_chunks),
            "degraded": diagnostics["degraded"],
        },
        latency_ms=latency_ms,
        stage_idx=3,
        extra={
            "executor": dict(diagnostics["executor"]),
            "errors_count": len(diagnostics["errors"]),
            "degraded": diagnostics["degraded"],
            "execution": "sequential",
            "aggregation": "none",
            "rrf_levels": 1,
            "hybrid_rrf_k": 60,
            "doc_boost_factor": DOC_BOOST_FACTOR,
            "boost_doc_ids_count": len(boost_ids),
            "doc_boost_applied": doc_boost_applied,
            # Warm-up/бэкенды: без секретов и путей.
            "prewarm": dict(diagnostics.get("prewarm") or {}),
        },
    )
    _publish_diagnostics(diagnostics_out, diagnostics)
    return final_chunks, step


def retrieve_chunks(
    query: str,
    top_k: int,
    event_date: str | None = None,
    intent: Intent | None = None,
    selected_doc_ids: set[str] | None = None,
) -> tuple[list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    mode = "stub" if config.USE_STUB_RAG else "real"
    intent_str = (
        intent.value
        if isinstance(intent, Intent)
        else (str(intent) if intent is not None else None)
    )
    chunks = _call_hybrid_search(
        query=query,
        top_k=top_k,
        event_date=event_date,
        selected_doc_ids=selected_doc_ids,
        intent_str=intent_str,
    )
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_RAG_RETRIEVAL,
        input={
            "query": query,
            "top_k": top_k,
            "event_date": event_date,
            "intent": intent_str,
            "selected_doc_ids": (
                sorted(selected_doc_ids)
                if selected_doc_ids is not None
                else None
            ),
            "mode": mode,
        },
        output={
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "scores": [chunk.score for chunk in chunks],
            "count": len(chunks),
        },
        latency_ms=latency_ms,
        stage_idx=3,
    )
    logger.info(
        "Step %s retrieved %d chunks in %.3f ms",
        config.STEP_RAG_RETRIEVAL,
        len(chunks),
        latency_ms,
    )
    logger.debug("Retrieved chunks: %s", chunks)
    return chunks, step


def _chunk_to_rerank_candidate(chunk: RetrievedChunk) -> dict:
    """Готовит кандидата для локального реранкера, сохраняя все поля чанка."""
    structure_path = list(chunk.structure_path)
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "embedding_text": (
            f"{chunk.document_title}. "
            f"{' > '.join(structure_path)}. {chunk.text}"
        ),
        "document_id": chunk.document_id,
        "version_id": chunk.version_id,
        "document_title": chunk.document_title,
        "structure": {
            "provision_id": chunk.provision_id,
            "path": structure_path,
        },
        "score": chunk.score,
        "effective_from": chunk.effective_from,
        "effective_to": chunk.effective_to,
        "effective_dates_known": chunk.effective_dates_known,
        "amends_document_id": chunk.amends_document_id,
        "document_type": chunk.document_type,
        "source_kind": chunk.source_kind,
        "version_status": chunk.version_status,
        "validity_uncertain": chunk.validity_uncertain,
        "effective_date_mode": chunk.effective_date_mode,
    }


def rerank_chunks(
    chunks: list[RetrievedChunk], query: str, top_n: int
) -> tuple[list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    if config.USE_STUB_RERANKER:
        mode = "stub"
        reranked = stub_reranker(chunks=chunks, query=query, top_n=top_n)
    elif len(chunks) <= 1:
        mode = "passthrough"
        reranked = list(chunks[:top_n])
        logger.info(
            "Reranker skipped (passthrough): недостаточно кандидатов (%d)",
            len(chunks),
        )
    else:
        candidates = [_chunk_to_rerank_candidate(chunk) for chunk in chunks]
        try:
            reranked_dicts = rerank_candidates(
                query=query,
                candidates=candidates,
                top_k=top_n,
            )
        except Exception as exc:  # noqa: BLE001 - реранкер необязателен
            logger.warning(
                "Реранкер (%s) недоступен, применяю passthrough: %s",
                getattr(config, "RERANKER_BACKEND", "unknown"),
                exc,
                exc_info=True,
            )
            mode = "passthrough"
            reranked = list(chunks[:top_n])
        else:
            if reranked_dicts:
                mode = "real"
                reranked = [_dict_to_chunk(raw) for raw in reranked_dicts]
            else:
                logger.warning(
                    "Реранкер вернул пустой результат, применяю passthrough"
                )
                mode = "passthrough"
                reranked = list(chunks[:top_n])
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_RERANKING,
        input={
            "query": query,
            "top_n": top_n,
            "input_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "mode": mode,
        },
        output={
            "chunk_ids": [chunk.chunk_id for chunk in reranked],
            "scores": [chunk.score for chunk in reranked],
            "count": len(reranked),
        },
        latency_ms=latency_ms,
        stage_idx=3,
    )
    logger.info(
        "Step %s kept %d chunks in %.3f ms",
        config.STEP_RERANKING,
        len(reranked),
        latency_ms,
    )
    return reranked, step


def filter_chunks_by_threshold(
    chunks: list[RetrievedChunk], threshold: float
) -> tuple[bool, list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    max_score = max(
        (c.score for c in chunks), default=config.DEFAULT_MIN_SCORE
    )
    effective_threshold = threshold
    ceiling = getattr(config, "RRF_FALLBACK_SCORE_CEILING", 0.1)
    fallback_thresh = getattr(config, "RRF_FALLBACK_THRESHOLD", 0.005)
    # Адаптивное снижение порога только для явно заданного ненулевого порога.
    # threshold=0.0 означает «принять всё» и не должен ужесточаться.
    if threshold > 0.0 and 0.0 < max_score < ceiling:
        effective_threshold = min(threshold, fallback_thresh)

    passed = bool(chunks) and max_score >= effective_threshold
    # Порог работает как gate: при прохождении сохраняются все кандидаты,
    # ранжирование/усечение выполняет реранкер и RAG_TOP_K.
    filtered = list(chunks) if passed else []
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_THRESHOLD_FILTER,
        input={
            "threshold": threshold,
            "effective_threshold": effective_threshold,
            "input_count": len(chunks),
            "max_score": max_score,
        },
        output={
            "passed": passed,
            "chunk_ids": [chunk.chunk_id for chunk in filtered],
            "count": len(filtered),
        },
        latency_ms=latency_ms,
        stage_idx=3,
    )
    logger.info(
        "Step %s evaluated passed=%s (max_score=%.3f, threshold=%.3f) in %.3f ms",
        config.STEP_THRESHOLD_FILTER,
        passed,
        max_score,
        effective_threshold,
        latency_ms,
    )
    return passed, filtered, step
