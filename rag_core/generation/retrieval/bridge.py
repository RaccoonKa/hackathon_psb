import concurrent.futures
from dataclasses import replace
<<<<<<< HEAD
import inspect
import logging
import time

from rag_core import config
from rag_core.generation.errors import RetrievalAPIError, RetrievalTimeoutError
=======
import datetime
import inspect
import logging
import threading
import time

from rag_core import config
from rag_core.generation.errors import (
    RetrievalAPIError,
    RetrievalBusyError,
    RetrievalTimeoutError,
)
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
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
}

<<<<<<< HEAD
=======
# Для задач, которые вовсе не используют эмбеддинги (bm25): provider none
# означает «канал не задействован», поэтому success не может быть true без
# измерения.
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
_EMPTY_EMBEDDINGS_DIAGNOSTICS: dict = {
    "provider": "none",
    "model": None,
    "dimension": None,
<<<<<<< HEAD
    "success": True,
    "fallback_used": False,
    "error_type": None,
}

_SIGNATURE_CACHE: dict = {}


=======
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

# Период опроса pending-futures, чтобы замечать старт задач и per-task
# дедлайны без busy-wait. Батч ограничен per-task дедлайнами и завершается
# сам; глобального бесконечного ожидания нет.
_DEADLINE_POLL_S: float = 0.05

# Учёт последнего пула: новый пул не создаётся, пока живы потоки предыдущего.
_EXECUTOR_LOCK = threading.Lock()
_LAST_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None

# Пер-поточное состояние текущей задачи (нужно _run_single_task для стадий),
# без изменения его сигнатуры для обратной совместимости.
_TASK_STATE_LOCAL = threading.local()

_SIGNATURE_CACHE: dict = {}


def _now_iso() -> str:
    """UTC-время в ISO 8601 для диагностики (без секретов)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="milliseconds"
    )


def _current_task_state() -> dict | None:
    return getattr(_TASK_STATE_LOCAL, "state", None)


def _mark_execution_start(state: dict) -> None:
    state["started_at_mono"] = time.monotonic()
    state["started_at"] = _now_iso()
    state["stage"] = "started"


def _mark_execution_finish(state: dict) -> None:
    state["finished_at_mono"] = time.monotonic()
    state["finished_at"] = _now_iso()


def _execute_single_task(
    task: RetrievalTask, task_state: dict
) -> tuple[list[RetrievedChunk], dict, dict, int]:
    """Обёртка исполнителя: фиксирует старт/финиш и thread-local состояние.

    Обратная совместимость: ``_run_single_task`` получает только ``task``,
    поэтому monkeypatch-функции с прежней сигнатурой продолжают работать.
    """
    _mark_execution_start(task_state)
    _TASK_STATE_LOCAL.state = task_state
    try:
        return _run_single_task(task)
    finally:
        _mark_execution_finish(task_state)
        _TASK_STATE_LOCAL.state = None


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


>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
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

    raw_chunks = search_func(**kwargs)
    return [_dict_to_chunk(raw) for raw in raw_chunks]


def _call_hybrid_search(
    query: str,
    top_k: int,
    event_date: str | None,
    selected_doc_ids: set[str] | None,
    intent_str: str | None,
    diagnostics: dict | None = None,
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
    )


def _call_lexical_search(
    query: str,
    top_k: int,
    event_date: str | None,
    selected_doc_ids: set[str] | None,
    intent_str: str | None,
    diagnostics: dict | None = None,
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
    )


def _call_dense_search(
    query: str,
    top_k: int,
    event_date: str | None,
    selected_doc_ids: set[str] | None,
    intent_str: str | None,
    diagnostics: dict | None = None,
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
    )


def _postprocess_task_chunks(
    chunks: list[RetrievedChunk], task: RetrievalTask
) -> list[RetrievedChunk]:
<<<<<<< HEAD
    """Единый постпроцессинг задач: дедуп по chunk_id + буст документов."""
=======
    """Дедуп по ``chunk_id`` с сохранением лучшего дубля (по score канала).

    ``task.boost_doc_ids`` здесь намеренно не применяется: score канала
    дальше полностью перезаписывается RRF-агрегацией, поэтому любое
    приращение на этом этапе было бы декоративным. Мягкий doc-boost
    выполняется один раз в :func:`_aggregate_task_results`.
    """
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
    best_by_id: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    for chunk in chunks:
        cid = chunk.chunk_id
        if cid not in best_by_id:
            order.append(cid)
            best_by_id[cid] = chunk
        elif chunk.score > best_by_id[cid].score:
            best_by_id[cid] = chunk

<<<<<<< HEAD
    result: list[RetrievedChunk] = []
    for cid in order:
        chunk = best_by_id[cid]
        if task.boost_doc_ids and chunk.document_id in task.boost_doc_ids:
            delta = 0.002 if chunk.score < 0.1 else 0.05
            chunk = replace(
                chunk, score=min(1.0, round(chunk.score + delta, 6))
            )
        result.append(chunk)
    return result
=======
    return [best_by_id[cid] for cid in order]
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)


def _run_single_task(
    task: RetrievalTask,
<<<<<<< HEAD
) -> tuple[list[RetrievedChunk], dict, dict, int]:
    """Выполняет одну задачу; возвращает (chunks, search, embeddings, ms)."""
=======
    task_state: dict | None = None,
) -> tuple[list[RetrievedChunk], dict, dict, int]:
    """Выполняет одну задачу; возвращает (chunks, search, embeddings, ms).

    ``task_state`` — необязательное пер-задачное состояние для диагностики
    стадий. Если не передан, используется thread-local состояние, которое
    выставляет обёртка пула; прямой вызов с одним аргументом (тесты,
    обратная совместимость) продолжает работать.
    """
    state = task_state if task_state is not None else _current_task_state()
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
    started = time.perf_counter()
    reset_embedding_diagnostics()

    mode = (task.mode or "hybrid").strip().lower()
<<<<<<< HEAD
    intent_str = _MODE_TO_HYBRID_INTENT.get(mode, "mixed_query")
    stats: dict = dict(_SEARCH_DIAGNOSTICS_DEFAULTS)
=======
    # H-2A: явный intent задачи имеет приоритет; fallback по mode нужен
    # только для обратной совместимости (например, follow-up задачи H-4).
    intent_str = task.intent or _MODE_TO_HYBRID_INTENT.get(mode, "mixed_query")
    stats: dict = dict(_SEARCH_DIAGNOSTICS_DEFAULTS)
    if state is not None:
        state["stage"] = "search"
        state["search_type"] = mode
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)

    if config.USE_STUB_RAG:
        chunks = stub_rag_retrieve(query=task.query, top_k=task.top_k)
    elif mode == "bm25":
        chunks = _call_lexical_search(
            query=task.query,
            top_k=task.top_k,
            event_date=task.event_date,
            selected_doc_ids=task.selected_doc_ids,
            intent_str=intent_str,
            diagnostics=stats,
        )
    elif mode == "dense":
        chunks = _call_dense_search(
            query=task.query,
            top_k=task.top_k,
            event_date=task.event_date,
            selected_doc_ids=task.selected_doc_ids,
            intent_str=intent_str,
            diagnostics=stats,
        )
    else:
        # hybrid и неизвестный mode — безопасный дефолт.
        chunks = _call_hybrid_search(
            query=task.query,
            top_k=task.top_k,
            event_date=task.event_date,
            selected_doc_ids=task.selected_doc_ids,
            intent_str=intent_str,
            diagnostics=stats,
        )

<<<<<<< HEAD
=======
    if state is not None:
        state["stage"] = "postprocess"
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
    chunks = _postprocess_task_chunks(chunks, task)

    if mode == "bm25":
        embeddings = dict(_EMPTY_EMBEDDINGS_DIAGNOSTICS)
    else:
        embeddings = get_last_embedding_diagnostics()

    stats["final_chunks_count"] = len(chunks)
    if stats.get("selected_doc_ids") is None and task.selected_doc_ids:
        stats["selected_doc_ids"] = sorted(task.selected_doc_ids)

    duration_ms = int(
        round((time.perf_counter() - started) * config.MS_PER_SECOND)
    )
<<<<<<< HEAD
    return list(chunks), dict(stats), embeddings, duration_ms


def _task_diag(
    index: int, task: RetrievalTask, status: str
) -> dict:
    return {
        "task_id": f"task_{index}",
        "mode": task.mode,
        "purpose": task.purpose,
        "query": task.query,
        "status": status,
        "duration_ms": 0,
        "error_type": None,
        "error_message": None,
        "search": dict(_SEARCH_DIAGNOSTICS_DEFAULTS),
        "embeddings": dict(_EMPTY_EMBEDDINGS_DIAGNOSTICS),
=======
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
        "error_type": state.get("error_type"),
        "error_message": state.get("error_message"),
        "search": search,
        "embeddings": embeddings,
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
    }


def _publish_diagnostics(
    diagnostics_out: dict | None, payload: dict
) -> None:
    if diagnostics_out is None:
        return
    diagnostics_out.update(payload)


<<<<<<< HEAD
def _join_executor_threads(executor, grace_s: float) -> None:
    """Ограниченное ожидание завершения потоков после единственного shutdown."""
=======
def _join_executor_threads(executor, grace_s: float) -> int:
    """Ограниченное ожидание завершения потоков; возвращает число живых.

    ``shutdown(wait=False, cancel_futures=True)`` не останавливает уже
    выполняющийся код, поэтому здесь только ожидаем потоки в пределах
    grace и честно возвращаем число не завершившихся (зомби) потоков.
    """
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
    threads = list(getattr(executor, "_threads", ()) or ())
    deadline = time.monotonic() + max(0.0, float(grace_s))
    for thread in threads:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)
    alive = [t for t in threads if t.is_alive()]
    if alive:
        logger.warning(
            "Retrieval executor: %d поток(ов) не завершились за %.1f с",
            len(alive),
            grace_s,
        )
<<<<<<< HEAD
=======
    return len(alive)


def _acquire_executor(
    max_workers: int, grace_s: float
) -> concurrent.futures.ThreadPoolExecutor:
    """Создаёт пул, только если потоки предыдущего батча завершились.

    Ожидание ограничено ``grace_s``; если потоки живы — контролируемая
    техническая ошибка :class:`RetrievalBusyError`, а не бесконечный lock.
    """
    global _LAST_EXECUTOR
    with _EXECUTOR_LOCK:
        previous = _LAST_EXECUTOR
        if previous is not None:
            alive = _join_executor_threads(previous, grace_s)
            if alive:
                error = RetrievalBusyError(
                    "Предыдущий батч retrieval всё ещё удерживает "
                    f"{alive} поток(ов) после ожидания {grace_s} с; "
                    "новый батч не запускается, чтобы не накапливать "
                    "зомби-потоки"
                )
                error.threads_alive = alive
                raise error
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="retrieval"
        )
        _LAST_EXECUTOR = executor
        return executor


# H-2B: константа RRF-агрегации рангов задач (стандартная формула RRF).
RRF_AGGREGATION_K: int = 60
# Мягкий doc-boost неоднозначных кандидатов (``boost_doc_ids``): это
# множитель к уже посчитанному RRF-скору, а не дополнительный RRF-вклад
# рангов. Чтобы порядок и score реально менялись, boost применяется после
# агрегации, один раз на чанк.
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


def _aggregate_task_results(
    task_results: list[tuple[RetrievalTask, list[RetrievedChunk]]],
) -> tuple[dict[str, RetrievedChunk], list[str]]:
    """Суммирующая RRF-агрегация результатов задач по ``chunk_id``.

    ``RRF(chunk) = Σ weight_i / (RRF_AGGREGATION_K + rank_i + 1)``,
    где ``rank_i`` — позиция чанка в списке конкретной задачи (с 0), а
    ``weight_i`` — вес задачи (``getattr(task, "weight", 1.0)``).

    Raw score каналов не смешиваются: ``hybrid_search`` внутри задачи уже
    нормализует dense/bm25 через собственный RRF, поэтому на уровне моста
    складываются только вклады рангов задач.

    Мягкий doc-boost (``boost_doc_ids`` всех успешных задач) применяется к
    итоговому RRF-скору один раз на чанк:
    ``score = round(RRF * DOC_BOOST_FACTOR, 6)``, если ``document_id`` чанка
    входит в объединение boost-документов. Это не RRF-вклад и не меняет
    суммирующую формулу.
    """
    merged_by_id: dict[str, RetrievedChunk] = {}
    rrf_scores: dict[str, float] = {}
    protected_top_ids: list[str] = []
    boost_ids = _collect_boost_doc_ids(
        task for task, _chunks in task_results
    )

    for task, chunk_list in task_results:
        try:
            weight = float(getattr(task, "weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        seen_in_task: set[str] = set()
        for rank_idx, chunk in enumerate(chunk_list):
            cid = chunk.chunk_id
            if cid in seen_in_task:
                # Внутри одного списка chunk_id уникален: повтор не даёт
                # дополнительного вклада и не создаёт дубликат.
                continue
            seen_in_task.add(cid)
            if rank_idx < 3 and cid not in protected_top_ids:
                protected_top_ids.append(cid)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
                weight / (RRF_AGGREGATION_K + rank_idx + 1)
            )
            if cid not in merged_by_id:
                merged_by_id[cid] = chunk

    deduped_map: dict[str, RetrievedChunk] = {}
    for cid, chunk in merged_by_id.items():
        score = rrf_scores[cid]
        document_id = getattr(chunk, "document_id", None)
        if document_id is not None and document_id in boost_ids:
            score *= DOC_BOOST_FACTOR
        deduped_map[cid] = replace(chunk, score=round(score, 6))
    return deduped_map, protected_top_ids
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)


def retrieve_tasks(
    tasks: list[RetrievalTask],
    *,
    diagnostics_out: dict | None = None,
) -> tuple[list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    diagnostics: dict = {
        "executor": {
            "submitted": len(tasks),
<<<<<<< HEAD
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "timed_out": 0,
=======
            "started": 0,
            "completed": 0,
            "failed": 0,
            "queued": 0,
            "cancelled": 0,
            "timed_out": 0,
            "queue_timeout": 0,
            "threads_alive_after_grace": None,
            "batch_duration_ms": None,
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
        },
        "tasks": [],
        "degraded": False,
        "errors": [],
<<<<<<< HEAD
    }

    if not tasks:
=======
        "prewarm": {},
    }

    if not tasks:
        diagnostics["executor"]["batch_duration_ms"] = 0
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
        step = TraceStep(
            step=config.STEP_RAG_RETRIEVAL,
            input={"tasks_count": 0},
            output={"chunk_ids": [], "scores": [], "count": 0},
            latency_ms=0.0,
            stage_idx=3,
        )
        _publish_diagnostics(diagnostics_out, diagnostics)
        return [], step

<<<<<<< HEAD
    # Предзагрузка до создания пула: гонок инициализации нет в принципе.
=======
    # Предзагрузка (включая warm-up forward) до создания пула: гонок
    # инициализации нет, а первый реальный forward уже «горячий».
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
    try:
        prewarm_report = prewarm_retrieval_backends()
    except Exception as exc:  # noqa: BLE001 - предзагрузка не фатальна
        logger.warning("Prewarm retrieval backends failed: %s", exc)
        diagnostics["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)[:300]}
        )
    else:
<<<<<<< HEAD
=======
        diagnostics["prewarm"] = {
            "local_model": prewarm_report.get("local_model"),
            "collection": prewarm_report.get("collection"),
            "warmup": prewarm_report.get("warmup"),
            "warmup_ms": prewarm_report.get("warmup_ms"),
        }
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
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
<<<<<<< HEAD
=======
    # Queue-бюджет — отдельная централизованная настройка; не маскирует
    # проблему увеличением общего тайм-аута.
    queue_timeout_s = float(
        getattr(config, "RETRIEVAL_QUEUE_TIMEOUT_S", timeout_s)
    )
    grace_s = float(
        getattr(config, "RETRIEVAL_SHUTDOWN_GRACE_S", 2.0)
    )
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
    configured_workers = int(
        getattr(config, "RETRIEVAL_MAX_WORKERS", 0)
        or getattr(config, "MAX_WORKERS", 1)
        or 1
    )
    max_workers = max(1, min(configured_workers, len(tasks)))

<<<<<<< HEAD
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="retrieval"
    )
    futures: dict[concurrent.futures.Future, tuple[int, RetrievalTask]] = {}
    submitted_at: dict[concurrent.futures.Future, float] = {}
    task_results: list[list[RetrievedChunk]] = []
    deadline = time.monotonic() + timeout_s

    try:
        for index, task in enumerate(tasks):
            future = executor.submit(_run_single_task, task)
            futures[future] = (index, task)
            submitted_at[future] = time.perf_counter()

        for future, (index, task) in futures.items():
            entry = _task_diag(index, task, "pending")
            elapsed_ms = int(
                round(
                    (time.perf_counter() - submitted_at[future])
                    * config.MS_PER_SECOND
                )
            )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                cancelled = future.cancel()
                entry["status"] = "cancelled" if cancelled else "timeout"
                entry["duration_ms"] = elapsed_ms
                entry["error_type"] = "BatchDeadlineExceeded"
                entry["error_message"] = (
                    f"Дедлайн батча {timeout_s} с исчерпан до запуска задачи"
                )
                if cancelled:
                    diagnostics["executor"]["cancelled"] += 1
                else:
                    diagnostics["executor"]["timed_out"] += 1
                diagnostics["errors"].append(
                    {
                        "type": entry["error_type"],
                        "message": entry["error_message"],
                    }
                )
                diagnostics["tasks"].append(entry)
                continue

            try:
                chunks, stats, embeddings, duration_ms = future.result(
                    timeout=remaining
                )
            except concurrent.futures.TimeoutError:
                cancelled = future.cancel()
                entry["status"] = "cancelled" if cancelled else "timeout"
                entry["duration_ms"] = int(
                    round(
                        (time.perf_counter() - submitted_at[future])
                        * config.MS_PER_SECOND
                    )
                )
                entry["error_type"] = "TimeoutError"
                entry["error_message"] = (
                    f"Задача не завершилась за {timeout_s} с"
                )
                if cancelled:
                    diagnostics["executor"]["cancelled"] += 1
                else:
                    diagnostics["executor"]["timed_out"] += 1
                diagnostics["errors"].append(
                    {
                        "type": entry["error_type"],
                        "message": entry["error_message"],
                    }
                )
            except concurrent.futures.CancelledError:
                entry["status"] = "cancelled"
                entry["duration_ms"] = elapsed_ms
                entry["error_type"] = "CancelledError"
                entry["error_message"] = "Задача отменена"
                diagnostics["executor"]["cancelled"] += 1
            except Exception as exc:  # noqa: BLE001 - изолируем задачи
                entry["status"] = "error"
                entry["duration_ms"] = int(
                    round(
                        (time.perf_counter() - submitted_at[future])
                        * config.MS_PER_SECOND
                    )
                )
                entry["error_type"] = type(exc).__name__
                entry["error_message"] = str(exc)[:300]
                diagnostics["executor"]["failed"] += 1
                diagnostics["errors"].append(
                    {
                        "type": entry["error_type"],
                        "message": entry["error_message"],
                    }
                )
                logger.warning(
                    "Retrieval task %s (%s) failed: %s: %s",
                    index,
                    task.purpose,
                    entry["error_type"],
                    entry["error_message"],
                )
            else:
                entry["status"] = "ok"
                entry["duration_ms"] = duration_ms
                entry["search"] = stats
                entry["embeddings"] = embeddings
                diagnostics["executor"]["completed"] += 1
                task_results.append(chunks)

            diagnostics["tasks"].append(entry)
    finally:
        for future in futures:
            future.cancel()
        # Ровно один shutdown; потоки дожидаются ограниченно.
        executor.shutdown(wait=False, cancel_futures=True)
        _join_executor_threads(
            executor,
            getattr(config, "RETRIEVAL_SHUTDOWN_GRACE_S", 2.0),
        )

    submitted = len(tasks)
    completed = diagnostics["executor"]["completed"]
    timed_out = diagnostics["executor"]["timed_out"]
    diagnostics["degraded"] = completed < submitted

    if completed == 0 and submitted > 0:
=======
    # Новый пул создаётся только после завершения потоков предыдущего батча.
    try:
        executor = _acquire_executor(max_workers, grace_s)
    except RetrievalBusyError as exc:
        alive = getattr(exc, "threads_alive", None)
        diagnostics["executor"]["threads_alive_after_grace"] = alive
        diagnostics["executor"]["batch_duration_ms"] = int(
            round((time.perf_counter() - start_time) * config.MS_PER_SECOND)
        )
        diagnostics["degraded"] = True
        diagnostics["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)[:300]}
        )
        _publish_diagnostics(diagnostics_out, diagnostics)
        raise

    futures: dict[concurrent.futures.Future, dict] = {}
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

    try:
        for index, task in enumerate(tasks):
            state = _new_state(index, task)
            task_states.append(state)
            try:
                future = executor.submit(_execute_single_task, task, state)
            except Exception as exc:  # noqa: BLE001 - пул мог быть закрыт
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
                continue
            futures[future] = state

        pending = set(futures)
        while pending:
            done, _ = concurrent.futures.wait(
                pending,
                timeout=_DEADLINE_POLL_S,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            # 1) Сначала забираем завершённые задачи. Проверка done() идёт
            #    раньше дедлайнов, поэтому готовый результат никогда не
            #    помечается timeout и не выбрасывается.
            for future in done:
                if future not in pending:
                    continue
                state = futures[future]
                if future.cancelled():
                    pending.discard(future)
                    _finalize(state, "cancelled")
                    diagnostics["executor"]["cancelled"] += 1
                    continue
                if not future.done():
                    continue
                pending.discard(future)
                exc = future.exception()
                if exc is not None:
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
                chunks, stats, embeddings, duration_ms = future.result()
                _finalize(state, "ok", stage="done")
                state["duration_ms"] = duration_ms
                _apply_diagnostics_to_state(
                    state, stats=stats, embeddings=embeddings
                )
                diagnostics["executor"]["completed"] += 1
                task_results.append((state["task"], chunks))

            if not pending:
                break

            now = time.monotonic()
            # 2) Только теперь проверяем per-task дедлайны. Завершённые
            #    future не трогаем — их заберём на следующей итерации.
            for future in list(pending):
                if future.done() or future.cancelled():
                    continue
                state = futures[future]
                started_mono = state.get("started_at_mono")
                if started_mono is not None:
                    if now - started_mono < timeout_s:
                        continue
                    # cancel() не прерывает уже выполняющийся код — поток
                    # останется зомби и будет учтён в grace-ожидании.
                    future.cancel()
                    pending.discard(future)
                    _finalize(
                        state, "timeout", timeout_reason="task_timeout"
                    )
                    state["error_type"] = "TimeoutError"
                    state["error_message"] = (
                        f"Задача не завершилась за {timeout_s} с от старта"
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
                    continue

                queued_mono = state.get("queued_at_mono")
                if (
                    queued_mono is None
                    or now - queued_mono < queue_timeout_s
                ):
                    continue
                cancelled = future.cancel()
                if not cancelled:
                    # Задача успела стартовать между проверкой и cancel —
                    # её судьбу решит per-task дедлайн от started_at.
                    continue
                pending.discard(future)
                _finalize(
                    state, "queue_timeout", timeout_reason="queue_timeout"
                )
                state["error_type"] = "QueueTimeoutError"
                state["error_message"] = (
                    "Задача не началась за queue-бюджет "
                    f"{queue_timeout_s} с"
                )
                diagnostics["executor"]["cancelled"] += 1
                diagnostics["executor"]["timed_out"] += 1
                diagnostics["executor"]["queue_timeout"] += 1
                diagnostics["errors"].append(
                    {
                        "type": state["error_type"],
                        "message": state["error_message"],
                    }
                )
                logger.warning(
                    "Retrieval task %s (%s): queue_timeout после %.2f с",
                    state["index"],
                    state["purpose"],
                    queue_timeout_s,
                )
    finally:
        for future in futures:
            future.cancel()
        # Ровно один shutdown; потоки дожидаются ограниченно. cancel() не
        # останавливает уже выполняющийся код — это фиксируется числом
        # threads_alive_after_grace, а не выдаётся за успешную отмену.
        executor.shutdown(wait=False, cancel_futures=True)
        diagnostics["executor"]["threads_alive_after_grace"] = (
            _join_executor_threads(executor, grace_s)
        )

    started_count = sum(
        1
        for state in task_states
        if state.get("started_at_mono") is not None
    )
    diagnostics["executor"]["started"] = started_count
    diagnostics["executor"]["queued"] = max(0, len(tasks) - started_count)
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
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
        _publish_diagnostics(diagnostics_out, diagnostics)
        error_types = sorted(
            {error["type"] for error in diagnostics["errors"]}
        ) or ["UnknownError"]
        summary = ", ".join(error_types)
<<<<<<< HEAD
        if timed_out == submitted:
            raise RetrievalTimeoutError(
                "Все retrieval-задачи превысили дедлайн "
                f"{timeout_s} с (причины: {summary})"
=======
        statuses = {state.get("status") for state in task_states}
        only_timeouts = statuses <= {
            "timeout", "queue_timeout", "cancelled", "not_started"
        } and bool(statuses & {"timeout", "queue_timeout"})
        if only_timeouts or timed_out >= len(tasks):
            raise RetrievalTimeoutError(
                "Все retrieval-задачи превысили дедлайн "
                f"(task={timeout_s} с, queue={queue_timeout_s} с; "
                f"причины: {summary})"
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
            )
        raise RetrievalAPIError(
            f"Все retrieval-задачи завершились ошибкой ({summary})"
        )

<<<<<<< HEAD
    merged_by_id: dict[str, RetrievedChunk] = {}
    hit_counts: dict[str, int] = {}
    protected_top_ids: list[str] = []

    for chunk_list in task_results:
        for rank_idx, ch in enumerate(chunk_list):
            cid = ch.chunk_id
            if rank_idx < 3 and cid not in protected_top_ids:
                protected_top_ids.append(cid)
            if cid not in merged_by_id:
                merged_by_id[cid] = ch
                hit_counts[cid] = 1
            else:
                hit_counts[cid] += 1
                prev = merged_by_id[cid]
                if ch.score > prev.score:
                    merged_by_id[cid] = ch

    deduped_map: dict[str, RetrievedChunk] = {}
    for cid, ch in merged_by_id.items():
        if hit_counts[cid] > 1:
            # Пропорциональный буст в шкале RRF/Reranker, не искажающий порядок на порядок величины
            increment = 0.0015 if ch.score < 0.1 else 0.05
            boosted_score = min(1.0, round(ch.score + increment, 6))
            deduped_map[cid] = replace(ch, score=boosted_score)
        else:
            deduped_map[cid] = ch
=======
    # H-2B: суммирующая RRF-агрегация рангов задач вместо max(score)+increment.
    deduped_map, protected_top_ids = _aggregate_task_results(task_results)
    boost_ids = _collect_boost_doc_ids(
        task for task, _chunks in task_results
    )
    doc_boost_applied = sum(
        1
        for chunk in deduped_map.values()
        if getattr(chunk, "document_id", None) in boost_ids
    )
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)

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
<<<<<<< HEAD
=======
            "intents": [t.intent for t in tasks],
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
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
<<<<<<< HEAD
=======
            "aggregation": "sum_rrf",
            "rrf_k": RRF_AGGREGATION_K,
            "doc_boost_factor": DOC_BOOST_FACTOR,
            "boost_doc_ids_count": len(boost_ids),
            "doc_boost_applied": doc_boost_applied,
            # Warm-up/бэкенды: без секретов и путей.
            "prewarm": dict(diagnostics.get("prewarm") or {}),
>>>>>>> a87f9eb (fix(rag): stabilize retrieval executor and restore hybrid ranking)
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
