import concurrent.futures
from dataclasses import replace
import inspect
import logging
import time

from src import config
from src.generation.errors import RetrievalAPIError, RetrievalTimeoutError
from src.generation.stubs import stub_rag_retrieve, stub_reranker
from src.generation.types import (
    Intent,
    RetrievalTask,
    RetrievedChunk,
    TraceStep,
)
from src.retrieval.hybrid import (
    bm25_search,
    dense_search,
    hybrid_search,
)
from src.retrieval.reranker import rerank_candidates
from src.retrieval.vector_store import (
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

_EMPTY_EMBEDDINGS_DIAGNOSTICS: dict = {
    "provider": "none",
    "model": None,
    "dimension": None,
    "success": True,
    "fallback_used": False,
    "error_type": None,
}

_SIGNATURE_CACHE: dict = {}


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
    """Единый постпроцессинг задач: дедуп по chunk_id + буст документов."""
    best_by_id: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    for chunk in chunks:
        cid = chunk.chunk_id
        if cid not in best_by_id:
            order.append(cid)
            best_by_id[cid] = chunk
        elif chunk.score > best_by_id[cid].score:
            best_by_id[cid] = chunk

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


def _run_single_task(
    task: RetrievalTask,
) -> tuple[list[RetrievedChunk], dict, dict, int]:
    """Выполняет одну задачу; возвращает (chunks, search, embeddings, ms)."""
    started = time.perf_counter()
    reset_embedding_diagnostics()

    mode = (task.mode or "hybrid").strip().lower()
    intent_str = _MODE_TO_HYBRID_INTENT.get(mode, "mixed_query")
    stats: dict = dict(_SEARCH_DIAGNOSTICS_DEFAULTS)

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
    }


def _publish_diagnostics(
    diagnostics_out: dict | None, payload: dict
) -> None:
    if diagnostics_out is None:
        return
    diagnostics_out.update(payload)


def _join_executor_threads(executor, grace_s: float) -> None:
    """Ограниченное ожидание завершения потоков после единственного shutdown."""
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


def retrieve_tasks(
    tasks: list[RetrievalTask],
    *,
    diagnostics_out: dict | None = None,
) -> tuple[list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    diagnostics: dict = {
        "executor": {
            "submitted": len(tasks),
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "timed_out": 0,
        },
        "tasks": [],
        "degraded": False,
        "errors": [],
    }

    if not tasks:
        step = TraceStep(
            step=config.STEP_RAG_RETRIEVAL,
            input={"tasks_count": 0},
            output={"chunk_ids": [], "scores": [], "count": 0},
            latency_ms=0.0,
            stage_idx=3,
        )
        _publish_diagnostics(diagnostics_out, diagnostics)
        return [], step

    # Предзагрузка до создания пула: гонок инициализации нет в принципе.
    try:
        prewarm_report = prewarm_retrieval_backends()
    except Exception as exc:  # noqa: BLE001 - предзагрузка не фатальна
        logger.warning("Prewarm retrieval backends failed: %s", exc)
        diagnostics["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)[:300]}
        )
    else:
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
    configured_workers = int(
        getattr(config, "RETRIEVAL_MAX_WORKERS", 0)
        or getattr(config, "MAX_WORKERS", 1)
        or 1
    )
    max_workers = max(1, min(configured_workers, len(tasks)))

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
        _publish_diagnostics(diagnostics_out, diagnostics)
        error_types = sorted(
            {error["type"] for error in diagnostics["errors"]}
        ) or ["UnknownError"]
        summary = ", ".join(error_types)
        if timed_out == submitted:
            raise RetrievalTimeoutError(
                "Все retrieval-задачи превысили дедлайн "
                f"{timeout_s} с (причины: {summary})"
            )
        raise RetrievalAPIError(
            f"Все retrieval-задачи завершились ошибкой ({summary})"
        )

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
