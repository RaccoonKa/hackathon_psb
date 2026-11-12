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
from src.retrieval.hybrid import hybrid_search

logger = logging.getLogger(__name__)

_HYBRID_SEARCH_ACCEPTS_INTENT = "intent" in inspect.signature(hybrid_search).parameters

_MODE_TO_HYBRID_INTENT: dict[str, str] = {
    "bm25": "exact_lookup",
    "dense": "semantic_query",
    "hybrid": "mixed_query",
}


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


def _call_hybrid_search(
    query: str,
    top_k: int,
    event_date: str | None,
    selected_doc_ids: set[str] | None,
    intent_str: str | None,
) -> list[RetrievedChunk]:
    if config.USE_STUB_RAG:
        return stub_rag_retrieve(query=query, top_k=top_k)

    kwargs: dict = {
        "query": query,
        "top_k": top_k,
        "event_date": event_date,
        "selected_doc_ids": selected_doc_ids,
    }
    if _HYBRID_SEARCH_ACCEPTS_INTENT and intent_str is not None:
        kwargs["intent"] = intent_str

    raw_chunks = hybrid_search(**kwargs)
    return [_dict_to_chunk(raw) for raw in raw_chunks]


def _run_single_task(task: RetrievalTask) -> list[RetrievedChunk]:
    intent_str = _MODE_TO_HYBRID_INTENT.get(task.mode, "mixed_query")
    chunks = _call_hybrid_search(
        query=task.query,
        top_k=task.top_k,
        event_date=task.event_date,
        selected_doc_ids=task.selected_doc_ids,
        intent_str=intent_str,
    )
    if task.boost_doc_ids:
        boosted: list[RetrievedChunk] = []
        for ch in chunks:
            if ch.document_id in task.boost_doc_ids:
                delta = 0.002 if ch.score < 0.1 else 0.05
                boosted.append(replace(ch, score=min(1.0, round(ch.score + delta, 6))))
            else:
                boosted.append(ch)
        return boosted
    return chunks


def retrieve_tasks(
    tasks: list[RetrievalTask],
) -> tuple[list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    if not tasks:
        step = TraceStep(
            step=config.STEP_RAG_RETRIEVAL,
            input={"tasks_count": 0},
            output={"chunk_ids": [], "scores": [], "count": 0},
            latency_ms=0.0,
            stage_idx=3,
        )
        return [], step

    timeout_s = float(config.RETRIEVAL_TIMEOUT_S)
    task_results: list[list[RetrievedChunk]] = []

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    try:
        futures = [executor.submit(_run_single_task, task) for task in tasks]
        for fut in futures:
            try:
                res = fut.result(timeout=timeout_s)
                task_results.append(res)
            except concurrent.futures.TimeoutError as exc:
                executor.shutdown(wait=False, cancel_futures=True)
                raise RetrievalTimeoutError(
                    f"Retrieval task timed out after {timeout_s}s"
                ) from exc
            except Exception as exc:
                executor.shutdown(wait=False, cancel_futures=True)
                raise RetrievalAPIError(f"Retrieval task failed: {exc}") from exc
    finally:
        executor.shutdown(wait=False)

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
        },
        latency_ms=latency_ms,
        stage_idx=3,
    )
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


def rerank_chunks(
    chunks: list[RetrievedChunk], query: str, top_n: int
) -> tuple[list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    if config.USE_STUB_RERANKER:
        mode = "stub"
        reranked = stub_reranker(chunks=chunks, query=query, top_n=top_n)
    else:
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
    if 0.0 < max_score < ceiling:
        effective_threshold = fallback_thresh

    passed = bool(chunks) and max_score >= effective_threshold
    filtered = [c for c in chunks if c.score >= effective_threshold] if passed else []
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