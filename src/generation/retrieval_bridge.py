import logging
import time
from src import config
from src.generation.stubs import stub_rag_retrieve, stub_reranker
from src.generation.types import Intent, RetrievedChunk, TraceStep
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank_candidates

logger = logging.getLogger(__name__)


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


def retrieve_chunks(
    query: str,
    top_k: int,
    event_date: str | None = None,
    intent: Intent | None = None,
    selected_doc_ids: set[str] | None = None,
) -> tuple[list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    if config.USE_STUB_RAG:
        mode = "stub"
        chunks = stub_rag_retrieve(query=query, top_k=top_k)
    else:
        mode = "real"
        # Получаем пул RRF кандидатов без преждевременного реранкинга,
        # чтобы этап реранкинга честно замерил задержку на следующем шаге
        raw_chunks = hybrid_search(
            query=query,
            top_k=max(top_k, 20),
            event_date=event_date,
            selected_doc_ids=selected_doc_ids,
            intent=intent.value if intent is not None else None,
            rerank=False,
        )
        chunks = [_dict_to_chunk(raw) for raw in raw_chunks]

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
            "intent": intent.value if intent is not None else None,
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
    )
    logger.info(
        "Step %s retrieved %d chunks in %.3f ms",
        config.STEP_RAG_RETRIEVAL,
        len(chunks),
        latency_ms,
    )
    return chunks, step


def rerank_chunks(
    chunks: list[RetrievedChunk], query: str, top_n: int
) -> tuple[list[RetrievedChunk], TraceStep]:
    start_time = time.perf_counter()
    if config.USE_STUB_RERANKER:
        mode = "stub"
        reranked = stub_reranker(chunks=chunks, query=query, top_n=top_n)
    else:
        mode = "real"
        cand_dicts = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "embedding_text": f"{c.document_title}. {' > '.join(c.structure_path)}. {c.text}",
                "document_id": c.document_id,
                "version_id": c.version_id,
                "document_title": c.document_title,
                "structure": {
                    "provision_id": c.provision_id,
                    "path": c.structure_path,
                },
                "score": c.score,
                "effective_from": c.effective_from,
                "effective_to": c.effective_to,
                "effective_dates_known": c.effective_dates_known,
                "amends_document_id": c.amends_document_id,
            }
            for c in chunks
        ]
        reranked_dicts = rerank_candidates(query=query, candidates=cand_dicts, top_k=top_n)
        reranked = [_dict_to_chunk(r) for r in reranked_dicts]

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
    if threshold > 0.0 and 0.0 < max_score < 0.1:
        effective_threshold = min(threshold, 0.005)
        logger.warning(
            "Обнаружены RRF-скоры (max=%.3f). Порог адаптивно снижен до %.3f",
            max_score,
            effective_threshold,
        )

    passed = bool(chunks) and max_score >= effective_threshold
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
    )
    logger.info(
        "Step %s evaluated passed=%s (max_score=%.3f, eff_threshold=%.3f) in %.3f ms",
        config.STEP_THRESHOLD_FILTER,
        passed,
        max_score,
        effective_threshold,
        latency_ms,
    )
    return passed, filtered, step