import logging
from src import config
from src.generation.context_builder import build_context_and_sources
from src.generation.generator import assemble_final_response, generate_answer
from src.generation.intent import detect_intent
from src.generation.refusal import (
    build_no_context_refusal_result,
    build_template_intent_result,
)
from src.generation.retrieval_bridge import (
    filter_chunks_by_threshold,
    rerank_chunks,
    retrieve_chunks,
)
from src.generation.types import AgentResult, Intent, TraceStep

logger = logging.getLogger(__name__)


def run_agent(query: str) -> AgentResult:
    trace: list[TraceStep] = []

    intent, intent_step = detect_intent(query=query)
    trace.append(intent_step)

    if intent in {Intent.OUT_OF_SCOPE, Intent.ABOUT_ASSISTANT}:
        return build_template_intent_result(
            query=query, intent=intent, prior_trace=trace
        )

    retrieved_chunks, retrieval_step = retrieve_chunks(
        query=query, top_k=config.RAG_TOP_K
    )
    trace.append(retrieval_step)

    reranked_chunks, rerank_step = rerank_chunks(
        chunks=retrieved_chunks, query=query, top_n=config.RERANKER_TOP_N
    )
    trace.append(rerank_step)

    passed, filtered_chunks, filter_step = filter_chunks_by_threshold(
        chunks=reranked_chunks, threshold=config.RAG_SCORE_THRESHOLD
    )
    trace.append(filter_step)

    if not passed:
        return build_no_context_refusal_result(
            query=query, intent=intent, prior_trace=trace
        )

    sources, context_str, context_step = build_context_and_sources(
        chunks=filtered_chunks
    )
    trace.append(context_step)

    answer_text, generation_step = generate_answer(
        query=query, context=context_str
    )
    trace.append(generation_step)

    return assemble_final_response(
        final_answer=answer_text,
        sources=sources,
        intent=intent,
        trace=trace,
    )