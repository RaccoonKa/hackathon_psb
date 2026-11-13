import logging
import re
import time
from typing import Any
from rag_core import config
from rag_core.generation import prompts
from rag_core.generation.llm_client import llm_complete_with_usage
from rag_core.generation.types import AgentResult, Intent, TraceStep

logger = logging.getLogger(__name__)

_SOURCE_TAG_PATTERN = re.compile(r"<source>(.*?)</source>", re.DOTALL)


def build_generation_messages(
    query: str, context: str
) -> list[dict[str, str]]:
    return [
        {
            "role": config.ROLE_SYSTEM,
            "content": prompts.GENERATION_SYSTEM,
        },
        {
            "role": config.ROLE_USER,
            "content": prompts.GENERATION_USER.format(
                context=context, query=query
            ),
        },
    ]


def generate_answer(query: str, context: str) -> tuple[str, TraceStep]:
    start_time = time.perf_counter()
    messages = build_generation_messages(query=query, context=context)
    answer, prompt_tokens, completion_tokens = llm_complete_with_usage(
        messages=messages,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_GENERATION,
        input={
            "query": query,
            "context_length": len(context),
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        },
        output={"answer_length": len(answer)},
        latency_ms=latency_ms,
        model=config.LLM_MODEL,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    logger.info(
        "Step %s generated answer (%d chars) in %.3f ms",
        config.STEP_GENERATION,
        len(answer),
        latency_ms,
    )
    return answer, step


def _sanitize_and_extract_citations(
    answer: str, available_sources: dict[str, dict[str, Any]]
) -> tuple[str, list[str]]:
    cited_ids: list[str] = []

    def _replace_tag(match: re.Match[str]) -> str:
        raw_inner = match.group(1)
        tokens = [t.strip() for t in raw_inner.split(",") if t.strip()]
        valid_tokens: list[str] = []
        for token in tokens:
            if token in available_sources:
                valid_tokens.append(token)
                if token not in cited_ids:
                    cited_ids.append(token)
        if not valid_tokens:
            return ""
        return f"<source>{', '.join(valid_tokens)}</source>"

    cleaned_answer = _SOURCE_TAG_PATTERN.sub(_replace_tag, answer)
    cleaned_answer = re.sub(r" +([.,;:])", r"\1", cleaned_answer)
    return cleaned_answer, cited_ids


def assemble_final_response(
    final_answer: str,
    sources: dict[str, dict[str, Any]],
    intent: Intent,
    trace: list[TraceStep],
) -> AgentResult:
    start_time = time.perf_counter()
    cleaned_answer, cited_ids = _sanitize_and_extract_citations(
        answer=final_answer, available_sources=sources
    )
    filtered_sources = (
        {sid: sources[sid] for sid in cited_ids} if cited_ids else dict(sources)
    )
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    assembly_step = TraceStep(
        step=config.STEP_RESPONSE_ASSEMBLY,
        input={
            "intent": intent.value,
            "source_ids": list(sources.keys()),
            "answer_length": len(final_answer),
        },
        output={
            "refused": False,
            "sources_count": len(filtered_sources),
            "cited_source_ids": cited_ids,
        },
        latency_ms=latency_ms,
    )
    full_trace = [*trace, assembly_step]
    logger.info(
        "Step %s completed in %.3f ms",
        config.STEP_RESPONSE_ASSEMBLY,
        latency_ms,
    )
    return AgentResult(
        final_answer=cleaned_answer,
        sources=filtered_sources,
        intent=intent.value,
        refused=False,
        trace=full_trace,
    )