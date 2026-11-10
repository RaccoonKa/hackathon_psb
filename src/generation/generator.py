import logging
import time
from typing import Any
from src import config
from src.generation import prompts
from src.generation.llm_client import llm_complete
from src.generation.types import AgentResult, Intent, TraceStep

logger = logging.getLogger(__name__)


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
    answer = llm_complete(
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
        prompt_tokens=config.STUB_PROMPT_TOKENS,
        completion_tokens=config.STUB_COMPLETION_TOKENS,
    )
    logger.info(
        "Step %s generated answer (%d chars) in %.3f ms",
        config.STEP_GENERATION,
        len(answer),
        latency_ms,
    )
    return answer, step


def assemble_final_response(
    final_answer: str,
    sources: dict[str, dict[str, Any]],
    intent: Intent,
    trace: list[TraceStep],
) -> AgentResult:
    start_time = time.perf_counter()
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
            "sources_count": len(sources),
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
        final_answer=final_answer,
        sources=sources,
        intent=intent.value,
        refused=False,
        trace=full_trace,
    )