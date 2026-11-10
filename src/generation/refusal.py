import logging
import time
from src import config
from src.generation import prompts
from src.generation.types import AgentResult, Intent, TraceStep

logger = logging.getLogger(__name__)

TEMPLATE_MAP: dict[str, str] = {
    config.RESPONSE_OUT_OF_SCOPE: prompts.RESPONSE_OUT_OF_SCOPE,
    config.RESPONSE_ABOUT_ASSISTANT: prompts.RESPONSE_ABOUT_ASSISTANT,
    config.RESPONSE_REFUSAL_NO_CONTEXT: prompts.RESPONSE_REFUSAL_NO_CONTEXT,
}


def resolve_template(template_key: str) -> str:
    return TEMPLATE_MAP[template_key]


def build_template_intent_result(
    query: str, intent: Intent, prior_trace: list[TraceStep]
) -> AgentResult:
    start_time = time.perf_counter()
    if intent == Intent.OUT_OF_SCOPE:
        template_key = config.RESPONSE_OUT_OF_SCOPE
        refused = True
    else:
        template_key = config.RESPONSE_ABOUT_ASSISTANT
        refused = False

    answer_text = resolve_template(template_key)
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_TEMPLATE_RESPONSE,
        input={
            "query": query,
            "intent": intent.value,
            "template_key": template_key,
        },
        output={
            "refused": refused,
            "answer_length": len(answer_text),
        },
        latency_ms=latency_ms,
    )
    logger.info(
        "Step %s returned template %s (refused=%s) in %.3f ms",
        config.STEP_TEMPLATE_RESPONSE,
        template_key,
        refused,
        latency_ms,
    )
    return AgentResult(
        final_answer=answer_text,
        sources={},
        intent=intent.value,
        refused=refused,
        trace=[*prior_trace, step],
    )


def build_no_context_refusal_result(
    query: str, intent: Intent, prior_trace: list[TraceStep]
) -> AgentResult:
    start_time = time.perf_counter()
    template_key = config.RESPONSE_REFUSAL_NO_CONTEXT
    answer_text = resolve_template(template_key)
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_REFUSAL,
        input={
            "query": query,
            "intent": intent.value,
            "template_key": template_key,
        },
        output={
            "refused": True,
            "answer_length": len(answer_text),
        },
        latency_ms=latency_ms,
    )
    logger.info(
        "Step %s triggered refusal (%s) in %.3f ms",
        config.STEP_REFUSAL,
        template_key,
        latency_ms,
    )
    return AgentResult(
        final_answer=answer_text,
        sources={},
        intent=intent.value,
        refused=True,
        trace=[*prior_trace, step],
    )