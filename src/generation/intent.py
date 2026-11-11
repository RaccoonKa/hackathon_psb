import logging
import re
import time
from src import config
from src.generation import prompts
from src.generation.llm_client import llm_complete_with_usage
from src.generation.types import Intent, TraceStep

logger = logging.getLogger(__name__)


def build_intent_messages(query: str) -> list[dict[str, str]]:
    return [
        {
            "role": config.ROLE_SYSTEM,
            "content": prompts.INTENT_CLASSIFIER_SYSTEM,
        },
        {
            "role": config.ROLE_USER,
            "content": prompts.INTENT_CLASSIFIER_USER.format(query=query),
        },
    ]


def parse_intent_label(raw_label: str) -> Intent:
    cleaned = raw_label.strip().lower()
    for item in Intent:
        if item.value == cleaned:
            return item
    for item in Intent:
        if re.search(rf"\b{re.escape(item.value)}\b", cleaned):
            return item
    for item in Intent:
        if item.value in cleaned:
            return item
    return config.INTENT_FALLBACK


def detect_intent(query: str) -> tuple[Intent, TraceStep]:
    start_time = time.perf_counter()
    messages = build_intent_messages(query=query)
    raw_output, prompt_tokens, completion_tokens = llm_complete_with_usage(
        messages=messages,
        model=config.LLM_MODEL,
        temperature=config.INTENT_CLASSIFIER_TEMPERATURE,
        max_tokens=config.INTENT_CLASSIFIER_MAX_TOKENS,
    )
    detected = parse_intent_label(raw_output)
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_INTENT_DETECTION,
        input={
            "query": query,
            "temperature": config.INTENT_CLASSIFIER_TEMPERATURE,
            "max_tokens": config.INTENT_CLASSIFIER_MAX_TOKENS,
        },
        output={"intent": detected.value},
        latency_ms=latency_ms,
        model=config.LLM_MODEL,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    logger.info(
        "Step %s finished with intent=%s in %.3f ms",
        config.STEP_INTENT_DETECTION,
        detected.value,
        latency_ms,
    )
    return detected, step