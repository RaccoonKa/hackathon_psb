import logging
from rag_core import config
from rag_core.generation.llm.client import llm_complete
from rag_core.generation.prompts import bailout as bailout_prompts
from rag_core.generation.prompts import templates
from rag_core.generation.types import BailoutContext

logger = logging.getLogger(__name__)


def explain(ctx: BailoutContext, timeout: float | None = None) -> str:
    if not ctx.llm_was_available or ctx.partial_spec is None:
        return templates.RESPONSE_CANNOT_PARSE

    effective_timeout = (
        min(float(config.LLM_REQUEST_TIMEOUT_S), float(timeout))
        if timeout is not None
        else float(config.LLM_REQUEST_TIMEOUT_S)
    )
    if effective_timeout <= 0.5:
        return templates.RESPONSE_CANNOT_PARSE

    spec = ctx.partial_spec
    partial_summary = (
        f"intent={spec.intent}, сущности={spec.entities_raw}, "
        f"статьи={spec.article_refs}, дата={spec.event_date}"
    )
    messages = [
        {"role": config.ROLE_SYSTEM, "content": bailout_prompts.BAILOUT_SYSTEM},
        {
            "role": config.ROLE_USER,
            "content": bailout_prompts.BAILOUT_USER.format(
                query=ctx.query,
                partial_spec_summary=partial_summary,
                last_stage=ctx.last_stage,
                last_error=ctx.last_error or "превышен лимит попыток",
            ),
        },
    ]
    try:
        explanation = llm_complete(
            messages=messages,
            model=config.LLM_MODEL,
            temperature=bailout_prompts.BAILOUT_TEMPERATURE,
            max_tokens=bailout_prompts.BAILOUT_MAX_TOKENS,
            timeout=effective_timeout,
        )
        return explanation.strip() or templates.RESPONSE_CANNOT_PARSE
    except Exception:
        return templates.RESPONSE_CANNOT_PARSE