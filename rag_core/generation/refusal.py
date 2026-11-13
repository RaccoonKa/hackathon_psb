import logging
import time
from rag_core import config
from rag_core.generation.prompts import templates
from rag_core.generation.types import (
    AgentResult,
    Intent,
    QuerySpec,
    TraceStep,
    VerificationReport,
)

logger = logging.getLogger(__name__)

# Технический сбой поиска — принципиально иное сообщение, чем отказ
# «недостаточно сведений»: пользователь должен понимать, что база знаний
# не проверялась из-за сбоя, и запрос имеет смысл повторить.
RESPONSE_RETRIEVAL_ERROR: str = (
    "Не удалось выполнить поиск по базе знаний из-за временной технической "
    "ошибки. Пожалуйста, повторите запрос через несколько секунд. "
    "Если ошибка повторится, уточните реквизиты документа и попробуйте ещё раз."
)


def build_template_intent_result(
    query: str,
    intent: Intent,
    prior_trace: list[TraceStep],
    template_key: str | None = None,
    reason: str | None = None,
) -> AgentResult:
    start_time = time.perf_counter()
    if template_key and hasattr(templates, template_key):
        answer_text = getattr(templates, template_key)
    elif intent == Intent.ABOUT_ASSISTANT:
        answer_text = templates.RESPONSE_ABOUT_ASSISTANT
    else:
        answer_text = templates.RESPONSE_OUT_OF_SCOPE

    refused = intent != Intent.ABOUT_ASSISTANT and reason != "about_assistant"
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_TEMPLATE_RESPONSE,
        input={"query": query, "intent": intent.value},
        output={"refused": refused, "answer_length": len(answer_text)},
        latency_ms=latency_ms,
        stage_idx=0,
    )
    return AgentResult(
        final_answer=answer_text,
        sources={},
        intent=intent.value,
        refused=refused,
        trace=[*prior_trace, step],
        refusal_reason=reason if refused else None,
        confidence="high",
    )


def build_no_context_refusal_result(
    query: str,
    intent: Intent | str,
    prior_trace: list[TraceStep],
    refusal_reason: str = "insufficient_evidence",
    query_spec: QuerySpec | None = None,
    missing_info: list[str] | None = None,
    degradation_reasons: list[str] | None = None,
    used_history: bool = False,
) -> AgentResult:
    start_time = time.perf_counter()
    intent_val = intent.value if isinstance(intent, Intent) else str(intent)
    answer_text = templates.RESPONSE_REFUSAL_NO_CONTEXT
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_REFUSAL,
        input={"query": query, "intent": intent_val, "reason": refusal_reason},
        output={"refused": True, "answer_length": len(answer_text)},
        latency_ms=latency_ms,
        stage_idx=4,
    )
    return AgentResult(
        final_answer=answer_text,
        sources={},
        intent=intent_val,
        refused=True,
        trace=[*prior_trace, step],
        refusal_reason=refusal_reason,
        confidence="low",
        assumptions=list(query_spec.assumptions) if query_spec else [],
        missing_info=list(missing_info) if missing_info else [],
        query_spec=query_spec,
        degradation_reasons=list(degradation_reasons) if degradation_reasons else [],
        used_history=used_history,
    )


def build_llm_unavailable_result(
    query: str,
    prior_trace: list[TraceStep],
    degradation_reasons: list[str] | None = None,
) -> AgentResult:
    start_time = time.perf_counter()
    answer_text = templates.RESPONSE_LLM_UNAVAILABLE
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_REFUSAL,
        input={"query": query, "reason": "llm_unavailable"},
        output={"refused": True, "answer_length": len(answer_text)},
        latency_ms=latency_ms,
        degradation="llm_unavailable",
    )
    return AgentResult(
        final_answer=answer_text,
        sources={},
        intent="semantic",
        refused=True,
        trace=[*prior_trace, step],
        refusal_reason="llm_unavailable",
        confidence="low",
        degradation_reasons=list(degradation_reasons or ["llm_unavailable"]),
    )


def build_cannot_parse_result(
    query: str,
    prior_trace: list[TraceStep],
    partial_spec: QuerySpec | None = None,
    explanation_text: str | None = None,
    degradation_reasons: list[str] | None = None,
) -> AgentResult:
    start_time = time.perf_counter()
    answer_text = explanation_text or templates.RESPONSE_CANNOT_PARSE
    intent_val = partial_spec.intent if partial_spec else "semantic"
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_REFUSAL,
        input={"query": query, "reason": "cannot_parse"},
        output={"refused": True, "answer_length": len(answer_text)},
        latency_ms=latency_ms,
        degradation="cannot_parse",
    )
    return AgentResult(
        final_answer=answer_text,
        sources={},
        intent=intent_val,
        refused=True,
        trace=[*prior_trace, step],
        refusal_reason="cannot_parse",
        confidence="low",
        query_spec=partial_spec,
        degradation_reasons=list(degradation_reasons or ["cannot_parse"]),
    )


def build_retrieval_error_result(
    query: str,
    prior_trace: list[TraceStep],
    error_types: list[str] | None = None,
    attempts: list[dict] | None = None,
    query_spec: QuerySpec | None = None,
    missing_info: list[str] | None = None,
    degradation_reasons: list[str] | None = None,
    used_history: bool = False,
) -> AgentResult:
    """Отказ из-за технического сбоя retrieval.

    В отличие от ``build_no_context_refusal_result`` сообщает о сбое поиска,
    а не об отсутствии сведений. В trace попадают только имена классов
    ошибок и номера попыток — без путей, ключей и стектрейсов.
    """
    start_time = time.perf_counter()
    intent_val = query_spec.intent if query_spec else "semantic"
    safe_types = sorted({str(t) for t in (error_types or []) if t})
    attempt_summaries = [
        {
            "attempt": item.get("attempt"),
            "status": item.get("status", "error"),
            "error_type": item.get("error_type"),
            "error_message": (
                str(item.get("error_message"))[:300]
                if item.get("error_message")
                else None
            ),
            "technical": item.get("technical"),
        }
        for item in (attempts or [])[:10]
    ]
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_REFUSAL,
        input={
            "query": query,
            "reason": "retrieval_error",
            "technical_retrieval_failure": True,
            "attempts": attempt_summaries,
        },
        output={
            "refused": True,
            "refusal_kind": "retrieval_error",
            "technical_retrieval_failure": True,
            "error_types": safe_types,
            "answer_length": len(RESPONSE_RETRIEVAL_ERROR),
        },
        latency_ms=latency_ms,
        stage_idx=3,
        degradation="retrieval_error",
    )
    return AgentResult(
        final_answer=RESPONSE_RETRIEVAL_ERROR,
        sources={},
        intent=intent_val,
        refused=True,
        trace=[*prior_trace, step],
        refusal_reason="retrieval_error",
        confidence="low",
        assumptions=list(query_spec.assumptions) if query_spec else [],
        missing_info=list(missing_info or []),
        query_spec=query_spec,
        degradation_reasons=list(degradation_reasons or ["retrieval_error"]),
        used_history=used_history,
    )


def build_verification_refusal_result(
    query: str,
    spec: QuerySpec,
    verification: VerificationReport,
    prior_trace: list[TraceStep],
    degradation_reasons: list[str] | None = None,
    used_history: bool = False,
) -> AgentResult:
    start_time = time.perf_counter()
    answer_text = templates.RESPONSE_REFUSAL_NO_CONTEXT
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_REFUSAL,
        input={"query": query, "reason": "verification_refused"},
        output={"refused": True, "final_verdict": verification.final_verdict.value},
        latency_ms=latency_ms,
        stage_idx=7,
    )
    return AgentResult(
        final_answer=answer_text,
        sources={},
        intent=spec.intent,
        refused=True,
        trace=[*prior_trace, step],
        refusal_reason="verification_failed",
        confidence="low",
        assumptions=list(spec.assumptions),
        missing_info=list(spec.required_slots),
        verification=verification,
        query_spec=spec,
        degradation_reasons=list(degradation_reasons or []),
        used_history=used_history,
    )