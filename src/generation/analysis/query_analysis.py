import json
import logging
import re
import time
from typing import Callable, Optional
from pydantic import ValidationError
from src import config
from src.generation.analysis.entity_resolver import (
    has_effective_dates_in_corpus,
    resolve_entities,
)
from src.generation.errors import LLMError, LLMFormatError
from src.generation.llm.client import llm_complete_with_usage
from src.generation.prompts import query_analysis as qa_prompts
from src.generation.types import (
    Ambiguity,
    Premise,
    QueryAnalysisSchema,
    QuerySpec,
    SubQuestion,
    TaskType,
    TraceStep,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CONTRACT_NUM_RE = re.compile(r"(?:№\s*|договор[ауеом]?\s+)([\w\d\-\/]+)", re.IGNORECASE)
_QUOTED_ENTITY_RE = re.compile(r"[«\"]([^»\"]+)[»\"]")
_ARTICLE_EXTRACT_RE = re.compile(
    r"(?:стать[яеиюй]|ст\.|пункт[ауеом]?|п\.)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_RU_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(20\d{2})\b")

_SLOT_DETECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("rent_amount", re.compile(r"арендн\w+\s+плат|стоимост\w+\s+аренд|размер\w*\s+аренд", re.IGNORECASE)),
    ("payment_deadline", re.compile(r"срок\w*\s+оплат|порядок\s+расчет|не\s+позднее\s+какого\s+числа", re.IGNORECASE)),
    ("probation_period", re.compile(r"испытательн\w+\s+срок", re.IGNORECASE)),
    ("overtime_limit", re.compile(r"сколько\s+часов\s+сверхурочн|продолжительност\w+\s+сверхурочн|120\s+часов", re.IGNORECASE)),
    ("overtime_pay", re.compile(r"оплач\w+\s+сверхурочн|оплат\w+\s+сверхурочн", re.IGNORECASE)),
    ("approval_process", re.compile(r"согласован\w+|порядок\s+подписан|визирован", re.IGNORECASE)),
    ("termination", re.compile(r"расторжен\w+|прекращен\w+\s+договор|односторонн\w+\s+отказ", re.IGNORECASE)),
    ("penalty", re.compile(r"неустойк\w+|пен[яию]|штраф", re.IGNORECASE)),
    ("tax_rate", re.compile(r"налог\w+\s+ставк|ндс|налог\s+на\s+прибыл", re.IGNORECASE)),
]


def resolve_history(query: str, history: list[dict]) -> tuple[str, bool]:
    if not history:
        return query, False
    last_turns = history[-3:]
    context_parts = [
        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
        for turn in last_turns
        if isinstance(turn, dict) and turn.get("content")
    ]
    if not context_parts:
        return query, False
    return f"{query} (Контекст диалога: {' | '.join(context_parts)})", True


def _extract_json_dict(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = _JSON_BLOCK_RE.search(raw_text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            raise LLMFormatError(f"Invalid JSON object in LLM output: {exc}") from exc

    raise LLMFormatError("No JSON object found in QueryAnalysis LLM output")


def _ensure_exact_tokens_in_bm25(query: str, bm25_queries: list[str]) -> list[str]:
    exact_tokens = _CONTRACT_NUM_RE.findall(query)
    if not exact_tokens:
        return bm25_queries
    combined = " ".join(bm25_queries).lower()
    missing = [tok for tok in exact_tokens if tok.lower() not in combined]
    if missing:
        bm25_queries[0] = f"{bm25_queries[0]} {' '.join(missing)}".strip()
    return bm25_queries


def _infer_slots_by_regex(query: str) -> list[str]:
    slots = [
        slot_name
        for slot_name, pattern in _SLOT_DETECTION_PATTERNS
        if pattern.search(query)
    ]
    return slots if slots else ["general"]


def _build_regex_fallback_spec(
    query: str,
    event_date: str | None,
    history_resolved: bool,
) -> QuerySpec:
    entities_raw: list[str] = []
    for m in _QUOTED_ENTITY_RE.findall(query):
        entities_raw.append(m.strip())
    for m in _CONTRACT_NUM_RE.findall(query):
        entities_raw.append(m.strip())

    article_refs = [m.strip(".") for m in _ARTICLE_EXTRACT_RE.findall(query) if m.strip(".")]

    dates: list[str] = list(_ISO_DATE_RE.findall(query))
    for day, month, year in _RU_DATE_RE.findall(query):
        iso_candidate = f"{year}-{int(month):02d}-{int(day):02d}"
        if iso_candidate not in dates:
            dates.append(iso_candidate)

    resolved_event_date = event_date or (dates[0] if dates else None)
    entities_resolved = resolve_entities(entities_raw)
    required_slots = _infer_slots_by_regex(query)

    return QuerySpec(
        intent="semantic",
        task_type=TaskType.EXPLAIN,
        entities_raw=entities_raw,
        entities_resolved=entities_resolved,
        article_refs=article_refs,
        dates=dates,
        event_date=resolved_event_date,
        premises=[],
        required_slots=required_slots,
        assumptions=[],
        ambiguity=Ambiguity.LOW,
        needs_clarification=False,
        decomposition=[],
        rewritten_queries_bm25=[query],
        rewritten_queries_dense=[query],
        history_resolved=history_resolved,
        degradation_reason="regex_fallback",
    )


def _apply_date_version_degradation(spec: QuerySpec) -> QuerySpec:
    if spec.intent == "date_version" and not has_effective_dates_in_corpus(
        spec.entities_resolved
    ):
        spec.intent = "semantic"
        spec.degradation_reason = "no_effective_dates"
        msg = "версия не определена, нужна дата"
        if msg not in spec.assumptions:
            spec.assumptions.append(msg)
    return spec


def analyze_query(
    query: str,
    event_date: str | None = None,
    chat_history: list[dict] | None = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
    timeout: float | None = None,
) -> tuple[QuerySpec, TraceStep, bool]:
    start_time = time.perf_counter()
    used_history = False
    effective_query = query

    if config.ENABLE_MEMORY and chat_history:
        effective_query, used_history = resolve_history(query, chat_history)

    max_attempts = max(1, int(config.MAX_ATTEMPTS_QUERY_ANALYSIS))
    retry_feedback = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    last_format_error: Optional[Exception] = None
    last_api_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        messages = [
            {"role": config.ROLE_SYSTEM, "content": qa_prompts.QUERY_ANALYSIS_SYSTEM},
            {
                "role": config.ROLE_USER,
                "content": qa_prompts.QUERY_ANALYSIS_USER.format(
                    query=effective_query,
                    event_date=event_date or "не указана",
                    retry_feedback=retry_feedback,
                ),
            },
        ]
        try:
            raw_output, p_tok, c_tok = llm_complete_with_usage(
                messages=messages,
                model=config.LLM_MODEL,
                temperature=qa_prompts.QUERY_ANALYSIS_TEMPERATURE,
                max_tokens=qa_prompts.QUERY_ANALYSIS_MAX_TOKENS,
                timeout=timeout,
            )
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            if on_llm_success:
                on_llm_success()

            data = _extract_json_dict(raw_output)
            try:
                parsed = QueryAnalysisSchema.model_validate(data)
            except ValidationError as val_exc:
                raise LLMFormatError(f"QueryAnalysis validation error: {val_exc}") from val_exc

            bm25_queries = _ensure_exact_tokens_in_bm25(
                effective_query, list(parsed.rewritten_queries_bm25)
            )
            entities_resolved = resolve_entities(parsed.entities_raw)
            required_slots = (
                list(parsed.required_slots)
                if parsed.required_slots
                else _infer_slots_by_regex(effective_query)
            )
            effective_event_date = event_date or parsed.event_date or (
                parsed.dates[0] if parsed.dates else None
            )

            spec = QuerySpec(
                intent=parsed.intent,
                task_type=parsed.task_type,
                entities_raw=list(parsed.entities_raw),
                entities_resolved=entities_resolved,
                article_refs=list(parsed.article_refs),
                dates=list(parsed.dates),
                event_date=effective_event_date,
                premises=[
                    Premise(text=p.text, kind=p.kind, value=p.value, verified=False)
                    for p in parsed.premises
                ],
                required_slots=required_slots,
                assumptions=list(parsed.assumptions),
                ambiguity=parsed.ambiguity,
                needs_clarification=False,
                decomposition=[
                    SubQuestion(
                        text=sq.text,
                        target_doc_raw=sq.target_doc_raw,
                        aspect=sq.aspect,
                    )
                    for sq in parsed.decomposition
                ],
                rewritten_queries_bm25=bm25_queries,
                rewritten_queries_dense=list(parsed.rewritten_queries_dense),
                history_resolved=used_history,
                degradation_reason=None,
            )
            spec = _apply_date_version_degradation(spec)

            latency_ms = round(
                (time.perf_counter() - start_time) * config.MS_PER_SECOND,
                config.LATENCY_ROUND_DIGITS,
            )
            step = TraceStep(
                step="query_analysis",
                input={
                    "query": effective_query,
                    "event_date": event_date,
                    "attempts": attempt,
                },
                output={
                    "intent": spec.intent,
                    "task_type": spec.task_type.value,
                    "required_slots": spec.required_slots,
                    "event_date": spec.event_date,
                    "degradation_reason": spec.degradation_reason,
                },
                latency_ms=latency_ms,
                model=config.LLM_MODEL,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                stage_idx=1,
                degradation=spec.degradation_reason,
            )
            return spec, step, used_history

        except LLMFormatError as exc:
            last_format_error = exc
            if on_failure:
                on_failure(exc)
            retry_feedback = (
                f"\nПредыдущий ответ содержал ошибку формата JSON: {exc}. "
                "Верни строго валидный JSON по схеме."
            )
        except LLMError as exc:
            last_api_error = exc
            if on_failure:
                on_failure(exc)

    if last_api_error is not None and last_format_error is None:
        raise last_api_error

    fallback_spec = _build_regex_fallback_spec(
        query=effective_query,
        event_date=event_date,
        history_resolved=used_history,
    )
    fallback_spec = _apply_date_version_degradation(fallback_spec)
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step="query_analysis",
        input={
            "query": effective_query,
            "event_date": event_date,
            "attempts": max_attempts,
        },
        output={
            "intent": fallback_spec.intent,
            "task_type": fallback_spec.task_type.value,
            "required_slots": fallback_spec.required_slots,
            "event_date": fallback_spec.event_date,
            "degradation_reason": fallback_spec.degradation_reason,
        },
        latency_ms=latency_ms,
        model=config.LLM_MODEL,
        prompt_tokens=total_prompt_tokens or None,
        completion_tokens=total_completion_tokens or None,
        stage_idx=1,
        degradation=fallback_spec.degradation_reason,
    )
    return fallback_spec, step, used_history