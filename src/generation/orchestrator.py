import inspect
import logging
import re
import time
import traceback
from typing import Any
from src import config
from src.generation import bailout, refusal
from src.generation.analysis.prefilter import run_prefilter
from src.generation.analysis.query_analysis import analyze_query
from src.generation.answer.context_builder import build_context_and_sources
from src.generation.answer.generator import generate_draft, regenerate_draft
from src.generation.answer.verification import verify
from src.generation.errors import OrchestratorError
from src.generation.planning import planner
from src.generation.policy.precheck import precheck
from src.generation.prompts import templates
from src.generation.retrieval import bridge, evidence
from src.generation.types import (
    AgentResult,
    BailoutContext,
    DraftAnswer,
    Intent,
    QuerySpec,
    Sufficiency,
    TraceStep,
    VerificationReport,
    VerificationVerdict,
)

logger = logging.getLogger(__name__)

_SOURCE_TAG_PATTERN = re.compile(r"<source>(.*?)</source>", re.DOTALL)
_NULL_LIKE_REFUSAL = {"null", "none", "nil", "false", "нет", "n/a", ""}

# Редакция trace: ключи, токены, абсолютные пути вне проекта не выводятся.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), "<redacted-key>"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+"), r"\1<redacted>"),
    (re.compile(r"(?i)(authorization\s*[=:]\s*)[^\s,;]+"), r"\1<redacted>"),
)
_ABS_PATH_RE = re.compile(
    r"/(?:Users|home|root|private|var|tmp|opt|etc)/[^\s:'\",;)]+"
)
_SEARCH_FIELDS: tuple[str, ...] = (
    "selected_doc_ids",
    "allowed_chunk_ids_count",
    "bm25_candidates_count",
    "dense_candidates_count",
    "candidates_before_filters",
    "candidates_after_date_filter",
    "candidates_after_document_filter",
    "final_chunks_count",
    "validity_unknown_count",
)
_EMBEDDING_FIELDS: tuple[str, ...] = (
    "provider",
    "model",
    "dimension",
    "success",
    "fallback_used",
    "error_type",
)


def _trace_level() -> str:
    return str(getattr(config, "TRACE_DETAIL_LEVEL", "basic") or "basic")


def _trace_detailed() -> bool:
    return _trace_level() == "detailed"


def _trace_diagnostics_enabled() -> bool:
    return bool(getattr(config, "TRACE_INCLUDE_DIAGNOSTICS", True)) and (
        _trace_level() != "off"
    )


def _max_error_chars() -> int:
    return int(getattr(config, "TRACE_MAX_ERROR_CHARS", 300))


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _redact_text(value: Any) -> str:
    """Удаляет секреты и абсолютные пути вне проекта из диагностики."""
    text = str(value or "")
    base_dir = str(getattr(config, "BASE_DIR", "") or "")
    if base_dir:
        text = text.replace(base_dir + "/", "").replace(base_dir, ".")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    text = _ABS_PATH_RE.sub("<path>", text)
    return text


def _safe_message(value: Any, limit: int | None = None) -> str:
    return _truncate(
        _redact_text(value), limit if limit is not None else _max_error_chars()
    )


def _prepare_task_diag(entry: dict[str, Any]) -> dict[str, Any]:
    search = entry.get("search") or {}
    embeddings = entry.get("embeddings") or {}
    return {
        "task_id": entry.get("task_id"),
        "mode": entry.get("mode"),
        "purpose": entry.get("purpose"),
        "query": _safe_message(entry.get("query"), 200),
        "status": entry.get("status"),
        "duration_ms": entry.get("duration_ms", 0),
        "error_type": entry.get("error_type"),
        "error_message": (
            _safe_message(entry.get("error_message"))
            if entry.get("error_message")
            else None
        ),
        "search": {
            key: search.get(key, [] if key == "selected_doc_ids" else 0)
            for key in _SEARCH_FIELDS
        },
        "embeddings": {key: embeddings.get(key) for key in _EMBEDDING_FIELDS},
    }


def _prepare_errors(errors: list[dict[str, Any]]) -> list[dict[str, str]]:
    limit = int(getattr(config, "TRACE_MAX_LIST_ITEMS", 20))
    prepared: list[dict[str, str]] = []
    for error in errors[:limit]:
        prepared.append(
            {
                "type": str(error.get("type", "UnknownError")),
                "message": _safe_message(error.get("message")),
            }
        )
    return prepared


def _failed_tasks(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    failed_statuses = {"error", "timeout", "cancelled"}
    result: list[dict[str, Any]] = []
    for entry in diagnostics.get("tasks") or []:
        if entry.get("status") in failed_statuses:
            result.append(
                {
                    "task_id": entry.get("task_id"),
                    "mode": entry.get("mode"),
                    "purpose": entry.get("purpose"),
                    "status": entry.get("status"),
                    "error_type": entry.get("error_type"),
                }
            )
    return result


def _enrich_retrieval_step(
    step: TraceStep, diagnostics: dict[str, Any], attempt: int
) -> TraceStep:
    """Дополняет шаг retrieval диагностикой из ``diagnostics_out``."""
    executor = dict(diagnostics.get("executor") or {})
    errors = list(diagnostics.get("errors") or [])
    degraded = bool(diagnostics.get("degraded"))
    failed = _failed_tasks(diagnostics)

    step.input["attempt"] = attempt
    step.output["executor"] = executor
    step.output["degraded"] = degraded
    step.output["status"] = "partial" if degraded else "ok"
    if failed:
        step.output["failed_tasks"] = failed
    if _trace_diagnostics_enabled():
        max_tasks = int(getattr(config, "TRACE_MAX_TASKS", 20))
        step.output["errors"] = _prepare_errors(errors)
        step.output["tasks"] = [
            _prepare_task_diag(entry)
            for entry in (diagnostics.get("tasks") or [])[:max_tasks]
        ]
    step.extra.update(
        {
            "executor": executor,
            "errors_count": len(errors),
            "degraded": degraded,
            "attempt": attempt,
            "status": step.output["status"],
        }
    )
    if degraded:
        step.degradation = "retrieval_partial_failure"
    return step


def _build_retrieval_error_step(
    exc: Exception,
    traceback_text: str,
    diagnostics: dict[str, Any],
    attempt: int,
    tasks: list[Any],
) -> TraceStep:
    """Шаг ``rag_retrieval`` со статусом error вместо маскировки пустым списком."""
    executor = dict(diagnostics.get("executor") or {})
    errors = list(diagnostics.get("errors") or [])
    error_type = type(exc).__name__
    tb_limit = (
        int(getattr(config, "TRACE_MAX_TRACEBACK_CHARS", 4000))
        if _trace_detailed()
        else min(800, int(getattr(config, "TRACE_MAX_TRACEBACK_CHARS", 4000)))
    )
    output: dict[str, Any] = {
        "status": "error",
        "chunk_ids": [],
        "scores": [],
        "count": 0,
        "executor": executor,
        "degraded": True,
        "degradation_reason": "retrieval_error",
        "error_type": error_type,
        "error_message": _safe_message(str(exc)),
        "error_types": sorted(
            {
                str(error.get("type", "UnknownError"))
                for error in errors
            }
            or {error_type}
        ),
        "attempt": attempt,
    }
    if _trace_diagnostics_enabled():
        output["error_traceback"] = _truncate(
            _redact_text(traceback_text), tb_limit
        )
        output["errors"] = _prepare_errors(errors)
        max_tasks = int(getattr(config, "TRACE_MAX_TASKS", 20))
        output["tasks"] = [
            _prepare_task_diag(entry)
            for entry in (diagnostics.get("tasks") or [])[:max_tasks]
        ]
    return TraceStep(
        step=config.STEP_RAG_RETRIEVAL,
        input={
            "tasks_count": len(tasks),
            "queries": [t.query for t in tasks[:3]],
            "attempt": attempt,
        },
        output=output,
        latency_ms=0.0,
        stage_idx=3,
        degradation="retrieval_error",
        extra={
            "executor": executor,
            "errors_count": len(errors),
            "degraded": True,
            "attempt": attempt,
            "status": "error",
            "error_type": error_type,
        },
    )


def _sanitize_and_extract_citations(
    answer: str,
    available_sources: dict[str, dict[str, Any]],
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


def _summarize_retrieval_trace(trace: list[TraceStep]) -> dict[str, Any]:
    """Агрегирует статусы/счётчики всех попыток retrieval по trace."""
    summary: dict[str, Any] = {
        "attempts": 0,
        "attempts_ok": 0,
        "attempts_partial": 0,
        "attempts_error": 0,
        "tasks_submitted": 0,
        "tasks_completed": 0,
        "tasks_failed": 0,
        "tasks_cancelled": 0,
        "tasks_timed_out": 0,
    }
    for step in trace:
        if step.step != config.STEP_RAG_RETRIEVAL:
            continue
        summary["attempts"] += 1
        status = str(step.output.get("status") or "ok")
        if status == "error":
            summary["attempts_error"] += 1
        elif status == "partial" or step.output.get("degraded"):
            summary["attempts_partial"] += 1
        else:
            summary["attempts_ok"] += 1
        executor = step.output.get("executor") or step.extra.get("executor") or {}
        summary["tasks_submitted"] += int(executor.get("submitted", 0) or 0)
        summary["tasks_completed"] += int(executor.get("completed", 0) or 0)
        summary["tasks_failed"] += int(executor.get("failed", 0) or 0)
        summary["tasks_cancelled"] += int(executor.get("cancelled", 0) or 0)
        summary["tasks_timed_out"] += int(executor.get("timed_out", 0) or 0)
    return summary


def _compute_confidence_and_reasons(
    refused: bool,
    degradation_reasons: list[str],
    missing_info: list[str],
    verification: VerificationReport | None,
) -> tuple[str, list[str]]:
    if refused:
        return "low", ["refused"]

    reasons: list[str] = []
    for deg in degradation_reasons:
        reasons.append(f"degradation:{deg}")
    if missing_info:
        reasons.append(f"missing_info:{len(missing_info)}")
    if verification is not None:
        if verification.citation_errors:
            reasons.append(f"citation_errors:{len(verification.citation_errors)}")
        if verification.injection_signals:
            reasons.append(f"injection_signals:{len(verification.injection_signals)}")
        if verification.final_verdict == VerificationVerdict.REFUSE:
            reasons.append("verification_verdict:refuse")

    if reasons:
        return "medium", reasons
    return "high", ["all_conditions_met"]


def assemble_final(
    draft: DraftAnswer,
    sources: dict[str, dict[str, Any]],
    spec: QuerySpec,
    verification: VerificationReport | None,
    trace: list[TraceStep],
    degradation_reasons: list[str],
    used_history: bool,
    failures_global: int,
) -> AgentResult:
    start_time = time.perf_counter()
    cleaned_answer, cited_ids = _sanitize_and_extract_citations(
        answer=draft.answer_text,
        available_sources=sources,
    )

    for claim in draft.claims:
        for sid in claim.source_ids:
            if sid in sources and sid not in cited_ids:
                cited_ids.append(sid)

    if draft.refusal_reason is not None:
        if (
            draft.refusal_reason.strip().lower() in _NULL_LIKE_REFUSAL
            or bool(cited_ids)
            or (verification is not None and bool(verification.supported_locked))
        ):
            draft.refusal_reason = None

    if (
        verification is not None
        and verification.final_verdict == VerificationVerdict.ACCEPT
        and draft.missing_info
    ):
        supported_slots = {
            c.slot
            for c in draft.claims
            if c.original_idx in verification.supported_locked and c.slot
        }
        if all(s in supported_slots for s in spec.required_slots if s != "general"):
            draft.missing_info = []

    refused = bool(draft.refusal_reason) or not bool(cleaned_answer.strip())
    filtered_sources = (
        {}
        if refused
        else {sid: sources[sid] for sid in cited_ids if sid in sources}
    )

    confidence, confidence_reasons = _compute_confidence_and_reasons(
        refused=refused,
        degradation_reasons=degradation_reasons,
        missing_info=draft.missing_info,
        verification=verification,
    )

    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    assembly_step = TraceStep(
        step=config.STEP_RESPONSE_ASSEMBLY,
        input={
            "intent": spec.intent,
            "source_ids": list(sources.keys()),
            "answer_length": len(draft.answer_text),
        },
        output={
            "refused": refused,
            "sources_count": len(filtered_sources),
            "cited_source_ids": cited_ids,
            "confidence": confidence,
            "confidence_reasons": confidence_reasons,
            "degradation_reasons": list(degradation_reasons),
            "retrieval_summary": _summarize_retrieval_trace(trace),
        },
        latency_ms=latency_ms,
        stage_idx=8,
        failures_global=failures_global,
        failures_step=0,
    )
    full_trace = [*trace, assembly_step]

    return AgentResult(
        final_answer=cleaned_answer,
        sources=filtered_sources,
        intent=spec.intent,
        refused=refused,
        trace=full_trace,
        refusal_reason=draft.refusal_reason,
        confidence=confidence,
        assumptions=list(draft.assumptions),
        missing_info=list(draft.missing_info),
        verification=verification,
        query_spec=spec,
        degradation_reasons=list(degradation_reasons),
        used_history=used_history,
    )


def _safe_no_context_refusal(
    query: str,
    intent_val: str,
    trace: list[TraceStep],
    refusal_reason: str,
    spec: QuerySpec | None,
    missing_info: list[str] | None,
    degradation_reasons: list[str],
    used_history: bool,
) -> AgentResult:
    fn = getattr(refusal, "build_no_context_refusal_result", None)
    if callable(fn):
        sig = inspect.signature(fn)
        if "refusal_reason" in sig.parameters:
            return fn(
                query=query,
                intent=intent_val,
                prior_trace=trace,
                refusal_reason=refusal_reason,
                query_spec=spec,
                missing_info=missing_info,
                degradation_reasons=degradation_reasons,
                used_history=used_history,
            )
    return AgentResult(
        final_answer=templates.RESPONSE_REFUSAL_NO_CONTEXT,
        sources={},
        intent=intent_val,
        refused=True,
        trace=trace,
        refusal_reason=refusal_reason,
        confidence="low",
        assumptions=list(spec.assumptions) if spec else [],
        missing_info=list(missing_info or []),
        query_spec=spec,
        degradation_reasons=list(degradation_reasons),
        used_history=used_history,
    )


def run_agent(
    query: str,
    event_date: str | None = None,
    selected_doc_ids: set[str] | None = None,
    chat_history: list[dict] | None = None,
) -> AgentResult:
    deadline = time.monotonic() + float(config.REQUEST_TOTAL_TIMEOUT_S)
    trace: list[TraceStep] = []
    degradation_reasons: list[str] = []
    failures_global = 0
    failures_step = 0
    llm_was_available = False
    last_stage = "init"
    last_error: str | None = None
    partial_spec: QuerySpec | None = None
    used_history = False

    def _remaining_time() -> float:
        return deadline - time.monotonic()

    def _llm_timeout() -> float:
        return max(1.0, min(float(config.LLM_REQUEST_TIMEOUT_S), _remaining_time()))

    def _record_failure(exc: Exception) -> None:
        nonlocal failures_global, failures_step, last_error
        failures_global += 1
        failures_step += 1
        last_error = f"{type(exc).__name__}: {exc}"

    def _record_llm_success() -> None:
        nonlocal llm_was_available
        llm_was_available = True

    def _append_step(step: TraceStep, stage_idx: int) -> None:
        step.stage_idx = stage_idx
        step.failures_global = failures_global
        step.failures_step = failures_step
        if step.degradation and step.degradation not in degradation_reasons:
            degradation_reasons.append(step.degradation)
        trace.append(step)

    def _trigger_bailout() -> AgentResult:
        try:
            ctx = BailoutContext(
                query=query,
                failures_total=failures_global,
                last_stage=last_stage,
                last_error=last_error,
                partial_spec=partial_spec,
                llm_was_available=llm_was_available,
            )
            if partial_spec is not None and llm_was_available:
                rem = _remaining_time()
                explanation = bailout.explain(
                    ctx, timeout=min(float(config.LLM_REQUEST_TIMEOUT_S), rem)
                )
                fn_cp = getattr(refusal, "build_cannot_parse_result", None)
                if callable(fn_cp):
                    return fn_cp(
                        query=query,
                        prior_trace=trace,
                        partial_spec=partial_spec,
                        explanation_text=explanation,
                        degradation_reasons=degradation_reasons,
                    )
                return AgentResult(
                    final_answer=explanation or templates.RESPONSE_CANNOT_PARSE,
                    sources={},
                    intent=partial_spec.intent,
                    refused=True,
                    trace=trace,
                    refusal_reason="cannot_parse",
                    confidence="low",
                    query_spec=partial_spec,
                    degradation_reasons=list(degradation_reasons),
                )
            fn_unavail = getattr(refusal, "build_llm_unavailable_result", None)
            if callable(fn_unavail):
                return fn_unavail(
                    query=query,
                    prior_trace=trace,
                    degradation_reasons=degradation_reasons,
                )
        except Exception as bail_exc:
            logger.warning("Fallback inside _trigger_bailout: %s", bail_exc)

        return AgentResult(
            final_answer=templates.RESPONSE_LLM_UNAVAILABLE,
            sources={},
            intent=partial_spec.intent if partial_spec else "semantic",
            refused=True,
            trace=trace,
            refusal_reason="llm_unavailable",
            confidence="low",
            query_spec=partial_spec,
            degradation_reasons=list(degradation_reasons or ["llm_unavailable"]),
        )

    def _budget_or_deadline_exceeded() -> bool:
        if time.monotonic() > deadline:
            nonlocal last_error
            last_error = "request_total_timeout_exceeded"
            degradation_reasons.append("deadline_exceeded")
            return True
        if failures_global >= int(config.MAX_LLM_FAILURES_PER_REQUEST):
            degradation_reasons.append("max_failures_per_request_exceeded")
            return True
        return False

    try:
        last_stage = "prefilter"
        failures_step = 0
        pf_result, pf_step = run_prefilter(query=query)
        _append_step(pf_step, stage_idx=0)
        if pf_result.triggered:
            mapped_intent = (
                Intent.ABOUT_ASSISTANT
                if pf_result.reason == "about_assistant"
                else Intent.OUT_OF_SCOPE
            )
            return refusal.build_template_intent_result(
                query=query,
                intent=mapped_intent,
                prior_trace=trace,
                template_key=pf_result.template_key,
                reason=pf_result.reason,
            )

        if _budget_or_deadline_exceeded():
            return _trigger_bailout()

        last_stage = "query_analysis"
        failures_step = 0
        try:
            spec, qa_step, used_history = analyze_query(
                query=query,
                event_date=event_date,
                chat_history=chat_history,
                on_failure=_record_failure,
                on_llm_success=_record_llm_success,
                timeout=_llm_timeout(),
            )
            partial_spec = spec
            # Сущности и слоты без сырых ответов LLM (секреты в trace не нужны).
            qa_step.output["entities"] = [
                {
                    "raw": ref.raw,
                    "document_id": ref.document_id,
                    "resolved": ref.resolved,
                }
                for ref in spec.entities_resolved
            ]
            qa_step.output["entities_raw"] = list(spec.entities_raw)
            _append_step(qa_step, stage_idx=1)
        except OrchestratorError as exc:
            last_error = str(exc)
            return _trigger_bailout()

        if _budget_or_deadline_exceeded():
            return _trigger_bailout()

        last_stage = "plan_retrieval"
        failures_step = 0
        tasks, plan_step = planner.plan(
            spec=spec, user_selected_doc_ids=selected_doc_ids
        )
        _append_step(plan_step, stage_idx=2)

        if _budget_or_deadline_exceeded():
            return _trigger_bailout()

        followup_cycles = 0
        accumulated_chunks: list[Any] = []
        bundle = None
        assessment = None
        retrieval_attempt = 0
        retrieval_attempt_log: list[dict[str, Any]] = []
        technical_retrieval_failure = False
        executed_queries: set[str] = {t.query for t in tasks}

        while True:
            last_stage = "retrieve"
            failures_step = 0
            retrieval_attempt += 1
            diagnostics: dict[str, Any] = {}
            try:
                new_chunks, ret_step = bridge.retrieve_tasks(
                    tasks=tasks, diagnostics_out=diagnostics
                )
                _enrich_retrieval_step(ret_step, diagnostics, retrieval_attempt)
                _append_step(ret_step, stage_idx=3)
            except Exception as exc:  # noqa: BLE001 - изолируем техсбой retrieval
                _record_failure(exc)
                # Ошибка НЕ маскируется пустым результатом: шаг rag_retrieval
                # попадает в trace со status=error и диагностикой.
                err_step = _build_retrieval_error_step(
                    exc=exc,
                    traceback_text=traceback.format_exc(limit=20),
                    diagnostics=diagnostics,
                    attempt=retrieval_attempt,
                    tasks=tasks,
                )
                _append_step(err_step, stage_idx=3)
                technical_retrieval_failure = True
                retrieval_attempt_log.append(
                    {
                        "attempt": retrieval_attempt,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "executor": dict(diagnostics.get("executor") or {}),
                    }
                )
                new_chunks = []
                logger.warning(
                    "Retrieval attempt %d failed: %s",
                    retrieval_attempt,
                    type(exc).__name__,
                )
            else:
                partial = bool(diagnostics.get("degraded"))
                if partial:
                    technical_retrieval_failure = True
                retrieval_attempt_log.append(
                    {
                        "attempt": retrieval_attempt,
                        "status": "partial" if partial else "ok",
                        "error_type": None,
                        "executor": dict(diagnostics.get("executor") or {}),
                    }
                )

            # Реранкинг — отдельный этап после объединения задач: внутри
            # hybrid_search он отключён (rerank=False), поэтому локальный
            # Qwen-reranker применяется здесь, до оценки доказательств.
            if new_chunks:
                before_count = len(new_chunks)
                try:
                    new_chunks, rerank_step = bridge.rerank_chunks(
                        chunks=new_chunks,
                        query=query,
                        top_n=int(config.RERANKER_TOP_N),
                    )
                    rerank_step.output["before_count"] = before_count
                    rerank_step.output["after_count"] = len(new_chunks)
                    rerank_step.output["top_n"] = int(config.RERANKER_TOP_N)
                    _append_step(rerank_step, stage_idx=3)
                except Exception as exc:  # noqa: BLE001 - реранкер необязателен
                    _record_failure(exc)
                    degradation_reasons.append("rerank_error")
                    logger.warning(
                        "Reranking failed, using retrieval order: %s", exc
                    )

            by_cid = {c.chunk_id: c for c in accumulated_chunks}
            for ch in new_chunks:
                if ch.chunk_id not in by_cid or ch.score > by_cid[ch.chunk_id].score:
                    by_cid[ch.chunk_id] = ch
            accumulated_chunks = sorted(
                by_cid.values(), key=lambda c: c.score, reverse=True
            )

            if _budget_or_deadline_exceeded():
                return _trigger_bailout()

            last_stage = "assess_evidence"
            failures_step = 0
            bundle, assessment, ev_step = evidence.assess(
                spec=spec,
                chunks=accumulated_chunks,
                on_failure=_record_failure,
                on_llm_success=_record_llm_success,
                timeout=_llm_timeout(),
            )
            _append_step(ev_step, stage_idx=4)

            if (
                (
                    assessment.sufficiency == Sufficiency.INSUFFICIENT
                    or bool(assessment.missing_slots)
                )
                and followup_cycles < int(config.MAX_FOLLOWUP_RETRIEVAL_CYCLES)
                and not _budget_or_deadline_exceeded()
            ):
                followup_cycles += 1
                tasks, followup_step = planner.followup(
                    missing_slots=assessment.missing_slots,
                    spec=spec,
                    user_selected_doc_ids=selected_doc_ids,
                    existing_queries=executed_queries,
                )
                followup_step.output["followup_cycle"] = followup_cycles
                _append_step(followup_step, stage_idx=2)
                executed_queries.update(t.query for t in tasks)
                continue
            break

        evidence_available = bool(bundle is not None and bundle.chunks)
        evidence_sufficient = (
            evidence_available
            and assessment is not None
            and assessment.sufficiency != Sufficiency.INSUFFICIENT
        )

        # Технический сбой поиска — отдельное сообщение, не «нет сведений».
        # Если в другой попытке доказательств достаточно — отвечаем по ним,
        # деградация остаётся видимой в trace.
        if technical_retrieval_failure and not evidence_sufficient:
            error_types = [
                str(step.output.get("error_type"))
                for step in trace
                if step.step == config.STEP_RAG_RETRIEVAL
                and step.output.get("error_type")
            ]
            if not error_types:
                error_types = ["PartialRetrievalFailure"]
            return refusal.build_retrieval_error_result(
                query=query,
                prior_trace=trace,
                error_types=error_types,
                attempts=retrieval_attempt_log,
                query_spec=spec,
                missing_info=(
                    assessment.missing_slots
                    if assessment
                    else spec.required_slots
                ),
                degradation_reasons=degradation_reasons or ["retrieval_error"],
                used_history=used_history,
            )

        if not evidence_available:
            # Пустая выдача retrieval без исключения — это отдельный случай
            # «документы не найдены», а не технический сбой и не частичное
            # покрытие слотов.
            no_context_reason = (
                "no_relevant_documents"
                if not accumulated_chunks
                else (
                    assessment.reason
                    if assessment
                    else "insufficient_evidence"
                )
            )
            return _safe_no_context_refusal(
                query=query,
                intent_val=spec.intent,
                trace=trace,
                refusal_reason=no_context_reason,
                spec=spec,
                missing_info=assessment.missing_slots if assessment else spec.required_slots,
                degradation_reasons=degradation_reasons,
                used_history=used_history,
            )

        if assessment is not None and assessment.sufficiency == Sufficiency.INSUFFICIENT:
            assessment.sufficiency = Sufficiency.PARTIAL
            assessment.core_slot_covered = True

        if _budget_or_deadline_exceeded():
            return _trigger_bailout()

        last_stage = "policy_precheck"
        failures_step = 0
        policy_verdict, policy_step = precheck(
            spec=spec, bundle=bundle, assessment=assessment
        )
        _append_step(policy_step, stage_idx=5)

        sources, context_str, ctx_step = build_context_and_sources(
            chunks=bundle.chunks
        )
        _append_step(ctx_step, stage_idx=5)

        if _budget_or_deadline_exceeded():
            return _trigger_bailout()

        last_stage = "generate_draft"
        failures_step = 0
        try:
            draft, gen_step = generate_draft(
                spec=spec,
                evidence=bundle,
                policy=policy_verdict,
                context=context_str,
                query=query,
                on_failure=_record_failure,
                on_llm_success=_record_llm_success,
                timeout=_llm_timeout(),
            )
            _append_step(gen_step, stage_idx=6)
        except OrchestratorError as exc:
            last_error = str(exc)
            return _trigger_bailout()

        if _budget_or_deadline_exceeded():
            return _trigger_bailout()

        regenerate_cycles = 0
        verification_report: VerificationReport | None = None

        while True:
            last_stage = "verify_draft"
            failures_step = 0
            can_regen = regenerate_cycles < int(config.MAX_REGENERATE_CYCLES)
            verification_report, ver_step = verify(
                draft=draft,
                chunks=bundle.chunks,
                spec=spec,
                can_regenerate=can_regen,
                on_failure=_record_failure,
                on_llm_success=_record_llm_success,
                timeout=_llm_timeout(),
            )
            _append_step(ver_step, stage_idx=7)

            if (
                verification_report.final_verdict == VerificationVerdict.REGENERATE
                and can_regen
                and not _budget_or_deadline_exceeded()
            ):
                regenerate_cycles += 1
                last_stage = "regenerate_draft"
                failures_step = 0
                draft, regen_step = regenerate_draft(
                    previous=draft,
                    report=verification_report,
                    chunks=bundle.chunks,
                    spec=spec,
                    on_failure=_record_failure,
                    on_llm_success=_record_llm_success,
                    timeout=_llm_timeout(),
                )
                _append_step(regen_step, stage_idx=6)
                continue
            break

        if (
            verification_report is not None
            and verification_report.final_verdict == VerificationVerdict.REFUSE
            and not policy_verdict.refusal_forbidden
            and bool(draft.refusal_reason)
        ):
            return refusal.build_verification_refusal_result(
                query=query,
                spec=spec,
                verification=verification_report,
                prior_trace=trace,
                degradation_reasons=degradation_reasons,
                used_history=used_history,
            )

        last_stage = "assemble_final"
        return assemble_final(
            draft=draft,
            sources=sources,
            spec=spec,
            verification=verification_report,
            trace=trace,
            degradation_reasons=degradation_reasons,
            used_history=used_history,
            failures_global=failures_global,
        )

    except Exception as exc:
        logger.exception("Unhandled error in run_agent: %s", exc)
        _record_failure(exc)
        return _trigger_bailout()