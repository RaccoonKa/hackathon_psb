import logging
import re
import time
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

    refused = bool(draft.refusal_reason) or not bool(cleaned_answer.strip())
    filtered_sources = (
        {}
        if refused
        else {sid: sources[sid] for sid in cited_ids if sid in sources}
    )

    if refused:
        confidence = "low"
    elif (
        degradation_reasons
        or draft.missing_info
        or (verification and verification.final_verdict == VerificationVerdict.PARTIAL)
    ):
        confidence = "medium"
    else:
        confidence = "high"

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
            return refusal.build_cannot_parse_result(
                query=query,
                prior_trace=trace,
                partial_spec=partial_spec,
                explanation_text=explanation,
                degradation_reasons=degradation_reasons,
            )
        return refusal.build_llm_unavailable_result(
            query=query,
            prior_trace=trace,
            degradation_reasons=degradation_reasons,
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
        accumulated_chunks = []
        bundle = None
        assessment = None

        while True:
            last_stage = "retrieve"
            failures_step = 0
            try:
                new_chunks, ret_step = bridge.retrieve_tasks(tasks=tasks)
                _append_step(ret_step, stage_idx=3)
            except OrchestratorError as exc:
                _record_failure(exc)
                degradation_reasons.append("retrieval_error")
                new_chunks = []

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
                assessment.sufficiency == Sufficiency.INSUFFICIENT
                and followup_cycles < int(config.MAX_FOLLOWUP_RETRIEVAL_CYCLES)
                and not _budget_or_deadline_exceeded()
            ):
                followup_cycles += 1
                tasks, followup_step = planner.followup(
                    missing_slots=assessment.missing_slots,
                    spec=spec,
                    user_selected_doc_ids=selected_doc_ids,
                )
                _append_step(followup_step, stage_idx=2)
                continue
            break

        if assessment is None or bundle is None or assessment.sufficiency == Sufficiency.INSUFFICIENT:
            return refusal.build_no_context_refusal_result(
                query=query,
                intent=spec.intent,
                prior_trace=trace,
                refusal_reason=assessment.reason if assessment else "insufficient_evidence",
                query_spec=spec,
                missing_info=assessment.missing_slots if assessment else spec.required_slots,
                degradation_reasons=degradation_reasons,
                used_history=used_history,
            )

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