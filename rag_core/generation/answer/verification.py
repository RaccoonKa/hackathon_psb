import json
import logging
import re
import time
from typing import Callable, Optional
from pydantic import ValidationError
from rag_core import config
from rag_core.generation.analysis.source_kind import (
    SOURCE_KIND_UNKNOWN,
    expected_source_kind_for_slot,
)
from rag_core.generation.errors import LLMError, LLMFormatError
from rag_core.generation.llm.client import llm_complete_with_usage
from rag_core.generation.prompts import verifier as ver_prompts
from rag_core.generation.types import (
    AnswerToQuestionCheck,
    AnswerToQuestionVerdict,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    DraftAnswer,
    PolicyVerdict,
    PremiseStatus,
    QuerySpec,
    RetrievedChunk,
    TraceStep,
    VerificationBatchSchema,
    VerificationReport,
    VerificationVerdict,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")
_SOURCE_TAG_RE = re.compile(r"<source>(.*?)</source>", re.DOTALL)
_SOURCE_TAG_ANY_RE = re.compile(r"</?source>", re.IGNORECASE)


def _parse_verifier_output(raw_text: str) -> VerificationBatchSchema:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, list):
            return VerificationBatchSchema.model_validate({"verdicts": loaded})
        if isinstance(loaded, dict):
            return VerificationBatchSchema.model_validate(loaded)
    except Exception:
        pass

    match = _JSON_BLOCK_RE.search(raw_text)
    if match:
        try:
            return VerificationBatchSchema.model_validate_json(match.group(0))
        except ValidationError as exc:
            raise LLMFormatError(f"Verifier validation error: {exc}") from exc
        except Exception as exc:
            raise LLMFormatError(f"Verifier JSON error: {exc}") from exc

    raise LLMFormatError("No valid JSON found in verifier output")


def _partition_core_and_non_core_claims(
    claims: list[Claim],
    required_slots: list[str],
    status_by_idx: dict[int, ClaimStatus],
) -> tuple[list[Claim], list[Claim]]:
    # Первый подтвержденный claim для каждого required_slot фиксируется как core;
    # вторичные claims с интерпретациями на том же слоте считаются non-core.
    req_set = set(required_slots) if required_slots else set()
    core_claims: list[Claim] = []
    non_core_claims: list[Claim] = []
    secured_slots: set[str] = set()

    for idx, c in enumerate(claims):
        orig_idx = c.original_idx if c.original_idx is not None else idx
        st = status_by_idx.get(orig_idx, ClaimStatus.UNVERIFIED)
        if c.slot in req_set and st == ClaimStatus.SUPPORTED and c.slot not in secured_slots:
            core_claims.append(c)
            secured_slots.add(c.slot)

    for idx, c in enumerate(claims):
        if c in core_claims:
            continue
        if c.slot in req_set and c.slot not in secured_slots:
            core_claims.append(c)
        else:
            non_core_claims.append(c)

    if not core_claims and claims:
        return [claims[0]], list(claims[1:])

    return core_claims, non_core_claims


def _extract_tag_source_ids(text_fragment: str) -> list[str]:
    """Извлекает source_id из тегов ``<source>S1</source>`` фрагмента."""
    ids: list[str] = []
    prefix = re.escape(config.SOURCE_ID_PREFIX)
    pattern = re.compile(rf"{prefix}\d+")
    for inner in _SOURCE_TAG_RE.findall(text_fragment or ""):
        for sid in pattern.findall(inner):
            if sid not in ids:
                ids.append(sid)
    return ids


def _synthesize_claims_from_answer(answer_text: str) -> list[Claim]:
    """Строит claims из предложений с тегами <source>...</source>.

    Предложение без тегов источником не подтверждено и claim не порождает.
    Если тегов нет ни в одном предложении — возвращается пустой список.
    """
    claims: list[Claim] = []
    for sentence in _SENTENCE_SPLIT_RE.split(answer_text or ""):
        source_ids = _extract_tag_source_ids(sentence)
        if not source_ids:
            continue
        clean_text = _SOURCE_TAG_ANY_RE.sub("", sentence).strip()
        clean_text = re.sub(r"\s+", " ", clean_text)
        if not clean_text:
            continue
        claims.append(
            Claim(
                text=clean_text,
                source_ids=source_ids,
                slot=None,
                original_idx=len(claims),
            )
        )
    return claims


def _prune_answer_to_supported_sources(
    answer_text: str,
    allowed_source_ids: set[str],
) -> str:
    """Удаляет предложения/теги, не подкреплённые подтверждёнными claims.

    Предложение без подтверждённых тегов отбрасывается; внутри предложения
    source_id, отсутствующие в ``allowed_source_ids``, удаляются из тегов.
    """

    def _replace_tag(match: re.Match[str]) -> str:
        tokens = [t.strip() for t in match.group(1).split(",") if t.strip()]
        kept_tokens = [t for t in tokens if t in allowed_source_ids]
        return f"<source>{', '.join(kept_tokens)}</source>" if kept_tokens else ""

    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(answer_text or ""):
        if not sentence.strip():
            continue
        tag_ids = _extract_tag_source_ids(sentence)
        if tag_ids and not any(sid in allowed_source_ids for sid in tag_ids):
            continue
        pruned_sentence = (
            _SOURCE_TAG_RE.sub(_replace_tag, sentence)
            if tag_ids
            else sentence
        )
        kept.append(pruned_sentence.strip())
    return " ".join(kept).strip()


def _normalize_claim_text(text: str) -> str:
    """Нормализует текст для технической сверки answer_text ↔ claims."""
    cleaned = _SOURCE_TAG_ANY_RE.sub(" ", text or "").lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Пробел перед знаком пунктуации не должен ломать дословную сверку.
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _check_answer_text_claim_consistency(
    draft: DraftAnswer,
    *,
    require_verbatim_claims: bool = False,
) -> list[str]:
    """Техническая сверка текста ответа со списком проверяемых claims.

    (a) Каждый ``<source>``-тег в ``answer_text`` обязан ссылаться на source_id
        из объединения ``claim.source_ids``.
    (b) Если claims синтезированы из ``answer_text`` (схема, в которой дословное
        присутствие гарантировано построением), текст каждого claim обязан
        присутствовать в нормализованном ``answer_text``. Для LLM-разметки
        claims дословность не гарантируется контрактом схемы, поэтому (b)
        применяется только к синтезированной схеме.
    (c) Пустой ответ при пустых claims обрабатывается отдельно до вызова
        этой проверки и здесь не дублируется.
    """
    issues: list[str] = []
    if not draft.claims:
        return issues

    claim_source_ids: set[str] = set()
    for claim in draft.claims:
        claim_source_ids.update(claim.source_ids)

    for sid in _extract_tag_source_ids(draft.answer_text):
        if sid not in claim_source_ids:
            issues.append(
                f"answer_text: source tag {sid} is not present in claims"
            )

    if require_verbatim_claims:
        normalized_answer = _normalize_claim_text(draft.answer_text)
        for pos, claim in enumerate(draft.claims):
            normalized_claim = _normalize_claim_text(claim.text)
            if normalized_claim and normalized_claim not in normalized_answer:
                issues.append(f"claim {pos}: text is not present in answer_text")

    return issues


def _build_premises_payload(spec: QuerySpec, policy: PolicyVerdict | None) -> list[dict]:
    payload: list[dict] = []
    if spec.premises:
        for premise in spec.premises:
            status = premise.status
            if not isinstance(status, PremiseStatus):
                try:
                    status = PremiseStatus(status)
                except ValueError:
                    status = PremiseStatus.UNKNOWN
            payload.append(
                {
                    "text": premise.text,
                    "kind": premise.kind,
                    "value": premise.value,
                    "status": status.value,
                    "reason": premise.status_reason,
                }
            )
        return payload
    if policy is not None and policy.premise_statuses:
        for entry in policy.premise_statuses:
            payload.append(
                {
                    "text": entry.text,
                    "kind": entry.kind,
                    "value": entry.value,
                    "status": entry.status.value
                    if isinstance(entry.status, PremiseStatus)
                    else str(entry.status),
                    "reason": entry.reason,
                }
            )
    return payload


def _has_contradicted_premise(spec: QuerySpec, policy: PolicyVerdict | None) -> bool:
    for premise in spec.premises:
        if premise.status == PremiseStatus.CONTRADICTED:
            return True
    if policy is not None:
        for entry in policy.premise_statuses:
            if entry.status == PremiseStatus.CONTRADICTED:
                return True
    return False


def _source_role_mismatch(
    claim: Claim,
    cited_chunks: list[RetrievedChunk],
    required_slots: list[str] | None = None,
) -> tuple[bool, str]:
    # Если слотов больше одного — claim мог быть размечен "не тем" слотом,
    # не отклоняем по роли источника.
    if required_slots and len(required_slots) > 1:
        return False, ""
    # Если хотя бы один чанк правильной роли — не отклоняем.
    expected = expected_source_kind_for_slot(claim.slot)
    if expected is None or not cited_chunks:
        return False, ""
    known = [str(getattr(c, "source_kind", "") or "").lower() for c in cited_chunks]
    known = [k for k in known if k and k != "unknown"]
    if not known:
        return False, ""
    if any(k == expected for k in known):
        return False, ""
    if all(k != expected for k in known):
        return True, "source_role_mismatch"
    return False, ""


def _build_answer_to_question_check(
    parsed: VerificationBatchSchema | None,
    draft: DraftAnswer,
    spec: QuerySpec,
    question: str | None,
    fallback_reason: str = "verifier_unavailable",
) -> AnswerToQuestionCheck:
    if parsed is None or parsed.answer_to_question is None:
        reason = fallback_reason if parsed is None else "answer_to_question_missing"
        return AnswerToQuestionCheck(
            verdict=AnswerToQuestionVerdict.UNDETERMINED,
            reason=reason,
        )
    return AnswerToQuestionCheck(
        verdict=parsed.answer_to_question.verdict,
        reason=parsed.answer_to_question.reason or "",
        quote=parsed.answer_to_question.quote,
    )


def verify(
    draft: DraftAnswer,
    chunks: list[RetrievedChunk],
    spec: QuerySpec,
    can_regenerate: bool = True,
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
    timeout: float | None = None,
    question: str | None = None,
    policy: PolicyVerdict | None = None,
) -> tuple[VerificationReport, TraceStep]:
    start_time = time.perf_counter()
    source_map: dict[str, RetrievedChunk] = {
        f"{config.SOURCE_ID_PREFIX}{idx}": ch
        for idx, ch in enumerate(chunks, start=config.SOURCE_START_INDEX)
    }

    if question is not None:
        question_text = question
        question_source = "original"
    else:
        question_text = (
            spec.rewritten_queries_dense[0] if spec.rewritten_queries_dense else ""
        )
        question_source = "rewritten"

    claims_synthesized = False
    pruned_claims_count = 0
    verifier_unavailable = False
    verifier_reason: str | None = None
    auto_accept_blocked = False

    def _finish(
        report: VerificationReport,
        *,
        claims_count: int,
        core_claims_count: int = 0,
        non_core_claims_count: int = 0,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        unverified_claims_count: int = 0,
    ) -> tuple[VerificationReport, TraceStep]:
        latency_ms = round(
            (time.perf_counter() - start_time) * config.MS_PER_SECOND,
            config.LATENCY_ROUND_DIGITS,
        )
        a2q = report.answer_to_question
        supported_claims_count = sum(
            1
            for v in report.batch_status
            if v.status == ClaimStatus.SUPPORTED
        )
        contradicted_claims_count = sum(
            1
            for v in report.batch_status
            if v.status == ClaimStatus.CONTRADICTED
        )
        not_in_source_claims_count = sum(
            1
            for v in report.batch_status
            if v.status == ClaimStatus.NOT_IN_SOURCE
        )
        step = TraceStep(
            step="verify_draft",
            input={
                "claims_count": claims_count,
                "core_claims_count": core_claims_count,
                "non_core_claims_count": non_core_claims_count,
                "question_source": question_source,
                "question_length": len(question_text),
                "answer_length": len(draft.answer_text),
            },
            output={
                "final_verdict": report.final_verdict.value,
                "supported_locked": report.supported_locked,
                "citation_errors_count": len(report.citation_errors),
                "injection_signals_count": len(report.injection_signals),
                "answer_to_question": {
                    "verdict": a2q.verdict.value,
                    "reason": a2q.reason,
                }
                if a2q is not None
                else None,
                "verifier_unavailable": report.verifier_unavailable,
                "verifier_unavailable_reason": verifier_reason,
                "auto_accept_blocked": report.auto_accept_blocked,
                "claims_synthesized": report.claims_synthesized,
                "supported_claims_count": supported_claims_count,
                "contradicted_claims_count": contradicted_claims_count,
                "not_in_source_claims_count": not_in_source_claims_count,
                "unverified_claims_count": unverified_claims_count,
                "text_claim_consistency": report.text_claim_consistency,
                "text_claim_issues_count": len(report.text_claim_issues),
                "source_role_issues": list(report.source_role_issues),
                "source_role_issues_count": len(report.source_role_issues),
                "pruned_claims_count": pruned_claims_count,
                "question_source": question_source,
                "question_length": len(question_text),
                "answer_length": len(draft.answer_text),
            },
            latency_ms=latency_ms,
            model=config.LLM_MODEL if prompt_tokens is not None else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            stage_idx=7,
        )
        return report, step

    # --- Пустые claims -------------------------------------------------
    if not draft.claims:
        if not draft.answer_text.strip() or draft.refusal_reason:
            report = VerificationReport(
                batch_status=[],
                citation_errors=[],
                injection_signals=[],
                supported_locked=[],
                final_verdict=VerificationVerdict.REFUSE,
                answer_to_question=AnswerToQuestionCheck(
                    verdict=AnswerToQuestionVerdict.UNDETERMINED,
                    reason="no_answer_text",
                ),
            )
            return _finish(report, claims_count=0)

        synthesized = _synthesize_claims_from_answer(draft.answer_text)
        if synthesized:
            draft.claims = synthesized
            claims_synthesized = True
        else:
            if "ответ не содержит проверяемых утверждений" not in draft.missing_info:
                draft.missing_info.append("ответ не содержит проверяемых утверждений")
            auto_accept_blocked = True
            report = VerificationReport(
                batch_status=[],
                citation_errors=[],
                injection_signals=[],
                supported_locked=[],
                final_verdict=VerificationVerdict.PARTIAL,
                answer_to_question=AnswerToQuestionCheck(
                    verdict=AnswerToQuestionVerdict.UNDETERMINED,
                    reason="no_verifiable_claims",
                ),
                verifier_unavailable=False,
                auto_accept_blocked=True,
                claims_synthesized=False,
            )
            return _finish(report, claims_count=0)

    claims_payload = [
        {
            "claim_idx": c.original_idx if c.original_idx is not None else idx,
            "text": c.text,
            "source_ids": c.source_ids,
            "slot": c.slot,
        }
        for idx, c in enumerate(draft.claims)
    ]
    chunks_text = "\n\n".join(
        f"[{sid}] (doc={ch.document_title}, from={ch.effective_from}, to={ch.effective_to}):\n{ch.text}"
        for sid, ch in source_map.items()
    )
    premises_payload = _build_premises_payload(spec, policy)

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    llm_verdicts_by_idx: dict[int, ClaimVerdict] = {}
    parsed_batch: VerificationBatchSchema | None = None

    verifier_active = config.ENABLE_LLM_VERIFIER and not (
        config.USE_STUB_LLM
        or getattr(config, "LLM_MODE", config.LLM_MODE_REAL) == config.LLM_MODE_STUB
    )

    if not verifier_active:
        verifier_unavailable = True
        verifier_reason = "verifier_disabled"
    else:
        messages = [
            {"role": config.ROLE_SYSTEM, "content": ver_prompts.VERIFIER_SYSTEM},
            {
                "role": config.ROLE_USER,
                "content": ver_prompts.VERIFIER_USER.format(
                    question=question_text,
                    answer_text=draft.answer_text,
                    required_slots=", ".join(spec.required_slots)
                    if spec.required_slots
                    else "",
                    premises_json=json.dumps(premises_payload, ensure_ascii=False),
                    claims_json=json.dumps(claims_payload, ensure_ascii=False),
                    chunks_text=chunks_text,
                ),
            },
        ]
        try:
            raw_output, p_tok, c_tok = llm_complete_with_usage(
                messages=messages,
                model=config.LLM_MODEL,
                temperature=ver_prompts.VERIFIER_TEMPERATURE,
                max_tokens=ver_prompts.VERIFIER_MAX_TOKENS,
                timeout=timeout,
            )
            prompt_tokens = p_tok
            completion_tokens = c_tok
            if on_llm_success:
                on_llm_success()

            parsed_batch = _parse_verifier_output(raw_output)
            for v in parsed_batch.verdicts:
                llm_verdicts_by_idx[v.claim_idx] = ClaimVerdict(
                    claim_idx=v.claim_idx,
                    status=v.status,
                    quote=v.quote,
                    source_id=v.source_id,
                )
        except Exception as exc:
            if on_failure:
                on_failure(LLMFormatError(str(exc)) if not isinstance(exc, LLMError) else exc)
            verifier_unavailable = True
            verifier_reason = (
                "parse_error" if isinstance(exc, LLMFormatError) else "verifier_error"
            )
            parsed_batch = None
            llm_verdicts_by_idx = {}

    batch_status: list[ClaimVerdict] = []
    citation_errors: list[str] = []
    injection_signals: list[str] = []
    source_role_issues: list[str] = []
    supported_locked: list[int] = []
    unverified_claims_count = 0

    for pos_idx, claim in enumerate(draft.claims):
        orig_idx = claim.original_idx if claim.original_idx is not None else pos_idx

        if verifier_unavailable:
            verdict_obj = ClaimVerdict(
                claim_idx=orig_idx,
                status=ClaimStatus.UNVERIFIED,
                quote="",
                source_id=None,
                reason=verifier_reason or "verifier_unavailable",
            )
        elif orig_idx not in llm_verdicts_by_idx:
            verdict_obj = ClaimVerdict(
                claim_idx=orig_idx,
                status=ClaimStatus.UNVERIFIED,
                quote="",
                source_id=None,
                reason="verdict_missing",
            )
        else:
            verdict_obj = llm_verdicts_by_idx[orig_idx]

        if verdict_obj.status == ClaimStatus.UNVERIFIED:
            unverified_claims_count += 1

        if verdict_obj.status == ClaimStatus.SUPPORTED:
            v_sid = verdict_obj.source_id
            if not v_sid or v_sid not in source_map:
                verdict_obj.status = ClaimStatus.NOT_IN_SOURCE
                verdict_obj.reason = verdict_obj.reason or "invalid_source_id"
                citation_errors.append(
                    f"Claim {orig_idx} marked supported with invalid source_id: {v_sid}"
                )

        invalid_sids = [sid for sid in claim.source_ids if sid not in source_map]
        if not claim.source_ids or invalid_sids:
            citation_errors.append(
                f"Claim {orig_idx} has invalid source_ids: {claim.source_ids}"
            )

        valid_cited_chunks = [
            source_map[sid] for sid in claim.source_ids if sid in source_map
        ]

        role_mismatch, role_mismatch_reason = _source_role_mismatch(
            claim, valid_cited_chunks, spec.required_slots
        )
        if role_mismatch and verdict_obj.status == ClaimStatus.SUPPORTED:
            verdict_obj.status = ClaimStatus.NOT_IN_SOURCE
            verdict_obj.reason = role_mismatch_reason
            source_role_issues.append(
                f"Claim {orig_idx}: источник не соответствует назначению "
                f"для слота {claim.slot}"
            )

        if (
            verdict_obj.status == ClaimStatus.SUPPORTED
            and valid_cited_chunks
            and any(ch.untrusted for ch in valid_cited_chunks)
        ):
            trusted_confirmations = [ch for ch in valid_cited_chunks if not ch.untrusted]
            if not trusted_confirmations:
                injection_signals.append(
                    f"Claim {orig_idx} relies solely on untrusted chunk"
                )
                verdict_obj.status = ClaimStatus.NOT_IN_SOURCE

        if verdict_obj.status == ClaimStatus.SUPPORTED and spec.event_date:
            for ch in valid_cited_chunks:
                if ch.effective_from and spec.event_date < ch.effective_from and ch.document_type != "law":
                    verdict_obj.status = ClaimStatus.CONTRADICTED
                if ch.effective_to and spec.event_date >= ch.effective_to:
                    verdict_obj.status = ClaimStatus.CONTRADICTED

        if (
            verdict_obj.status == ClaimStatus.SUPPORTED
            and not invalid_sids
            and claim.source_ids
        ):
            supported_locked.append(orig_idx)

        if verdict_obj.status == ClaimStatus.CONTRADICTED:
            verdict_obj.reason = verdict_obj.reason or "contradicted"
        if verdict_obj.status == ClaimStatus.NOT_IN_SOURCE and not verdict_obj.reason:
            verdict_obj.reason = "not_in_source"

        batch_status.append(verdict_obj)

    status_by_idx = {v.claim_idx: v.status for v in batch_status}
    core_claims, non_core_claims = _partition_core_and_non_core_claims(
        draft.claims, spec.required_slots, status_by_idx
    )

    a2q = _build_answer_to_question_check(
        parsed_batch,
        draft,
        spec,
        question_text,
        fallback_reason=verifier_reason or "verifier_unavailable",
    )

    # Техническая сверка: текст ответа согласован со списком claims.
    # Несогласованность запрещает ACCEPT (допустимы PARTIAL после очистки
    # либо REGENERATE, если одновременно есть actionable-проблемы).
    text_claim_issues = _check_answer_text_claim_consistency(
        draft, require_verbatim_claims=claims_synthesized
    )
    text_claim_consistency = not text_claim_issues

    has_claims = bool(draft.claims)
    statuses = [v.status for v in batch_status]
    all_supported = has_claims and all(s == ClaimStatus.SUPPORTED for s in statuses)
    any_supported = any(s == ClaimStatus.SUPPORTED for s in statuses)
    has_contradicted = any(s == ClaimStatus.CONTRADICTED for s in statuses)
    has_not_in_source = any(s == ClaimStatus.NOT_IN_SOURCE for s in statuses)
    has_unverified = any(s == ClaimStatus.UNVERIFIED for s in statuses)
    actionable = has_contradicted or has_not_in_source or bool(citation_errors)
    premise_contradicted = _has_contradicted_premise(spec, policy)
    regen_available = can_regenerate and int(config.MAX_REGENERATE_CYCLES) > 0

    verdict_reason = ""
    if (
        a2q.verdict == AnswerToQuestionVerdict.MISMATCH
        and regen_available
        and not verifier_unavailable
    ):
        final_verdict = VerificationVerdict.REGENERATE
        verdict_reason = "answer_question_mismatch"
    elif actionable and regen_available:
        final_verdict = VerificationVerdict.REGENERATE
        verdict_reason = "claim_failures"
    elif a2q.verdict == AnswerToQuestionVerdict.MISMATCH:
        final_verdict = VerificationVerdict.REFUSE
        verdict_reason = "answer_question_mismatch"
        draft.refusal_reason = "answer_question_mismatch"
    elif (
        has_claims
        and all_supported
        and not has_contradicted
        and not has_not_in_source
        and not has_unverified
        and not citation_errors
        and a2q.verdict == AnswerToQuestionVerdict.MATCH
        and not premise_contradicted
        and not verifier_unavailable
        and not auto_accept_blocked
        and text_claim_consistency
    ):
        final_verdict = VerificationVerdict.ACCEPT
        verdict_reason = "all_verified"
    elif any_supported and a2q.verdict in (
        AnswerToQuestionVerdict.MATCH,
        AnswerToQuestionVerdict.UNDETERMINED,
    ):
        final_verdict = VerificationVerdict.PARTIAL
        verdict_reason = "partial_support"
        original_claim_count = len(draft.claims)
        supported_slots = {
            c.slot for c in draft.claims if c.original_idx in supported_locked and c.slot
        }
        kept_claims: list[Claim] = []
        for c in draft.claims:
            if c.original_idx in supported_locked:
                kept_claims.append(c)
            else:
                interp_note = f"интерпретация: {c.text}"
                if interp_note not in draft.assumptions:
                    draft.assumptions.append(interp_note)
                if (
                    c.slot
                    and c.slot in spec.required_slots
                    and c.slot not in supported_slots
                    and c.slot not in draft.missing_info
                ):
                    draft.missing_info.append(f"не подтверждено источниками: {c.slot}")
        draft.claims = kept_claims
        pruned_claims_count = original_claim_count - len(kept_claims)

        allowed_sids: set[str] = set()
        for c in kept_claims:
            allowed_sids.update(c.source_ids)
        pruned_answer = _prune_answer_to_supported_sources(
            draft.answer_text, allowed_sids
        )
        draft.answer_text = pruned_answer

        if a2q.verdict == AnswerToQuestionVerdict.UNDETERMINED:
            note = "answer_to_question:undetermined"
            if note not in draft.missing_info:
                draft.missing_info.append(note)

        if not draft.answer_text.strip():
            final_verdict = VerificationVerdict.REFUSE
            verdict_reason = "empty_after_prune"
            if not draft.refusal_reason:
                draft.refusal_reason = "insufficient_evidence"
    elif has_unverified and not any_supported and not actionable:
        final_verdict = VerificationVerdict.PARTIAL
        verdict_reason = "verification_unavailable"
        auto_accept_blocked = True
        supported_locked = []
    else:
        final_verdict = VerificationVerdict.REFUSE
        verdict_reason = "no_support"

    # Причина решения не является полем VerificationReport, но попадает в trace.
    report = VerificationReport(
        batch_status=batch_status,
        citation_errors=citation_errors,
        injection_signals=injection_signals,
        supported_locked=supported_locked,
        final_verdict=final_verdict,
        answer_to_question=a2q,
        verifier_unavailable=verifier_unavailable,
        auto_accept_blocked=auto_accept_blocked,
        claims_synthesized=claims_synthesized,
        text_claim_consistency=text_claim_consistency,
        text_claim_issues=text_claim_issues,
        source_role_issues=source_role_issues,
    )

    report_step, step = _finish(
        report,
        claims_count=len(claims_payload),
        core_claims_count=len(core_claims),
        non_core_claims_count=len(non_core_claims),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        unverified_claims_count=unverified_claims_count,
    )
    if verdict_reason:
        step.output["verdict_reason"] = verdict_reason
    return report_step, step
