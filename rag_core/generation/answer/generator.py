from dataclasses import asdict
import json
import logging
import re
import time
from typing import Callable, Optional
from pydantic import ValidationError
from rag_core import config
from rag_core.generation.answer.context_builder import format_single_chunk
from rag_core.generation.errors import LLMError, LLMFormatError
from rag_core.generation.llm.client import llm_complete_with_usage
from rag_core.generation.prompts import generator as gen_prompts
from rag_core.generation.prompts import regenerate as regen_prompts
from rag_core.generation.types import (
    Claim,
    ClaimStatus,
    DraftAnswer,
    DraftAnswerSchema,
    EvidenceBundle,
    PolicyVerdict,
    QuerySpec,
    RetrievedChunk,
    TraceStep,
    VerificationReport,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_NULL_REFUSAL_VALUES = {"null", "none", "nil", "false", "нет", "n/a", ""}

_HISTORY_SOURCE_TAG_RE = re.compile(r"\s*<source>.*?</source>", re.DOTALL)
_INJECTION_TOKENS_RE = re.compile(
    r"(?:<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<system>|</system>|<\|endoftext\|>|---\s*system)",
    re.IGNORECASE,
)


def _format_chat_history_block(chat_history: list[dict] | None) -> str:
    if not getattr(config, "ENABLE_MEMORY", True) or not chat_history:
        return ""

    max_turns = max(1, int(getattr(config, "MAX_HISTORY_TURNS", 6)))
    max_chars = max(100, int(getattr(config, "HISTORY_TURN_MAX_CHARS", 1200)))
    recent_turns = chat_history[-max_turns:]

    lines: list[str] = []
    for turn in recent_turns:
        if not isinstance(turn, dict):
            continue
        raw_content = str(turn.get("content") or "").strip()
        if not raw_content:
            continue
        # Защита от prompt injection и утечки старых тегов источников
        raw_content = _INJECTION_TOKENS_RE.sub("", raw_content).strip()
        raw_content = _HISTORY_SOURCE_TAG_RE.sub("", raw_content).strip()
        if not raw_content:
            continue
        if len(raw_content) > max_chars:
            raw_content = raw_content[:max_chars].rstrip() + "..."
        raw_content = " ".join(raw_content.split())
        role_raw = str(turn.get("role") or "user").strip().lower()
        role_label = "Пользователь" if role_raw == "user" else "Ассистент"
        lines.append(f"{role_label}: {raw_content}")

    if not lines:
        return ""

    history_body = "\n".join(lines)
    return (
        "История предыдущих сообщений диалога (используй только для понимания контекста беседы; "
        "все факты и ссылки на источники бери СТРОГО из раздела «Контекст (источники)»):\n"
        f"{history_body}\n\n"
    )

def _normalize_refusal_reason(
    reason: Optional[str],
    has_claims: bool,
    refusal_forbidden: bool,
) -> Optional[str]:
    if refusal_forbidden or has_claims or reason is None:
        return None
    cleaned = reason.strip()
    if cleaned.lower() in _NULL_REFUSAL_VALUES:
        return None
    return cleaned


def _parse_draft_schema(raw_text: str) -> DraftAnswerSchema:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return DraftAnswerSchema.model_validate_json(cleaned)
    except Exception:
        pass

    match = _JSON_BLOCK_RE.search(raw_text)
    if match:
        try:
            return DraftAnswerSchema.model_validate_json(match.group(0))
        except ValidationError as exc:
            raise LLMFormatError(f"DraftAnswer schema validation failed: {exc}") from exc
        except Exception as exc:
            raise LLMFormatError(f"DraftAnswer JSON parse failed: {exc}") from exc

    raise LLMFormatError("No valid JSON object found in generator output")


def generate_draft(
    spec: QuerySpec,
    evidence: EvidenceBundle,
    policy: PolicyVerdict,
    context: str,
    query: str | None = None,
    chat_history: list[dict] | None = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
    timeout: float | None = None,
) -> tuple[DraftAnswer, TraceStep]:
    start_time = time.perf_counter()
    effective_query = query or (
        spec.rewritten_queries_dense[0] if spec.rewritten_queries_dense else ""
    )
    chat_history_block = _format_chat_history_block(chat_history)
    history_turns_count = (
        len(chat_history)
        if (getattr(config, "ENABLE_MEMORY", True) and chat_history)
        else 0
    )
    premise_statuses_payload = [
        {
            "kind": entry.kind,
            "value": entry.value,
            "status": getattr(entry.status, "value", entry.status),
            "reason": entry.reason,
        }
        for entry in policy.premise_statuses
    ]

    max_attempts = max(1, int(config.MAX_ATTEMPTS_GENERATE_INITIAL))
    retry_feedback = ""
    raw_last_output = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    last_api_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        messages = [
            {"role": config.ROLE_SYSTEM, "content": gen_prompts.GENERATION_SYSTEM},
            {
                "role": config.ROLE_USER,
                "content": gen_prompts.GENERATION_USER.format(
                    chat_history_block=chat_history_block,
                    query=effective_query,
                    event_date=spec.event_date or "не указана",
                    required_slots=", ".join(spec.required_slots),
                    premise_statuses=json.dumps(
                        premise_statuses_payload, ensure_ascii=False
                    ),
                    premise_details=json.dumps(policy.premise_details, ensure_ascii=False),
                    version_issues=json.dumps(
                        list(evidence.version_issues)
                        + list(
                            getattr(evidence, "generation_warnings", None) or []
                        ),
                        ensure_ascii=False,
                    ),
                    refusal_forbidden=policy.refusal_forbidden,
                    context=context,
                    retry_feedback=retry_feedback,
                ),
            },
        ]
        try:
            raw_output, p_tok, c_tok = llm_complete_with_usage(
                messages=messages,
                model=config.LLM_MODEL,
                temperature=gen_prompts.GENERATOR_TEMPERATURE,
                max_tokens=gen_prompts.GENERATOR_MAX_TOKENS,
                timeout=timeout,
            )
            raw_last_output = raw_output
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            if on_llm_success:
                on_llm_success()

            parsed = _parse_draft_schema(raw_output)
            claims: list[Claim] = []
            for i, c_schema in enumerate(parsed.claims):
                raw_sids = [s.strip() for s in c_schema.source_ids if s.strip()]
                claims.append(
                    Claim(
                        text=c_schema.text.strip(),
                        source_ids=raw_sids,
                        slot=c_schema.slot or (spec.required_slots[0] if spec.required_slots else None),
                        original_idx=i,
                    )
                )

            merged_assumptions = list(dict.fromkeys(spec.assumptions + parsed.assumptions))
            missing_info = list(parsed.missing_info)
            if spec.degradation_reason == "no_effective_dates":
                note = "версия не определена, нужна дата"
                if note not in missing_info:
                    missing_info.append(note)

            refusal_reason = _normalize_refusal_reason(
                reason=parsed.refusal_reason,
                has_claims=bool(claims),
                refusal_forbidden=policy.refusal_forbidden,
            )

            draft = DraftAnswer(
                answer_text=parsed.answer_text.strip(),
                claims=claims,
                assumptions=merged_assumptions,
                missing_info=missing_info,
                refusal_reason=refusal_reason,
            )

            latency_ms = round(
                (time.perf_counter() - start_time) * config.MS_PER_SECOND,
                config.LATENCY_ROUND_DIGITS,
            )
            step = TraceStep(
                step=config.STEP_GENERATION,
                input={
                    "query": effective_query,
                    "context_length": len(context),
                    "history_turns": history_turns_count,
                    "attempts": attempt,
                },
                output={
                    "answer_length": len(draft.answer_text),
                    "claims_count": len(draft.claims),
                    "assumptions_count": len(draft.assumptions),
                    "missing_info_count": len(draft.missing_info),
                    "refusal_reason": draft.refusal_reason,
                },
                latency_ms=latency_ms,
                model=config.LLM_MODEL,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                stage_idx=6,
            )
            return draft, step

        except LLMFormatError as exc:
            if on_failure:
                on_failure(exc)
            retry_feedback = (
                f"\nОшибка парсинга JSON в предыдущей попытке: {exc}. "
                "Верни строго валидный JSON по схеме DraftAnswer."
            )
        except LLMError as exc:
            last_api_error = exc
            if on_failure:
                on_failure(exc)

    if last_api_error is not None and not raw_last_output:
        raise last_api_error

    fallback_missing = (
        ["версия не определена, нужна дата"]
        if spec.degradation_reason == "no_effective_dates"
        else []
    )
    fallback_draft = DraftAnswer(
        answer_text=raw_last_output.strip(),
        claims=[],
        assumptions=list(spec.assumptions),
        missing_info=fallback_missing,
        refusal_reason=None,
    )
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_GENERATION,
        input={
            "query": effective_query,
            "context_length": len(context),
            "history_turns": history_turns_count,
            "attempts": attempt,
        },
        output={
            "answer_length": len(fallback_draft.answer_text),
            "claims_count": 0,
            "fallback": "raw_text",
        },
        latency_ms=latency_ms,
        model=config.LLM_MODEL,
        prompt_tokens=total_prompt_tokens or None,
        completion_tokens=total_completion_tokens or None,
        extra={"confidence": "low"},
        stage_idx=6,
        degradation="draft_raw_fallback",
    )
    return fallback_draft, step


def _enforce_supported_locked(
    new_draft: DraftAnswer,
    previous: DraftAnswer,
    report: VerificationReport,
    available_sources: set[str],
) -> DraftAnswer:
    by_orig = {
        c.original_idx: c
        for c in previous.claims
        if c.original_idx is not None
    }
    previous_supported = {
        idx: by_orig[idx]
        for idx in report.supported_locked
        if idx in by_orig
    }

    new_by_orig: dict[int, Claim] = {
        c.original_idx: c
        for c in new_draft.claims
        if c.original_idx is not None
    }

    for idx, prev_claim in previous_supported.items():
        if idx in new_by_orig:
            candidate = new_by_orig[idx]
            valid_new_sources = (
                bool(candidate.source_ids)
                and all(sid in available_sources for sid in candidate.source_ids)
            )
            final_sources = (
                list(candidate.source_ids)
                if valid_new_sources
                else list(prev_claim.source_ids)
            )
            if candidate.text != prev_claim.text:
                new_by_orig[idx] = Claim(
                    text=prev_claim.text,
                    source_ids=final_sources,
                    slot=prev_claim.slot,
                    original_idx=idx,
                )
            else:
                new_by_orig[idx] = Claim(
                    text=candidate.text,
                    source_ids=final_sources,
                    slot=candidate.slot or prev_claim.slot,
                    original_idx=idx,
                )
        else:
            new_by_orig[idx] = Claim(
                text=prev_claim.text,
                source_ids=list(prev_claim.source_ids),
                slot=prev_claim.slot,
                original_idx=idx,
            )

    unindexed_claims = [c for c in new_draft.claims if c.original_idx is None]
    next_orig_idx = (
        max(
            [c.original_idx for c in previous.claims if c.original_idx is not None]
            + list(new_by_orig.keys())
            + [-1]
        )
        + 1
    )
    for c in unindexed_claims:
        new_by_orig[next_orig_idx] = Claim(
            text=c.text,
            source_ids=list(c.source_ids),
            slot=c.slot,
            original_idx=next_orig_idx,
        )
        next_orig_idx += 1

    ordered_claims = [new_by_orig[k] for k in sorted(new_by_orig.keys())]
    new_draft.claims = ordered_claims
    return new_draft


_SOURCE_TAG_RE = re.compile(r"<source>(.*?)</source>", re.DOTALL)
_SOURCE_ID_TOKEN_RE = re.compile(
    re.escape(config.SOURCE_ID_PREFIX) + r"\d+"
)

_FALLBACK_MISSING_DROPPED = "часть ответа не подтверждена источниками и удалена"
_FALLBACK_MISSING_UNVERIFIED = "ответ не подтверждён источниками"


def _extract_answer_tag_ids(answer_text: str) -> list[str]:
    """Извлекает source_id из тегов ``<source>S...</source>`` текста ответа."""
    ids: list[str] = []
    for inner in _SOURCE_TAG_RE.findall(answer_text or ""):
        for sid in _SOURCE_ID_TOKEN_RE.findall(inner):
            if sid not in ids:
                ids.append(sid)
    return ids


def _locked_claims_map(
    previous: DraftAnswer,
    report: VerificationReport,
    available_sources: set[str],
) -> dict[int, Claim]:
    """Подтверждённые источники: claims previous с валидными source_ids."""
    by_orig = {
        c.original_idx: c
        for c in previous.claims
        if c.original_idx is not None
    }
    locked: dict[int, Claim] = {}
    for idx in report.supported_locked:
        claim = by_orig.get(idx)
        if claim is None:
            continue
        valid_sources = [sid for sid in claim.source_ids if sid in available_sources]
        if not valid_sources:
            continue
        locked[idx] = Claim(
            text=claim.text,
            source_ids=valid_sources,
            slot=claim.slot,
            original_idx=idx,
        )
    return locked


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _validate_draft_consistency(
    draft: DraftAnswer,
    report: VerificationReport,
    available_sources: set[str],
    locked_claims: Optional[dict[int, Claim]] = None,
) -> list[str]:
    """Технические инварианты согласованности answer_text / claims / источников.

    Позиционный контракт — ``(draft, report, available_sources)``; дополнительные
    ``locked_claims`` позволяют проверить дословное сохранение подтверждённых
    утверждений (text/slot).
    """
    issues: list[str] = []

    # (1) все source_ids claims должны существовать в текущем контексте.
    for pos, claim in enumerate(draft.claims):
        if not claim.source_ids:
            issues.append(f"claim {pos}: empty source_ids")
            continue
        for sid in claim.source_ids:
            if sid not in available_sources:
                issues.append(f"claim {pos}: unknown source_id {sid}")

    tag_ids = set(_extract_answer_tag_ids(draft.answer_text))
    claim_ids: set[str] = set()
    for claim in draft.claims:
        claim_ids.update(claim.source_ids)

    # (2) каждый <source>-тег в answer_text подкреплён claims.
    for sid in sorted(tag_ids):
        if sid not in claim_ids:
            issues.append(f"answer_text: tag source_id {sid} not backed by claims")

    # (3) supported_locked сохранены (text/slot/источники).
    by_orig: dict[int, Claim] = {
        c.original_idx: c
        for c in draft.claims
        if c.original_idx is not None
    }
    for idx in report.supported_locked:
        locked = None if locked_claims is None else locked_claims.get(idx)
        candidate = by_orig.get(idx)
        if candidate is None:
            issues.append(f"supported_locked claim {idx} missing")
            continue
        if not candidate.source_ids or any(
            sid not in available_sources for sid in candidate.source_ids
        ):
            issues.append(f"supported_locked claim {idx} has invalid sources")
        if locked is not None:
            if candidate.text != locked.text:
                issues.append(f"supported_locked claim {idx} text changed")
            if candidate.slot != locked.slot:
                issues.append(f"supported_locked claim {idx} slot changed")

    # (4) claims и answer_text согласованы по непустоте.
    if draft.claims and not draft.answer_text.strip():
        issues.append("claims present but answer_text is empty")

    # (5) обоснованный отказ несовместим с утверждениями.
    if draft.refusal_reason and draft.claims:
        issues.append("refusal_reason present together with claims")

    # (6) original_idx не должны дублироваться.
    seen: set[int] = set()
    for claim in draft.claims:
        if claim.original_idx is None:
            continue
        if claim.original_idx in seen:
            issues.append(f"duplicate original_idx {claim.original_idx}")
        seen.add(claim.original_idx)

    return issues


def _sanitize_regenerated_answer(
    draft: DraftAnswer,
    allowed_source_ids: set[str],
) -> DraftAnswer:
    """Убирает незадействованные теги и невалидные source_ids.

    Новые сведения не добавляются: только фильтрация уже существующих
    claim/тегов. Claims без валидных источников удаляются как непроверяемые.
    """
    sanitized_claims: list[Claim] = []
    claim_sources: set[str] = set()
    for claim in draft.claims:
        valid_sources = [sid for sid in claim.source_ids if sid in allowed_source_ids]
        if not valid_sources:
            continue
        sanitized_claims.append(
            Claim(
                text=claim.text,
                source_ids=valid_sources,
                slot=claim.slot,
                original_idx=claim.original_idx,
            )
        )
        claim_sources.update(valid_sources)

    def _replace_tag(match: re.Match[str]) -> str:
        tokens = [t.strip() for t in match.group(1).split(",") if t.strip()]
        kept = [
            token
            for token in tokens
            if token in allowed_source_ids and token in claim_sources
        ]
        return f"<source>{', '.join(kept)}</source>" if kept else ""

    answer = _SOURCE_TAG_RE.sub(_replace_tag, draft.answer_text or "")
    answer = re.sub(r" {2,}", " ", answer)
    answer = re.sub(r" +([.,;:])", r"\1", answer).strip()

    return DraftAnswer(
        answer_text=answer,
        claims=sanitized_claims,
        assumptions=list(draft.assumptions),
        missing_info=list(draft.missing_info),
        refusal_reason=draft.refusal_reason,
    )


def _build_safe_regen_fallback(
    previous: DraftAnswer,
    report: VerificationReport,
    available_sources: set[str],
    scenario: str = "A",
) -> DraftAnswer:
    """Детерминированный безопасный результат регенерации.

    Сценарий A — есть подтверждённые claims: answer_text собирается ТОЛЬКО
    из их текстов, старый текст не возвращается.
    Сценарий B — подтверждённых claims нет: пустой ответ и обоснованный отказ.
    ``scenario="C"`` выбирает A или B по ``report.supported_locked``.
    """
    locked = _locked_claims_map(previous, report, available_sources)
    effective = scenario if scenario in {"A", "B"} else ("A" if locked else "B")

    if effective == "A":
        claims = [locked[idx] for idx in sorted(locked)]
        parts = [
            f"{claim.text} <source>{', '.join(claim.source_ids)}</source>"
            for claim in claims
        ]
        return DraftAnswer(
            answer_text=" ".join(parts).strip(),
            claims=claims,
            assumptions=list(previous.assumptions),
            missing_info=_dedupe(
                list(previous.missing_info) + [_FALLBACK_MISSING_DROPPED]
            ),
            refusal_reason=None,
        )

    return DraftAnswer(
        answer_text="",
        claims=[],
        assumptions=list(previous.assumptions),
        missing_info=_dedupe(
            list(previous.missing_info) + [_FALLBACK_MISSING_UNVERIFIED]
        ),
        refusal_reason="verification_failed",
    )


def _finalize_regen_invariants(draft: DraftAnswer) -> DraftAnswer:
    """Последняя техническая нормализация инвариантов регенерации."""
    if draft.refusal_reason and draft.claims:
        draft.refusal_reason = None
    if not draft.answer_text.strip() and not draft.refusal_reason:
        draft.refusal_reason = "verification_failed"
        draft.claims = []
    return draft


def regenerate_draft(
    previous: DraftAnswer,
    report: VerificationReport,
    chunks: list[RetrievedChunk],
    spec: QuerySpec,
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
    timeout: float | None = None,
) -> tuple[DraftAnswer, TraceStep]:
    start_time = time.perf_counter()
    available_sources = {
        f"{config.SOURCE_ID_PREFIX}{idx}"
        for idx in range(config.SOURCE_START_INDEX, config.SOURCE_START_INDEX + len(chunks))
    }
    context_str = config.CONTEXT_CHUNK_SEPARATOR.join(
        format_single_chunk(f"{config.SOURCE_ID_PREFIX}{idx}", ch)
        for idx, ch in enumerate(chunks, start=config.SOURCE_START_INDEX)
    )

    # UNVERIFIED — не actionable-ошибка: переписывание инициируют только
    # подтверждённые противоречия и утверждения вне источника.
    failed_verdicts = [
        {
            "original_idx": v.claim_idx,
            "status": v.status.value,
            "quote": v.quote,
            "source_id": v.source_id,
        }
        for v in report.batch_status
        if v.status in {ClaimStatus.CONTRADICTED, ClaimStatus.NOT_IN_SOURCE}
    ]

    a2q = report.answer_to_question
    answer_question_feedback = {
        "verdict": a2q.verdict.value if a2q is not None else None,
        "reason": (a2q.reason if a2q is not None else "") or "",
        "quote": a2q.quote if a2q is not None else None,
    }

    messages = [
        {"role": config.ROLE_SYSTEM, "content": regen_prompts.REGENERATE_SYSTEM},
        {
            "role": config.ROLE_USER,
            "content": regen_prompts.REGENERATE_USER.format(
                query=spec.rewritten_queries_dense[0] if spec.rewritten_queries_dense else "",
                previous_draft_json=json.dumps(asdict(previous), ensure_ascii=False),
                supported_locked=json.dumps(report.supported_locked),
                failed_claims_feedback=json.dumps(failed_verdicts, ensure_ascii=False),
                citation_errors=json.dumps(report.citation_errors, ensure_ascii=False),
                answer_question_feedback=json.dumps(
                    answer_question_feedback, ensure_ascii=False
                ),
                context=context_str,
            ),
        },
    ]

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    fallback_used = False
    fallback_scenario: Optional[str] = None
    consistency_issues: list[str] = []
    answer_rewritten = False

    try:
        raw_output, p_tok, c_tok = llm_complete_with_usage(
            messages=messages,
            model=config.LLM_MODEL,
            temperature=regen_prompts.REGENERATE_TEMPERATURE,
            max_tokens=regen_prompts.REGENERATE_MAX_TOKENS,
            timeout=timeout,
        )
        prompt_tokens = p_tok
        completion_tokens = c_tok
        if on_llm_success:
            on_llm_success()

        parsed = _parse_draft_schema(raw_output)
        new_claims: list[Claim] = []
        for i, cs in enumerate(parsed.claims):
            orig_idx = cs.claim_idx if cs.claim_idx is not None else i
            new_claims.append(
                Claim(
                    text=cs.text.strip(),
                    source_ids=[s.strip() for s in cs.source_ids if s.strip()],
                    slot=cs.slot,
                    original_idx=orig_idx,
                )
            )

        refusal_reason = _normalize_refusal_reason(
            reason=parsed.refusal_reason,
            has_claims=bool(new_claims),
            refusal_forbidden=False,
        )
        candidate_draft = DraftAnswer(
            answer_text=parsed.answer_text.strip(),
            claims=new_claims,
            assumptions=list(dict.fromkeys(previous.assumptions + parsed.assumptions)),
            missing_info=list(dict.fromkeys(previous.missing_info + parsed.missing_info)),
            refusal_reason=refusal_reason,
        )
        validated_draft = _enforce_supported_locked(
            new_draft=candidate_draft,
            previous=previous,
            report=report,
            available_sources=available_sources,
        )
        locked_claims = _locked_claims_map(previous, report, available_sources)

        consistency_issues = _validate_draft_consistency(
            validated_draft,
            report,
            available_sources,
            locked_claims=locked_claims,
        )
        sanitized_draft = _sanitize_regenerated_answer(
            validated_draft, available_sources
        )

        if consistency_issues:
            # Сценарий C: LLM вернула JSON, но согласованность нарушена.
            fallback_used = True
            fallback_scenario = "C"
            answer_rewritten = True
            post_issues = _validate_draft_consistency(
                sanitized_draft,
                report,
                available_sources,
                locked_claims=locked_claims,
            )
            if post_issues:
                consistency_issues.extend(post_issues)
                validated_draft = _build_safe_regen_fallback(
                    previous,
                    report,
                    available_sources,
                    scenario="C",
                )
            else:
                validated_draft = sanitized_draft
        else:
            validated_draft = sanitized_draft
            answer_rewritten = (
                validated_draft.answer_text.strip() != previous.answer_text.strip()
            )
    except Exception as exc:
        if on_failure:
            on_failure(LLMFormatError(str(exc)) if not isinstance(exc, LLMError) else exc)
        # Полный старый answer_text не возвращается: текст пересобирается
        # детерминированно только из подтверждённых claims либо ответ пуст.
        fallback_used = True
        fallback_scenario = "A" if report.supported_locked else "B"
        answer_rewritten = True
        validated_draft = _build_safe_regen_fallback(
            previous,
            report,
            available_sources,
            scenario=fallback_scenario,
        )

    validated_draft = _finalize_regen_invariants(validated_draft)

    supported_locked_set = set(report.supported_locked)
    supported_claims_kept = sum(
        1
        for claim in validated_draft.claims
        if claim.original_idx in supported_locked_set
    )

    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step="regenerate_draft",
        input={
            "supported_locked": report.supported_locked,
            "failed_claims_count": len(failed_verdicts),
            "answer_question_verdict": answer_question_feedback["verdict"],
        },
        output={
            "claims_count": len(validated_draft.claims),
            "answer_length": len(validated_draft.answer_text),
            "fallback_used": fallback_used,
            "fallback_scenario": fallback_scenario,
            "consistency_issues_count": len(consistency_issues),
            "supported_claims_kept": supported_claims_kept,
            "answer_rewritten": answer_rewritten,
            "answer_question_feedback": answer_question_feedback,
        },
        latency_ms=latency_ms,
        model=config.LLM_MODEL,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stage_idx=6,
    )
    if fallback_used:
        step.degradation = "regen_fallback"
        step.extra["fallback_scenario"] = fallback_scenario
    return validated_draft, step