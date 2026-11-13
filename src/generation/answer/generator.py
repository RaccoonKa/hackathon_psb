from dataclasses import asdict
import json
import logging
import re
import time
from typing import Callable, Optional
from pydantic import ValidationError
from src import config
from src.generation.answer.context_builder import format_single_chunk
from src.generation.errors import LLMError, LLMFormatError
from src.generation.llm.client import llm_complete_with_usage
from src.generation.prompts import generator as gen_prompts
from src.generation.prompts import regenerate as regen_prompts
from src.generation.types import (
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
        role_raw = str(turn.get("role") or "user").strip().lower()
        if role_raw != "user":
            raw_content = _HISTORY_SOURCE_TAG_RE.sub("", raw_content).strip()
        if not raw_content:
            continue
        if len(raw_content) > max_chars:
            raw_content = raw_content[:max_chars].rstrip() + "..."
        role_label = "Пользователь" if role_raw == "user" else "Ассистент"
        lines.append(f"{role_label}: {raw_content}")

    if not lines:
        return ""

    history_body = "\n".join(lines)
    return (
        "История предыдущих сообщений диалога (используй только для понимания контекста беседы; "
        "все факты и теги <source> бери СТРОГО из раздела «Контекст (источники)»):\n"
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
                    premise_details=json.dumps(policy.premise_details, ensure_ascii=False),
                    version_issues=json.dumps(evidence.version_issues, ensure_ascii=False),
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
                context=context_str,
            ),
        },
    ]

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

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
    except Exception as exc:
        if on_failure:
            on_failure(LLMFormatError(str(exc)) if not isinstance(exc, LLMError) else exc)
        kept_claims = [
            c for c in previous.claims if c.original_idx in set(report.supported_locked)
        ]
        validated_draft = DraftAnswer(
            answer_text=previous.answer_text,
            claims=kept_claims,
            assumptions=list(previous.assumptions),
            missing_info=list(previous.missing_info),
            refusal_reason=None if kept_claims else previous.refusal_reason,
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
        },
        output={
            "claims_count": len(validated_draft.claims),
            "answer_length": len(validated_draft.answer_text),
        },
        latency_ms=latency_ms,
        model=config.LLM_MODEL,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stage_idx=6,
    )
    return validated_draft, step