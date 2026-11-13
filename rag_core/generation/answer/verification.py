import json
import logging
import re
import time
from typing import Callable, Optional
from pydantic import ValidationError
from rag_core import config
from rag_core.generation.errors import LLMError, LLMFormatError
from rag_core.generation.llm.client import llm_complete_with_usage
from rag_core.generation.prompts import verifier as ver_prompts
from rag_core.generation.types import (
    Claim,
    ClaimStatus,
    ClaimVerdict,
    DraftAnswer,
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
    # вторичные claims с интерпретациями на том же слоте считаются non-core
    req_set = set(required_slots) if required_slots else set()
    core_claims: list[Claim] = []
    non_core_claims: list[Claim] = []
    secured_slots: set[str] = set()

    for idx, c in enumerate(claims):
        orig_idx = c.original_idx if c.original_idx is not None else idx
        st = status_by_idx.get(orig_idx, ClaimStatus.NOT_IN_SOURCE)
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


def verify(
    draft: DraftAnswer,
    chunks: list[RetrievedChunk],
    spec: QuerySpec,
    can_regenerate: bool = True,
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
    timeout: float | None = None,
) -> tuple[VerificationReport, TraceStep]:
    start_time = time.perf_counter()
    source_map: dict[str, RetrievedChunk] = {
        f"{config.SOURCE_ID_PREFIX}{idx}": ch
        for idx, ch in enumerate(chunks, start=config.SOURCE_START_INDEX)
    }

    if not draft.claims:
        has_text = bool(draft.answer_text.strip()) and not draft.refusal_reason
        verdict = VerificationVerdict.ACCEPT if has_text else VerificationVerdict.REFUSE
        report = VerificationReport(
            batch_status=[],
            citation_errors=[],
            injection_signals=[],
            supported_locked=[],
            final_verdict=verdict,
        )
        step = TraceStep(
            step="verify_draft",
            input={"claims_count": 0},
            output={"final_verdict": verdict.value},
            latency_ms=0.0,
            stage_idx=7,
        )
        return report, step

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

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    llm_verdicts_by_idx: dict[int, ClaimVerdict] = {}

    if config.ENABLE_LLM_VERIFIER and not (
        config.USE_STUB_LLM or getattr(config, "LLM_MODE", "real") == config.LLM_MODE_STUB
    ):
        messages = [
            {"role": config.ROLE_SYSTEM, "content": ver_prompts.VERIFIER_SYSTEM},
            {
                "role": config.ROLE_USER,
                "content": ver_prompts.VERIFIER_USER.format(
                    query=spec.rewritten_queries_dense[0] if spec.rewritten_queries_dense else "",
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
            for c_item in claims_payload:
                c_idx = c_item["claim_idx"]
                first_sid = c_item["source_ids"][0] if c_item["source_ids"] else None
                llm_verdicts_by_idx[c_idx] = ClaimVerdict(
                    claim_idx=c_idx,
                    status=ClaimStatus.SUPPORTED if first_sid in source_map else ClaimStatus.NOT_IN_SOURCE,
                    quote="",
                    source_id=first_sid,
                )
    else:
        for c_item in claims_payload:
            c_idx = c_item["claim_idx"]
            first_sid = c_item["source_ids"][0] if c_item["source_ids"] else None
            llm_verdicts_by_idx[c_idx] = ClaimVerdict(
                claim_idx=c_idx,
                status=ClaimStatus.SUPPORTED if first_sid in source_map else ClaimStatus.NOT_IN_SOURCE,
                quote="",
                source_id=first_sid,
            )

    batch_status: list[ClaimVerdict] = []
    citation_errors: list[str] = []
    injection_signals: list[str] = []
    supported_locked: list[int] = []

    for pos_idx, claim in enumerate(draft.claims):
        orig_idx = claim.original_idx if claim.original_idx is not None else pos_idx
        verdict_obj = llm_verdicts_by_idx.get(
            orig_idx,
            ClaimVerdict(
                claim_idx=orig_idx,
                status=ClaimStatus.NOT_IN_SOURCE,
                quote="",
                source_id=None,
            ),
        )

        invalid_sids = [sid for sid in claim.source_ids if sid not in source_map]
        if not claim.source_ids or invalid_sids:
            citation_errors.append(
                f"Claim {orig_idx} has invalid source_ids: {claim.source_ids}"
            )

        valid_cited_chunks = [
            source_map[sid] for sid in claim.source_ids if sid in source_map
        ]

        if valid_cited_chunks and any(ch.untrusted for ch in valid_cited_chunks):
            trusted_confirmations = [ch for ch in valid_cited_chunks if not ch.untrusted]
            if not trusted_confirmations:
                injection_signals.append(
                    f"Claim {orig_idx} relies solely on untrusted chunk"
                )
                verdict_obj.status = ClaimStatus.NOT_IN_SOURCE

        claim_dates = _ISO_DATE_RE.findall(claim.text)
        if claim_dates and valid_cited_chunks:
            for c_date in claim_dates:
                for ch in valid_cited_chunks:
                    if ch.effective_from and c_date < ch.effective_from:
                        verdict_obj.status = ClaimStatus.CONTRADICTED
                    if ch.effective_to and c_date >= ch.effective_to:
                        verdict_obj.status = ClaimStatus.CONTRADICTED

        if verdict_obj.status == ClaimStatus.SUPPORTED and not invalid_sids and claim.source_ids:
            supported_locked.append(orig_idx)

        batch_status.append(verdict_obj)

    status_by_idx = {v.claim_idx: v.status for v in batch_status}
    core_claims, non_core_claims = _partition_core_and_non_core_claims(
        draft.claims, spec.required_slots, status_by_idx
    )

    all_core_supported = all(
        status_by_idx.get(c.original_idx if c.original_idx is not None else 0) == ClaimStatus.SUPPORTED
        for c in core_claims
    )
    any_core_supported = any(
        status_by_idx.get(c.original_idx if c.original_idx is not None else 0) == ClaimStatus.SUPPORTED
        for c in core_claims
    )
    has_contradicted = any(v.status == ClaimStatus.CONTRADICTED for v in batch_status)
    all_non_core_supported = all(
        status_by_idx.get(c.original_idx if c.original_idx is not None else 0) == ClaimStatus.SUPPORTED
        for c in non_core_claims
    )

    if all_core_supported and all_non_core_supported and not citation_errors and not has_contradicted:
        final_verdict = VerificationVerdict.ACCEPT
    elif (not all_core_supported or has_contradicted or citation_errors) and can_regenerate and config.MAX_REGENERATE_CYCLES > 0:
        final_verdict = VerificationVerdict.REGENERATE
    elif any_core_supported:
        final_verdict = VerificationVerdict.PARTIAL
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
    else:
        final_verdict = VerificationVerdict.REFUSE

    report = VerificationReport(
        batch_status=batch_status,
        citation_errors=citation_errors,
        injection_signals=injection_signals,
        supported_locked=supported_locked,
        final_verdict=final_verdict,
    )

    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step="verify_draft",
        input={
            "claims_count": len(claims_payload),
            "core_claims_count": len(core_claims),
            "non_core_claims_count": len(non_core_claims),
        },
        output={
            "final_verdict": final_verdict.value,
            "supported_locked": supported_locked,
            "citation_errors_count": len(citation_errors),
            "injection_signals_count": len(injection_signals),
        },
        latency_ms=latency_ms,
        model=config.LLM_MODEL if prompt_tokens is not None else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stage_idx=7,
    )
    return report, step