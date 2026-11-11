from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Callable, Optional
from src import config
from src.generation.errors import LLMError, LLMFormatError
from src.generation.llm.client import llm_complete_with_usage
from src.generation.prompts import evidence as ev_prompts
from src.generation.types import (
    ContradictionResolutionSchema,
    EvidenceAssessment,
    EvidenceBundle,
    QuerySpec,
    RetrievedChunk,
    Sufficiency,
    TraceStep,
)
from src.retrieval import hybrid as hybrid_module

logger = logging.getLogger(__name__)

CHUNKS_JSONL_PATH = Path("data/chunks.jsonl")

SLOT_KEYWORDS: dict[str, list[str]] = {
    "rent_amount": ["арендн", "плат", "рубл", "ндс", "стоимост", "месяц"],
    "payment_deadline": ["не позднее", "числ", "оплат", "расчет", "рабочих дней", "срок"],
    "probation_period": ["испытани", "испытательн", "месяц", "прием", "бухгалтер", "руководител"],
    "overtime_limit": ["сверхурочн", "120", "часов", "год", "четырех часов"],
    "overtime_pay": ["сверхурочн", "полуторн", "двойн", "первые два часа", "оплат"],
    "approval_process": ["согласован", "визирован", "департамент", "подписан", "регламент"],
    "termination": ["расторжен", "прекращен", "отказ", "уведомлен", "сторон"],
    "penalty": ["неустойк", "пен", "штраф", "просрочк", "процент"],
    "liability": ["ответственност", "убытк", "возмещен", "штраф"],
    "tax_rate": ["ставк", "налог", "процент", "ндс", "прибыл", "освобожден"],
    "effective_date": ["редакц", "действ", "вступает", "соглашен", "дат"],
}

_NUMERIC_CONFLICT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("money_rub", re.compile(r"(\d[\d\s]{2,12})\s*(?:руб|₽)", re.IGNORECASE)),
    ("percent", re.compile(r"(\d+(?:[.,]\d+)?)\s*%")),
    ("days_or_months", re.compile(r"(\d+)\s*(?:рабочих\s+дней|календарных\s+дней|месяц\w*|часов)", re.IGNORECASE)),
]


@lru_cache(maxsize=1)
def _load_chunks_for_version_check() -> dict[str, dict]:
    loader = getattr(hybrid_module, "load_canonical_chunks_map", None)
    if callable(loader):
        try:
            loaded = loader()
            if isinstance(loaded, dict) and loaded:
                return loaded
            if loaded:
                return {
                    (ch.get("chunk_id") or ch.get("id")): ch
                    for ch in loaded
                    if isinstance(ch, dict) and (ch.get("chunk_id") or ch.get("id"))
                }
        except Exception as exc:
            logger.warning("load_canonical_chunks_map failed: %s", exc)

    candidate_paths = [CHUNKS_JSONL_PATH]
    if hasattr(config, "PROCESSED_DATA_DIR"):
        candidate_paths.insert(0, Path(config.PROCESSED_DATA_DIR) / "chunks.jsonl")

    store: dict[str, dict] = {}
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    cid = chunk.get("chunk_id") or chunk.get("id")
                    if cid:
                        store[str(cid)] = chunk
            if store:
                break
        except Exception as exc:
            logger.warning("Fallback chunks.jsonl reader failed on %s: %s", path, exc)
    return store


def _detect_contradictions(
    chunks: list[RetrievedChunk],
    slot_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    by_id = {c.chunk_id: c for c in chunks}
    contradictions: list[dict[str, Any]] = []

    for slot, cids in slot_map.items():
        if len(cids) < 2:
            continue
        for pat_name, pattern in _NUMERIC_CONFLICT_PATTERNS:
            values_by_chunk: dict[str, set[str]] = {}
            for cid in cids:
                ch = by_id.get(cid)
                if not ch:
                    continue
                matches = {
                    re.sub(r"\s+", "", m)
                    for m in pattern.findall(ch.text)
                    if m.strip()
                }
                if matches:
                    values_by_chunk[cid] = matches

            if len(values_by_chunk) >= 2:
                distinct_sets = list(values_by_chunk.values())
                first_set = distinct_sets[0]
                if any(s != first_set for s in distinct_sets[1:]):
                    contradictions.append(
                        {
                            "slot": slot,
                            "pattern": pat_name,
                            "chunk_ids": list(values_by_chunk.keys()),
                            "values": {k: sorted(v) for k, v in values_by_chunk.items()},
                        }
                    )
    return contradictions


def _resolve_contradictions_with_llm(
    spec: QuerySpec,
    chunks: list[RetrievedChunk],
    contradictions: list[dict[str, Any]],
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
    timeout: float | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    chunks_text = "\n\n".join(
        f"[{c.chunk_id}] (doc={c.document_id}, from={c.effective_from}, to={c.effective_to}): {c.text}"
        for c in chunks
    )
    messages = [
        {"role": config.ROLE_SYSTEM, "content": ev_prompts.EVIDENCE_CONTRADICTION_SYSTEM},
        {
            "role": config.ROLE_USER,
            "content": ev_prompts.EVIDENCE_CONTRADICTION_USER.format(
                query=spec.rewritten_queries_dense[0] if spec.rewritten_queries_dense else "",
                event_date=spec.event_date or "не указана",
                contradictions_text=json.dumps(contradictions, ensure_ascii=False),
                chunks_text=chunks_text,
            ),
        },
    ]
    try:
        raw, p_tok, c_tok = llm_complete_with_usage(
            messages=messages,
            model=config.LLM_MODEL,
            temperature=ev_prompts.EVIDENCE_TEMPERATURE,
            max_tokens=ev_prompts.EVIDENCE_MAX_TOKENS,
            timeout=timeout,
        )
        if on_llm_success:
            on_llm_success()
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = ContradictionResolutionSchema.model_validate_json(cleaned)
        res_by_slot = {r.slot: r for r in parsed.resolutions}
        unresolved: list[dict[str, Any]] = []
        for item in contradictions:
            res = res_by_slot.get(item["slot"])
            if res is None or res.is_real_contradiction:
                if res and res.explanation:
                    item["explanation"] = res.explanation
                unresolved.append(item)
        return unresolved, p_tok, c_tok
    except Exception as exc:
        if on_failure:
            on_failure(LLMFormatError(str(exc)) if not isinstance(exc, LLMError) else exc)
        return contradictions, 0, 0


def assess(
    spec: QuerySpec,
    chunks: list[RetrievedChunk],
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
    timeout: float | None = None,
) -> tuple[EvidenceBundle, EvidenceAssessment, TraceStep]:
    start_time = time.perf_counter()
    required_slots = spec.required_slots or ["general"]
    slot_map: dict[str, list[str]] = {slot: [] for slot in required_slots}
    covered_slots: list[str] = []
    missing_slots: list[str] = []

    if chunks:
        for slot in required_slots:
            keywords = SLOT_KEYWORDS.get(slot)
            if not keywords:
                slot_map[slot] = [c.chunk_id for c in chunks]
                covered_slots.append(slot)
                continue

            matched_cids = [
                c.chunk_id
                for c in chunks
                if any(kw in c.text.lower() for kw in keywords)
            ]
            if matched_cids:
                slot_map[slot] = matched_cids
                covered_slots.append(slot)
            elif len(required_slots) == 1:
                slot_map[slot] = [c.chunk_id for c in chunks]
                covered_slots.append(slot)
            else:
                missing_slots.append(slot)
    else:
        missing_slots = list(required_slots)

    core_slot = required_slots[0]
    core_slot_covered = bool(chunks) and (core_slot in covered_slots)

    if not spec.event_date:
        version_issues: list[dict[str, Any]] = []
    else:
        chunks_map = _load_chunks_for_version_check()
        version_issues = []
        for doc_ref in spec.entities_resolved:
            if not doc_ref.document_id:
                continue
            doc_chunks = [
                c
                for c in chunks_map.values()
                if c.get("document_id") == doc_ref.document_id
            ]
            if not doc_chunks:
                continue
            all_missing_meta = all(
                c.get("effective_from") is None for c in doc_chunks
            )
            if all_missing_meta:
                version_issues.append(
                    {
                        "kind": "no_effective_metadata",
                        "document_id": doc_ref.document_id,
                    }
                )
                continue
            covered = any(
                (c.get("effective_from") is None or c["effective_from"] <= spec.event_date)
                and (c.get("effective_to") is None or spec.event_date < c["effective_to"])
                for c in doc_chunks
            )
            if not covered:
                version_issues.append(
                    {
                        "kind": "no_matching_version",
                        "document_id": doc_ref.document_id,
                        "needed_date": spec.event_date,
                        "found_from": sorted(
                            {
                                c.get("effective_from")
                                for c in doc_chunks
                                if c.get("effective_from")
                            }
                        ),
                    }
                )

    contradictions = _detect_contradictions(chunks, slot_map)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    if contradictions:
        contradictions, p_tok, c_tok = _resolve_contradictions_with_llm(
            spec=spec,
            chunks=chunks,
            contradictions=contradictions,
            on_failure=on_failure,
            on_llm_success=on_llm_success,
            timeout=timeout,
        )
        if p_tok or c_tok:
            prompt_tokens = p_tok
            completion_tokens = c_tok

    if not core_slot_covered:
        sufficiency = Sufficiency.INSUFFICIENT
        reason = "core_slot_not_covered"
    elif missing_slots or contradictions or version_issues:
        sufficiency = Sufficiency.PARTIAL
        reason = "partial_slots_or_conflicts"
    else:
        sufficiency = Sufficiency.SUFFICIENT
        reason = "all_required_slots_covered"

    degraded_intents: list[str] = []
    if version_issues:
        degraded_intents.append("version_mismatch")

    bundle = EvidenceBundle(
        chunks=chunks,
        slot_map=slot_map,
        contradictions=contradictions,
        version_issues=version_issues,
    )
    assessment = EvidenceAssessment(
        covered_slots=covered_slots,
        missing_slots=missing_slots,
        core_slot_covered=core_slot_covered,
        sufficiency=sufficiency,
        degraded_intents=degraded_intents,
        reason=reason,
    )

    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step="assess_evidence",
        input={
            "chunks_count": len(chunks),
            "required_slots": required_slots,
            "event_date": spec.event_date,
        },
        output={
            "sufficiency": sufficiency.value,
            "covered_slots": covered_slots,
            "missing_slots": missing_slots,
            "contradictions_count": len(contradictions),
            "version_issues_count": len(version_issues),
        },
        latency_ms=latency_ms,
        model=config.LLM_MODEL if prompt_tokens is not None else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stage_idx=4,
    )
    return bundle, assessment, step