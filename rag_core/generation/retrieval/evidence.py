from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import statistics
import time
from typing import Any, Callable, Optional
from rag_core import config
from rag_core.generation.errors import LLMError, LLMFormatError
from rag_core.generation.llm.client import llm_complete_with_usage
from rag_core.generation.prompts import evidence as ev_prompts
from rag_core.generation.types import (
    ContradictionResolutionSchema,
    EvidenceAssessment,
    EvidenceBundle,
    QuerySpec,
    RetrievedChunk,
    Sufficiency,
    TraceStep,
)
from rag_core.retrieval import hybrid as hybrid_module

logger = logging.getLogger(__name__)

CHUNKS_JSONL_PATH = Path("data/chunks.jsonl")

# Сильные термины слота: одиночного совпадения достаточно для покрытия.
# Если для слота не задано отдельного списка, все его ключевые слова
# считаются сильными (обратная совместимость).
SLOT_KEYWORDS_STRONG: dict[str, list[str]] = {
    "rent_amount": [
        "арендная плата составляет",
        "рублей в месяц",
        "размер арендной платы",
        "стоимость аренды",
    ],
    "payment_deadline": [
        "не позднее",
        "числа текущего месяца",
        "срок внесения арендной платы",
        "срок оплаты",
    ],
}

SLOT_KEYWORDS: dict[str, list[str]] = {
    "security_deposit": ["обеспечительн", "депозит", "месячной арендной платы", "при подписании договора"],
    "penalty": ["пен", "неустойк", "штраф", "%", "процент", "просрочк"],
    "accrual_period": ["за каждый день", "день просрочки", "период просрочки", "со дня", "начисляются за"],
    "liability_cap": ["огранич", "ответственност", "реальн", "ущерб", "упущенн", "предел", "совокупн", "не несут", "убытк", "размер ответственности"],
    "rent_amount": ["арендная плата составляет", "рублей в месяц", "включая ндс", "450 000", "размер арендной платы по настоящему"],
    "payment_deadline": ["не позднее", "числа текущего месяца", "вносится", "ежемесячн", "срок внесения арендной платы", "срок оплаты", "порядок расчетов", "пункт 3.2"],
    "probation_period": ["испытани", "испытательн", "шести месяцев", "трех месяцев", "двух недель"],
    "overtime_limit": ["сверхурочн", "120 часов", "четырех часов", "двух дней подряд"],
    "overtime_pay": ["сверхурочн", "полуторн", "двойн", "первые два часа", "последующие часы"],
    "approval_process": ["согласован", "генеральн", "директор", "одобрен", "инициатор", "руководитель подразделения"],
    "termination": ["расторжен", "прекращен", "односторонн", "уведомлен"],
    "liability": ["ответственност", "убытк", "возмещен", "реальный ущерб"],
    "tax_rate": ["ставк", "налог", "процент", "ндс", "прибыл", "освобожден"],
    "effective_date": ["вступает в силу", "применяется с", "действует с", "редакц"],
    "delivery_terms": ["адрес доставки", "включает стоимость упаковки", "доставк", "склад покупателя"],
    "acceptance_terms": ["приемк", "по качеству", "рабочих дней", "претензи", "акт о выявленных недостатках"],
    "warranty_period": ["гарантийн", "гарантия", "талон", "производител", "месяц"],
    "prepayment": ["предоплат", "аванс", "по соглашению сторон", "в счете поставщика"],
    "return_procedure": ["возвращ", "по акту", "ключ", "нормального износа"],
    "document_handover": ["оригинал", "юридический отдел", "трех рабочих дней", "по заявке в сэд"],
    "consent_withdrawal": ["отозвать согласие", "продолжить обработку", "без согласия", "персональных данных"],
    "seasonal_work": ["сезонн", "срочный трудовой договор", "определенного периода"],
}


def _field(chunk: Any, name: str, default: Any = None) -> Any:
    """Доступ к полю чанка, совместимый с dict и pydantic/dataclass."""
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    if hasattr(chunk, name):
        return getattr(chunk, name)
    model_dump = getattr(chunk, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump().get(name, default)
        except Exception:  # noqa: BLE001 - защищаемся от чужих моделей
            return default
    return default


def _slot_matches(body: str, slot: str) -> bool:
    """Покрытие слота подтверждается текстом: ≥1 сильный или ≥2 слабых."""
    keywords = SLOT_KEYWORDS.get(slot) or []
    if not keywords:
        return False
    strong = SLOT_KEYWORDS_STRONG.get(slot) or keywords
    if any(term in body for term in strong):
        return True
    weak_hits = sum(
        1
        for term in keywords
        if term not in strong and term in body
    )
    return weak_hits >= 2


_HEADER_LINE_RE = re.compile(
    r"^\s*(?:раздел\s+\d+|глава\s+\d+|статья\s+\d+|пункт\s+\d+(?:\.\d+)*|\d+\.\s+[А-ЯЁA-Z][^\n.;:]{0,70})\s*$",
    re.IGNORECASE,
)

_NUMERIC_CONFLICT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("money_rub", re.compile(r"(\d[\d\s]{2,12})\s*(?:руб|₽)", re.IGNORECASE)),
    ("percent", re.compile(r"(\d+(?:[.,]\d+)?)\s*%")),
    ("days_or_months", re.compile(r"(\d+)\s*(?:рабочих\s+дней|календарных\s+дней|месяц\w*|часов)", re.IGNORECASE)),
]


def _extract_substantive_body(chunk: RetrievedChunk) -> str:
    text = str(_field(chunk, "text", "") or "")
    structure_path = list(_field(chunk, "structure_path", []) or [])
    document_title = str(_field(chunk, "document_title", "") or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    path_titles = {str(p).strip().lower() for p in structure_path if str(p).strip()}
    doc_title_lower = document_title.strip().lower()

    body_lines: list[str] = []
    for line in lines:
        low = line.lower().rstrip(".")
        if low == doc_title_lower or low in path_titles:
            continue
        if _HEADER_LINE_RE.match(line) and not re.search(r"\d+\s*(?:%|руб|дн|мес|час)", low):
            continue
        body_lines.append(line)

    body = " ".join(body_lines).strip()
    if len(body) < 25:
        return ""
    return body.lower()


def _detect_retrieval_fallback(chunks: list[RetrievedChunk]) -> bool:
    if not chunks:
        return False
    scores = [float(_field(c, "score", 0.0) or 0.0) for c in chunks]
    max_score = max(scores)
    if max_score >= 0.1:
        return False
    if len(scores) >= 2:
        med = statistics.median(scores)
        if abs(max_score - med) <= 0.015:
            return True
    return max_score < 0.05


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
    # Сравниваем числа только между чанками с одинаковым provision_id (например, старая и новая редакции одного пункта),
    # чтобы не принимать разные пункты одного договора за противоречие
    by_id = {str(_field(c, "chunk_id")): c for c in chunks}
    contradictions: list[dict[str, Any]] = []

    for slot, cids in slot_map.items():
        if len(cids) < 2:
            continue
        by_provision: dict[str, list[RetrievedChunk]] = {}
        for cid in cids:
            ch = by_id.get(cid)
            provision_id = _field(ch, "provision_id") if ch is not None else None
            if ch is not None and provision_id:
                by_provision.setdefault(str(provision_id), []).append(ch)

        for prov_id, group_chunks in by_provision.items():
            if len(group_chunks) < 2:
                continue
            for pat_name, pattern in _NUMERIC_CONFLICT_PATTERNS:
                values_by_chunk: dict[str, set[str]] = {}
                for ch in group_chunks:
                    text = str(_field(ch, "text", "") or "")
                    matches = {
                        re.sub(r"\s+", "", m)
                        for m in pattern.findall(text)
                        if m.strip()
                    }
                    if matches:
                        values_by_chunk[str(_field(ch, "chunk_id"))] = matches

                if len(values_by_chunk) >= 2:
                    distinct_sets = list(values_by_chunk.values())
                    first_set = distinct_sets[0]
                    if any(s != first_set for s in distinct_sets[1:]):
                        contradictions.append(
                            {
                                "slot": slot,
                                "provision_id": prov_id,
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

    retrieval_fallback = _detect_retrieval_fallback(chunks)
    min_viable_score = 0.005

    substantive_bodies: dict[str, str] = {}
    viable_chunks: list[RetrievedChunk] = []
    for c in chunks:
        chunk_id = str(_field(c, "chunk_id"))
        score = float(_field(c, "score", 0.0) or 0.0)
        if not chunk_id or score < min_viable_score:
            continue
        body = _extract_substantive_body(c)
        if body:
            substantive_bodies[chunk_id] = body
            viable_chunks.append(c)

    if viable_chunks:
        for slot in required_slots:
            if slot not in SLOT_KEYWORDS:
                custom_terms = [w.lower() for w in slot.split("_") if len(w) >= 4]
                if custom_terms:
                    matched_cids = [
                        cid
                        for cid, body in substantive_bodies.items()
                        if any(t in body for t in custom_terms)
                    ]
                else:
                    matched_cids = list(substantive_bodies)

                if matched_cids or slot == "general":
                    slot_map[slot] = matched_cids or list(substantive_bodies)
                    covered_slots.append(slot)
                else:
                    missing_slots.append(slot)
                continue

            matched_cids = [
                cid
                for cid, body in substantive_bodies.items()
                if _slot_matches(body, slot)
            ]
            if matched_cids:
                slot_map[slot] = matched_cids
                covered_slots.append(slot)
            else:
                missing_slots.append(slot)
    else:
        missing_slots = list(required_slots)

    core_slot = required_slots[0]
    core_slot_covered = bool(viable_chunks) and (core_slot in covered_slots)

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

    contradictions = _detect_contradictions(viable_chunks, slot_map)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    if contradictions:
        contradictions, p_tok, c_tok = _resolve_contradictions_with_llm(
            spec=spec,
            chunks=viable_chunks,
            contradictions=contradictions,
            on_failure=on_failure,
            on_llm_success=on_llm_success,
            timeout=timeout,
        )
        if p_tok or c_tok:
            prompt_tokens = p_tok
            completion_tokens = c_tok

    degraded_intents: list[str] = []
    if version_issues:
        degraded_intents.append("version_mismatch")
    if retrieval_fallback:
        degraded_intents.append("reranker_fallback_rrf")

    if not core_slot_covered:
        sufficiency = Sufficiency.INSUFFICIENT
        reason = "core_slot_not_covered"
    elif missing_slots:
        sufficiency = Sufficiency.PARTIAL
        reason = "missing_required_slots"
    elif contradictions or version_issues:
        sufficiency = Sufficiency.PARTIAL
        reason = "conflicts_or_version_issues"
    elif retrieval_fallback:
        sufficiency = Sufficiency.PARTIAL
        reason = "reranker_fallback_rrf_mode"
    else:
        sufficiency = Sufficiency.SUFFICIENT
        reason = "all_required_slots_covered"

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
            "viable_chunks_count": len(viable_chunks),
            "required_slots": required_slots,
            "event_date": spec.event_date,
        },
        output={
            "sufficiency": sufficiency.value,
            "reason": reason,
            "required_slots": required_slots,
            "covered_slots": covered_slots,
            "uncovered_slots": missing_slots,
            "missing_slots": missing_slots,
            "core_slot_covered": core_slot_covered,
            "min_viable_score": min_viable_score,
            "checked_chunks_count": len(viable_chunks),
            "slot_match_counts": {
                slot: len(cids) for slot, cids in slot_map.items()
            },
            "retrieval_fallback": retrieval_fallback,
            "contradictions_count": 0 if not contradictions else len(contradictions),
            "version_issues_count": len(version_issues),
        },
        latency_ms=latency_ms,
        model=config.LLM_MODEL if prompt_tokens is not None else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        stage_idx=4,
    )
    return bundle, assessment, step