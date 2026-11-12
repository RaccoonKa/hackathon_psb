from dataclasses import replace
import logging
import re
import time
from src import config
from src.generation.types import (
    EvidenceAssessment,
    EvidenceBundle,
    PolicyVerdict,
    QuerySpec,
    RetrievedChunk,
    Sufficiency,
    TraceStep,
)

logger = logging.getLogger(__name__)

_CHUNK_INJECTION_PATTERNS = [
    re.compile(r"игнорируй(?:те)?\s+(?:все\s+)?(?:другие|предыдущие|остальные)\s+(?:документы|инструкции|правила)", re.IGNORECASE),
    re.compile(r"новое\s+системное\s+(?:правило|указание)", re.IGNORECASE),
    re.compile(r"внимание\s+(?:для\s+)?(?:ии|ассистента|модели)", re.IGNORECASE),
    re.compile(r"измени(?:ть)?\s+порядок\s+ответов", re.IGNORECASE),
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
]

_ISO_DATE_FULL_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DOC_OR_AMEND_REF_RE = re.compile(
    r"(?:дополнительн\w+\s+соглашен|договор|регламент|приказ|письм|записк|№|\bno\b)",
    re.IGNORECASE,
)


def _normalize_for_match(val: str) -> str:
    cleaned = val.strip().lower().replace("ё", "е")
    cleaned = re.sub(r"\bno\.?\s*(?=\d)", "№", cleaned)
    return re.sub(r"\s+", "", cleaned)


def _candidate_forms(val: str) -> list[str]:
    norm = _normalize_for_match(val)
    forms = [norm]
    m_iso = _ISO_DATE_FULL_RE.match(val.strip())
    if m_iso:
        year, month, day = m_iso.groups()
        forms.append(f"{day}.{month}.{year}")
        forms.append(f"{int(day)}.{month}.{year}")
    return forms


def precheck(
    spec: QuerySpec,
    bundle: EvidenceBundle,
    assessment: EvidenceAssessment,
) -> tuple[PolicyVerdict, TraceStep]:
    start_time = time.perf_counter()
    premise_mismatch = False
    premise_details: list[str] = []

    if config.ENABLE_PREMISE_CHECK and spec.premises and bundle.chunks:
        combined_raw = " ".join(
            f"{c.document_title} {c.effective_from or ''} {c.effective_to or ''} {c.text}"
            for c in bundle.chunks
        )
        combined_text_norm = _normalize_for_match(combined_raw)
        entity_norms = {_normalize_for_match(e) for e in spec.entities_raw if e.strip()}

        for premise in spec.premises:
            raw_val = (premise.value or "").strip()
            if not raw_val or not re.search(r"\d", raw_val):
                premise.verified = True
                continue
            if spec.event_date and raw_val == spec.event_date:
                premise.verified = True
                continue
            if _DOC_OR_AMEND_REF_RE.search(raw_val) or _normalize_for_match(raw_val) in entity_norms:
                premise.verified = True
                continue

            forms = _candidate_forms(raw_val)
            if any(f and f in combined_text_norm for f in forms):
                premise.verified = True
            else:
                premise.verified = False
                premise_mismatch = True
                premise_details.append(
                    f"Предпосылка пользователя '{premise.text}' (значение '{raw_val}') "
                    "не подтверждается текстом найденных документов."
                )

    injection_flagged = False
    updated_chunks: list[RetrievedChunk] = []

    for ch in bundle.chunks:
        if config.ENABLE_INJECTION_GUARD and any(
            pat.search(ch.text) for pat in _CHUNK_INJECTION_PATTERNS
        ):
            injection_flagged = True
            updated_chunks.append(replace(ch, untrusted=True))
        else:
            updated_chunks.append(ch)

    bundle.chunks = updated_chunks
    refusal_forbidden = assessment.sufficiency == Sufficiency.SUFFICIENT

    verdict = PolicyVerdict(
        premise_mismatch=premise_mismatch,
        premise_details=premise_details,
        injection_flagged=injection_flagged,
        refusal_forbidden=refusal_forbidden,
    )

    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step="policy_precheck",
        input={
            "premises_count": len(spec.premises),
            "chunks_count": len(bundle.chunks),
            "sufficiency": assessment.sufficiency.value,
        },
        output={
            "premise_mismatch": premise_mismatch,
            "premise_details": premise_details,
            "injection_flagged": injection_flagged,
            "refusal_forbidden": refusal_forbidden,
        },
        latency_ms=latency_ms,
        stage_idx=5,
    )
    return verdict, step