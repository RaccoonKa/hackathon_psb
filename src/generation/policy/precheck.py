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


def _normalize_digits(val: str) -> str:
    return re.sub(r"\s+", "", val.strip().lower())


def precheck(
    spec: QuerySpec,
    bundle: EvidenceBundle,
    assessment: EvidenceAssessment,
) -> tuple[PolicyVerdict, TraceStep]:
    start_time = time.perf_counter()
    premise_mismatch = False
    premise_details: list[str] = []

    if config.ENABLE_PREMISE_CHECK and spec.premises and bundle.chunks:
        combined_text_norm = _normalize_digits(
            " ".join(c.text for c in bundle.chunks)
        )
        for premise in spec.premises:
            if not premise.value:
                premise.verified = True
                continue
            val_norm = _normalize_digits(premise.value)
            if val_norm and val_norm in combined_text_norm:
                premise.verified = True
            else:
                premise.verified = False
                premise_mismatch = True
                premise_details.append(
                    f"Предпосылка пользователя '{premise.text}' (значение '{premise.value}') "
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