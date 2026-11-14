from dataclasses import replace
import logging
import re
import time
from rag_core import config
from rag_core.generation.types import (
    EvidenceAssessment,
    EvidenceBundle,
    PolicyVerdict,
    Premise,
    PremiseStatus,
    PremiseStatusEntry,
    QuerySpec,
    RetrievedChunk,
    Sufficiency,
    TraceStep,
)

logger = logging.getLogger(__name__)

_CHUNK_INJECTION_PATTERNS = [
    re.compile(r"игнорируй(?:те)?\s+(?:все\s+)?(?:другие|предыдущие|остальные)\s+(?:документы|инструкции|правила)", re.IGNORECASE),
    re.compile(r"новое\s+системное\s+(?:правило|указание)", re.IGNORECASE),
    re.compile(r"измени(?:ть)?\s+порядок\s+ответов", re.IGNORECASE),
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
]

_COMPARABLE_KINDS = frozenset({"amount", "term", "rate", "date"})

_TERM_UNIT_SUFFIX = {
    "day": "d",
    "week": "w",
    "month": "m",
    "year": "y",
    "hour": "h",
    "minute": "min",
}

# Основы служебных слов, которые не должны участвовать в проверке anchors:
# они встречаются почти в любом предложении и не идентифицируют параметр.
_ANCHOR_STOPWORD_ROOTS = frozenset({
    "дого", "стор", "разм", "сумм", "сост", "рубл", "меся", "срок", "дата",
    "быть", "имет", "имее", "такж", "кото", "наст", "согл", "данн", "указ",
    "опре", "явля", "иной", "этот", "если", "посл", "пере", "чере", "всег",
    "копе", "дней", "день", "проц", "пери", "числ", "обща", "общи", "може",
    "долж",
})
_ANCHOR_STOPWORDS_SHORT = frozenset({"для", "при", "над", "под", "без", "или"})

_MONTH_STEMS: tuple[tuple[str, str], ...] = (
    ("январ", "m01"),
    ("феврал", "m02"),
    ("март", "m03"),
    ("апрел", "m04"),
    ("май", "m05"),
    ("мая", "m05"),
    ("мае", "m05"),
    ("маю", "m05"),
    ("июн", "m06"),
    ("июл", "m07"),
    ("август", "m08"),
    ("сентябр", "m09"),
    ("октябр", "m10"),
    ("ноябр", "m11"),
    ("декабр", "m12"),
)
_MONTH_TOKEN_TO_NUMBER = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "май": 5, "мая": 5,
    "мае": 5, "маю": 5, "июн": 6, "июл": 7, "август": 8, "сентябр": 9,
    "октябр": 10, "ноябр": 11, "декабр": 12,
}

_PERIOD_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
_TEXT_DATE_RE = re.compile(r"\b(\d{1,2})\s+([а-я]+)\s+(\d{4})\b")

_AMOUNT_RE = re.compile(
    r"\d[\d\s\u00a0]*(?:[.,]\d+)?\s*"
    r"(?:млрд|млн|тысяч[а-я]*|тыс\.?|руб[а-я]*\.?|р\.|₽)",
    re.IGNORECASE,
)
_TERM_RE = re.compile(
    r"\d[\d\s\u00a0]*(?:[.,]\d+)?\s*"
    r"(?:дн[а-я]*|ден[а-я]*|сут[а-я]*|недел[а-я]*|месяц[а-я]*|мес\.?|"
    r"час[а-я]*|минут[а-я]*|год[а-я]*|лет)",
    re.IGNORECASE,
)
_RATE_RE = re.compile(
    r"\d[\d\s\u00a0]*(?:[.,]\d+)?\s*(?:%|процент[а-я]*)",
    re.IGNORECASE,
)
_DATE_RES = (_ISO_DATE_RE, _DMY_DATE_RE, _TEXT_DATE_RE)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def _normalize_text(value: str | None) -> str:
    return (value or "").lower().replace("ё", "е")


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part and part.strip()]


def _is_anchor_stopword(token: str) -> bool:
    if token in _ANCHOR_STOPWORDS_SHORT:
        return True
    if token in _ANCHOR_STOPWORD_ROOTS:
        return True
    return token[:4] in _ANCHOR_STOPWORD_ROOTS


def _extract_anchors(text: str) -> set[str]:
    """Основы значимых слов (до 5 символов) для проверки совпадения параметра."""
    normalized = _normalize_text(text)
    anchors: set[str] = set()
    for token in re.findall(r"[a-zа-я]+", normalized):
        if len(token) < 4 or _is_anchor_stopword(token):
            continue
        anchors.add(token[:5])
    return anchors


def _parse_number(raw: str) -> float | None:
    compact = re.sub(r"[\s\u00a0]", "", _normalize_text(raw))
    match = re.search(r"\d+(?:[.,]\d+)?", compact)
    if not match:
        return None
    number = float(match.group(0).replace(",", "."))
    if "млрд" in compact:
        number *= 1_000_000_000
    elif "млн" in compact:
        number *= 1_000_000
    elif "тыс" in compact:
        number *= 1_000
    return number


def _format_number(number: float) -> str:
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _month_number(token: str) -> int | None:
    for stem, number in _MONTH_TOKEN_TO_NUMBER.items():
        if token.startswith(stem):
            return number
    return None


def _normalize_date(raw: str) -> str | None:
    value = raw.strip().lower().replace("ё", "е")
    iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
    if iso:
        year, month, day = (int(g) for g in iso.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
        return None
    dmy = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", value)
    if dmy:
        day, month, year = (int(g) for g in dmy.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
        return None
    text_date = re.match(r"^(\d{1,2})\s+([а-я]+)\s+(\d{4})$", value)
    if text_date:
        day = int(text_date.group(1))
        month = _month_number(text_date.group(2))
        year = int(text_date.group(3))
        if month and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _value_unit_class(value: str, kind: str) -> str | None:
    """Класс единицы измерения значения; None — единица не распознана."""
    if kind not in _COMPARABLE_KINDS:
        return None
    normalized = _normalize_text(value)
    if kind == "amount":
        return "amount"
    if kind == "rate":
        return "rate"
    if kind == "date":
        return "date" if _normalize_date(normalized) else None
    if kind == "term":
        if "час" in normalized:
            return "hour"
        if "минут" in normalized:
            return "minute"
        if re.search(r"дн|ден|сут", normalized):
            return "day"
        if "недел" in normalized:
            return "week"
        if "мес" in normalized:
            return "month"
        if "год" in normalized or "лет" in normalized:
            return "year"
    return None


def _normalize_value(value: str | None, kind: str) -> str | None:
    """Каноническое представление значения или None, если единица не распознана."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if kind == "date":
        return _normalize_date(raw)
    if kind == "term":
        unit = _value_unit_class(raw, "term")
        if unit is None:
            return None
        number = _parse_number(raw)
        if number is None:
            return None
        return f"{_format_number(number)}{_TERM_UNIT_SUFFIX[unit]}"
    if kind in {"amount", "rate"}:
        number = _parse_number(raw)
        if number is None:
            return None
        return _format_number(number)
    return None


def _extract_values(text: str, kind: str) -> list[str]:
    if not text:
        return []
    normalized = _normalize_text(text)
    if kind == "amount":
        return [m.group(0) for m in _AMOUNT_RE.finditer(normalized)]
    if kind == "term":
        return [m.group(0) for m in _TERM_RE.finditer(normalized)]
    if kind == "rate":
        return [m.group(0) for m in _RATE_RE.finditer(normalized)]
    if kind == "date":
        found: list[str] = []
        for pattern in _DATE_RES:
            found.extend(m.group(0) for m in pattern.finditer(normalized))
        return found
    return []


def _extract_period_tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    tokens: set[str] = set(_PERIOD_YEAR_RE.findall(normalized))
    for stem, label in _MONTH_STEMS:
        if stem in normalized:
            tokens.add(label)
    for year, month, _day in _ISO_DATE_RE.findall(normalized):
        tokens.add(year)
        tokens.add(f"m{month}")
    for _day, month, year in _DMY_DATE_RE.findall(normalized):
        tokens.add(year)
        tokens.add(f"m{int(month):02d}")
    return tokens


def _period_compatible(
    premise_period: set[str],
    candidate_period: set[str],
    premise_has_period: bool,
) -> bool:
    """True, если предложение допустимо рассматривать для периода предпосылки."""
    if not premise_has_period:
        return True
    if not candidate_period:
        # Период в предложении не указан: VERIFIED по точному значению допустим,
        # запрет на CONTRADICTED проверяется отдельно в _evaluate_premise.
        return True
    return bool(premise_period & candidate_period)


def _entry(
    premise: Premise,
    status: PremiseStatus,
    *,
    reason: str | None = None,
    evidence_source_id: str | None = None,
    conflicting_value: str | None = None,
) -> PremiseStatusEntry:
    return PremiseStatusEntry(
        text=premise.text,
        kind=premise.kind,
        value=premise.value,
        status=status,
        reason=reason,
        evidence_source_id=evidence_source_id,
        conflicting_value=conflicting_value,
    )


def _evaluate_premise(
    premise: Premise,
    trusted_chunks: list[RetrievedChunk],
    spec: QuerySpec,
) -> PremiseStatusEntry:
    """Независимая оценка одной предпосылки только по доверенным чанкам."""
    if not config.ENABLE_PREMISE_CHECK:
        return _entry(premise, PremiseStatus.UNKNOWN, reason="check_disabled")
    if not trusted_chunks:
        return _entry(premise, PremiseStatus.UNKNOWN, reason="no_evidence")

    kind = _normalize_text(premise.kind).strip()
    raw_value = (premise.value or "").strip()
    if kind not in _COMPARABLE_KINDS or not re.search(r"\d", raw_value):
        # Нечисловые/несравнимые предпосылки не считаются подтверждёнными.
        return _entry(premise, PremiseStatus.UNKNOWN, reason="not_comparable")

    premise_value = _normalize_value(raw_value, kind)
    if premise_value is None:
        return _entry(premise, PremiseStatus.UNKNOWN, reason="non_normalizable_value")

    premise_anchors = _extract_anchors(f"{premise.text} {raw_value}")
    premise_period = _extract_period_tokens(premise.text)
    premise_has_period = bool(premise_period)
    premise_unit = _value_unit_class(raw_value, kind)

    any_anchored = False
    candidates: list[tuple[str, str | None, bool]] = []

    for chunk in trusted_chunks:
        for sentence in _split_sentences(chunk.text):
            candidate_anchors = _extract_anchors(sentence)
            if not (premise_anchors & candidate_anchors):
                continue
            candidate_period = _extract_period_tokens(sentence)
            if not _period_compatible(premise_period, candidate_period, premise_has_period):
                continue
            any_anchored = True
            sentence_values: list[str] = []
            for raw_candidate in _extract_values(sentence, kind):
                if kind == "term":
                    candidate_unit = _value_unit_class(raw_candidate, kind)
                    if candidate_unit is None:
                        continue
                    if premise_unit and candidate_unit != premise_unit:
                        # Разные единицы срока (дни/месяцы/часы) несопоставимы.
                        continue
                normalized = _normalize_value(raw_candidate, kind)
                if normalized is not None:
                    sentence_values.append(normalized)
            for normalized in sentence_values:
                candidates.append((normalized, chunk.chunk_id, bool(candidate_period)))

    if not any_anchored:
        return _entry(premise, PremiseStatus.UNKNOWN, reason="no_anchored_match")

    for value, source_id, _has_period in candidates:
        if value == premise_value:
            return _entry(
                premise,
                PremiseStatus.VERIFIED,
                reason="value_confirmed",
                evidence_source_id=source_id,
            )

    # CONTRADICTED допускается только при безопасных условиях:
    # - значение действительно найдено и нормализовано;
    # - нет периода предпосылки или период предложения тоже известен.
    safe_conflicts = [
        (value, source_id)
        for value, source_id, has_period in candidates
        if value != premise_value and (not premise_has_period or has_period)
    ]
    unique_values = {value for value, _source_id in safe_conflicts}
    if len(unique_values) == 1:
        conflicting_value = next(iter(unique_values))
        source_id = next(
            sid for value, sid in safe_conflicts if value == conflicting_value
        )
        return _entry(
            premise,
            PremiseStatus.CONTRADICTED,
            reason="value_mismatch",
            evidence_source_id=source_id,
            conflicting_value=conflicting_value,
        )
    if len(unique_values) > 1:
        return _entry(premise, PremiseStatus.UNKNOWN, reason="ambiguous_values")
    return _entry(premise, PremiseStatus.UNKNOWN, reason="no_anchored_match")


def _entry_to_dict(entry: PremiseStatusEntry) -> dict[str, object]:
    return {
        "text": entry.text,
        "kind": entry.kind,
        "value": entry.value,
        "status": entry.status.value,
        "reason": entry.reason,
        "evidence_source_id": entry.evidence_source_id,
        "conflicting_value": entry.conflicting_value,
    }


def precheck(
    spec: QuerySpec,
    bundle: EvidenceBundle,
    assessment: EvidenceAssessment,
) -> tuple[PolicyVerdict, TraceStep]:
    start_time = time.perf_counter()

    # (1) injection-guard по всем чанкам ДО оценки предпосылок, чтобы
    # недоверенный чанк не мог «опровергнуть» предпосылку.
    injection_flagged = False
    updated_chunks: list[RetrievedChunk] = []
    for ch in bundle.chunks:
        if config.ENABLE_INJECTION_GUARD and any(
            pat.search(ch.text) for pat in _CHUNK_INJECTION_PATTERNS
        ):
            injection_flagged = True
            safe_lines = [
                line for line in ch.text.splitlines()
                if not any(pat.search(line) for pat in _CHUNK_INJECTION_PATTERNS)
            ]
            cleaned_text = "\n".join(safe_lines).strip()
            if len(cleaned_text) >= 40:
                updated_chunks.append(replace(ch, text=cleaned_text, untrusted=False))
            else:
                updated_chunks.append(replace(ch, untrusted=True))
        else:
            updated_chunks.append(ch)
    bundle.chunks = updated_chunks

    # (2) только доверенные чанки участвуют в проверке предпосылок.
    trusted_chunks = [ch for ch in bundle.chunks if not ch.untrusted]
    untrusted_chunks_count = len(bundle.chunks) - len(trusted_chunks)

    # (3) каждая предпосылка оценивается отдельно.
    statuses: list[PremiseStatusEntry] = []
    for premise in spec.premises:
        entry = _evaluate_premise(premise, trusted_chunks, spec)
        if (
            config.ENABLE_PREMISE_CHECK
            and not trusted_chunks
            and bundle.chunks
            and entry.status == PremiseStatus.UNKNOWN
        ):
            entry = _entry(
                premise,
                PremiseStatus.UNKNOWN,
                reason="untrusted_source_only",
            )
        premise.mark_status(
            entry.status,
            reason=entry.reason,
            evidence_source_id=entry.evidence_source_id,
            conflicting_value=entry.conflicting_value,
        )
        statuses.append(entry)

    # (4) Семантика: mismatch — ТОЛЬКО подтверждённые противоречия.
    premise_mismatch = any(
        entry.status == PremiseStatus.CONTRADICTED for entry in statuses
    )
    premise_details = [
        (
            f"Предпосылка пользователя '{entry.text}' содержит значение "
            f"'{entry.value}'; подтверждённое значение в источнике: "
            f"'{entry.conflicting_value}'"
            + (
                f" (источник {entry.evidence_source_id})"
                if entry.evidence_source_id
                else ""
            )
        )
        for entry in statuses
        if entry.status == PremiseStatus.CONTRADICTED
    ]

    refusal_forbidden = assessment.sufficiency == Sufficiency.SUFFICIENT

    # (5) PolicyVerdict + trace.
    verdict = PolicyVerdict(
        premise_mismatch=premise_mismatch,
        premise_details=premise_details,
        injection_flagged=injection_flagged,
        refusal_forbidden=refusal_forbidden,
        premise_statuses=statuses,
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
            "trusted_chunks_count": len(trusted_chunks),
            "untrusted_chunks_count": untrusted_chunks_count,
            "sufficiency": assessment.sufficiency.value,
        },
        output={
            "premise_mismatch": premise_mismatch,
            "premise_details": premise_details,
            "premise_unknown_count": sum(
                1 for entry in statuses if entry.status == PremiseStatus.UNKNOWN
            ),
            "premises": [_entry_to_dict(entry) for entry in statuses],
            "injection_flagged": injection_flagged,
            "refusal_forbidden": refusal_forbidden,
        },
        latency_ms=latency_ms,
        stage_idx=5,
    )
    return verdict, step
