"""Консервативное определение назначения источника.

Модуль сопоставляет слоты запроса и текстовые маркеры с назначением
источника: нормативное основание (``normative``) или документ сделки
(``specific``). Не вводит отдельной классификационной подсистемы и
опирается только на уже существующие слоты и текстовые признаки.
"""

from typing import Iterable

from rag_core.generation.types import QuerySpec

SOURCE_KIND_NORMATIVE = "normative"
SOURCE_KIND_SPECIFIC = "specific"
SOURCE_KIND_UNKNOWN = "unknown"

# Нормативные слоты: вопросы о правилах, порядке и пределах.
NORMATIVE_SLOTS: frozenset[str] = frozenset(
    {
        "approval_process",
        "tax_rate",
        "overtime_limit",
        "overtime_pay",
        "probation_period",
        "consent_withdrawal",
        "seasonal_work",
        "document_handover",
    }
)

# Предметные слоты: конкретные условия конкретной сделки.
SPECIFIC_SLOTS: frozenset[str] = frozenset(
    {
        "rent_amount",
        "security_deposit",
        "payment_deadline",
        "penalty",
        "accrual_period",
        "liability_cap",
        "termination",
        "delivery_terms",
        "acceptance_terms",
        "warranty_period",
        "prepayment",
        "return_procedure",
        "effective_date",
    }
)

# Единый маппинг слот -> назначение источника.
STRICT_SLOT_SOURCE_KIND: dict[str, str] = {
    **{slot: SOURCE_KIND_NORMATIVE for slot in NORMATIVE_SLOTS},
    **{slot: SOURCE_KIND_SPECIFIC for slot in SPECIFIC_SLOTS},
}

# Текстовые маркеры нормативного характера (lower-case, подстроки).
_NORMATIVE_MARKERS: tuple[str, ...] = (
    "регламент",
    "положение",
    "инструкция",
    "правила",
    "порядок согласования",
    "порядок одобрения",
    "кто должен",
    "обязан",
    "вправе",
    "разрешено",
    "допускается",
    "порог",
    "одобрение",
    "визирование",
    "нормативный",
    "закон",
    "статья",
)

# Текстовые маркеры предметного характера (lower-case, подстроки).
_SPECIFIC_MARKERS: tuple[str, ...] = (
    "условия договора",
    "условия поставки",
    "условия аренды",
    "условия оплаты",
    "стороны",
    "контрагент",
    "поставщик",
    "покупатель",
    "арендодатель",
    "арендатор",
    "сумма",
    "цена",
    "стоимость",
    "товар",
    "поставка",
    "гарантия",
    "адрес доставки",
    "подписали",
    "заключили",
    "фактически",
)

_DOCUMENT_TYPE_TO_SOURCE_KIND: dict[str, str] = {
    "law": SOURCE_KIND_NORMATIVE,
    "internal": SOURCE_KIND_NORMATIVE,
    "contract": SOURCE_KIND_SPECIFIC,
}

_SOURCE_KIND_TO_DOCUMENT_TYPES: dict[str, frozenset[str]] = {
    SOURCE_KIND_NORMATIVE: frozenset({"law", "internal"}),
    SOURCE_KIND_SPECIFIC: frozenset({"contract"}),
}

_SOURCE_KIND_LABELS: dict[str, str] = {
    SOURCE_KIND_NORMATIVE: "нормативное основание",
    SOURCE_KIND_SPECIFIC: "документ сделки",
    SOURCE_KIND_UNKNOWN: "тип источника не определён",
}


def _normalize(value: str | None) -> str:
    """Приводит значение к lower-case без окружающих пробелов."""
    if not value:
        return ""
    return str(value).strip().lower()


def _normalize_slot(slot: str | None) -> str:
    """Нормализует идентификатор слота."""
    return _normalize(slot)


def _query_text(spec: QuerySpec) -> str:
    """Собирает доступный текст запроса для поиска маркеров."""
    parts: list[str] = []
    for attr in ("rewritten_queries_bm25", "rewritten_queries_dense"):
        values = getattr(spec, attr, None) or []
        for value in values:
            text = _normalize(value)
            if text:
                parts.append(text)
    return " ".join(parts)


def detect_required_source_kinds(spec: QuerySpec) -> list[str]:
    """Определяет требуемые назначения источника для запроса.

    Сначала учитываются слоты ``spec.required_slots``, затем — текстовые
    маркеры в сохранённых переформулировках запроса. Порядок результата
    фиксирован: сначала ``normative``, затем ``specific``. Пустой список
    означает, что фильтр по назначению источника не применяется.
    """
    slots = {
        _normalize_slot(slot)
        for slot in (getattr(spec, "required_slots", None) or [])
        if slot
    }
    has_normative = bool(slots & NORMATIVE_SLOTS)
    has_specific = bool(slots & SPECIFIC_SLOTS)

    text = _query_text(spec)
    if text:
        if any(marker in text for marker in _NORMATIVE_MARKERS):
            has_normative = True
        if any(marker in text for marker in _SPECIFIC_MARKERS):
            has_specific = True

    result: list[str] = []
    if has_normative:
        result.append(SOURCE_KIND_NORMATIVE)
    if has_specific:
        result.append(SOURCE_KIND_SPECIFIC)
    return result


def source_kind_for_document_type(document_type: str | None) -> str:
    """Сопоставляет тип документа с назначением источника."""
    normalized = _normalize(document_type)
    return _DOCUMENT_TYPE_TO_SOURCE_KIND.get(normalized, SOURCE_KIND_UNKNOWN)


def document_types_for_source_kind(kind: str) -> set[str] | None:
    """Возвращает типы документов для назначения или ``None`` без фильтра."""
    document_types = _SOURCE_KIND_TO_DOCUMENT_TYPES.get(_normalize(kind))
    if document_types is None:
        return None
    return set(document_types)


def document_types_for_source_kinds(kinds: Iterable[str]) -> set[str] | None:
    """Объединяет типы документов для нескольких назначений источника.

    Возвращает ``None``, если список пуст или не содержит известных
    назначений (фильтр в таком случае не применяется).
    """
    document_types: set[str] = set()
    for kind in kinds or ():
        resolved = document_types_for_source_kind(kind)
        if resolved:
            document_types.update(resolved)
    return document_types or None


def expected_source_kind_for_slot(slot: str | None) -> str | None:
    """Возвращает ожидаемое назначение источника для слота."""
    return STRICT_SLOT_SOURCE_KIND.get(_normalize_slot(slot))


def source_kind_label(kind: str | None) -> str:
    """Человекочитаемая подпись назначения источника."""
    return _SOURCE_KIND_LABELS.get(_normalize(kind), "")
