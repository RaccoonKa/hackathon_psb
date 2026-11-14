import logging
import time
from rag_core import config
from rag_core.generation.analysis.entity_resolver import (
    extract_doc_routing_sets,
    resolve_single_entity,
)
from rag_core.generation.analysis.source_kind import detect_required_source_kinds
from rag_core.generation.types import QuerySpec, RetrievalTask, TraceStep

logger = logging.getLogger(__name__)

_SLOT_SEARCH_EXPANSIONS: dict[str, str] = {
    "security_deposit": "размер обеспечительного платежа условия внесения и возврата депозита",
    "penalty": "размер пеней неустойки штрафа процент за каждый день просрочки обязательств",
    "accrual_period": "период начисления пеней неустойки за каждый день просрочки исполнения обязательств",
    "liability_cap": "ограничение ответственности сторон предел ответственности реальный ущерб упущенная выгода",
    "rent_amount": "размер арендной платы стоимость аренды сумма ежемесячной платы",
    "payment_deadline": "срок внесения арендной платы порядок расчетов дата ежемесячной оплаты",
    "probation_period": "испытательный срок при приеме на работу продолжительность испытания",
    "overtime_limit": "продолжительность сверхурочной работы предельный лимит часов в год",
    "overtime_pay": "размер оплаты сверхурочной работы повышенная оплата часов",
    "approval_process": "порядок согласования договора одобрение руководителем пороговая сумма",
    "termination": "порядок расторжения договора односторонний отказ срок уведомления",
    "tax_rate": "налоговая ставка налог на прибыль НДС условия применения",
    "effective_date": "срок действия редакция вступает в силу дополнительное соглашение",
    "delivery_terms": "условия и адрес доставки распределение расходов на упаковку и перевозку",
    "acceptance_terms": "порядок и срок приемки товара по качеству претензия акт недостатков",
    "warranty_period": "гарантийный срок на товар условия и порядок исчисления гарантии",
    "prepayment": "условия предоплаты и авансового платежа порядок расчетов",
    "return_procedure": "порядок возврата имущества помещения при прекращении договора по акту",
    "document_handover": "выдача и передача оригиналов договоров в юридический отдел срок",
    "consent_withdrawal": "отзыв согласия на обработку персональных данных прекращение обработки",
    "seasonal_work": "срочный трудовой договор для выполнения сезонных работ условия",
}

# Служебные слоты не являются поисковыми терминами: они описывают
# "общий" запрос и не должны попадать в hybrid-query (ни в основной
# план, ни в followup), иначе разбавляют реальные слоты.
_SERVICE_SLOTS: frozenset[str] = frozenset({"general"})

# Арендные слоты: позволяют определить тип документа без отраслевых догадок.
_RENTAL_SLOTS: frozenset[str] = frozenset(
    {
        "rent_amount",
        "security_deposit",
        "return_procedure",
    }
)
_SUPPLY_SLOTS: frozenset[str] = frozenset(
    {
        "delivery_terms",
        "acceptance_terms",
        "warranty_period",
        "prepayment",
    }
)

# Нейтральная формулировка срока оплаты для договора поставки: без переноса
# терминов аренды и без «оплачивает товар» из другого шаблона.
_PAYMENT_DEADLINE_NEUTRAL = (
    "срок оплаты порядок расчетов не позднее какого числа месяца"
)

# Маркеры родительского договора/допсоглашений для date_version: раньше это
# был отдельный лексический канал, теперь добавляются части единого запроса.
_AMENDMENT_QUERY_PHRASE = (
    "родительский договор дополнительное соглашение редакция изменения"
)

# H-2A: единый маппинг intent анализа → hybrid-intent поиска.
# compare/date_version/checklist используют смешанные веса, поскольку
# требуют и точных, и смысловых совпадений.
_HYBRID_INTENT_MAP: dict[str, str] = {
    "exact_lookup": "exact_lookup",
    "mixed": "mixed_query",
    "mixed_query": "mixed_query",
    "semantic": "semantic_query",
    "semantic_query": "semantic_query",
    "compare": "mixed_query",
    "date_version": "mixed_query",
    "checklist": "mixed_query",
}

# Префиксы purpose, которые проверяются тестами и внешними контрактами.
_PURPOSE_BY_INTENT: dict[str, str] = {
    "exact_lookup": "exact_lookup_hybrid",
    "mixed": "mixed_hybrid",
    "mixed_query": "mixed_hybrid",
    "semantic": "semantic_hybrid",
    "semantic_query": "semantic_hybrid",
    "compare": "compare_hybrid",
    "date_version": "date_version_hybrid",
    "checklist": "checklist_hybrid",
}

# Ограничение размера списков в trace, чтобы план не раздувал лог.
_TRACE_PARTS_LIMIT = 8
# Верхняя граница частей единого запроса (дедуп + вложения уже сокращают).
_MAX_COMBINED_PARTS = 8


def _detect_document_type(spec: QuerySpec) -> str:
    """Возвращает ``rental``/``supply``/``unknown`` по контексту запроса.

    Опирается только на уже нормализованный QuerySpec (rewritten queries,
    сущности, слоты) и разрешённые document_id — без чтения индекса.
    """
    text = " ".join(
        [
            *spec.rewritten_queries_bm25,
            *spec.rewritten_queries_dense,
            *spec.entities_raw,
        ]
    ).lower()
    doc_ids = " ".join(
        (ref.document_id or "") for ref in spec.entities_resolved
    ).lower()
    slot_set = set(spec.required_slots)

    if (
        "аренд" in text
        or "аренд" in doc_ids
        or bool(slot_set & _RENTAL_SLOTS)
    ):
        return "rental"
    if (
        "поставк" in text
        or "товар" in text
        or bool(slot_set & _SUPPLY_SLOTS)
    ):
        return "supply"
    return "unknown"


def _slot_expansion(slot: str, document_type: str) -> str:
    """Расширение слота без терминов из чужих типов документов."""
    if slot in _SERVICE_SLOTS:
        return ""
    if slot == "payment_deadline" and document_type == "supply":
        return _PAYMENT_DEADLINE_NEUTRAL
    return _SLOT_SEARCH_EXPANSIONS.get(slot, slot.replace("_", " "))


def _normalize_for_dedup(text: str) -> str:
    return " ".join((text or "").lower().split()).strip(" ?!.,;:")


def _hybrid_intent(intent: str | None) -> str:
    """Возвращает hybrid-intent; неизвестное значение → ``mixed_query``."""
    return _HYBRID_INTENT_MAP.get((intent or "").strip().lower(), "mixed_query")


def _purpose_for_intent(intent: str | None) -> str:
    """Префикс purpose единой hybrid-задачи для данного intent."""
    return _PURPOSE_BY_INTENT.get(
        (intent or "").strip().lower(), "semantic_hybrid"
    )


def _unique_queries(*groups) -> list[str]:
    """Объединяет группы запросов без дублей (нормализация регистра/пробелов)."""
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for query in group:
            key = _normalize_for_dedup(query)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(query)
    return result


def _merge_query_parts(parts, limit: int) -> str:
    """Склеивает части запроса без дублей и вложенных фрагментов.

    Порядок частей сохраняется. Повтор (после нормализации) и вложение
    (когда новая часть уже содержится в предыдущей или наоборот) не
    добавляются, чтобы единый hybrid-запрос не раздувался.
    """
    merged: list[str] = []
    seen_keys: list[str] = []
    for part in parts:
        cleaned = (part or "").strip()
        key = _normalize_for_dedup(cleaned)
        if not cleaned or not key:
            continue
        if any(
            key == existing
            or key in existing
            or existing in key
            for existing in seen_keys
        ):
            continue
        seen_keys.append(key)
        merged.append(cleaned)
        if len(merged) >= max(1, int(limit)):
            break
    return " ".join(merged).strip()


def _collect_query_parts(
    spec: QuerySpec, document_type: str, limit: int
) -> dict[str, list[str]]:
    """Детерминированные части единого запроса (для сборки и trace)."""
    intent = (spec.intent or "").strip().lower()
    bm25_q = spec.rewritten_queries_bm25 or (
        spec.rewritten_queries_dense[:1] or [""]
    )
    dense_q = spec.rewritten_queries_dense or (
        spec.rewritten_queries_bm25[:1] or [""]
    )
    searchable_slots = [
        slot for slot in spec.required_slots if slot not in _SERVICE_SLOTS
    ][:3]
    slot_expansions = [
        _slot_expansion(slot, document_type) for slot in searchable_slots
    ]
    decomposition = [
        (sq.text or "").strip()
        for sq in spec.decomposition[:3]
        if (sq.text or "").strip()
    ]
    amendment_markers: list[str] = []
    compare_parts: list[str] = []

    if intent == "date_version":
        amendment_markers = [
            *[entity for entity in spec.entities_raw if entity],
            _AMENDMENT_QUERY_PHRASE,
        ]
        if "effective_date" in spec.required_slots:
            slot_expansions.append(
                _slot_expansion("effective_date", document_type)
            )

    if intent == "compare" and spec.decomposition:
        for sq in spec.decomposition[:3]:
            text = (sq.text or "").strip()
            if text:
                compare_parts.append(text)
            slot_key = sq.aspect or (
                spec.required_slots[0] if spec.required_slots else ""
            )
            if slot_key and slot_key not in _SERVICE_SLOTS:
                compare_parts.append(_slot_expansion(slot_key, document_type))
            if sq.target_doc_raw:
                compare_parts.append(sq.target_doc_raw)

    return {
        "bm25_rewrites": [q for q in bm25_q[:limit] if q],
        "dense_rewrites": [q for q in dense_q[:limit] if q],
        "slot_expansions": [part for part in slot_expansions if part],
        "decomposition": decomposition,
        "amendment_markers": [part for part in amendment_markers if part],
        "compare_parts": [part for part in compare_parts if part],
    }


def _combined_hybrid_query(
    spec: QuerySpec, document_type: str, max_parts: int
) -> str:
    """Единый последовательный hybrid-запрос для конкретного intent."""
    intent = (spec.intent or "").strip().lower()
    query_limit = max(2, min(4, len(spec.required_slots) + 1))
    parts = _collect_query_parts(spec, document_type, query_limit)

    if intent == "exact_lookup":
        ordered = [
            *parts["bm25_rewrites"],
            *parts["dense_rewrites"][:1],
            *parts["slot_expansions"],
        ]
    elif intent == "checklist":
        ordered = [
            *parts["dense_rewrites"],
            *parts["slot_expansions"],
        ]
    elif intent == "date_version":
        ordered = [
            *parts["bm25_rewrites"][:1],
            *parts["dense_rewrites"],
            *parts["amendment_markers"],
            *parts["slot_expansions"],
        ]
    elif intent == "compare":
        if spec.decomposition:
            ordered = list(parts["compare_parts"])
        else:
            ordered = [
                *parts["dense_rewrites"],
                *parts["bm25_rewrites"],
                *parts["slot_expansions"],
            ]
    elif intent in {"semantic", "semantic_query"}:
        ordered = [
            *parts["dense_rewrites"],
            *parts["bm25_rewrites"],
            *parts["slot_expansions"],
            *parts["decomposition"],
        ]
    else:
        # mixed/mixed_query и неизвестный intent — смысловой запрос первым.
        ordered = [
            *parts["dense_rewrites"],
            *parts["bm25_rewrites"],
            *parts["slot_expansions"],
        ]
    return _merge_query_parts(ordered, max_parts)


def _merge_routing_sets(
    spec: QuerySpec,
    user_selected_doc_ids: set[str] | None = None,
) -> tuple[set[str] | None, set[str] | None]:
    """Маршрутизация единой hybrid-задачи.

    Для compare дополнительно учитываются target-документы подвопросов:
    однозначные цели сужают поиск, неоднозначные кандидаты идут в boost.
    """
    base_selected, base_boost = extract_doc_routing_sets(
        spec.entities_resolved, user_selected_doc_ids
    )
    intent = (spec.intent or "").strip().lower()
    if intent != "compare" or not spec.decomposition:
        return base_selected, base_boost

    resolved_targets: set[str] = set()
    target_candidates: set[str] = set()
    all_resolved_single = True
    for sq in spec.decomposition[:3]:
        if not sq.target_doc_raw:
            all_resolved_single = False
            continue
        resolved = resolve_single_entity(sq.target_doc_raw)
        if resolved.document_id:
            resolved_targets.add(resolved.document_id)
        elif resolved.candidates:
            target_candidates.update(resolved.candidates)
            all_resolved_single = False
        else:
            all_resolved_single = False

    if (
        all_resolved_single
        and len(resolved_targets) == 1
        and not user_selected_doc_ids
    ):
        return resolved_targets, base_boost

    selected = set(base_selected or set()) | resolved_targets
    boost = (
        set(base_boost or set())
        | target_candidates
        | resolved_targets
    )
    return (selected or None), (boost or None)


def _trace_query_parts(parts: dict[str, list[str]]) -> dict[str, list[str]]:
    """Ограничивает размер частей запроса в trace."""
    result: dict[str, list[str]] = {}
    for key in (
        "dense_rewrites",
        "bm25_rewrites",
        "slot_expansions",
        "decomposition",
        "amendment_markers",
    ):
        result[key] = list(parts.get(key, []))[:_TRACE_PARTS_LIMIT]
    return result


def _task_trace_entry(idx: int, task: RetrievalTask, prefix: str = "task") -> dict:
    return {
        "task_id": f"{prefix}_{idx}",
        "query": task.query,
        "mode": task.mode,
        "purpose": task.purpose,
        "intent": task.intent,
        "source_kinds": list(task.source_kinds or []),
    }


def plan(
    spec: QuerySpec,
    user_selected_doc_ids: set[str] | None = None,
) -> tuple[list[RetrievalTask], TraceStep]:
    """Готовит РОВНО ОДНУ последовательную hybrid-задачу.

    Пустой/out-of-scope запрос возвращает 0 задач. Все внутриинтентные
    варианты (bm25/dense/подвопросы) объединяются в единый запрос.
    """
    start_time = time.perf_counter()
    selected_doc_ids, boost_doc_ids = _merge_routing_sets(
        spec, user_selected_doc_ids
    )
    top_k = int(config.RAG_TOP_K)
    document_type = _detect_document_type(spec)
    required_source_kinds = detect_required_source_kinds(spec)
    spec.required_source_kinds = required_source_kinds
    intent = (spec.intent or "").strip().lower()
    max_slot_queries = max(2, min(4, len(spec.required_slots) + 1))
    tasks: list[RetrievalTask] = []

    if intent == "out_of_scope" or (
        not spec.rewritten_queries_bm25 and not spec.rewritten_queries_dense
    ):
        latency_ms = round(
            (time.perf_counter() - start_time) * config.MS_PER_SECOND,
            config.LATENCY_ROUND_DIGITS,
        )
        step = TraceStep(
            step="plan_retrieval",
            input={
                "intent": spec.intent,
                "required_slots": spec.required_slots,
                "event_date": spec.event_date,
            },
            output={
                "tasks_count": 0,
                "tasks": [],
                "hybrid_intent": None,
                "query_parts": {
                    "dense_rewrites": [],
                    "bm25_rewrites": [],
                    "slot_expansions": [],
                    "decomposition": [],
                    "amendment_markers": [],
                },
                "combined_query_chars": 0,
                "required_source_kinds": list(required_source_kinds),
                "limits": {
                    "top_k": top_k,
                    "max_slot_queries": 0,
                    "task_timeout_s": float(config.RETRIEVAL_TASK_TIMEOUT_S),
                    "max_followup_cycles": int(
                        config.MAX_FOLLOWUP_RETRIEVAL_CYCLES
                    ),
                },
                "document_type": document_type,
            },
            latency_ms=latency_ms,
            stage_idx=2,
        )
        return tasks, step

    hybrid_intent = _hybrid_intent(intent)
    parts = _collect_query_parts(spec, document_type, max_slot_queries)
    combined_query = _combined_hybrid_query(
        spec, document_type, _MAX_COMBINED_PARTS
    )

    if combined_query:
        tasks.append(
            RetrievalTask(
                query=combined_query,
                mode="hybrid",
                purpose=_purpose_for_intent(intent),
                selected_doc_ids=selected_doc_ids,
                boost_doc_ids=boost_doc_ids,
                event_date=spec.event_date,
                top_k=top_k,
                intent=hybrid_intent,
                source_kinds=required_source_kinds or None,
            )
        )

    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step="plan_retrieval",
        input={
            "intent": spec.intent,
            "required_slots": spec.required_slots,
            "event_date": spec.event_date,
        },
        output={
            "tasks_count": len(tasks),
            "tasks": [
                _task_trace_entry(idx, task)
                for idx, task in enumerate(tasks)
            ],
            "hybrid_intent": hybrid_intent,
            "query_parts": _trace_query_parts(parts),
            "combined_query_chars": len(combined_query),
            "required_source_kinds": list(required_source_kinds),
            "limits": {
                "top_k": top_k,
                "max_slot_queries": max_slot_queries,
                "task_timeout_s": float(config.RETRIEVAL_TASK_TIMEOUT_S),
                "max_followup_cycles": int(
                    config.MAX_FOLLOWUP_RETRIEVAL_CYCLES
                ),
            },
            "document_type": document_type,
        },
        latency_ms=latency_ms,
        stage_idx=2,
    )
    return tasks, step


def followup(
    missing_slots: list[str],
    spec: QuerySpec,
    user_selected_doc_ids: set[str] | None = None,
    existing_queries: set[str] | None = None,
) -> tuple[list[RetrievalTask], TraceStep]:
    """Расширяющая follow-up выборка по непокрытым слотам.

    Формируется РОВНО ОДНА hybrid-задача: исходная формулировка запроса
    сохраняется в начале, затем добавляются контекст сущностей и расширения
    до трёх недублирующихся слотов. Дубликаты уже выполненных запросов
    отбрасываются.
    """
    start_time = time.perf_counter()
    selected_doc_ids, boost_doc_ids = _merge_routing_sets(
        spec, user_selected_doc_ids
    )
    top_k = int(config.RAG_TOP_K)
    required_source_kinds = detect_required_source_kinds(spec)
    spec.required_source_kinds = required_source_kinds
    entity_context = " ".join(spec.entities_raw + spec.article_refs).strip()
    base_query = (
        spec.rewritten_queries_bm25[0]
        if spec.rewritten_queries_bm25
        else (spec.rewritten_queries_dense[0] if spec.rewritten_queries_dense else "")
    )
    document_type = _detect_document_type(spec)

    tasks: list[RetrievalTask] = []
    target_slots = list(
        (missing_slots or spec.required_slots or ["general"])[:3]
    )
    # В поисковую часть идут только реальные слоты: служебный ``general``
    # остаётся в purpose/slots_used для совместимости, но не становится
    # поисковым термином.
    expansion_slots = [
        slot for slot in target_slots if slot not in _SERVICE_SLOTS
    ]
    slot_expansions = [
        _slot_expansion(slot, document_type) for slot in expansion_slots
    ]
    combined_query = _merge_query_parts(
        [base_query, entity_context, *slot_expansions],
        max(2, len(target_slots) + 2),
    )

    seen = {
        _normalize_for_dedup(q)
        for q in (existing_queries or set())
        if q and q.strip()
    }
    skipped_duplicates: list[str] = []

    if not combined_query:
        slots_used: list[str] = []
    elif _normalize_for_dedup(combined_query) in seen:
        skipped_duplicates = list(target_slots)
        slots_used = []
    else:
        slots_used = list(target_slots)
        tasks.append(
            RetrievalTask(
                query=combined_query,
                mode="hybrid",
                purpose="followup_hybrid:" + ",".join(target_slots),
                selected_doc_ids=selected_doc_ids,
                boost_doc_ids=boost_doc_ids,
                event_date=spec.event_date,
                top_k=top_k,
                intent=_hybrid_intent(spec.intent),
                source_kinds=required_source_kinds or None,
            )
        )

    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step="plan_followup_retrieval",
        input={
            "missing_slots": missing_slots,
            "existing_queries_count": len(existing_queries or set()),
        },
        output={
            "tasks_count": len(tasks),
            "tasks": [
                _task_trace_entry(idx, task, prefix="followup_task")
                for idx, task in enumerate(tasks)
            ],
            "hybrid_intent": _hybrid_intent(spec.intent),
            "query_parts": {
                "slot_expansions": [
                    part
                    for part in slot_expansions
                    if part
                ][:_TRACE_PARTS_LIMIT],
            },
            "combined_query_chars": len(combined_query),
            "slots_used": slots_used,
            "required_source_kinds": list(required_source_kinds),
            "document_type": document_type,
            "skipped_duplicate_slots": skipped_duplicates,
            "limits": {
                "top_k": top_k,
                "max_slots": 3,
            },
        },
        latency_ms=latency_ms,
        stage_idx=2,
    )
    return tasks, step
