import logging
import time
from rag_core import config
from rag_core.generation.analysis.entity_resolver import (
    extract_doc_routing_sets,
    resolve_single_entity,
)
from rag_core.generation.types import QuerySpec, RetrievalTask, TraceStep

logger = logging.getLogger(__name__)

_SLOT_SEARCH_EXPANSIONS: dict[str, str] = {
    "security_deposit": "обеспечительный платеж размер месячной арендной платы при подписании договора пункт 3.5",
    "penalty": "размер пеней неустойки штраф процент за каждый день просрочки пункт 5.1",
    "accrual_period": "период начисления пеней за каждый день просрочки исполнения обязательств",
    "liability_cap": "ограничение ответственности сторон предел ответственности реальный ущерб упущенная выгода",
    "rent_amount": "размер арендной платы стоимость аренды в месяц рублей НДС пункт 3.1",
    "payment_deadline": "срок внесения арендной платы арендная плата размер арендной платы стоимость аренды ежемесячная оплата не позднее какого числа месяца пункт договора о расчетах пункт договора об арендной плате",
    "probation_period": "испытательный срок при приеме на работу максимальный срок испытания",
    "overtime_limit": "продолжительность сверхурочной работы лимит часов в год",
    "overtime_pay": "оплата сверхурочной работы первые два часа последующие часы",
    "approval_process": "порядок согласования договора одобрение генеральным директором",
    "termination": "порядок расторжения договора односторонний отказ уведомление",
    "tax_rate": "налоговая ставка налог на прибыль НДС освобождение",
    "effective_date": "срок действия редакция вступает в силу дополнительное соглашение",
    "delivery_terms": "адрес доставки стоимость упаковки и доставки в цену товара",
    "acceptance_terms": "приемка товара по качеству срок рабочих дней претензия акт недостатков",
    "warranty_period": "гарантийный срок на товар гарантийный талон производителя",
    "prepayment": "предоплата аванс по соглашению сторон счет поставщика",
    "return_procedure": "возврат помещения по акту передача ключей при прекращении договора",
    "document_handover": "выдача и передача оригиналов договоров в юридический отдел срок рабочих дней",
    "consent_withdrawal": "отзыв согласия на обработку персональных данных продолжение обработки",
    "seasonal_work": "срочный трудовой договор для выполнения сезонных работ",
}

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
    if slot == "payment_deadline" and document_type == "supply":
        return _PAYMENT_DEADLINE_NEUTRAL
    return _SLOT_SEARCH_EXPANSIONS.get(slot, slot.replace("_", " "))


def _normalize_for_dedup(text: str) -> str:
    return " ".join((text or "").lower().split()).strip(" ?!.,;:")


def plan(
    spec: QuerySpec,
    user_selected_doc_ids: set[str] | None = None,
) -> tuple[list[RetrievalTask], TraceStep]:
    start_time = time.perf_counter()
    selected_doc_ids, boost_doc_ids = extract_doc_routing_sets(
        spec.entities_resolved, user_selected_doc_ids
    )
    top_k = int(config.RAG_TOP_K)
    tasks: list[RetrievalTask] = []

    bm25_q = spec.rewritten_queries_bm25 or [spec.rewritten_queries_dense[0]]
    dense_q = spec.rewritten_queries_dense or [bm25_q[0]]
    max_slot_queries = max(2, min(4, len(spec.required_slots) + 1))

    intent = spec.intent
    if intent == "exact_lookup":
        for q in bm25_q[:max_slot_queries]:
            tasks.append(
                RetrievalTask(
                    query=q,
                    mode="bm25",
                    purpose="exact_lookup_bm25",
                    selected_doc_ids=selected_doc_ids,
                    boost_doc_ids=boost_doc_ids,
                    event_date=spec.event_date,
                    top_k=top_k,
                )
            )
        tasks.append(
            RetrievalTask(
                query=dense_q[0],
                mode="dense",
                purpose="exact_lookup_dense",
                selected_doc_ids=selected_doc_ids,
                boost_doc_ids=boost_doc_ids,
                event_date=spec.event_date,
                top_k=top_k,
            )
        )
    elif intent in {"mixed", "mixed_query"}:
        for q in bm25_q[:max_slot_queries]:
            tasks.append(
                RetrievalTask(
                    query=q,
                    mode="bm25",
                    purpose="mixed_bm25",
                    selected_doc_ids=selected_doc_ids,
                    boost_doc_ids=boost_doc_ids,
                    event_date=spec.event_date,
                    top_k=top_k,
                )
            )
        for q in dense_q[:max_slot_queries]:
            tasks.append(
                RetrievalTask(
                    query=q,
                    mode="dense",
                    purpose="mixed_dense",
                    selected_doc_ids=selected_doc_ids,
                    boost_doc_ids=boost_doc_ids,
                    event_date=spec.event_date,
                    top_k=top_k,
                )
            )
    elif intent == "compare":
        if spec.decomposition:
            for sq in spec.decomposition:
                sq_selected = selected_doc_ids
                sq_boost = set(boost_doc_ids) if boost_doc_ids else set()
                if sq.target_doc_raw:
                    resolved_target = resolve_single_entity(sq.target_doc_raw)
                    if resolved_target.document_id:
                        sq_selected = (
                            {resolved_target.document_id}
                            if not user_selected_doc_ids
                            else (user_selected_doc_ids & {resolved_target.document_id}) or {resolved_target.document_id}
                        )
                        sq_boost.add(resolved_target.document_id)
                    elif resolved_target.candidates:
                        sq_boost.update(resolved_target.candidates)

                slot_key = sq.aspect or (spec.required_slots[0] if spec.required_slots else "")
                expansion = _SLOT_SEARCH_EXPANSIONS.get(slot_key, "")
                tasks.append(
                    RetrievalTask(
                        query=sq.text,
                        mode="hybrid",
                        purpose=f"compare:{slot_key or 'subquestion'}",
                        selected_doc_ids=sq_selected,
                        boost_doc_ids=sq_boost or None,
                        event_date=spec.event_date,
                        top_k=top_k,
                    )
                )
                if expansion:
                    enriched_bm25 = f"{sq.target_doc_raw or ''} {expansion}".strip()
                    tasks.append(
                        RetrievalTask(
                            query=enriched_bm25,
                            mode="bm25",
                            purpose=f"compare_bm25:{slot_key}",
                            selected_doc_ids=sq_selected,
                            boost_doc_ids=sq_boost or None,
                            event_date=spec.event_date,
                            top_k=top_k,
                        )
                    )
        else:
            for q in dense_q[:max_slot_queries]:
                tasks.append(
                    RetrievalTask(
                        query=q,
                        mode="dense",
                        purpose="compare_fallback",
                        selected_doc_ids=selected_doc_ids,
                        boost_doc_ids=boost_doc_ids,
                        event_date=spec.event_date,
                        top_k=top_k,
                    )
                )
            for q in bm25_q[:max_slot_queries]:
                tasks.append(
                    RetrievalTask(
                        query=q,
                        mode="bm25",
                        purpose="compare_fallback_bm25",
                        selected_doc_ids=selected_doc_ids,
                        boost_doc_ids=boost_doc_ids,
                        event_date=spec.event_date,
                        top_k=top_k,
                    )
                )
    elif intent == "date_version":
        tasks.append(
            RetrievalTask(
                query=bm25_q[0],
                mode="bm25",
                purpose="date_version_bm25",
                selected_doc_ids=selected_doc_ids,
                boost_doc_ids=boost_doc_ids,
                event_date=spec.event_date,
                top_k=top_k,
            )
        )
        for q in dense_q[:max_slot_queries]:
            tasks.append(
                RetrievalTask(
                    query=q,
                    mode="dense",
                    purpose="date_version_dense",
                    selected_doc_ids=selected_doc_ids,
                    boost_doc_ids=boost_doc_ids,
                    event_date=spec.event_date,
                    top_k=top_k,
                )
            )
        entity_prefix = " ".join(spec.entities_raw)
        amendment_query = f"{entity_prefix} родительский договор дополнительное соглашение редакция изменения".strip()
        tasks.append(
            RetrievalTask(
                query=amendment_query,
                mode="bm25",
                purpose="parent_contract_and_amendments",
                selected_doc_ids=selected_doc_ids,
                boost_doc_ids=boost_doc_ids,
                event_date=None,
                top_k=top_k,
            )
        )
    elif intent == "checklist":
        for q in dense_q[:max_slot_queries]:
            tasks.append(
                RetrievalTask(
                    query=q,
                    mode="dense",
                    purpose="checklist_steps",
                    selected_doc_ids=selected_doc_ids,
                    boost_doc_ids=boost_doc_ids,
                    event_date=spec.event_date,
                    top_k=top_k,
                )
            )
    else:
        for q in dense_q[:max_slot_queries]:
            tasks.append(
                RetrievalTask(
                    query=q,
                    mode="dense",
                    purpose="semantic_dense",
                    selected_doc_ids=selected_doc_ids,
                    boost_doc_ids=boost_doc_ids,
                    event_date=spec.event_date,
                    top_k=top_k,
                )
            )
        for q in bm25_q[:max_slot_queries]:
            tasks.append(
                RetrievalTask(
                    query=q,
                    mode="bm25",
                    purpose="semantic_bm25",
                    selected_doc_ids=selected_doc_ids,
                    boost_doc_ids=boost_doc_ids,
                    event_date=spec.event_date,
                    top_k=top_k,
                )
            )
        if spec.decomposition:
            existing_queries = {t.query.lower() for t in tasks}
            for sq in spec.decomposition[:3]:
                if sq.text.lower() not in existing_queries:
                    tasks.append(
                        RetrievalTask(
                            query=sq.text,
                            mode="dense",
                            purpose=f"subquestion:{sq.aspect or 'part'}",
                            selected_doc_ids=selected_doc_ids,
                            boost_doc_ids=boost_doc_ids,
                            event_date=spec.event_date,
                            top_k=top_k,
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
                {
                    "task_id": f"task_{idx}",
                    "query": t.query,
                    "mode": t.mode,
                    "purpose": t.purpose,
                }
                for idx, t in enumerate(tasks)
            ],
            "limits": {
                "top_k": top_k,
                "max_slot_queries": max_slot_queries,
                "max_workers": int(config.RETRIEVAL_MAX_WORKERS),
                "task_timeout_s": float(config.RETRIEVAL_TASK_TIMEOUT_S),
                "max_followup_cycles": int(
                    config.MAX_FOLLOWUP_RETRIEVAL_CYCLES
                ),
            },
            "document_type": _detect_document_type(spec),
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

    Исходная формулировка запроса сохраняется в начале follow-up задачи,
    расширение добавляется как дополнение (OR-семантика для BM25/hybrid),
    а не заменяет её. Дубликаты уже выполненных запросов отбрасываются.
    """
    start_time = time.perf_counter()
    selected_doc_ids, boost_doc_ids = extract_doc_routing_sets(
        spec.entities_resolved, user_selected_doc_ids
    )
    top_k = int(config.RAG_TOP_K)
    entity_context = " ".join(spec.entities_raw + spec.article_refs).strip()
    base_query = (
        spec.rewritten_queries_bm25[0]
        if spec.rewritten_queries_bm25
        else (spec.rewritten_queries_dense[0] if spec.rewritten_queries_dense else "")
    )
    document_type = _detect_document_type(spec)

    tasks: list[RetrievalTask] = []
    target_slots = missing_slots or spec.required_slots or ["general"]
    seen = {
        _normalize_for_dedup(q)
        for q in (existing_queries or set())
        if q and q.strip()
    }
    skipped_duplicates: list[str] = []

    for slot in target_slots[:3]:
        expansion = _slot_expansion(slot, document_type)
        parts: list[str] = []
        seen_parts: set[str] = set()
        for part in (base_query, entity_context, expansion):
            cleaned = (part or "").strip()
            key = _normalize_for_dedup(cleaned)
            if not cleaned or not key or key in seen_parts:
                continue
            # Не дублируем уже присутствующий фрагмент (например, название
            # контрагента, которое уже есть в исходном запросе).
            if any(key in existing for existing in seen_parts):
                continue
            seen_parts.add(key)
            parts.append(cleaned)
        followup_query = " ".join(parts).strip()
        if not followup_query:
            continue
        normalized_query = _normalize_for_dedup(followup_query)
        if normalized_query in seen:
            skipped_duplicates.append(slot)
            continue
        seen.add(normalized_query)
        tasks.append(
            RetrievalTask(
                query=followup_query,
                mode="dense",
                purpose=f"followup_dense:{slot}",
                selected_doc_ids=selected_doc_ids,
                boost_doc_ids=boost_doc_ids,
                event_date=spec.event_date,
                top_k=top_k,
            )
        )
        tasks.append(
            RetrievalTask(
                query=followup_query,
                mode="bm25",
                purpose=f"followup_bm25:{slot}",
                selected_doc_ids=selected_doc_ids,
                boost_doc_ids=boost_doc_ids,
                event_date=spec.event_date,
                top_k=top_k,
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
                {
                    "task_id": f"followup_task_{idx}",
                    "query": t.query,
                    "mode": t.mode,
                    "purpose": t.purpose,
                }
                for idx, t in enumerate(tasks)
            ],
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