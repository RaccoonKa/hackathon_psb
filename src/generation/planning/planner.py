import logging
import time
from src import config
from src.generation.analysis.entity_resolver import (
    extract_doc_routing_sets,
    resolve_single_entity,
)
from src.generation.types import QuerySpec, RetrievalTask, TraceStep

logger = logging.getLogger(__name__)

_SLOT_SEARCH_EXPANSIONS: dict[str, str] = {
    "rent_amount": "размер арендной платы стоимость аренды в месяц НДС",
    "payment_deadline": "срок оплаты порядок расчетов не позднее числа месяца",
    "probation_period": "испытательный срок при приеме на работу максимальный срок испытания",
    "overtime_limit": "продолжительность сверхурочной работы лимит часов в год",
    "overtime_pay": "оплата сверхурочной работы первые два часа последующие часы",
    "approval_process": "порядок согласования договора визирование юридический департамент",
    "termination": "порядок расторжения договора односторонний отказ уведомление",
    "penalty": "неустойка пеня штраф за просрочку исполнения обязательств",
    "tax_rate": "налоговая ставка налог на прибыль НДС освобождение",
    "effective_date": "срок действия редакция вступает в силу дополнительное соглашение",
}


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

    intent = spec.intent
    if intent == "exact_lookup":
        for q in bm25_q[:2]:
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
        for q in (bm25_q + bm25_q)[:2]:
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
        for q in (dense_q + dense_q)[:2]:
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
                sq_boost = set(boost_doc_ids) if boost_doc_ids else set()
                if sq.target_doc_raw:
                    resolved_target = resolve_single_entity(sq.target_doc_raw)
                    if resolved_target.document_id:
                        sq_boost.add(resolved_target.document_id)
                    else:
                        sq_boost.update(resolved_target.candidates)
                tasks.append(
                    RetrievalTask(
                        query=sq.text,
                        mode="hybrid",
                        purpose=f"compare:{sq.aspect or 'subquestion'}",
                        selected_doc_ids=selected_doc_ids,
                        boost_doc_ids=sq_boost or None,
                        event_date=spec.event_date,
                        top_k=top_k,
                    )
                )
        else:
            for q in dense_q[:2]:
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
        tasks.append(
            RetrievalTask(
                query=dense_q[0],
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
        for q in dense_q[:2]:
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
        for q in dense_q[:2]:
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
        tasks.append(
            RetrievalTask(
                query=bm25_q[0],
                mode="bm25",
                purpose="semantic_bm25",
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
                {"query": t.query, "mode": t.mode, "purpose": t.purpose}
                for t in tasks
            ],
        },
        latency_ms=latency_ms,
        stage_idx=2,
    )
    return tasks, step


def followup(
    missing_slots: list[str],
    spec: QuerySpec,
    user_selected_doc_ids: set[str] | None = None,
) -> tuple[list[RetrievalTask], TraceStep]:
    start_time = time.perf_counter()
    selected_doc_ids, boost_doc_ids = extract_doc_routing_sets(
        spec.entities_resolved, user_selected_doc_ids
    )
    top_k = int(config.RAG_TOP_K)
    entity_context = " ".join(spec.entities_raw + spec.article_refs).strip()
    base_query = spec.rewritten_queries_bm25[0] if spec.rewritten_queries_bm25 else ""

    tasks: list[RetrievalTask] = []
    target_slots = missing_slots or spec.required_slots or ["general"]

    for slot in target_slots[:3]:
        expansion = _SLOT_SEARCH_EXPANSIONS.get(slot, slot)
        followup_query = f"{entity_context} {expansion} {base_query}".strip()
        tasks.append(
            RetrievalTask(
                query=followup_query,
                mode="hybrid",
                purpose=f"followup:{slot}",
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
        input={"missing_slots": missing_slots},
        output={
            "tasks_count": len(tasks),
            "tasks": [
                {"query": t.query, "mode": t.mode, "purpose": t.purpose}
                for t in tasks
            ],
        },
        latency_ms=latency_ms,
        stage_idx=2,
    )
    return tasks, step