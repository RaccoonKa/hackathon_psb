"""Сквозные интеграционные тесты диалоговых сценариев (Multi-Turn Test Suite).

Проверяет ключевые требования пункта 3 дорожной карты:
1. Тест 1 (Анафоры и уточнения): цепочка из 3 реплик (аренда -> срок внесения -> неустойка);
2. Тест 2 (Смена темы): переключение с договора на статью КоАП РФ со сбросом фильтра;
3. Тест 3 (Уточнение даты): наследование даты события из предыдущей реплики в RetrievalTask;
4. Тест 4 (Обоснованный отказ): корректный отказ без галлюцинаций при отсутствии факта в договоре;
5. Тест 5 (Интеграция solution.py): стенограммы диалогов, групповая обработка и сохранение порядка.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from rag_core import config
from rag_core.generation import orchestrator as orch_mod
from rag_core.generation.analysis import entity_resolver
from rag_core.generation.analysis.query_analysis import (
    analyze_query,
    extract_date_from_history,
    is_independent_query,
)
from rag_core.generation.planning import planner
from rag_core.generation.types import (
    AgentResult,
    Ambiguity,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    DocRef,
    DraftAnswer,
    EvidenceAssessment,
    EvidenceBundle,
    PreFilterResult,
    QuerySpec,
    RetrievedChunk,
    Sufficiency,
    TaskType,
    TraceStep,
    VerificationReport,
    VerificationVerdict,
)
import solution


def _step(name: str) -> TraceStep:
    return TraceStep(step=name, input={}, output={}, latency_ms=0.0)


def _make_chunk(
    chunk_id: str,
    text: str,
    doc_id: str = "contract_а-08-2024",
    title: str = "Договор аренды складского комплекса № А-08/2024",
    prov: str = "3.1",
    path: list[str] | None = None,
    effective_from: str | None = "2024-01-01",
    effective_to: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=doc_id,
        version_id="v1",
        document_title=title,
        provision_id=prov,
        structure_path=path or ["Раздел 3", f"Пункт {prov}"],
        score=0.95,
        effective_from=effective_from,
        effective_to=effective_to,
        effective_dates_known=True,
    )


def test_dialog_scenario_1_anaphora_and_clarification_chain():
    """Тест 1 (Анафоры и уточнения): цепочка из 3 реплик.

    1. «Какая арендная плата по договору с ООО «Северный Склад»?» -> ответ со ссылкой на S1;
    2. «А в какой срок её нужно вносить?» -> подтягивание сущности из истории, ответ со ссылкой;
    3. «Какая неустойка предусмотрена по нему?» -> удержание договора через «по нему», ответ со ссылкой.
    """
    chunk_rent = _make_chunk(
        "c_rent",
        "3.1. Арендная плата составляет 450 000 (четыреста пятьдесят тысяч) рублей в месяц, включая НДС.",
        prov="3.1",
    )
    chunk_deadline = _make_chunk(
        "c_deadline",
        "3.2. Арендная плата вносится Арендатором ежемесячно не позднее 5-го числа оплачиваемого месяца.",
        prov="3.2",
    )
    chunk_penalty = _make_chunk(
        "c_penalty",
        "5.1. За нарушение срока внесения арендной платы Арендатор уплачивает пени в размере 0,1% от суммы задолженности за каждый день просрочки.",
        prov="5.1",
    )

    history: list[dict[str, str]] = []

    def fake_verify(draft, chunks, spec, **kwargs):
        report = VerificationReport(
            batch_status=[],
            citation_errors=[],
            injection_signals=[],
            supported_locked=[draft.claims[0]] if draft.claims else [],
            final_verdict=VerificationVerdict.ACCEPT,
        )
        return report, _step("verify")

    # --- Turn 1: Арендная плата ---
    q1 = "Какая арендная плата по договору с ООО «Северный Склад»?"
    payload_1 = {
        "intent": "exact_lookup",
        "task_type": "lookup",
        "entities_raw": ["ООО «Северный Склад»"],
        "article_refs": [],
        "dates": [],
        "event_date": None,
        "premises": [],
        "required_slots": ["rent_amount"],
        "assumptions": [],
        "ambiguity": "low",
        "needs_clarification": False,
        "decomposition": [],
        "rewritten_queries_bm25": ["арендная плата ООО «Северный Склад»"],
        "rewritten_queries_dense": ["арендная плата ООО «Северный Склад»"],
    }

    def fake_retrieve_1(tasks, **kwargs):
        return [chunk_rent], _step("retrieval")

    def fake_assess_1(spec, chunks, **kwargs):
        bundle = EvidenceBundle(chunks=chunks, slot_map={"rent_amount": ["c_rent"]}, contradictions=[], version_issues=[])
        assessment = EvidenceAssessment(
            covered_slots=["rent_amount"],
            missing_slots=[],
            core_slot_covered=True,
            sufficiency=Sufficiency.SUFFICIENT,
            degraded_intents=[],
            reason="covered",
        )
        return bundle, assessment, _step("assess")

    def fake_generate_1(spec, evidence, policy, context, **kwargs):
        ans = "Арендная плата по договору с ООО «Северный Склад» составляет 450 000 руб./мес. <source>S1</source>."
        claims = [Claim(text="Арендная плата 450 000 руб./мес.", source_ids=["S1"], slot="rent_amount", original_idx=0)]
        return DraftAnswer(ans, claims, [], [], None), _step("generation")

    with patch.object(orch_mod, "run_prefilter", return_value=(PreFilterResult(False, None, None), _step("prefilter"))), \
         patch("rag_core.generation.analysis.query_analysis.llm_complete_with_usage", return_value=(json.dumps(payload_1), 40, 20)), \
         patch("rag_core.generation.retrieval.bridge.retrieve_tasks", side_effect=fake_retrieve_1), \
         patch.object(orch_mod.evidence, "assess", side_effect=fake_assess_1), \
         patch.object(orch_mod, "generate_draft", side_effect=fake_generate_1), \
         patch.object(orch_mod, "verify", side_effect=fake_verify):

        res1 = orch_mod.run_agent(q1, chat_history=history)
        assert res1.refused is False
        assert "450 000" in res1.final_answer
        assert "S1" in res1.sources
        assert res1.sources["S1"]["chunk_id"] == "c_rent"

        history.append({"role": "user", "content": q1})
        history.append({"role": "assistant", "content": res1.final_answer})

    # --- Turn 2: Срок внесения (анафора «её») ---
    q2 = "А в какой срок её нужно вносить?"
    resolved_q2 = "В какой срок нужно вносить арендную плату по договору с ООО «Северный Склад»?"
    payload_2 = {
        "intent": "exact_lookup",
        "task_type": "lookup",
        "entities_raw": ["ООО «Северный Склад»"],
        "article_refs": [],
        "dates": [],
        "event_date": None,
        "premises": [],
        "required_slots": ["payment_deadline"],
        "assumptions": [],
        "ambiguity": "low",
        "needs_clarification": False,
        "decomposition": [],
        "rewritten_queries_bm25": ["срок внесения арендной платы ООО «Северный Склад»"],
        "rewritten_queries_dense": ["срок внесения арендной платы ООО «Северный Склад»"],
    }

    def fake_retrieve_2(tasks, **kwargs):
        assert any(t.selected_doc_ids and "contract_а-08-2024" in t.selected_doc_ids for t in tasks)
        return [chunk_deadline], _step("retrieval")

    def fake_assess_2(spec, chunks, **kwargs):
        bundle = EvidenceBundle(chunks=chunks, slot_map={"payment_deadline": ["c_deadline"]}, contradictions=[], version_issues=[])
        assessment = EvidenceAssessment(
            covered_slots=["payment_deadline"],
            missing_slots=[],
            core_slot_covered=True,
            sufficiency=Sufficiency.SUFFICIENT,
            degraded_intents=[],
            reason="covered",
        )
        return bundle, assessment, _step("assess")

    def fake_generate_2(spec, evidence, policy, context, **kwargs):
        ans = "Арендная плата вносится не позднее 5-го числа оплачиваемого месяца <source>S1</source>."
        claims = [Claim(text="Оплата не позднее 5-го числа", source_ids=["S1"], slot="payment_deadline", original_idx=0)]
        return DraftAnswer(ans, claims, [], [], None), _step("generation")

    with patch.object(orch_mod, "run_prefilter", return_value=(PreFilterResult(False, None, None), _step("prefilter"))), \
         patch("rag_core.generation.analysis.query_analysis.llm_complete_with_usage", side_effect=[
             (resolved_q2, 20, 10),
             (json.dumps(payload_2), 40, 20),
         ]), \
         patch("rag_core.generation.retrieval.bridge.retrieve_tasks", side_effect=fake_retrieve_2), \
         patch.object(orch_mod.evidence, "assess", side_effect=fake_assess_2), \
         patch.object(orch_mod, "generate_draft", side_effect=fake_generate_2), \
         patch.object(orch_mod, "verify", side_effect=fake_verify):

        res2 = orch_mod.run_agent(q2, chat_history=history)
        assert res2.refused is False
        assert res2.used_history is True
        assert "5-го числа" in res2.final_answer
        assert "S1" in res2.sources
        assert res2.sources["S1"]["chunk_id"] == "c_deadline"

        history.append({"role": "user", "content": q2})
        history.append({"role": "assistant", "content": res2.final_answer})

    # --- Turn 3: Неустойка («по нему») ---
    q3 = "Какая неустойка предусмотрена по нему?"
    resolved_q3 = "Какая неустойка предусмотрена по договору с ООО «Северный Склад»?"
    payload_3 = {
        "intent": "exact_lookup",
        "task_type": "lookup",
        "entities_raw": ["ООО «Северный Склад»"],
        "article_refs": [],
        "dates": [],
        "event_date": None,
        "premises": [],
        "required_slots": ["penalty"],
        "assumptions": [],
        "ambiguity": "low",
        "needs_clarification": False,
        "decomposition": [],
        "rewritten_queries_bm25": ["неустойка пени ООО «Северный Склад»"],
        "rewritten_queries_dense": ["неустойка пени ООО «Северный Склад»"],
    }

    def fake_retrieve_3(tasks, **kwargs):
        assert any(t.selected_doc_ids and "contract_а-08-2024" in t.selected_doc_ids for t in tasks)
        return [chunk_penalty], _step("retrieval")

    def fake_assess_3(spec, chunks, **kwargs):
        bundle = EvidenceBundle(chunks=chunks, slot_map={"penalty": ["c_penalty"]}, contradictions=[], version_issues=[])
        assessment = EvidenceAssessment(
            covered_slots=["penalty"],
            missing_slots=[],
            core_slot_covered=True,
            sufficiency=Sufficiency.SUFFICIENT,
            degraded_intents=[],
            reason="covered",
        )
        return bundle, assessment, _step("assess")

    def fake_generate_3(spec, evidence, policy, context, **kwargs):
        ans = "По договору предусмотрена неустойка 0,1% за каждый день просрочки <source>S1</source>."
        claims = [Claim(text="Неустойка 0,1% за день просрочки", source_ids=["S1"], slot="penalty", original_idx=0)]
        return DraftAnswer(ans, claims, [], [], None), _step("generation")

    with patch.object(orch_mod, "run_prefilter", return_value=(PreFilterResult(False, None, None), _step("prefilter"))), \
         patch("rag_core.generation.analysis.query_analysis.llm_complete_with_usage", side_effect=[
             (resolved_q3, 20, 10),
             (json.dumps(payload_3), 40, 20),
         ]), \
         patch("rag_core.generation.retrieval.bridge.retrieve_tasks", side_effect=fake_retrieve_3), \
         patch.object(orch_mod.evidence, "assess", side_effect=fake_assess_3), \
         patch.object(orch_mod, "generate_draft", side_effect=fake_generate_3), \
         patch.object(orch_mod, "verify", side_effect=fake_verify):

        res3 = orch_mod.run_agent(q3, chat_history=history)
        assert res3.refused is False
        assert res3.used_history is True
        assert "0,1%" in res3.final_answer
        assert "S1" in res3.sources
        assert res3.sources["S1"]["chunk_id"] == "c_penalty"


def test_dialog_scenario_2_topic_shift_clears_contract_filter():
    """Тест 2 (Смена темы): переключение с договора на КоАП РФ.

    После обсуждения договора вопрос про статью закона:
    - Детектируется topic shift (is_independent_query = True);
    - Фильтр по старому договору сбрасывается;
    - Поисковые задачи не содержат selected_doc_ids={'contract_а-08-2024'}.
    """
    history = [
        {"role": "user", "content": "Какая арендная плата по договору с ООО «Северный Склад»?"},
        {"role": "assistant", "content": "Арендная плата составляет 450 000 руб./мес. <source>S1</source>"},
    ]

    query_law = "А какая ответственность предусмотрена по ст. 13.11 КоАП РФ?"

    assert is_independent_query(query_law) is True

    payload_law = {
        "intent": "semantic",
        "task_type": "explain",
        "entities_raw": [],
        "article_refs": ["13.11 КоАП РФ"],
        "dates": [],
        "event_date": None,
        "premises": [],
        "required_slots": ["penalty"],
        "assumptions": [],
        "ambiguity": "low",
        "needs_clarification": False,
        "decomposition": [],
        "rewritten_queries_bm25": ["ответственность статья 13.11 КоАП РФ"],
        "rewritten_queries_dense": ["ответственность статья 13.11 КоАП РФ"],
    }

    with patch(
        "rag_core.generation.analysis.query_analysis.llm_complete_with_usage",
        return_value=(json.dumps(payload_law), 40, 20),
    ):
        spec, step, used_history = analyze_query(query=query_law, chat_history=history)

    assert used_history is False
    assert "ООО «Северный Склад»" not in spec.entities_raw
    assert not any(ref.document_id == "contract_а-08-2024" for ref in spec.entities_resolved)
    assert "13.11 КоАП РФ" in spec.article_refs or "13.11" in " ".join(spec.article_refs)

    tasks, _ = planner.plan(spec=spec, user_selected_doc_ids=None)
    assert len(tasks) > 0
    for task in tasks:
        assert task.selected_doc_ids is None or "contract_а-08-2024" not in task.selected_doc_ids


def test_dialog_scenario_3_date_inheritance_in_dialog():
    """Тест 3 (Уточнение даты в диалоге): наследование даты события из предыдущей реплики.

    1. В реплике 1 зафиксирована дата инцидента: «15.05.2024»;
    2. В реплике 2 задан вопрос по редакции соглашения без указания даты;
    3. Дата 2024-05-15 наследуется в spec.event_date и передается в RetrievalTask.
    """
    history = [
        {
            "role": "user",
            "content": "У нас инцидент произошел 15.05.2024 по договору с ООО «Северный Склад»",
        },
        {
            "role": "assistant",
            "content": "Дата события 15 мая 2024 года зафиксирована. Чем я могу помочь?",
        },
    ]

    query_date = "Действует ли дополнительное соглашение № 2 к договору?"

    payload_date = {
        "intent": "date_version",
        "task_type": "date_check",
        "entities_raw": ["ООО «Северный Склад»"],
        "article_refs": [],
        "dates": [],
        "event_date": None,
        "premises": [],
        "required_slots": ["effective_date"],
        "assumptions": [],
        "ambiguity": "low",
        "needs_clarification": False,
        "decomposition": [],
        "rewritten_queries_bm25": ["дополнительное соглашение № 2"],
        "rewritten_queries_dense": ["дополнительное соглашение № 2"],
    }

    with patch(
        "rag_core.generation.analysis.query_analysis.llm_complete_with_usage",
        side_effect=[
            ("Действует ли дополнительное соглашение № 2 к договору с ООО «Северный Склад»?", 20, 10),
            (json.dumps(payload_date), 40, 20),
        ],
    ):
        spec, step, used_history = analyze_query(query=query_date, chat_history=history)

    assert spec.event_date == "2024-05-15"
    assert "2024-05-15" in spec.dates
    assert any("2024-05-15" in asm for asm in spec.assumptions)

    tasks, _ = planner.plan(spec=spec, user_selected_doc_ids=None)
    assert len(tasks) > 0
    date_tasks = [t for t in tasks if t.purpose.startswith("date_version")]
    assert len(date_tasks) > 0
    assert all(task.event_date == "2024-05-15" for task in date_tasks)


def test_dialog_scenario_4_justified_refusal_in_dialog():
    """Тест 4 (Отказ внутри диалога): корректный обоснованный отказ без галлюцинаций.

    После обсуждения договора пользователь спрашивает о факте, отсутствующем в документах:
    - Агент выдает обоснованный отказ;
    - refused=True, refusal_reason не пустой;
    - sources пусты ({});
    - В тексте ответа нет тегов <source>;
    - Нет выдуманных фактов или ложных заимствований из предыдущего шага диалога.
    """
    history = [
        {"role": "user", "content": "Какая арендная плата по договору с ООО «Северный Склад»?"},
        {"role": "assistant", "content": "Арендная плата составляет 450 000 руб./мес. <source>S1</source>"},
    ]

    query_absent = "Предусмотрена ли договором компенсация за питание сотрудников охраны?"

    payload_absent = {
        "intent": "exact_lookup",
        "task_type": "lookup",
        "entities_raw": ["ООО «Северный Склад»"],
        "article_refs": [],
        "dates": [],
        "event_date": None,
        "premises": [],
        "required_slots": ["general"],
        "assumptions": [],
        "ambiguity": "low",
        "needs_clarification": False,
        "decomposition": [],
        "rewritten_queries_bm25": ["питание сотрудников охраны"],
        "rewritten_queries_dense": ["питание сотрудников охраны"],
    }

    def fake_retrieve_absent(tasks, **kwargs):
        # Поиск не нашел релевантных фрагментов
        return [], _step("retrieval")

    with patch.object(orch_mod, "run_prefilter", return_value=(PreFilterResult(False, None, None), _step("prefilter"))), \
         patch("rag_core.generation.analysis.query_analysis.llm_complete_with_usage", side_effect=[
             ("Предусмотрена ли договором с ООО «Северный Склад» компенсация за питание сотрудников охраны?", 20, 10),
             (json.dumps(payload_absent), 40, 20),
         ]), \
         patch("rag_core.generation.retrieval.bridge.retrieve_tasks", side_effect=fake_retrieve_absent):

        res = orch_mod.run_agent(query_absent, chat_history=history)
        assert res.refused is True
        assert res.refusal_reason in ("no_relevant_documents", "insufficient_evidence", "cannot_parse")
        assert res.sources == {}
        assert "<source>" not in res.final_answer
        assert "450 000" not in res.final_answer


def test_dialog_scenario_5_inline_transcript_parsing():
    """Тест 5.1: Разбор стенограммы диалога из ячейки CSV (parse_dialog_transcript)."""
    raw_transcript = (
        "Пользователь: Мы арендуем склад у «Северного Склада». Планируем съезжать.\n"
        "Ассистент: Подскажите номер договора, чтобы я мог проверить условия возврата помещения.\n"
        "Пользователь: А-08/2024. Можно просто сдать ключи или обязательно оформлять акт?"
    )

    query, inline_history = solution.parse_dialog_transcript(raw_transcript)
    assert query == "А-08/2024. Можно просто сдать ключи или обязательно оформлять акт?"
    assert inline_history is not None
    assert len(inline_history) == 2
    assert inline_history[0] == {
        "role": "user",
        "content": "Мы арендуем склад у «Северного Склада». Планируем съезжать.",
    }
    assert inline_history[1] == {
        "role": "assistant",
        "content": "Подскажите номер договора, чтобы я мог проверить условия возврата помещения.",
    }


def test_dialog_scenario_6_solution_process_group_safe_sequential():
    """Тест 6: Последовательная обработка диалога через solution._process_group_safe.

    Проверяет, что:
    - chat_history накапливается от шага к шагу;
    - теги <source>S1</source> заменяются на читаемые цитаты через replace_source_tags;
    - ответ обрезается limit_answer;
    - сохраняются исходные индексы.
    """
    group = [
        (0, solution.QuestionItem(question_id="101", question="Какая ставка аренды?", dialog_id="d1", turn=1)),
        (1, solution.QuestionItem(question_id="102", question="А до какого числа платить?", dialog_id="d1", turn=2)),
    ]

    history_snapshots: list[list[dict]] = []

    def mock_run_agent(query, chat_history=None, **kwargs):
        history_snapshots.append(list(chat_history) if chat_history else [])
        if "ставка" in query:
            ans = "Ставка составляет 450 000 руб. <source>S1</source>"
            sources = {"S1": {"document_title": "Договор аренды", "structure_path": ["п. 3.1"]}}
        else:
            ans = "Платить нужно до 5-го числа <source>S1</source>"
            sources = {"S1": {"document_title": "Договор аренды", "structure_path": ["п. 3.2"]}}

        return AgentResult(
            final_answer=ans,
            sources=sources,
            intent="lookup",
            refused=False,
            trace=[],
        )

    with patch("solution.run_agent", side_effect=mock_run_agent):
        collected, error = solution._process_group_safe(group)
        assert error is None
        assert len(collected) == 2

        # Первый шаг выполнялся с пустой историей
        assert history_snapshots[0] == []

        # Второй шаг получил историю с вопросом и ответом первого шага
        assert len(history_snapshots[1]) == 2
        assert history_snapshots[1][0]["role"] == "user"
        assert history_snapshots[1][0]["content"] == "Какая ставка аренды?"
        assert history_snapshots[1][1]["role"] == "assistant"
        # Проверяем, что в истории уже отформатированный ответ со ссылкой на документ
        assert "Договор аренды" in history_snapshots[1][1]["content"]

        # Проверяем результат работы группы
        q_id1, ans1 = collected[0][1]
        assert q_id1 == "101"
        assert "<source>" not in ans1
        assert "Договор аренды" in ans1

        q_id2, ans2 = collected[1][1]
        assert q_id2 == "102"
        assert "<source>" not in ans2
        assert "5-го числа" in ans2


def test_dialog_scenario_7_solution_execution_groups_order():
    """Тест 7: Сохранение исходного порядка строк при группировке вопросов."""
    items = [
        solution.QuestionItem(question_id="q1", question="Одиночный вопрос 1"),
        solution.QuestionItem(question_id="q2", question="Диалог реплика 2", dialog_id="d1", turn=2),
        solution.QuestionItem(question_id="q3", question="Одиночный вопрос 2"),
        solution.QuestionItem(question_id="q4", question="Диалог реплика 1", dialog_id="d1", turn=1),
    ]

    groups = solution._build_execution_groups(items)
    assert len(groups) == 3

    d1_group = next(grp for grp in groups if len(grp) == 2)
    assert d1_group[0][1].question_id == "q4"  # turn 1
    assert d1_group[1][1].question_id == "q2"  # turn 2
