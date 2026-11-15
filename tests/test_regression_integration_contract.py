"""Тяжёлый интеграционный регрессионный тест контрольного вопроса.

Использует РЕАЛЬНЫЙ retrieval по реальному индексу (BM25 + Chroma +
llama.cpp эмбеддинг-сервер + chunks.jsonl). Замоканы только LLM-этапы
(анализ запроса, генератор-стаб и верификатор), сам retrieval — настоящий.

Генератор-стаб формирует ответ ИЗ ПЕРЕДАННОГО КОНТЕКСТА: находит source-метки
блоков, содержащих оба факта, и цитирует именно их. Если артефакты индекса
или модели отсутствуют — корректный skip с причиной.
"""

import re
from pathlib import Path
from unittest import SkipTest
from unittest.mock import patch

from rag_core import config
from rag_core.generation import orchestrator as orch_mod
from rag_core.generation.analysis.entity_resolver import resolve_single_entity
from rag_core.generation.retrieval import bridge as bridge_mod
from rag_core.generation.retrieval import evidence as evidence_mod
from rag_core.generation.types import (
    Ambiguity,
    Claim,
    DraftAnswer,
    PreFilterResult,
    QuerySpec,
    TaskType,
    TraceStep,
)
from rag_core.retrieval import vector_store as vs_mod
from tests._common import make_verification_report

# Тест помечен как тяжёлый/интеграционный: требует реального индекса и модели.
INTEGRATION = True
HEAVY = True

ROOT = Path(__file__).resolve().parent.parent
BM25_FILE = ROOT / "storage" / "bm25_index" / "bm25.pkl"
CHROMA_DIR = ROOT / "storage" / "chroma_db"
CHUNKS_FILE = ROOT / "data" / "processed" / "chunks.jsonl"

CONTRACT_DOC_ID = "contract_а-08-2024"
RENT_CHUNK_ID = "contract_а-08-2024_v1_0006_f0f28c05"
DEADLINE_CHUNK_ID = "contract_а-08-2024_v1_0007_e4e9b757"
CONTROL_QUESTION = (
    "Какая арендная плата по договору с ООО «Северный Склад» "
    "и до какого числа её нужно вносить?"
)


def _skip_if_artifacts_missing() -> None:
    missing = [
        str(path)
        for path in (BM25_FILE, CHROMA_DIR, CHUNKS_FILE)
        if not path.exists()
    ]
    if missing:
        raise SkipTest(
            "Интеграционный тест пропущен: отсутствуют артефакты " + ", ".join(missing)
        )

    # Эмбеддинги считает внешний llama.cpp-сервер: без него реальный
    # retrieval невозможен, поэтому тест честно пропускается.
    try:
        vs_mod.get_query_embedding("проверка доступности сервера эмбеддингов")
    except Exception as exc:  # noqa: BLE001 - нет сервера → skip
        raise SkipTest(
            "Интеграционный тест пропущен: сервер эмбеддингов недоступен "
            f"({type(exc).__name__})"
        )
    diagnostics = vs_mod.get_last_embedding_diagnostics()
    if diagnostics.get("provider") != "api":
        raise SkipTest(
            "Интеграционный тест пропущен: сервер эмбеддингов недоступен "
            f"(провайдер {diagnostics.get('provider')})"
        )

    try:
        collection = vs_mod.get_chroma_collection()
        count = int(collection.count())
    except Exception as exc:  # noqa: BLE001 - окружение без Chroma → skip
        raise SkipTest(f"Интеграционный тест пропущен: Chroma недоступна ({exc})")
    if count <= 0:
        raise SkipTest("Интеграционный тест пропущен: коллекция Chroma пуста")


def _control_spec() -> QuerySpec:
    ref = resolve_single_entity("ООО «Северный Склад»")
    assert ref.resolved is True
    assert ref.document_id == CONTRACT_DOC_ID
    return QuerySpec(
        intent="mixed",
        task_type=TaskType.LOOKUP,
        entities_raw=["ООО «Северный Склад»"],
        entities_resolved=[ref],
        article_refs=[],
        dates=[],
        event_date=None,
        premises=[],
        required_slots=["rent_amount", "payment_deadline"],
        assumptions=[],
        ambiguity=Ambiguity.LOW,
        needs_clarification=False,
        decomposition=[],
        rewritten_queries_bm25=[CONTROL_QUESTION],
        rewritten_queries_dense=[CONTROL_QUESTION],
        history_resolved=False,
        degradation_reason=None,
    )


def _step(name: str) -> TraceStep:
    return TraceStep(step=name, input={}, output={}, latency_ms=0.0)


def _context_blocks(context: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    for part in context.split(config.CONTEXT_CHUNK_SEPARATOR):
        stripped = part.strip()
        match = re.match(r"^\[(S\d+)\]", stripped)
        if match:
            blocks[match.group(1)] = stripped
    return blocks


def test_control_question_integration_real_retrieval():
    _skip_if_artifacts_missing()

    spec = _control_spec()
    captured: dict = {}

    real_build_context = orch_mod.build_context_and_sources
    real_assess = evidence_mod.assess

    def wrapped_build_context(chunks):
        sources, context, step = real_build_context(chunks)
        captured["sources"] = sources
        captured["context"] = context
        captured["context_chunks"] = list(chunks)
        return sources, context, step

    def wrapped_assess(*args, **kwargs):
        bundle, assessment, step = real_assess(*args, **kwargs)
        captured["bundle"] = bundle
        captured["assessment"] = assessment
        return bundle, assessment, step

    def passthrough_rerank(chunks, query, top_n):
        return list(chunks[:top_n]), _step(config.STEP_RERANKING)

    def generator_stub(
        spec,
        evidence,
        policy,
        context,
        query=None,
        chat_history=None,
        on_failure=None,
        on_llm_success=None,
        timeout=None,
        **kwargs,
    ):
        # Ответ строится исключительно из переданного контекста.
        blocks = _context_blocks(context)
        rent_sid = next(
            (sid for sid, text in blocks.items() if "450 000" in text), None
        )
        deadline_sid = next(
            (sid for sid, text in blocks.items() if "5-го числа" in text), None
        )
        assert rent_sid is not None, "В контексте генератора нет факта про 450 000"
        assert deadline_sid is not None, "В контексте генератора нет срока оплаты"
        answer = (
            "Арендная плата по договору с ООО «Северный Склад» составляет "
            "450 000 руб./мес. (включая НДС), вносить её нужно не позднее "
            "5-го числа текущего месяца "
            f"<source>{rent_sid}, {deadline_sid}</source>."
        )
        claims = [
            Claim(
                text="Арендная плата 450 000 руб./мес. включая НДС",
                source_ids=[rent_sid],
                slot="rent_amount",
                original_idx=0,
            ),
            Claim(
                text="Оплата не позднее 5-го числа текущего месяца",
                source_ids=[deadline_sid],
                slot="payment_deadline",
                original_idx=1,
            ),
        ]
        step = TraceStep(
            step=config.STEP_GENERATION,
            input={"query": query, "context_length": len(context)},
            output={"claims_count": len(claims)},
            latency_ms=0.0,
        )
        return DraftAnswer(answer, claims, [], [], None), step

    with patch.object(
        orch_mod,
        "run_prefilter",
        return_value=(PreFilterResult(False, None, None), _step("prefilter")),
    ), patch.object(
        orch_mod, "analyze_query", return_value=(spec, _step("query_analysis"), False)
    ), patch.object(
        bridge_mod, "rerank_chunks", side_effect=passthrough_rerank
    ), patch.object(
        evidence_mod, "assess", side_effect=wrapped_assess
    ), patch.object(
        orch_mod, "build_context_and_sources", side_effect=wrapped_build_context
    ), patch.object(
        orch_mod, "generate_draft", side_effect=generator_stub
    ), patch.object(
        orch_mod,
        "verify",
        return_value=(make_verification_report(), _step("verify_draft")),
    ), patch.object(
        orch_mod.config, "MAX_FOLLOWUP_RETRIEVAL_CYCLES", 0
    ), patch.object(
        orch_mod.config, "RETRIEVAL_TASK_TIMEOUT_S", 180.0
    ), patch.object(
        orch_mod.config, "RETRIEVAL_SHUTDOWN_GRACE_S", 10.0
    ):
        result = orch_mod.run_agent(CONTROL_QUESTION)

    bundle = captured["bundle"]
    assessment = captured["assessment"]
    retrieved_ids = [chunk.chunk_id for chunk in bundle.chunks]
    context = captured["context"]
    sources = captured["sources"]

    # Печать фактического результата (полезно в отчёте run_all).
    print("integration retrieved:", retrieved_ids)
    print("integration covered_slots:", assessment.covered_slots, assessment.reason)
    print("integration answer:", result.final_answer[:200])

    # Договор найден, оба факта присутствуют в реальной выдаче.
    assert any(
        chunk.document_id == CONTRACT_DOC_ID for chunk in bundle.chunks
    ), "В выдаче нет договора А-08/2024"
    assert RENT_CHUNK_ID in retrieved_ids
    assert DEADLINE_CHUNK_ID in retrieved_ids

    # Evidence покрывает оба слота.
    assert "rent_amount" in assessment.covered_slots
    assert "payment_deadline" in assessment.covered_slots
    assert assessment.reason != "core_slot_not_covered"

    # Контекст генератора содержит оба факта.
    assert "450 000" in context
    assert "5-го числа" in context

    # Генератор-стаб ответил без отказа, факты и цитаты — из контекста.
    assert result.refused is False
    assert "450 000" in result.final_answer
    assert "5-го числа" in result.final_answer
    assert "<source>" in result.final_answer
    assert result.sources

    # Source-метки соответствуют найденным chunk_id.
    retrieved_set = set(retrieved_ids)
    provision_to_chunk = {}
    for entry in result.sources.values():
        assert entry["chunk_id"] in retrieved_set
        if entry.get("provision_id"):
            provision_to_chunk[entry["provision_id"]] = entry["chunk_id"]
    assert provision_to_chunk.get("3.1") == RENT_CHUNK_ID
    assert provision_to_chunk.get("3.2") == DEADLINE_CHUNK_ID

    # Context/source-карта согласованы с реальными чанками.
    for source_id, entry in sources.items():
        assert f"[{source_id}]" in context
        assert entry["chunk_id"] in retrieved_set
