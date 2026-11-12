from unittest.mock import MagicMock, patch

from src import config
from src.generation import prompts
from src.generation.answer.context_builder import (
    _format_effective_dates,
    build_context_and_sources,
    build_source_entry,
    format_single_chunk,
)
from src.generation.llm.client import (
    _extract_token_usage,
    llm_complete,
    llm_complete_with_usage,
)
from src.generation.orchestrator import (
    _sanitize_and_extract_citations,
    assemble_final,
)
from src.generation.refusal import (
    build_no_context_refusal_result,
    build_template_intent_result,
)
from src.generation.types import AgentResult, DraftAnswer, Intent, TraceStep
from tests._common import (
    make_query_spec,
    make_retrieved_chunk,
    make_verification_report,
)


def _draft(answer="Текст <source>S1</source>", claims=None, refusal_reason=None):
    return DraftAnswer(
        answer_text=answer,
        claims=claims or [],
        assumptions=[],
        missing_info=[],
        refusal_reason=refusal_reason,
    )


def _assemble(draft, sources, intent="semantic_query", trace=None, verification=None):
    return assemble_final(
        draft=draft,
        sources=sources,
        spec=make_query_spec(intent=intent),
        verification=verification,
        trace=list(trace or []),
        degradation_reasons=[],
        used_history=False,
        failures_global=0,
    )


def test_templates_non_empty():
    for attr in (
        "RESPONSE_OUT_OF_SCOPE",
        "RESPONSE_ABOUT_ASSISTANT",
        "RESPONSE_REFUSAL_NO_CONTEXT",
    ):
        assert getattr(prompts, attr).strip()


def test_template_key_resolves_config_constant():
    result = build_template_intent_result(
        query="?",
        intent=Intent.OUT_OF_SCOPE,
        prior_trace=[],
        template_key=config.RESPONSE_OUT_OF_SCOPE,
    )
    assert result.final_answer == prompts.RESPONSE_OUT_OF_SCOPE


def test_format_effective_dates_both():
    ch = make_retrieved_chunk(effective_from="2024-01-01", effective_to="2025-01-01")
    assert _format_effective_dates(ch) == "действует с 2024-01-01 по 2025-01-01"


def test_format_effective_dates_only_from():
    ch = make_retrieved_chunk(effective_from="2024-01-01", effective_to=None)
    assert "по настоящее время" in _format_effective_dates(ch)


def test_format_effective_dates_only_to():
    ch = make_retrieved_chunk(effective_from=None, effective_to="2025-01-01")
    assert "действует по 2025-01-01" in _format_effective_dates(ch)


def test_format_effective_dates_unknown():
    ch = make_retrieved_chunk(effective_dates_known=False)
    assert _format_effective_dates(ch) == ""


def test_format_single_chunk_full():
    ch = make_retrieved_chunk(
        provision_id="3.2",
        structure_path=["3. Порядок поставки"],
        effective_from="2024-02-12",
        effective_to="2026-12-31",
    )
    out = format_single_chunk("S1", ch)
    assert "[S1]" in out
    assert f"{prompts.PROVISION_LINE_PREFIX} 3.2" in out
    assert "Редакция:" in out
    assert "2024-02-12" in out


def test_format_single_chunk_no_provision():
    ch = make_retrieved_chunk(provision_id=None)
    assert "Пункт:" not in format_single_chunk("S1", ch)


def test_format_single_chunk_no_effective():
    ch = make_retrieved_chunk(effective_dates_known=False)
    assert "Редакция:" not in format_single_chunk("S1", ch)


def test_build_source_entry_full():
    ch = make_retrieved_chunk(
        chunk_id="c1",
        document_id="d1",
        version_id="v1",
        document_title="Договор",
        provision_id="3.2",
        structure_path=["3. Порядок", "3.2. Адрес"],
        score=0.9,
        effective_from="2024-01-01",
        amends_document_id="parent_doc",
    )
    entry = build_source_entry(ch)
    assert entry["chunk_id"] == "c1"
    assert entry["document_id"] == "d1"
    assert entry["provision_id"] == "3.2"
    assert entry["structure_path"] == ["3. Порядок", "3.2. Адрес"]
    assert entry["score"] == 0.9
    assert entry["amends_document_id"] == "parent_doc"


def test_build_context_two_chunks():
    chunks = [
        make_retrieved_chunk(chunk_id="a", text="A"),
        make_retrieved_chunk(chunk_id="b", text="B"),
    ]
    sources, ctx, step = build_context_and_sources(chunks)
    assert list(sources.keys()) == ["S1", "S2"]
    assert sources["S1"]["chunk_id"] == "a"
    assert sources["S2"]["chunk_id"] == "b"
    assert "A" in ctx
    assert "B" in ctx
    assert step.output["source_ids"] == ["S1", "S2"]
    assert step.step == config.STEP_CONTEXT_ASSEMBLY


def test_citations_single_valid():
    sources = {"S1": {}, "S2": {}}
    cleaned, cited = _sanitize_and_extract_citations(
        "Текст <source>S1</source>.", sources
    )
    assert cited == ["S1"]
    assert "<source>S1</source>" in cleaned


def test_citations_multiple_valid():
    sources = {"S1": {}, "S2": {}}
    _, cited = _sanitize_and_extract_citations("A <source>S1, S2</source> B", sources)
    assert cited == ["S1", "S2"]


def test_citations_invalid_removed():
    sources = {"S1": {}}
    cleaned, cited = _sanitize_and_extract_citations(
        "A <source>X9</source> B", sources
    )
    assert cited == []
    assert "<source>" not in cleaned


def test_citations_mixed_keeps_valid():
    sources = {"S1": {}}
    cleaned, cited = _sanitize_and_extract_citations(
        "A <source>S1, X9</source> B", sources
    )
    assert cited == ["S1"]
    assert "<source>S1</source>" in cleaned
    assert "X9" not in cleaned


def test_citations_deduplicates():
    sources = {"S1": {}}
    _, cited = _sanitize_and_extract_citations(
        "A <source>S1</source> B <source>S1</source> C", sources
    )
    assert cited == ["S1"]


def test_citations_space_before_punct_cleaned():
    cleaned, _ = _sanitize_and_extract_citations(
        "Текст <source>X9</source> .", {}
    )
    assert " ." not in cleaned
    assert "." in cleaned


def test_citations_none_present():
    sources = {"S1": {}}
    cleaned, cited = _sanitize_and_extract_citations("Просто текст.", sources)
    assert cited == []
    assert cleaned == "Просто текст."


def test_assemble_filters_sources_by_citations():
    sources = {"S1": {"chunk_id": "c1"}, "S2": {"chunk_id": "c2"}}
    result = _assemble(_draft("Текст <source>S1</source>"), sources)
    assert isinstance(result, AgentResult)
    assert result.refused is False
    assert list(result.sources.keys()) == ["S1"]
    assert result.intent == "semantic_query"
    assert result.trace[-1].step == config.STEP_RESPONSE_ASSEMBLY


def test_assemble_uses_claim_sources():
    from src.generation.types import Claim

    sources = {"S1": {}, "S2": {}}
    draft = _draft("Без тега", claims=[Claim(text="факт", source_ids=["S2"], slot=None)])
    result = _assemble(draft, sources)
    assert list(result.sources.keys()) == ["S2"]
    assert result.refused is False


def test_assemble_no_citations_and_no_claims_returns_no_sources():
    # Новая политика: в sources попадают только фактически процитированные
    # источники (и источники claims), а не весь контекст.
    sources = {"S1": {}, "S2": {}}
    result = _assemble(_draft("Без цитат."), sources)
    assert result.sources == {}
    assert result.refused is False


def test_assemble_appends_assembly_step():
    result = _assemble(_draft("x"), {})
    assert result.trace[-1].step == config.STEP_RESPONSE_ASSEMBLY
    assert result.trace[-1].output["refused"] is False


def test_assemble_preserves_prior_trace():
    prior = [TraceStep(step="x", input={}, output={}, latency_ms=0.0)]
    result = _assemble(_draft("x"), {}, trace=prior)
    assert len(result.trace) == 2
    assert result.trace[0].step == "x"


def test_assemble_step_output_sources_count():
    result = _assemble(_draft("<source>S1</source>"), {"S1": {}})
    step = result.trace[-1]
    assert step.output["refused"] is False
    assert step.output["sources_count"] == 1


def test_assemble_empty_answer_refused():
    result = _assemble(_draft(""), {"S1": {}})
    assert result.refused is True
    assert result.sources == {}


def test_assemble_exact_intent_kept():
    result = _assemble(_draft("x"), {}, intent="exact_lookup")
    assert result.intent == "exact_lookup"


def test_assemble_mixed_intent_kept():
    result = _assemble(_draft("x"), {}, intent="mixed_query")
    assert result.intent == "mixed_query"


def test_assemble_clears_missing_info_for_supported_slots():
    from src.generation.types import Claim

    draft = _draft(
        "<source>S1</source>",
        claims=[Claim(text="факт", source_ids=["S1"], slot="rent_amount", original_idx=0)],
    )
    draft.missing_info = ["версия не определена, нужна дата"]
    result = assemble_final(
        draft=draft,
        sources={"S1": {}},
        spec=make_query_spec(intent="semantic_query", required_slots=["rent_amount"]),
        verification=make_verification_report(supported_locked=[0]),
        trace=[],
        degradation_reasons=[],
        used_history=False,
        failures_global=0,
    )
    assert result.missing_info == []


def test_resolve_template_keys_via_refusal_module():
    result = build_template_intent_result(
        query="?",
        intent=Intent.ABOUT_ASSISTANT,
        prior_trace=[],
        template_key=config.RESPONSE_ABOUT_ASSISTANT,
    )
    assert result.final_answer == prompts.RESPONSE_ABOUT_ASSISTANT


def test_template_out_of_scope():
    result = build_template_intent_result(
        query="?", intent=Intent.OUT_OF_SCOPE, prior_trace=[]
    )
    assert result.refused is True
    assert result.final_answer == prompts.RESPONSE_OUT_OF_SCOPE
    assert result.sources == {}
    assert result.intent == "out_of_scope"


def test_template_about_assistant():
    result = build_template_intent_result(
        query="?", intent=Intent.ABOUT_ASSISTANT, prior_trace=[]
    )
    assert result.refused is False
    assert result.final_answer == prompts.RESPONSE_ABOUT_ASSISTANT


def test_no_context_refusal():
    result = build_no_context_refusal_result(
        query="?", intent=Intent.SEMANTIC_QUERY, prior_trace=[]
    )
    assert result.refused is True
    assert result.final_answer == prompts.RESPONSE_REFUSAL_NO_CONTEXT
    assert result.sources == {}
    assert result.trace[-1].step == config.STEP_REFUSAL


def test_extract_token_usage_from_metadata():
    resp = MagicMock()
    resp.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    resp.response_metadata = None
    p, c = _extract_token_usage(resp)
    assert p == 100
    assert c == 50


def test_extract_token_usage_from_response_metadata():
    resp = MagicMock()
    resp.usage_metadata = None
    resp.response_metadata = {
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }
    p, c = _extract_token_usage(resp)
    assert p == 10
    assert c == 5


def test_extract_token_usage_fallback():
    resp = MagicMock()
    resp.usage_metadata = None
    resp.response_metadata = None
    p, c = _extract_token_usage(resp)
    assert p == config.STUB_PROMPT_TOKENS
    assert c == config.STUB_COMPLETION_TOKENS


def test_llm_complete_with_usage_mocked():
    with patch(
        "src.generation.llm.client._invoke_real_llm",
        return_value=("Ответ", 10, 5),
    ):
        text, p, c = llm_complete_with_usage(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            temperature=0.0,
            max_tokens=100,
        )
    assert text == "Ответ"
    assert p == 10
    assert c == 5


def test_llm_complete_mocked():
    with patch(
        "src.generation.llm.client._invoke_real_llm",
        return_value=("Ответ", 10, 5),
    ):
        text = llm_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            temperature=0.0,
            max_tokens=100,
        )
    assert text == "Ответ"


def test_run_agent_invokes_reranker_before_evidence():
    """Регрессия: retrieve_tasks вызывает hybrid_search с rerank=False,
    поэтому оркестратор обязан явно применить rerank_chunks до evidence.assess.
    """
    from src.generation import orchestrator as orch_mod
    from src.generation.retrieval import bridge as bridge_mod
    from src.generation.retrieval import evidence as evidence_mod
    from dataclasses import replace

    from src.generation.types import (
        DraftAnswer,
        EvidenceAssessment,
        EvidenceBundle,
        PolicyVerdict,
        PreFilterResult,
        Sufficiency,
    )

    raw = [
        make_retrieved_chunk(chunk_id="c1", score=0.02),
        make_retrieved_chunk(chunk_id="c2", score=0.01),
    ]
    # Реранкер возвращает новый порядок и обновлённые score; оркестратор
    # пересортировывает накопленный пул по score.
    reranked = [
        replace(raw[1], score=0.95),
        replace(raw[0], score=0.85),
    ]
    spec = make_query_spec(
        intent="semantic",
        event_date="2025-01-01",
        required_slots=["rent_amount"],
    )
    assessment = EvidenceAssessment(
        covered_slots=["rent_amount"],
        missing_slots=[],
        core_slot_covered=True,
        sufficiency=Sufficiency.SUFFICIENT,
        degraded_intents=[],
        reason="all_required_slots_covered",
    )
    step = lambda name: TraceStep(step=name, input={}, output={}, latency_ms=0.0)

    with patch.object(
        orch_mod, "run_prefilter",
        return_value=(PreFilterResult(False, None, None), step("prefilter")),
    ), patch.object(
        orch_mod, "analyze_query", return_value=(spec, step("query_analysis"), False),
    ), patch.object(
        bridge_mod, "retrieve_tasks", return_value=(raw, step(config.STEP_RAG_RETRIEVAL)),
    ), patch.object(
        bridge_mod, "rerank_chunks", return_value=(reranked, step(config.STEP_RERANKING)),
    ) as mock_rerank, patch.object(
        evidence_mod, "assess",
        return_value=(
            EvidenceBundle(
                chunks=reranked,
                slot_map={"rent_amount": ["c2"]},
                contradictions=[],
                version_issues=[],
            ),
            assessment,
            step("assess_evidence"),
        ),
    ) as mock_assess, patch.object(
        orch_mod, "precheck",
        return_value=(
            PolicyVerdict(False, [], False, True),
            step("policy_precheck"),
        ),
    ), patch.object(
        orch_mod, "build_context_and_sources",
        return_value=({"S1": {"chunk_id": "c2"}}, "ctx", step(config.STEP_CONTEXT_ASSEMBLY)),
    ), patch.object(
        orch_mod, "generate_draft",
        return_value=(
            DraftAnswer("Ответ", [], [], [], None),
            step(config.STEP_GENERATION),
        ),
    ), patch.object(
        orch_mod, "verify",
        return_value=(make_verification_report(), step("verify_draft")),
    ):
        result = orch_mod.run_agent("Какой срок оплаты?", event_date="2025-01-01")

    mock_rerank.assert_called_once()
    rerank_kwargs = mock_rerank.call_args.kwargs
    assert rerank_kwargs["chunks"] == raw
    assert rerank_kwargs["query"] == "Какой срок оплаты?"
    assert rerank_kwargs["top_n"] == config.RERANKER_TOP_N
    assert mock_assess.call_args.kwargs["chunks"] == reranked
    assert result.refused is False
    assert any(s.step == config.STEP_RERANKING for s in result.trace)
