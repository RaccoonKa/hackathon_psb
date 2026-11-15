"""Тесты аддитивных структур S0 в ``rag_core.generation.types``.

Проверяют только объявленные типы и обратную совместимость конструкторов.
Ни один рабочий сценарий здесь не подключается — поведение системы S0 не
меняет.
"""

from rag_core.generation.types import (
    AgentResult,
    ChunkTrust,
    Claim,
    ClaimStatus,
    ClaimVerdict,
    DocRef,
    DocTrust,
    EvidenceAssessment,
    InjectionFinding,
    QuerySpec,
    RegenOutcome,
    RetrievedChunk,
    RetrievalTask,
    SlotDrop,
    SlotDropReason,
    SlotOrigin,
    SourceMetadata,
    TraceStep,
)


def _chunk(**kwargs) -> RetrievedChunk:
    data = {
        "chunk_id": "c1",
        "text": "факт и инструкция",
        "document_id": "d1",
        "version_id": "d1_v1",
        "document_title": "Документ",
        "provision_id": None,
        "structure_path": [],
        "score": 0.5,
    }
    data.update(kwargs)
    return RetrievedChunk(**data)


def test_doctrust_values():
    assert DocTrust.TRUSTED_FACTS.value == "trusted_facts"
    assert DocTrust.MIXED.value == "mixed"
    assert DocTrust.INSTRUCTION_ONLY.value == "instruction_only"


def test_regen_outcome_values():
    assert RegenOutcome.OK.value == "ok"
    assert RegenOutcome.EMPTY_TEXT.value == "empty_text"
    assert RegenOutcome.EMPTY_TEXT_WITH_SUPPORTED.value == "empty_text_with_supported"
    assert RegenOutcome.PARSE_FAILED.value == "parse_failed"
    assert RegenOutcome.INCONSISTENT.value == "inconsistent"


def test_slot_drop_reason_values():
    assert SlotDropReason.UNKNOWN_SLOT.value == "unknown_slot"
    assert SlotDropReason.NOT_ASKED.value == "not_asked"
    assert SlotDropReason.EXCLUDED_BY_RULE.value == "excluded_by_rule"
    assert SlotDropReason.CONFLICT_WITH_OTHER_SLOT.value == "conflict_with_other_slot"


def test_slot_origin_values_are_strings():
    # Поле Claim.slot_origin остаётся строковым; enum — удобные константы.
    assert SlotOrigin.LLM == "llm"
    assert SlotOrigin.REGISTRY_RESOLVED == "registry_resolved"
    assert SlotOrigin.RECLASSIFIED == "reclassified"
    assert isinstance(SlotOrigin.LLM, str)


def test_injection_finding_defaults_and_frozen():
    finding = InjectionFinding(
        pattern_id="ai_attention", span_start=0, span_end=5, text="текст"
    )
    assert finding.severity == "high"
    try:
        finding.pattern_id = "other"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("InjectionFinding должен быть frozen")


def test_chunk_trust_defaults_and_coercion():
    trust = ChunkTrust(doc_trust=DocTrust.TRUSTED_FACTS)
    assert trust.findings == ()
    assert trust.instruction_spans == ()
    assert trust.fact_text == ""
    assert trust.instruction_text == ""

    # Списки и строковое значение enum принимаются (устойчивость к S2).
    coerced = ChunkTrust(
        doc_trust="mixed",  # type: ignore[arg-type]
        findings=[InjectionFinding("p", 0, 1, "x")],  # type: ignore[arg-type]
        instruction_spans=[[0, 3]],  # type: ignore[arg-type]
        fact_text="факт",
        instruction_text="инструкция",
    )
    assert coerced.doc_trust is DocTrust.MIXED
    assert isinstance(coerced.findings, tuple)
    assert coerced.instruction_spans == ((0, 3),)
    assert coerced.fact_text == "факт"


def test_slot_drop_defaults_and_reason_coercion():
    drop = SlotDrop(slot="claim_attachments", reason=SlotDropReason.NOT_ASKED, detail="d")
    assert drop.matched_markers == []
    coerced = SlotDrop(slot="x", reason="unknown_slot", detail="d")  # type: ignore[arg-type]
    assert coerced.reason is SlotDropReason.UNKNOWN_SLOT
    coerced.matched_markers.append("маркер")
    assert coerced.matched_markers == ["маркер"]


def test_retrieved_chunk_trust_defaults_backward_compatible():
    chunk = _chunk()
    assert chunk.doc_trust is DocTrust.TRUSTED_FACTS
    assert chunk.instruction_spans == ()
    assert chunk.instruction_findings == ()
    assert chunk.fact_text is None
    assert chunk.untrusted is False

    # Позиционная совместимость старого конструктора (8 обязательных полей).
    positional = RetrievedChunk("c", "t", "d", "v", "T", None, [], 0.1)
    assert positional.doc_trust is DocTrust.TRUSTED_FACTS
    assert positional.fact_text is None


def test_docref_defaults_positional_compatible():
    ref = DocRef("Альфа-Трейд", None, ["a", "b"], False)
    assert ref.ambiguous is False
    assert ref.match_level == "none"
    assert ref.match_scores == {}
    assert ref.routing_note is None
    ref.match_scores["a"] = 0.9
    assert ref.match_scores == {"a": 0.9}


def test_claim_lifecycle_defaults():
    claim = Claim(text="факт", source_ids=["S1"], slot=None)
    assert claim.slot_origin is None
    assert claim.slot_confidence == 0.0
    assert claim.status is None
    assert claim.status_reason is None
    assert claim.evidence_quote is None
    assert claim.evidence_source_id is None
    # Позиционная совместимость 4 полей сохранена.
    legacy = Claim("факт", ["S1"], "rent_amount", 0)
    assert legacy.original_idx == 0
    assert legacy.slot_origin is None


def test_claim_verdict_scope_defaults():
    verdict = ClaimVerdict(claim_idx=0, status=ClaimStatus.SUPPORTED, quote="q", source_id="S1")
    assert verdict.slot is None
    assert verdict.expected_source_kind is None
    assert verdict.observed_source_kind is None
    assert verdict.evidence_quote is None
    assert verdict.fact_part_used is False


def test_query_spec_slot_drops_default():
    from tests._common import make_query_spec

    spec = make_query_spec()
    assert spec.slot_drops == []
    spec.slot_drops.append(SlotDrop("x", SlotDropReason.NOT_ASKED, "d"))


def test_retrieval_task_new_defaults():
    from tests._common import make_retrieval_task

    task = make_retrieval_task()
    assert task.required_slots == []
    assert task.slot_queries == []
    assert task.pinned_doc_ids is None


def test_evidence_assessment_fallback_defaults():
    assessment = EvidenceAssessment(
        covered_slots=[],
        missing_slots=[],
        core_slot_covered=False,
        sufficiency="insufficient",
        degraded_intents=[],
        reason="test",
    )
    assert assessment.slot_coverage_reasons == {}
    assert assessment.retrieval_fallback is False
    assert assessment.retrieval_fallback_reason is None


def test_source_metadata_accepts_trust_keys():
    meta: SourceMetadata = {
        "chunk_id": "c1",
        "text": "факт",
        "doc_trust": "mixed",
        "has_instructions": True,
        "instruction_spans": [[0, 3]],
        "fact_text": "факт",
    }
    assert meta["doc_trust"] == "mixed"
    # total=False: старые словари без новых ключей валидны.
    legacy: SourceMetadata = {"chunk_id": "c2", "text": "t"}
    assert "doc_trust" not in legacy


def test_trust_structures_json_compatible():
    trust = ChunkTrust(
        doc_trust=DocTrust.MIXED,
        findings=(InjectionFinding("ai_attention", 0, 3, "abc", "high"),),
        instruction_spans=((0, 3),),
        fact_text="факт",
        instruction_text="abc",
    )
    step = TraceStep(step="x", input={}, output={"trust": trust}, latency_ms=1.0)
    result = AgentResult(
        final_answer="ok", sources={}, intent="semantic", refused=False, trace=[step]
    )
    payload = result.to_dict()
    serialized = payload["trace"][0]["output"]["trust"]
    assert serialized["doc_trust"] == "mixed"
    assert serialized["instruction_spans"] == [[0, 3]]
    assert serialized["findings"][0]["pattern_id"] == "ai_attention"
    assert serialized["findings"][0]["severity"] == "high"
