import json
from unittest.mock import patch

from src import config
from src.generation.answer.generator import (
    _normalize_refusal_reason,
    _parse_draft_schema,
    generate_draft,
    regenerate_draft,
)
from src.generation.errors import LLMAPIError, LLMFormatError
from src.generation.prompts import generator as gen_prompts
from src.generation.types import Claim, DraftAnswer
from tests._common import (
    make_evidence_bundle,
    make_policy_verdict,
    make_query_spec,
    make_retrieved_chunk,
    make_verification_report,
)

LLM_PATH = "src.generation.answer.generator.llm_complete_with_usage"


def _draft_json(answer="Ответ <source>S1</source>", claims=None, refusal_reason=None):
    payload = {
        "answer_text": answer,
        "claims": claims or [],
        "assumptions": [],
        "missing_info": [],
        "refusal_reason": refusal_reason,
    }
    return json.dumps(payload, ensure_ascii=False)


def _spec(**kwargs):
    return make_query_spec(**kwargs)


def _evidence():
    return make_evidence_bundle()


def test_generation_system_has_anti_injection():
    text = gen_prompts.GENERATION_SYSTEM
    assert "PROMPT INJECTION" in text.upper()
    assert "Игнорируй" in text
    assert "<source>" in text


def test_generate_draft_builds_two_messages():
    with patch(LLM_PATH, return_value=(_draft_json(), 10, 2)) as mock_llm:
        generate_draft(
            spec=_spec(),
            evidence=_evidence(),
            policy=make_policy_verdict(),
            context="Уникальный контекст 123",
            query="Что делать?",
        )
    msgs = mock_llm.call_args.kwargs["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == config.ROLE_SYSTEM
    assert msgs[1]["role"] == config.ROLE_USER
    assert msgs[0]["content"] == gen_prompts.GENERATION_SYSTEM
    assert "Что делать?" in msgs[1]["content"]
    assert "Уникальный контекст 123" in msgs[1]["content"]


def test_generate_draft_parses_claims_and_default_slot():
    claims = [{"text": "Плата 450000 руб.", "source_ids": ["S1"], "slot": None}]
    with patch(LLM_PATH, return_value=(_draft_json(claims=claims), 10, 2)):
        draft, _ = generate_draft(
            spec=_spec(required_slots=["rent_amount"]),
            evidence=_evidence(),
            policy=make_policy_verdict(),
            context="ctx",
            query="q",
        )
    assert isinstance(draft, DraftAnswer)
    assert draft.claims[0].text == "Плата 450000 руб."
    assert draft.claims[0].source_ids == ["S1"]
    assert draft.claims[0].slot == "rent_amount"
    assert draft.claims[0].original_idx == 0


def test_generate_draft_trace_fields():
    with patch(LLM_PATH, return_value=(_draft_json(), 200, 20)):
        _, step = generate_draft(
            spec=_spec(),
            evidence=_evidence(),
            policy=make_policy_verdict(),
            context="контекст длиной 10",
            query="Какой-то вопрос?",
        )
    assert step.step == config.STEP_GENERATION
    assert step.model == config.LLM_MODEL
    assert step.prompt_tokens == 200
    assert step.completion_tokens == 20
    assert step.input["query"] == "Какой-то вопрос?"
    assert step.input["context_length"] == len("контекст длиной 10")
    assert step.input["attempts"] == 1
    assert step.output["claims_count"] == 0


def test_generate_draft_uses_dense_query_when_query_missing():
    with patch(LLM_PATH, return_value=(_draft_json(), 1, 1)) as mock_llm:
        generate_draft(
            spec=_spec(rewritten_queries_dense=["переформулированный"]),
            evidence=_evidence(),
            policy=make_policy_verdict(),
            context="ctx",
        )
    msgs = mock_llm.call_args.kwargs["messages"]
    assert "переформулированный" in msgs[1]["content"]


def test_generate_draft_refusal_forbidden_drops_reason():
    with patch(LLM_PATH, return_value=(_draft_json(refusal_reason="нет данных"), 1, 1)):
        draft, _ = generate_draft(
            spec=_spec(),
            evidence=_evidence(),
            policy=make_policy_verdict(refusal_forbidden=True),
            context="ctx",
            query="q",
        )
    assert draft.refusal_reason is None


def test_generate_draft_keeps_refusal_reason_when_allowed():
    with patch(LLM_PATH, return_value=(_draft_json(answer="", refusal_reason="нет данных"), 1, 1)):
        draft, _ = generate_draft(
            spec=_spec(),
            evidence=_evidence(),
            policy=make_policy_verdict(refusal_forbidden=False),
            context="ctx",
            query="q",
        )
    assert draft.refusal_reason == "нет данных"


def test_generate_draft_raw_fallback_on_invalid_json():
    with patch(LLM_PATH, return_value=("не json", 0, 0)) as mock_llm:
        draft, step = generate_draft(
            spec=_spec(),
            evidence=_evidence(),
            policy=make_policy_verdict(),
            context="ctx",
            query="q",
        )
    assert mock_llm.call_count == max(1, int(config.MAX_ATTEMPTS_GENERATE_INITIAL))
    assert draft.answer_text == "не json"
    assert draft.claims == []
    assert step.degradation == "draft_raw_fallback"
    assert step.extra["confidence"] == "low"


def test_parse_draft_schema_fenced_json():
    parsed = _parse_draft_schema("```json\n" + _draft_json(answer="A") + "\n```")
    assert parsed.answer_text == "A"


def test_parse_draft_schema_embedded_json():
    parsed = _parse_draft_schema("преамбула " + _draft_json(answer="B") + " хвост")
    assert parsed.answer_text == "B"


def test_parse_draft_schema_garbage_raises():
    try:
        _parse_draft_schema("abcdef")
    except LLMFormatError:
        return
    raise AssertionError("Ожидался LLMFormatError")


def test_normalize_refusal_reason_null_like():
    assert _normalize_refusal_reason("нет", has_claims=False, refusal_forbidden=False) is None
    assert _normalize_refusal_reason(None, has_claims=False, refusal_forbidden=False) is None


def test_normalize_refusal_reason_dropped_when_claims_present():
    assert _normalize_refusal_reason(
        "нет данных", has_claims=True, refusal_forbidden=False
    ) is None


def test_normalize_refusal_reason_kept():
    assert _normalize_refusal_reason(
        "нет данных", has_claims=False, refusal_forbidden=False
    ) == "нет данных"


def _previous_draft():
    return DraftAnswer(
        answer_text="старый ответ <source>S1</source>",
        claims=[Claim(
            text="исходный факт",
            source_ids=["S1"],
            slot="rent_amount",
            original_idx=0,
        )],
        assumptions=[],
        missing_info=[],
        refusal_reason=None,
    )


def test_regenerate_draft_locks_supported_claims():
    payload = json.dumps({
        "answer_text": "новый ответ <source>S1</source>",
        "claims": [{
            "claim_idx": 0,
            "text": "изменённый факт",
            "source_ids": ["S1"],
            "slot": "rent_amount",
        }],
        "assumptions": [],
        "missing_info": [],
        "refusal_reason": None,
    }, ensure_ascii=False)
    with patch(LLM_PATH, return_value=(payload, 5, 3)):
        draft, step = regenerate_draft(
            previous=_previous_draft(),
            report=make_verification_report(supported_locked=[0]),
            chunks=[make_retrieved_chunk(chunk_id="c1")],
            spec=_spec(intent="semantic_query"),
        )
    assert step.step == "regenerate_draft"
    assert draft.claims[0].text == "исходный факт"
    assert draft.claims[0].source_ids == ["S1"]


def test_regenerate_draft_keeps_previous_on_error():
    with patch(LLM_PATH, side_effect=LLMAPIError("boom")):
        draft, step = regenerate_draft(
            previous=_previous_draft(),
            report=make_verification_report(supported_locked=[0]),
            chunks=[make_retrieved_chunk(chunk_id="c1")],
            spec=_spec(intent="semantic_query"),
        )
    assert step.step == "regenerate_draft"
    assert draft.answer_text == _previous_draft().answer_text
    assert len(draft.claims) == 1
    assert draft.claims[0].text == "исходный факт"
