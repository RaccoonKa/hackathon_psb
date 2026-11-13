from rag_core import config
from rag_core.generation import prompts
from rag_core.generation.answer.context_builder import (
    _format_effective_dates,
    build_context_and_sources,
    build_source_entry,
    format_single_chunk,
)
from tests._common import make_retrieved_chunk


def test_effective_dates_both():
    ch = make_retrieved_chunk(effective_from="2024-01-01", effective_to="2025-12-31")
    assert _format_effective_dates(ch) == "действует с 2024-01-01 по 2025-12-31"


def test_effective_dates_only_from():
    ch = make_retrieved_chunk(effective_from="2024-01-01", effective_to=None)
    assert _format_effective_dates(ch) == "действует с 2024-01-01 по настоящее время"


def test_effective_dates_only_to():
    ch = make_retrieved_chunk(effective_from=None, effective_to="2025-01-01")
    assert _format_effective_dates(ch) == "действует по 2025-01-01"


def test_effective_dates_none():
    ch = make_retrieved_chunk(effective_from=None, effective_to=None)
    assert _format_effective_dates(ch) == ""


def test_effective_dates_unknown():
    ch = make_retrieved_chunk(
        effective_from="2024-01-01", effective_to="2025-01-01",
        effective_dates_known=False,
    )
    assert _format_effective_dates(ch) == ""


def test_format_single_chunk_has_source_id():
    ch = make_retrieved_chunk()
    out = format_single_chunk("S1", ch)
    assert "[S1]" in out


def test_format_single_chunk_has_title():
    ch = make_retrieved_chunk(document_title="Договор поставки")
    out = format_single_chunk("S1", ch)
    assert "Договор поставки" in out


def test_format_single_chunk_has_path():
    ch = make_retrieved_chunk(structure_path=["Раздел 1", "Пункт 1.1"])
    out = format_single_chunk("S1", ch)
    assert "Раздел 1" in out
    assert "Пункт 1.1" in out


def test_format_single_chunk_has_provision():
    ch = make_retrieved_chunk(provision_id="3.2")
    out = format_single_chunk("S1", ch)
    assert f"{prompts.PROVISION_LINE_PREFIX} 3.2" in out


def test_format_single_chunk_no_provision_line():
    ch = make_retrieved_chunk(provision_id=None)
    out = format_single_chunk("S1", ch)
    assert prompts.PROVISION_LINE_PREFIX not in out


def test_format_single_chunk_empty_provision_line():
    ch = make_retrieved_chunk(provision_id="")
    out = format_single_chunk("S1", ch)
    assert prompts.PROVISION_LINE_PREFIX not in out


def test_format_single_chunk_no_effective_line():
    ch = make_retrieved_chunk(effective_dates_known=False)
    out = format_single_chunk("S1", ch)
    assert prompts.EFFECTIVE_LINE_PREFIX not in out


def test_format_single_chunk_no_amends_line():
    ch = make_retrieved_chunk(amends_document_id=None)
    out = format_single_chunk("S1", ch)
    assert prompts.AMENDS_LINE_PREFIX not in out


def test_format_single_chunk_has_amends_line():
    ch = make_retrieved_chunk(amends_document_id="parent_1")
    out = format_single_chunk("S1", ch)
    assert "Изменяет документ: parent_1" in out


def test_format_single_chunk_has_text():
    ch = make_retrieved_chunk(text="Уникальный текст 123")
    out = format_single_chunk("S1", ch)
    assert "Уникальный текст 123" in out


def test_format_single_chunk_line_order():
    ch = make_retrieved_chunk(
        provision_id="3.2",
        structure_path=["Раздел 3", "3.2. Пункт"],
        effective_from="2024-02-12",
        effective_to="2026-12-31",
        amends_document_id="parent_1",
        text="Тело текста",
    )
    out = format_single_chunk("S1", ch)
    pos_title = out.index("[S1]")
    pos_path = out.index("Раздел:")
    pos_prov = out.index(prompts.PROVISION_LINE_PREFIX)
    pos_eff = out.index(prompts.EFFECTIVE_LINE_PREFIX)
    pos_amend = out.index(prompts.AMENDS_LINE_PREFIX)
    pos_text = out.index("Текст:")
    assert pos_title < pos_path < pos_prov < pos_eff < pos_amend < pos_text


def test_build_source_entry_basic():
    ch = make_retrieved_chunk(
        chunk_id="c1", document_id="d1", version_id="v1",
        document_title="Договор", provision_id="3.2",
        structure_path=["3. Порядок"], score=0.9,
    )
    entry = build_source_entry(ch)
    assert entry["chunk_id"] == "c1"
    assert entry["document_id"] == "d1"
    assert entry["version_id"] == "v1"
    assert entry["document_title"] == "Договор"
    assert entry["provision_id"] == "3.2"
    assert entry["structure_path"] == ["3. Порядок"]
    assert entry["score"] == 0.9


def test_build_source_entry_none_provision():
    ch = make_retrieved_chunk(provision_id=None)
    entry = build_source_entry(ch)
    assert entry["provision_id"] is None


def test_build_source_entry_structure_path_copied():
    path = ["Раздел"]
    ch = make_retrieved_chunk(structure_path=path)
    entry = build_source_entry(ch)
    entry["structure_path"].append("Мутация")
    assert path == ["Раздел"]


def test_build_source_entry_amends():
    ch = make_retrieved_chunk(amends_document_id="parent_1")
    entry = build_source_entry(ch)
    assert entry["amends_document_id"] == "parent_1"


def test_build_context_empty():
    sources, ctx, step = build_context_and_sources([])
    assert sources == {}
    assert ctx == ""
    assert step.step == config.STEP_CONTEXT_ASSEMBLY


def test_build_context_single():
    chunks = [make_retrieved_chunk(chunk_id="c1", text="A")]
    sources, ctx, _ = build_context_and_sources(chunks)
    assert list(sources.keys()) == ["S1"]
    assert "A" in ctx


def test_build_context_sequential_ids():
    chunks = [
        make_retrieved_chunk(chunk_id="c1", text="A"),
        make_retrieved_chunk(chunk_id="c2", text="B"),
        make_retrieved_chunk(chunk_id="c3", text="C"),
    ]
    sources, ctx, step = build_context_and_sources(chunks)
    assert list(sources.keys()) == ["S1", "S2", "S3"]
    assert sources["S1"]["chunk_id"] == "c1"
    assert sources["S2"]["chunk_id"] == "c2"
    assert sources["S3"]["chunk_id"] == "c3"


def test_build_context_separated_by_double_newline():
    chunks = [
        make_retrieved_chunk(chunk_id="c1", text="A"),
        make_retrieved_chunk(chunk_id="c2", text="B"),
    ]
    _, ctx, _ = build_context_and_sources(chunks)
    assert config.CONTEXT_CHUNK_SEPARATOR in ctx


def test_build_context_trace_output():
    chunks = [make_retrieved_chunk(chunk_id="c1")]
    _, _, step = build_context_and_sources(chunks)
    assert step.output["source_ids"] == ["S1"]
    assert step.output["context_length"] > 0


def test_build_context_trace_input_chunk_ids():
    chunks = [make_retrieved_chunk(chunk_id="c1"), make_retrieved_chunk(chunk_id="c2")]
    _, _, step = build_context_and_sources(chunks)
    assert step.input["chunk_ids"] == ["c1", "c2"]