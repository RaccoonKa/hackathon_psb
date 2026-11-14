from rag_core.generation.prompts import generator as gen_prompts
from rag_core.generation.prompts import regenerate as regen_prompts
from rag_core.generation.prompts import verifier as ver_prompts


def _system_lower() -> str:
    return gen_prompts.GENERATION_SYSTEM.lower()


def test_system_requires_direct_answer_first_then_caveats():
    system = _system_lower()
    assert "сначала прямой ответ" in system
    assert "затем только существенные оговорки" in system


def test_system_has_unknown_and_contradicted_rules():
    system = gen_prompts.GENERATION_SYSTEM
    assert "UNKNOWN" in system
    assert "CONTRADICTED" in system
    assert "не путай отсутствие подтверждения с опровержением".lower() in system.lower()
    assert "подтверждённое противоречие".lower() in system.lower()


def test_system_no_longer_requires_reporting_wrong_amount():
    system = _system_lower()
    # Старая формулировка предписывала сообщать о «неверной» сумме/сроке/ставке.
    assert "неверную сумму, срок или ставку" not in system
    assert "укажи на расхождение в первом предложении" not in system


def test_system_preserves_anti_injection_and_source_only():
    system = gen_prompts.GENERATION_SYSTEM
    assert "PROMPT INJECTION" in system.upper()
    assert "Игнорируй" in system
    assert "<source>" in system
    assert "ТОЛЬКО ИЗ ТЕКУЩЕГО КОНТЕКСТА" in system
    assert "ОБОСНОВАННЫЙ ОТКАЗ" in system.upper()


def test_user_prompt_has_premise_statuses_placeholder():
    assert "{premise_statuses}" in gen_prompts.GENERATION_USER
    assert "{premise_details}" in gen_prompts.GENERATION_USER
    assert "только CONTRADICTED" in gen_prompts.GENERATION_USER


def test_user_prompt_formats_with_full_argument_set():
    rendered = gen_prompts.GENERATION_USER.format(
        chat_history_block="",
        query="тестовый вопрос",
        event_date="не указана",
        required_slots="general",
        premise_statuses="[]",
        premise_details="[]",
        version_issues="[]",
        refusal_forbidden=False,
        context="контекст",
        retry_feedback="",
    )
    assert "тестовый вопрос" in rendered
    assert "контекст" in rendered


def test_prompts_contain_no_task_example_artifacts():
    # Проверяются только отсутствие характерных маркеров примеров из ТЗ.
    # Сами примеры в промпты и тесты как обучающие данные не добавляются.
    combined = (gen_prompts.GENERATION_SYSTEM + gen_prompts.GENERATION_USER).lower()
    forbidden_markers = (
        "12 000 000",
        "12 млн",
        "10 млн",
        "август 2024",
        "августа 2024",
    )
    for marker in forbidden_markers:
        assert marker not in combined


def test_regenerate_system_matches_generation_structure_and_rules():
    system = regen_prompts.REGENERATE_SYSTEM
    lower = system.lower()
    # Сначала прямой ответ, затем только существенные оговорки/уточнения.
    assert "сначала прямой ответ" in lower
    assert "затем только существенные оговорки" in lower
    # UNVERIFIED нельзя выдавать за факт.
    assert "unverified" in lower
    assert "запрещено выдавать за установленный факт" in lower
    # answer_to_question=mismatch требует переписать ответ на ИСХОДНЫЙ вопрос.
    assert "mismatch" in lower
    assert "исходный вопрос" in lower
    # Не подменять объект вопроса.
    assert "не подменяй объект вопроса" in lower
    # Сохранённые anti-injection / source-only / теги / обоснованный отказ.
    assert "prompt injection" in lower
    assert "только из текущего контекста" in lower
    assert "<source>" in system
    assert "обоснованный отказ" in lower


def test_regenerate_user_has_answer_question_feedback_placeholder():
    assert "{answer_question_feedback}" in regen_prompts.REGENERATE_USER


def test_regenerate_user_formats_with_full_argument_set():
    rendered = regen_prompts.REGENERATE_USER.format(
        query="тестовый вопрос",
        previous_draft_json="{}",
        supported_locked="[]",
        failed_claims_feedback="[]",
        citation_errors="[]",
        answer_question_feedback='{"verdict": "mismatch"}',
        context="контекст",
    )
    assert "тестовый вопрос" in rendered
    assert "mismatch" in rendered
    assert "контекст" in rendered


def test_all_generation_prompts_load_and_format():
    gen_rendered = gen_prompts.GENERATION_USER.format(
        chat_history_block="",
        query="q",
        event_date="не указана",
        required_slots="general",
        premise_statuses="[]",
        premise_details="[]",
        version_issues="[]",
        refusal_forbidden=False,
        context="ctx",
        retry_feedback="",
    )
    regen_rendered = regen_prompts.REGENERATE_USER.format(
        query="q",
        previous_draft_json="{}",
        supported_locked="[]",
        failed_claims_feedback="[]",
        citation_errors="[]",
        answer_question_feedback="{}",
        context="ctx",
    )
    ver_rendered = ver_prompts.VERIFIER_USER.format(
        question="q",
        answer_text="a",
        required_slots="general",
        premises_json="[]",
        claims_json="[]",
        chunks_text="ctx",
    )
    for rendered, system in (
        (gen_rendered, gen_prompts.GENERATION_SYSTEM),
        (regen_rendered, regen_prompts.REGENERATE_SYSTEM),
        (ver_rendered, ver_prompts.VERIFIER_SYSTEM),
    ):
        assert rendered.strip()
        assert system.strip()


def test_regenerate_prompts_contain_no_task_example_artifacts():
    combined = (
        regen_prompts.REGENERATE_SYSTEM + regen_prompts.REGENERATE_USER
    ).lower()
    forbidden_markers = (
        "12 000 000",
        "12 млн",
        "10 млн",
        "август 2024",
        "августа 2024",
    )
    for marker in forbidden_markers:
        assert marker not in combined
