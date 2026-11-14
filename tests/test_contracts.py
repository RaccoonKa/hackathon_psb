# 1. final_answer никогда не пуст
@pytest.mark.parametrize("query", [
    "«Альфа-Трейд» привёз мебель с дефектами. Что нужно приложить к претензии?",
    "Хочу взять оригинал договора из архива на четыре рабочих дня. Это разрешено?",
    "Пользователь: Подписали новый договор, оригинал пока у меня.\nАссистент: Вы про порядок передачи...?\nПользователь: Да. В юротдел нужно отнести сразу?",
])
def test_final_answer_never_empty(query):
    result = run_agent(query)
    assert result.final_answer.strip() != "", \
        f"empty answer for {query!r}, refusal_reason={result.refusal_reason!r}"
    assert isinstance(result.final_answer, str)

# 2. refused=True ⇒ non-empty template answer
def test_refused_implies_template():
    result = run_agent("Какая погода в Москве?")
    if result.refused:
        assert result.final_answer.strip()
        assert result.refusal_reason  # причина обязательна

# 3. verdict=REFUSE ⇒ refused=True (нельзя протащить ответ)
def test_refuse_verdict_blocks_leak():
    # Q11-репродукция: injection + REFUSE verdict
    result = run_agent("Хочу взять оригинал договора из архива на четыре рабочих дня. Это разрешено?")
    assert not (result.refused is False and result.verification and
                result.verification.final_verdict.value == "refuse")

# 4. Multi-turn: chat_history доходит до assistant
def test_history_used_when_provided():
    transcript = "Пользователь: Какой срок?\nАссистент: 5 дней\nПользователь: А для другого договора?"
    result = run_agent(transcript)
    assert result.used_history is True, \
        "run_agent должен сам парсить стенограмму (BUG-06)"

# 5. Injection на документе: чанк не должен быть и «untrusted», и «процитирован»
def test_injection_sanitizes_not_kills():
    result = run_agent("Хочу взять оригинал из архива на 4 дня")
    if result.verification and result.verification.injection_signals:
        # если injection_signals > 0, то либо chunk вырезан, либо ответ не пуст
        assert result.final_answer.strip() != ""

# 6. Premise mismatch: «500 000» должно дать CONTRADICTED, а не UNKNOWN
#    (нужен доступ к spec из result.query_spec)
def test_premise_contradicted():
    result = run_agent("По договору с ООО «Северный Склад» арендная плата 500 000 рублей в месяц?")
    status = result.query_spec.premises[0].status.value if result.query_spec.premises else None
    assert status == "contradicted", f"got {status}, expected contradicted"