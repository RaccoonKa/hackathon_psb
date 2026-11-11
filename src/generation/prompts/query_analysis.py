QUERY_ANALYSIS_TEMPERATURE: float = 0.0
QUERY_ANALYSIS_MAX_TOKENS: int = 768

QUERY_ANALYSIS_JSON_SCHEMA: str = """{
  "intent": "exact_lookup | semantic | mixed | compare | date_version | checklist",
  "task_type": "lookup | explain | compare | checklist | date_check | premise_check",
  "entities_raw": ["строки с названиями документов, номерами договоров, контрагентами или кодексами из запроса"],
  "article_refs": ["только цифры и точки, например: 70, 3.2, 284.1"],
  "dates": ["YYYY-MM-DD"],
  "event_date": "YYYY-MM-DD или null",
  "premises": [
    {"text": "утверждение пользователя в вопросе", "kind": "amount | date | term | rate | fact", "value": "числовое значение или дата"}
  ],
  "required_slots": ["ключевые правовые аспекты: rent_amount, payment_deadline, probation_period, overtime_limit, overtime_pay, approval_process, termination, penalty, liability, tax_rate, effective_date"],
  "assumptions": ["допущения, если вопрос содержит неоднозначность"],
  "ambiguity": "low | medium | high",
  "needs_clarification": false,
  "decomposition": [
    {"text": "подвопрос", "target_doc_raw": "документ/контрагент или null", "aspect": "искомый аспект"}
  ],
  "rewritten_queries_bm25": ["1-2 поисковых запроса с сохранением ТОЧНЫХ номеров договоров, статей и терминов"],
  "rewritten_queries_dense": ["1-2 развернутых семантических поисковых запроса"]
}"""

QUERY_ANALYSIS_SYSTEM: str = (
    "Ты — аналитический модуль юридического RAG-ассистента.\n"
    "Твоя задача — разобрать запрос юриста и вернуть СТРОГО валидный JSON по заданной схеме без markdown-оберток и пояснений.\n"
    "Правила:\n"
    "1. НЕ ВЫДУМЫВАЙ факты, номера договоров, статьи или даты. Если их нет в запросе — оставь списки пустыми или null.\n"
    "2. Все даты переводи в формат ISO YYYY-MM-DD. Если указан только месяц и год (например, март 2025), используй первое число месяца (2025-03-01).\n"
    "3. В article_refs записывай ТОЛЬКО номера из цифр и точек (например, '70', '3.2', '152'). Не пиши слова 'статья' или 'пункт'.\n"
    "4. В rewritten_queries_bm25 обязательно сохраняй точные номера договоров (например, '12-А', 'П-17/2024'), названия контрагентов и номера статей.\n"
    "5. Если в вопросе спрашивается про редакцию на дату или действие допсоглашения во времени — ставь intent='date_version'.\n"
    "6. Если требуется сравнить два договора/нормы — ставь intent='compare', task_type='compare' и заполни decomposition.\n"
    "7. Если вопрос содержит встроенную предпосылку пользователя с конкретной цифрой/суммой/сроком (например: 'Правда ли, что аренда составляет 500 000 руб.?') — добавь её в premises.\n"
    "8. Поле needs_clarification всегда должно быть false.\n\n"
    f"Схема JSON:\n{QUERY_ANALYSIS_JSON_SCHEMA}"
)

QUERY_ANALYSIS_USER: str = (
    "Запрос пользователя: {query}\n"
    "Внешняя дата события (event_date): {event_date}\n"
    "{retry_feedback}"
)