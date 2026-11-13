from rag_core.generation.prompts.generator import DRAFT_ANSWER_JSON_SCHEMA

REGENERATE_TEMPERATURE: float = 0.2
REGENERATE_MAX_TOKENS: int = 2048

REGENERATE_SYSTEM: str = (
    "Ты — модуль исправления ответа юридического ИИ-ассистента после автоматической верификации.\n"
    "Твоя задача — исправить черновик ответа (`DraftAnswer`), устранив неподтвержденные или противоречащие источникам утверждения.\n\n"
    "ЖЕСТКИЕ ТРЕБОВАНИЯ:\n"
    "1. Для всех утверждений (`claims`) с индексами из списка `supported_locked` выведи их обратно с теми же `claim_idx` и ДОСЛОВНО тем же `text`.\n"
    "2. Для утверждений со статусом `contradicted` — либо удали claim, либо исправь его строго на значение из приведенной точной цитаты источника.\n"
    "3. Для утверждений со статусом `not_in_source` — удали их из `claims` и из `answer_text`, а соответствующий пробел отрази в `missing_info`.\n"
    "4. Все ссылки `<source>S...</source>` в `answer_text` и `source_ids` должны ссылаться только на реально существующие источники из контекста.\n\n"
    "Верни СТРОГО валидный JSON по схеме (в каждом объекте `claims` обязательно сохраняй поле `claim_idx`):\n"
    f"{DRAFT_ANSWER_JSON_SCHEMA}"
)

REGENERATE_USER: str = (
    "Вопрос пользователя: {query}\n\n"
    "Предыдущий DraftAnswer целиком:\n{previous_draft_json}\n\n"
    "Индексы подтвержденных утверждений (supported_locked), которые нужно сохранить с теми же claim_idx и дословно тем же text:\n"
    "{supported_locked}\n\n"
    "Проблемные утверждения (contradicted / not_in_source) с индексами original_idx и точными цитатами из источников:\n"
    "{failed_claims_feedback}\n\n"
    "Ошибки цитирования: {citation_errors}\n\n"
    "Доступные фрагменты источников:\n{context}"
)