"""Токенизация для структурно-адаптивного чанкинга.

Единственный источник истины по количеству токенов — tiktoken с кодировкой
``cl100k_base``. Любые приблизительные оценки (например, ``len(text) // 3``)
запрещены: они незаметно искажают границы чанков и нарушают HARD_MAX_TOKENS.
Инициализация кодировки выполняется на этапе импорта модуля, чтобы отсутствие
зависимости проявлялось сразу и с понятным сообщением.
"""

try:
    import tiktoken
except Exception as exc:  # noqa: BLE001 - tiktoken отсутствует или не импортируется
    raise RuntimeError(
        "Не удалось инициализировать tiktoken с кодировкой cl100k_base. "
        "Проверьте установку зависимости tiktoken."
    ) from exc

try:
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception as exc:  # noqa: BLE001 - кодировка недоступна/повреждена
    raise RuntimeError(
        "Не удалось инициализировать tiktoken с кодировкой cl100k_base. "
        "Проверьте установку зависимости tiktoken."
    ) from exc


def count_tokens(text: str) -> int:
    """Возвращает точное количество токенов ``text`` в кодировке cl100k_base."""
    if not text:
        return 0
    return len(_encoder.encode(text, disallowed_special=()))
