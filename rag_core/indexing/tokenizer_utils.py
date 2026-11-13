try:
    import tiktoken

    try:
        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _encoder = None
except ImportError:  # pragma: no cover - tiktoken не установлен
    _encoder = None


def count_tokens(text: str) -> int:
    """Вычисляет количество токенов в тексте."""
    if not text:
        return 0
    if _encoder:
        return len(_encoder.encode(text, disallowed_special=()))
    # Фоллбэк: в среднем 2.8-3 символа на токен для кириллицы
    return max(1, len(text) // 3)