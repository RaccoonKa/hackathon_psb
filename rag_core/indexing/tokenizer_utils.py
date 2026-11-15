"""Токенизация для структурно-адаптивного чанкинга.

Единственный источник истины по количеству токенов — tiktoken с кодировкой
``cl100k_base``. Любые приблизительные оценки (например, ``len(text) // 3``)
запрещены: они незаметно искажают границы чанков и нарушают HARD_MAX_TOKENS.
Инициализация кодировки выполняется на этапе импорта модуля, чтобы отсутствие
зависимости проявлялось сразу и с понятным сообщением.

Файл BPE (``9b5ad71b2ce5302211f9c61530b329a4922fc6a4``) поставляется вместе с
проектом, поэтому инициализация не требует сети:

* LOCAL-режим — ``<корень проекта>/tiktoken_cache/``;
* SUBMIT-режим — корень артефакта, рядом с плоскими модулями.

Каталог кэша подключается через ``TIKTOKEN_CACHE_DIR`` до вызова
``tiktoken.get_encoding``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

ENCODING_NAME = "cl100k_base"

# Имя файла в кэше tiktoken — sha1 от URL таблицы BPE:
# https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
TIKTOKEN_CACHE_FILENAME = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"

# Ожидаемая sha256 содержимого файла BPE. Ровно это значение tiktoken
# проверяет внутри ``openai_public.cl100k_base``; дублируем его здесь,
# чтобы офлайн-кэш можно было провалидировать до импорта tiktoken.
TIKTOKEN_CACHE_SHA256 = (
    "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
)


def _cache_candidates() -> Tuple[Path, ...]:
    """Каталоги, в которых может лежать заранее скачанный файл BPE.

    Порядок важен: в SUBMIT-режиме файл лежит прямо в корне артефакта, в
    LOCAL-режиме — в отдельном каталоге ``tiktoken_cache`` корня проекта.
    """
    module_dir = Path(__file__).resolve().parent
    return (
        module_dir,
        module_dir / "tiktoken_cache",
        module_dir.parent.parent / "tiktoken_cache",
    )


def resolve_cache_dir() -> Optional[Path]:
    """Возвращает каталог с готовым файлом BPE или ``None``.

    Функция ничего не скачивает и не изменяет окружение — это позволяет
    использовать её в диагностике и проверках сборки.
    """
    for candidate in _cache_candidates():
        try:
            if (candidate / TIKTOKEN_CACHE_FILENAME).is_file():
                return candidate
        except OSError:  # недоступный путь не должен ломать импорт
            continue
    return None


_CACHE_DIR = resolve_cache_dir()
if _CACHE_DIR is not None:
    os.environ['TIKTOKEN_CACHE_DIR'] = str(_CACHE_DIR)

try:
    import tiktoken
except Exception as exc:  # noqa: BLE001 - tiktoken отсутствует или не импортируется
    raise RuntimeError(
        "Не удалось инициализировать tiktoken с кодировкой cl100k_base. "
        "Проверьте установку зависимости tiktoken."
    ) from exc

try:
    _encoder = tiktoken.get_encoding(ENCODING_NAME)
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
