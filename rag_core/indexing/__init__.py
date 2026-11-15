"""Пакет индексации: парсинг, чанкинг и диагностика.

Реэкспорт ``CHUNKER_VERSION`` даёт удобную единую точку импорта без
циклических зависимостей: ``version`` не зависит от ``parser``/``chunker``.
"""

from rag_core.indexing.version import CHUNKER_VERSION

__all__ = ["CHUNKER_VERSION"]
