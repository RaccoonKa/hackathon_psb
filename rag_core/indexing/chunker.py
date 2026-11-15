"""Структурно-адаптивный чанкинг юридических документов.

Алгоритм:
1. Блоки извлекаются ``parser.extract_document_structure`` с иерархией.
2. Большие блоки (> HARD_MAX_TOKENS по полному embedding_text) делятся
   структурно: вложенные положения -> перечисления -> абзацы -> предложения,
   и только в крайнем случае — по токенам.
3. Маленькие соседние блоки одного крупного раздела (уровень <= 2) склеиваются
   до TARGET_MAX_TOKENS (жёсткий предел — HARD_MAX_TOKENS).
4. Границы крупных разделов приоритетнее целевого размера: положения из разных
   статей/разделов не смешиваются.

Все измерения размера выполняются по полному ``embedding_text`` (с префиксом
названия документа и родительского пути) через ``count_tokens`` (tiktoken).
"""

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from rag_core.config import PROCESSED_DATA_DIR
from rag_core.indexing.parser import extract_document_structure
from rag_core.indexing.tokenizer_utils import count_tokens
from rag_core.indexing.version import CHUNKER_VERSION

TARGET_MIN_TOKENS = 300
TARGET_MAX_TOKENS = 600
HARD_MAX_TOKENS = 750
OVERLAP_TOKENS = 60

# Константы сохранены для обратной совместимости с тестами перекрытия
SUB_CHUNK_SIZE_CHARS = HARD_MAX_TOKENS * 3
SUB_CHUNK_OVERLAP_CHARS = OVERLAP_TOKENS * 3

PRELUDE_SECTION_KEY = "__document_prelude__"
PRELUDE_PATH = ["Преамбула"]

_oversized_splitter = None

# --- Шаблоны смысловых границ для структурного разделения больших блоков ---
_NESTED_ITEM_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s+\S")
_PAREN_ITEM_RE = re.compile(r"^\s*\d+\)\s+\S")
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*[.)]|[а-яёa-z][)]|[-—•])\s+\S")
# Граница предложения только после знака, не являющегося частью номера пункта
# ("1.1.", "ст. 13.11") или инициала: цифра перед точкой исключается.
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?;])(?<![0-9][.!?;])\s+(?=[А-ЯЁA-Z«\"(])"
)
_BLANK_LINE_RE = re.compile(r"\n[ \t\xa0\u2000-\u200a]*\n")

_SPLIT_STAGE_NESTED = 0
_SPLIT_STAGE_LIST = 1
_SPLIT_STAGE_PARAGRAPH = 2
_SPLIT_STAGE_SENTENCE = 3
_SPLIT_STAGE_TOKENS = 4
_OVERLAP_STAGES = {_SPLIT_STAGE_PARAGRAPH, _SPLIT_STAGE_SENTENCE}


def _get_oversized_splitter():
    """Ленивая инициализация langchain-сплиттера по токенам (не нужна при импорте)."""
    global _oversized_splitter
    if _oversized_splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        _oversized_splitter = RecursiveCharacterTextSplitter(
            chunk_size=HARD_MAX_TOKENS - 15,
            chunk_overlap=OVERLAP_TOKENS,
            length_function=count_tokens,
            separators=["\n\n", "\n", ";\n", r"(?<!\d)\.\s+", " "],
            is_separator_regex=True,
            keep_separator=True,
        )
    return _oversized_splitter


def generate_chunk_id(
    doc_id: str,
    ver_id: str,
    source_hash: Optional[str],
    chunk_index: int,
    chunker_ver: str = CHUNKER_VERSION,
) -> str:
    """Генерирует стабильный chunk_id по разделу 3.4 rag.md."""
    s_hash = source_hash or "nohash"
    raw = f"{doc_id}_{ver_id}_{s_hash}_{chunker_ver}_{chunk_index:04d}"
    hash_suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return f"{ver_id}_{chunk_index:04d}_{hash_suffix}"


def _resolve_page_number(
    content: str,
    c_start: int,
    c_end: int,
    page_spans: Optional[List[Tuple[int, int, int]]],
) -> Optional[int]:
    """Определяет номер страницы (1-based) для PDF по символьному смещению; None для TXT/DOCX."""
    if not page_spans:
        return None

    # Смещаем точку поиска к первому непробельному символу фрагмента,
    # чтобы ведущие переносы строк не относили чанк к предыдущей странице.
    target_pos = c_start
    limit = min(len(content), max(c_start, c_end))
    while target_pos < limit and content[target_pos].isspace():
        target_pos += 1

    for p_start, p_end, page_num in page_spans:
        if p_start <= target_pos < p_end:
            return page_num

    if target_pos < page_spans[0][0]:
        return page_spans[0][2]
    return page_spans[-1][2]


def _is_bare_heading_block(b: Dict[str, Any]) -> bool:
    """Определяет, является ли блок лишь структурным заголовком без содержательного текста нормы.

    Заголовки разделов договоров ('1. Предмет договора') или статей законов ('Статья 112...'),
    за которыми сразу следуют подпункты, должны сохраняться в path дочерних элементов, но не
    порождать пустые чанки-пустышки из 3 слов (п. 2 CHUNKING_TASKS.md).
    """
    path = b.get("path") or []
    if not path:
        return False

    clean_text = b.get("text", "").strip()
    if not clean_text:
        return True

    if len(clean_text) > 300 or len(clean_text.split()) > 40:
        return False

    last_heading = path[-1].strip()
    norm_text = re.sub(r"[\s\.\:\;\-]+", " ", clean_text).strip().lower()
    norm_h = re.sub(r"[\s\.\:\;\-]+", " ", last_heading).strip().lower()

    if norm_text == norm_h:
        return True

    if clean_text.rstrip(".:") == last_heading.rstrip(".:"):
        return True

    return False


# --- Префикс embedding_text и структурные ключи ---


def _build_embedding_prefix(meta: Dict[str, Any], path: List[str]) -> str:
    """Формирует префикс embedding_text: название, маркер изменения и родительский путь."""
    prefix_parts = [meta["document_title"]]
    if meta.get("amends_document_id"):
        prefix_parts.append(f"[ИЗМЕНЕНИЕ ДЛЯ {str(meta['amends_document_id']).upper()}]")
    prefix_parts.extend(path or [])
    return ". ".join(prefix_parts) + ". "


def _fragment_tokens(prefix: str, text: str) -> int:
    """Точный размер фрагмента по полному embedding_text (с префиксом)."""
    return count_tokens(prefix + text)


def _block_section_key(block: Dict[str, Any]) -> str:
    """Ключ крупного структурного раздела: общий родитель до уровня <= 2.

    Для блоков без такого уровня (преамбула/неразмеченный хвост) используется
    единый ключ, чтобы их можно было склеивать между собой.
    """
    section_id = block.get("section_provision_id")
    if section_id:
        return f"prov:{section_id}"
    return PRELUDE_SECTION_KEY


def _block_section_path(block: Dict[str, Any]) -> List[str]:
    """Путь до уровня <= 2 (Раздел/Глава/Статья/крупный раздел договора)."""
    levels = block.get("path_levels") or []
    path = block.get("path") or []
    if levels and len(levels) == len(path):
        result = [title for level, title in zip(levels, path) if level <= 2]
        if result:
            return result
    if len(path) > 1:
        return list(path[:-1])
    return list(path) if path else list(PRELUDE_PATH)


def _common_path(blocks: List[Dict[str, Any]]) -> List[str]:
    """Общий родительский путь группы блоков (LCP), без конкретных пунктов."""
    paths = [list(b.get("path") or []) for b in blocks]
    if not paths:
        return list(PRELUDE_PATH)

    prefix = list(paths[0])
    for other in paths[1:]:
        i = 0
        while i < len(prefix) and i < len(other) and prefix[i] == other[i]:
            i += 1
        prefix = prefix[:i]

    if prefix:
        return prefix

    # Группа пересекает границу (например, маленькая преамбула + первый раздел):
    # используем путь крупного раздела, к которому относится содержательная часть.
    for block in blocks:
        if _block_section_key(block) != PRELUDE_SECTION_KEY:
            return _block_section_path(block)
    return list(PRELUDE_PATH)


def _slice_group_text(content: str, blocks: List[Dict[str, Any]]) -> str:
    """Точный текст группы из исходного content (полные тексты положений, исходный порядок)."""
    if not blocks:
        return ""
    start = int(blocks[0]["char_start"])
    end = int(blocks[-1]["char_end"])
    if 0 <= start < end <= len(content):
        text = content[start:end].strip()
        if text:
            return text
    return "\n\n".join(b["text"] for b in blocks).strip()


def _group_tokens(content: str, blocks: List[Dict[str, Any]], meta: Dict[str, Any]) -> int:
    path = _common_path(blocks)
    return _fragment_tokens(_build_embedding_prefix(meta, path), _slice_group_text(content, blocks))


def _group_is_prelude(blocks: List[Dict[str, Any]]) -> bool:
    return all(_block_section_key(b) == PRELUDE_SECTION_KEY for b in blocks)


def _actual_block_start(content: str, block: Dict[str, Any]) -> int:
    """Реальное начало текста блока в content (текст блока хранится без внешних пробелов)."""
    text = block.get("text", "")
    if text:
        found = content.find(text, int(block["char_start"]), int(block["char_end"]) + 1)
        if found != -1:
            return found
    return int(block["char_start"])


# --- Структурное разделение больших блоков ---


def _make_fragment(text: str, abs_start: int, block: Dict[str, Any], content: str) -> Dict[str, Any]:
    """Создаёт фрагмент с точными координатами в исходном content."""
    stripped = text.strip()
    if stripped:
        lead = len(text) - len(text.lstrip())
        abs_start = max(0, abs_start + lead)
        abs_end = min(len(content), abs_start + len(stripped))
    else:
        abs_end = abs_start

    if content and 0 <= abs_start <= len(content):
        start_line = content.count("\n", 0, abs_start) + 1
        end_line = content.count("\n", 0, max(abs_start, min(abs_end, len(content)) - 1)) + 1
    else:
        start_line = int(block.get("start_line", 1))
        end_line = int(block.get("end_line", start_line))

    return {
        "text": stripped,
        "abs_start": abs_start,
        "abs_end": abs_end,
        "start_line": start_line,
        "end_line": end_line,
    }


def _segments_by_line_pattern(text: str, pattern: re.Pattern) -> List[Tuple[str, int]]:
    """Делит блок по строкам-границам, оставляя строку-заголовок в первом сегменте."""
    newline_pos = text.find("\n")
    if newline_pos == -1:
        return []

    rest = text[newline_pos + 1:]
    boundaries = [0]
    offset = 0
    for line in rest.splitlines(keepends=True):
        if line.strip() and pattern.match(line):
            boundaries.append(newline_pos + 1 + offset)
        offset += len(line)

    if len(boundaries) <= 1:
        return []

    boundaries.append(len(text))
    segments: List[Tuple[str, int]] = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if idx == 0:
            # Первый сегмент всегда включает строку-заголовок исходного положения.
            start = 0
        segment = text[start:end]
        if segment.strip():
            segments.append((segment, start))
    return segments


def _segments_by_blank_lines(text: str) -> List[Tuple[str, int]]:
    """Делит блок по границам абзацев (пустым строкам)."""
    matches = list(_BLANK_LINE_RE.finditer(text))
    if not matches:
        return []

    segments: List[Tuple[str, int]] = []
    prev = 0

    def _append(segment: str, start: int) -> None:
        # Whitespace-only сегменты (например, строка из NBSP) приклеиваем к
        # предыдущему, чтобы текст оставался непрерывным и совпадал со смещениями.
        if not segment.strip():
            if segments:
                last_text, last_start = segments[-1]
                segments[-1] = (last_text + segment, last_start)
            return
        segments.append((segment, start))

    for match in matches:
        end = match.end()
        _append(text[prev:end], prev)
        prev = end
    if prev < len(text):
        _append(text[prev:], prev)

    if len(segments) <= 1:
        return []
    return segments


def _segments_by_sentences(text: str) -> List[Tuple[str, int]]:
    """Делит блок по границам предложений."""
    boundaries = [0]
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        boundaries.append(match.end())
    boundaries.append(len(text))

    segments: List[Tuple[str, int]] = []
    for idx in range(len(boundaries) - 1):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        if end <= start:
            continue
        segment = text[start:end]
        if segment.strip():
            segments.append((segment, start))

    if len(segments) <= 1:
        return []
    return segments


def _segments_for_stage(text: str, stage: int) -> List[Tuple[str, int]]:
    if stage == _SPLIT_STAGE_NESTED:
        return _segments_by_line_pattern(text, _NESTED_ITEM_RE) or _segments_by_line_pattern(
            text, _PAREN_ITEM_RE
        )
    if stage == _SPLIT_STAGE_LIST:
        return _segments_by_line_pattern(text, _LIST_ITEM_RE)
    if stage == _SPLIT_STAGE_PARAGRAPH:
        return _segments_by_blank_lines(text)
    if stage == _SPLIT_STAGE_SENTENCE:
        return _segments_by_sentences(text)
    return []


def _overlap_suffix(text: str, max_tokens: int) -> Tuple[str, int]:
    """Возвращает суффикс текста для перекрытия, не разрывая предложение без нужды."""
    text = text.rstrip()
    if not text:
        return "", 0

    sentence_starts = [0]
    for match in re.finditer(r"(?<=[.!?;])\s+", text):
        sentence_starts.append(match.end())

    for pos in reversed(sentence_starts):
        suffix = text[pos:]
        if count_tokens(suffix) <= max_tokens:
            return suffix, pos

    # Последнее предложение длиннее лимита: аккуратно отступаем по словам.
    # Смещение берётся из реальных span'ов слов, чтобы суффикс был точным
    # срезом исходного текста (join(" ") не совпадает с текстом при NBSP,
    # переносах строк и т.п. и раньше давал offset=0 — полное дублирование
    # предыдущего фрагмента).
    word_spans = [match.span() for match in re.finditer(r"\S+", text)]
    if len(word_spans) < 2:
        return "", 0

    start_pos = word_spans[-1][0]
    for span_start, _span_end in reversed(word_spans[:-1]):
        if count_tokens(text[span_start:]) > max_tokens:
            break
        start_pos = span_start

    if start_pos <= 0:
        return "", 0
    return text[start_pos:], start_pos


def _apply_overlap(
    fragments: List[Dict[str, Any]],
    prefix: str,
    content: str,
) -> List[Dict[str, Any]]:
    """Добавляет перекрытие OVERLAP_TOKENS между фрагментами одного положения."""
    for idx in range(1, len(fragments)):
        prev = fragments[idx - 1]
        current = fragments[idx]
        if current["abs_start"] < prev["abs_end"]:
            # Перекрытие уже обеспечено (например, сплиттером по токенам).
            continue

        suffix, offset = _overlap_suffix(prev["text"], OVERLAP_TOKENS)
        if not suffix or offset <= 0:
            continue

        # Начало суффикса задаётся его смещением внутри текста предыдущего
        # фрагмента, а не len(suffix): между фрагментами отбрасывается
        # разделитель ("\n\n", пробелы), и арифметика по длине сдвигала
        # границу на 1-2 символа, разрывая слово пополам.
        new_start = prev["abs_start"] + offset
        if not prev["abs_start"] < new_start < current["abs_start"]:
            # Перекрытие не даёт нового текста (суффикс совпал с фрагментом
            # целиком) — пропускаем, чтобы не дублировать целый фрагмент.
            continue
        slice_text = content[new_start:current["abs_end"]]
        candidate = slice_text.strip() if slice_text.strip() else suffix + current["text"]
        if _fragment_tokens(prefix, candidate) > HARD_MAX_TOKENS:
            # Не выходим за HARD_MAX ради точного размера перекрытия.
            continue

        current["text"] = candidate
        current["abs_start"] = new_start
        if content and 0 <= new_start <= len(content):
            current["start_line"] = content.count("\n", 0, new_start) + 1
    return fragments


def _resplit_part(part: str, max_text_tokens: int) -> List[str]:
    """Делит слишком крупную часть токенного сплиттера, сохраняя перекрытие."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Не делим по знакам препинания: иначе номер пункта ("4.") отрывается
        # от собственного текста и порождает микро-чанк.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_text_tokens,
            chunk_overlap=OVERLAP_TOKENS,
            length_function=count_tokens,
            separators=["\n\n", "\n", " "],
            keep_separator=True,
        )
        result = splitter.split_text(part)
        if result:
            return result
    except Exception:  # noqa: BLE001 - резервный путь не должен ломать чанкинг
        pass
    return [part]


def _split_by_tokens(
    text: str,
    abs_start: int,
    block: Dict[str, Any],
    content: str,
    prefix: str,
) -> List[Dict[str, Any]]:
    """Крайний случай: разделение по токенам с перекрытием OVERLAP_TOKENS.

    Сохраняется fallback-арифметика координат для случая, когда ``split_text``
    возвращает части, которых нет в исходном content (моки/перекодирование).
    """
    raw_parts = _get_oversized_splitter().split_text(text)

    max_text_tokens = HARD_MAX_TOKENS - count_tokens(prefix) - 10
    if max_text_tokens < 80:
        max_text_tokens = 80

    parts: List[str] = []
    for part in raw_parts:
        if count_tokens(part) > max_text_tokens:
            parts.extend(_resplit_part(part, max_text_tokens))
        else:
            parts.append(part)
    if not parts:
        parts = [text]

    fragments: List[Dict[str, Any]] = []
    running_char = abs_start
    search_cursor = abs_start
    block_end = min(len(content), abs_start + len(text))

    for part in parts:
        clean_part = part.strip()
        found_idx = (
            content.find(clean_part, search_cursor, block_end + 50)
            if len(clean_part) >= 15
            else -1
        )

        if found_idx != -1:
            c_start = found_idx
            c_end = found_idx + len(clean_part)
            search_cursor = max(c_start + 1, c_end - SUB_CHUNK_OVERLAP_CHARS * 3)
            start_line = content.count("\n", 0, c_start) + 1
            end_line = content.count("\n", 0, max(c_start, c_end - 1)) + 1
        else:
            c_start = running_char
            c_end = running_char + len(part)
            start_line = int(block.get("start_line", 1))
            end_line = int(block.get("end_line", start_line))

        fragments.append(
            {
                "text": clean_part,
                "abs_start": c_start,
                "abs_end": c_end,
                "start_line": start_line,
                "end_line": end_line,
            }
        )
        running_char = c_start + max(1, len(part) - SUB_CHUNK_OVERLAP_CHARS)

    return fragments


def _split_text_structurally(
    text: str,
    abs_start: int,
    stage: int,
    prefix: str,
    block: Dict[str, Any],
    content: str,
) -> List[Dict[str, Any]]:
    """Рекурсивно делит текст по приоритету смысловых границ."""
    if _fragment_tokens(prefix, text) <= HARD_MAX_TOKENS:
        return [_make_fragment(text, abs_start, block, content)]

    if stage >= _SPLIT_STAGE_TOKENS:
        return _split_by_tokens(text, abs_start, block, content, prefix)

    segments = _segments_for_stage(text, stage)
    if not segments:
        return _split_text_structurally(text, abs_start, stage + 1, prefix, block, content)

    use_overlap = stage in _OVERLAP_STAGES
    fragments: List[Dict[str, Any]] = []
    pack_text = ""
    pack_start = abs_start
    pack_tokens = 0

    for segment_text, segment_rel in segments:
        segment_abs = abs_start + segment_rel
        segment_tokens = _fragment_tokens(prefix, segment_text)

        if segment_tokens > HARD_MAX_TOKENS:
            if pack_text:
                fragments.append(_make_fragment(pack_text, pack_start, block, content))
                pack_text = ""
                pack_tokens = 0
            fragments.extend(
                _split_text_structurally(
                    segment_text, segment_abs, stage + 1, prefix, block, content
                )
            )
            continue

        if not pack_text:
            pack_text = segment_text
            pack_start = segment_abs
            pack_tokens = segment_tokens
            continue

        combined_text = pack_text + segment_text
        combined_tokens = _fragment_tokens(prefix, combined_text)
        fits_target = combined_tokens <= TARGET_MAX_TOKENS
        fits_hard_with_growth = (
            pack_tokens < TARGET_MIN_TOKENS and combined_tokens <= HARD_MAX_TOKENS
        )
        if fits_target or fits_hard_with_growth:
            pack_text = combined_text
            pack_tokens = combined_tokens
        else:
            fragments.append(_make_fragment(pack_text, pack_start, block, content))
            pack_text = segment_text
            pack_start = segment_abs
            pack_tokens = segment_tokens

    if pack_text:
        fragments.append(_make_fragment(pack_text, pack_start, block, content))

    if use_overlap:
        fragments = _apply_overlap(fragments, prefix, content)
    return fragments


# --- Сборка чанков ---


def _make_spec_from_blocks(
    content: str,
    blocks: List[Dict[str, Any]],
    meta: Dict[str, Any],
    section_key: str,
) -> Dict[str, Any]:
    """Формирует описание чанка из группы блоков одного крупного раздела."""
    path = _common_path(blocks)
    if len(blocks) == 1:
        path = list(blocks[0].get("path") or path)
        provision_id: Optional[str] = blocks[0].get("provision_id")
    else:
        provision_id = None

    text = _slice_group_text(content, blocks)
    char_start = int(blocks[0]["char_start"])
    char_end = int(blocks[-1]["char_end"])
    if content and 0 <= char_start < len(content) and char_start < char_end <= len(content):
        start_line = content.count("\n", 0, char_start) + 1
        end_line = content.count("\n", 0, max(char_start, char_end - 1)) + 1
    else:
        start_line = int(blocks[0].get("start_line", 1))
        end_line = int(blocks[-1].get("end_line", start_line))

    return {
        "text": text,
        "path": path,
        "provision_id": provision_id,
        "blocks": list(blocks),
        "section_key": section_key,
        "is_fragment": False,
        "char_start": char_start,
        "char_end": char_end,
        "start_line": start_line,
        "end_line": end_line,
    }


def _build_chunk_specs(
    content: str,
    meta: Dict[str, Any],
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Строит чанки: структурное разделение oversize и склейка маленьких блоков."""
    filtered = [
        block
        for block in blocks
        if not (len(blocks) > 1 and _is_bare_heading_block(block))
    ]

    specs: List[Dict[str, Any]] = []
    index = 0
    total = len(filtered)

    while index < total:
        block = filtered[index]
        block_text = block["text"].strip()
        block_tokens = _fragment_tokens(_build_embedding_prefix(meta, block["path"]), block_text)

        # 1. Большое положение: структурное разделение по смыслу, токены — крайний случай.
        if block_tokens > HARD_MAX_TOKENS:
            actual_start = _actual_block_start(content, block)
            prefix = _build_embedding_prefix(meta, block["path"])
            fragments = _split_text_structurally(
                block_text, actual_start, _SPLIT_STAGE_NESTED, prefix, block, content
            )
            for fragment in fragments:
                specs.append(
                    {
                        "text": fragment["text"],
                        "path": list(block["path"]),
                        "provision_id": block.get("provision_id"),
                        "blocks": [block],
                        "section_key": _block_section_key(block),
                        "is_fragment": True,
                        "char_start": fragment["abs_start"],
                        "char_end": fragment["abs_end"],
                        "start_line": fragment["start_line"],
                        "end_line": fragment["end_line"],
                    }
                )
            index += 1
            continue

        # 2. Склейка маленьких соседних блоков одного крупного раздела.
        group: List[Dict[str, Any]] = [block]
        group_key = _block_section_key(block)
        next_index = index + 1

        while next_index < total:
            candidate = filtered[next_index]
            candidate_key = _block_section_key(candidate)
            candidate_path = _common_path(group + [candidate])
            candidate_prefix = _build_embedding_prefix(meta, candidate_path)
            candidate_text = candidate["text"].strip()
            candidate_tokens = _fragment_tokens(candidate_prefix, candidate_text)

            # Большой блок обрабатывается отдельно и не смешивается с соседями.
            if candidate_tokens > HARD_MAX_TOKENS:
                break

            combined_text = _slice_group_text(content, group + [candidate])
            combined_tokens = _fragment_tokens(candidate_prefix, combined_text)

            same_section = candidate_key == group_key
            if not same_section:
                # Маленькая преамбула может быть присоединена к первому крупному разделу.
                can_lead_prelude = (
                    group_key == PRELUDE_SECTION_KEY
                    and _group_is_prelude(group)
                    and _group_tokens(content, group, meta) < TARGET_MIN_TOKENS
                    and combined_tokens <= TARGET_MAX_TOKENS
                )
                if not can_lead_prelude:
                    break
                group.append(candidate)
                group_key = candidate_key
                next_index += 1
                continue

            current_tokens = _group_tokens(content, group, meta)
            can_grow = (
                combined_tokens <= TARGET_MAX_TOKENS
                or (current_tokens < TARGET_MIN_TOKENS and combined_tokens <= HARD_MAX_TOKENS)
            )
            if not can_grow:
                break

            group.append(candidate)
            next_index += 1

        specs.append(_make_spec_from_blocks(content, group, meta, group_key))
        index = next_index

    return _balance_specs(content, meta, specs)


def _balance_specs(
    content: str,
    meta: Dict[str, Any],
    specs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Повторная балансировка: склеивает оставшиеся маленькие совместимые чанки.

    Не пересекает границы крупных разделов и не объединяет фрагменты большого
    положения (они сохраняют исходный provision_id и path).
    """
    if not specs:
        return specs

    result: List[Dict[str, Any]] = []
    for spec in specs:
        if (
            result
            and not spec["is_fragment"]
            and not result[-1]["is_fragment"]
        ):
            previous = result[-1]
            same_section = previous["section_key"] == spec["section_key"]
            prelude_lead = (
                previous["section_key"] == PRELUDE_SECTION_KEY
                and _group_is_prelude(previous["blocks"])
            )
            if same_section or prelude_lead:
                combined_blocks = previous["blocks"] + spec["blocks"]
                path = _common_path(combined_blocks)
                prefix = _build_embedding_prefix(meta, path)
                combined_text = _slice_group_text(content, combined_blocks)
                combined_tokens = _fragment_tokens(prefix, combined_text)

                previous_tokens = _fragment_tokens(
                    _build_embedding_prefix(meta, previous["path"]), previous["text"]
                )
                spec_tokens = _fragment_tokens(
                    _build_embedding_prefix(meta, spec["path"]), spec["text"]
                )
                if (
                    combined_tokens <= HARD_MAX_TOKENS
                    and min(previous_tokens, spec_tokens) < TARGET_MIN_TOKENS
                ):
                    result[-1] = _make_spec_from_blocks(
                        content, combined_blocks, meta, spec["section_key"]
                    )
                    continue

        result.append(spec)

    return result


def chunk_document_structural(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = doc["content"]
    meta = doc["metadata"]
    page_spans: Optional[List[Tuple[int, int, int]]] = doc.get("page_spans")
    total_doc_tokens = count_tokens(content)

    chunks: List[Dict[str, Any]] = []
    chunk_idx = 0

    def make_chunk_payload(
        text: str,
        path: List[str],
        provision_id: Optional[str],
        start_line: int,
        end_line: int,
        c_start: int,
        c_end: int,
    ) -> Dict[str, Any]:
        nonlocal chunk_idx

        prefix = _build_embedding_prefix(meta, path)
        clean_text = text.strip()
        page_num = _resolve_page_number(content, c_start, c_end, page_spans)

        if content and 0 <= c_start <= len(content) and c_start <= c_end <= len(content):
            start_line = content.count("\n", 0, c_start) + 1
            end_line = content.count("\n", 0, max(c_start, c_end - 1)) + 1

        cid = generate_chunk_id(
            doc_id=meta["document_id"],
            ver_id=meta["version_id"],
            source_hash=meta.get("source_hash"),
            chunk_index=chunk_idx,
            chunker_ver=meta.get("chunker_version", CHUNKER_VERSION),
        )

        payload = {
            "chunk_id": cid,
            "document_id": meta["document_id"],
            "version_id": meta["version_id"],
            "chunk_index": chunk_idx,
            "text": clean_text,
            "embedding_text": prefix + clean_text,
            "structure": {
                "path": list(path),
                "provision_id": provision_id,
            },
            "location": {
                "start_line": start_line,
                "end_line": end_line,
                "char_start": c_start,
                "char_end": c_end,
                "page": page_num,
            },
            "document_title": meta["document_title"],
            "document_type": meta["document_type"],
            "scope": meta.get("scope", "global"),
            "chat_id": meta.get("chat_id"),
            "effective_from": meta.get("effective_from"),
            "effective_to": meta.get("effective_to"),
            "effective_dates_known": meta.get("effective_dates_known", True),
            "source_hash": meta.get("source_hash"),
            "amends_document_id": meta.get("amends_document_id"),
            "chunker_version": meta.get("chunker_version", CHUNKER_VERSION),
        }
        chunk_idx += 1
        return payload

    # Защита коротких допников и служебных писем: сохраняем целиком, если даже
    # полный embedding_text (с префиксом) укладывается в HARD_MAX_TOKENS.
    full_prefix = _build_embedding_prefix(meta, ["Полный текст документа"])
    bypass_fits = _fragment_tokens(full_prefix, content.strip()) <= HARD_MAX_TOKENS
    if total_doc_tokens <= HARD_MAX_TOKENS and bypass_fits and (
        meta.get("amends_document_id") or meta.get("document_type") == "internal"
    ):
        chunks.append(
            make_chunk_payload(
                text=content,
                path=["Полный текст документа"],
                provision_id=None,
                start_line=1,
                end_line=len(content.splitlines()),
                c_start=0,
                c_end=len(content),
            )
        )
        return chunks

    blocks = extract_document_structure(content)
    if not blocks:
        return chunks

    specs = _build_chunk_specs(content, meta, blocks)
    for spec in specs:
        if not spec["text"]:
            continue
        chunks.append(
            make_chunk_payload(
                text=spec["text"],
                path=spec["path"],
                provision_id=spec["provision_id"],
                start_line=spec["start_line"],
                end_line=spec["end_line"],
                c_start=spec["char_start"],
                c_end=spec["char_end"],
            )
        )

    return chunks


def save_canonical_chunks(all_chunks: List[Dict[str, Any]]) -> Path:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PROCESSED_DATA_DIR / "chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
    return out_file
