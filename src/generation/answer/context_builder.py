import json
import logging
import re
import time
from typing import Any
from src import config
from src.generation.prompts import templates as prompts
from src.generation.types import RetrievedChunk, SourceMetadata, TraceStep

logger = logging.getLogger(__name__)

_CHUNK_LOCATIONS_CACHE: dict[str, dict[str, Any]] = {}
_CHUNK_LOCATIONS_MTIME: float | None = None


def _get_chunk_location(chunk_id: str) -> dict[str, Any]:
    global _CHUNK_LOCATIONS_CACHE, _CHUNK_LOCATIONS_MTIME
    if not chunk_id:
        return {}

    chunks_path = config.PROCESSED_DATA_DIR / "chunks.jsonl"
    if not chunks_path.is_file():
        return {}

    try:
        current_mtime = chunks_path.stat().st_mtime
        if _CHUNK_LOCATIONS_MTIME != current_mtime:
            loaded: dict[str, dict[str, Any]] = {}
            with open(chunks_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    record = json.loads(raw)
                    cid = str(record.get("chunk_id") or "")
                    loc = record.get("location")
                    if cid and isinstance(loc, dict):
                        loaded[cid] = dict(loc)
            _CHUNK_LOCATIONS_CACHE = loaded
            _CHUNK_LOCATIONS_MTIME = current_mtime
    except Exception as exc:  # noqa: BLE001 - не ломаем сборку контекста из-за кэша координат
        logger.debug("Не удалось загрузить location из chunks.jsonl: %s", exc)
        return {}

    return dict(_CHUNK_LOCATIONS_CACHE.get(chunk_id) or {})


def _format_effective_dates(chunk: RetrievedChunk) -> str:
    if not chunk.effective_dates_known:
        return ""
    if chunk.effective_from and chunk.effective_to:
        return f"действует с {chunk.effective_from} по {chunk.effective_to}"
    if chunk.effective_from:
        return f"действует с {chunk.effective_from} по настоящее время"
    if chunk.effective_to:
        return f"действует по {chunk.effective_to}"
    return ""


def format_single_chunk(source_id: str, chunk: RetrievedChunk) -> str:
    has_provision: bool = (
        chunk.provision_id is not None and bool(chunk.provision_id.strip())
    )
    effective_str = _format_effective_dates(chunk)
    has_effective: bool = bool(effective_str)
    has_amends: bool = (
        chunk.amends_document_id is not None
        and bool(chunk.amends_document_id.strip())
    )

    structure_path_str = config.STRUCTURE_PATH_SEPARATOR.join(
        chunk.structure_path
    )
    rendered = prompts.CONTEXT_ITEM_TEMPLATE.format(
        source_id=source_id,
        document_title=chunk.document_title,
        structure_path_str=structure_path_str,
        provision_id=chunk.provision_id or "",
        effective_str=effective_str,
        amends_document_id=chunk.amends_document_id or "",
        text=chunk.text,
    )

    filtered_lines: list[str] = []
    for line in rendered.splitlines():
        if not has_provision and line.startswith(prompts.PROVISION_LINE_PREFIX):
            continue
        if not has_effective and line.startswith(prompts.EFFECTIVE_LINE_PREFIX):
            continue
        if not has_amends and line.startswith(prompts.AMENDS_LINE_PREFIX):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def build_source_entry(chunk: RetrievedChunk) -> SourceMetadata:
    return {
        "document_id": chunk.document_id,
        "version_id": chunk.version_id,
        "chunk_id": chunk.chunk_id,
        "document_title": chunk.document_title,
        "provision_id": chunk.provision_id,
        "structure_path": list(chunk.structure_path),
        "text": chunk.text,
        "location": _get_chunk_location(chunk.chunk_id),
        "score": chunk.score,
        "effective_from": chunk.effective_from,
        "effective_to": chunk.effective_to,
        "effective_dates_known": chunk.effective_dates_known,
        "amends_document_id": chunk.amends_document_id,
    }


def build_context_and_sources(
    chunks: list[RetrievedChunk],
) -> tuple[dict[str, dict[str, Any]], str, TraceStep]:
    start_time = time.perf_counter()
    sources: dict[str, dict[str, Any]] = {}
    formatted_items: list[str] = []

    for idx, chunk in enumerate(chunks, start=config.SOURCE_START_INDEX):
        source_id = f"{config.SOURCE_ID_PREFIX}{idx}"
        sources[source_id] = dict(build_source_entry(chunk))
        formatted_items.append(format_single_chunk(source_id, chunk))

    context_str = config.CONTEXT_CHUNK_SEPARATOR.join(formatted_items)
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step=config.STEP_CONTEXT_ASSEMBLY,
        input={"chunk_ids": [chunk.chunk_id for chunk in chunks]},
        output={
            "source_ids": list(sources.keys()),
            "context_length": len(context_str),
        },
        latency_ms=latency_ms,
    )
    logger.info(
        "Step %s assembled %d sources in %.3f ms",
        config.STEP_CONTEXT_ASSEMBLY,
        len(sources),
        latency_ms,
    )
    logger.debug("Assembled context string:\n%s", context_str)
    return sources, context_str, step

_SOURCE_TAG_RE = re.compile(r"<source>(.*?)</source>", re.DOTALL)

_SKIP_STRUCTURE_ITEMS = {
    "введение / общие положения",
    "введение",
}


def _parse_source_ids(raw_ids: str) -> list[str]:
    return [sid.strip() for sid in (raw_ids or "").split(",") if sid.strip()]


def _is_covered_by_title(structure_item: str, title_lower: str) -> bool:
    """True, если элемент structure_path уже «растворён» в document_title.

    Работает не по подстроке, а по значимым токенам: заголовок
    «…Статья 70. ТК РФ Испытание при приеме на работу…» перекрывает
    структуру «Статья 70. Испытание при приеме на работу».
    """
    text = structure_item.lower().strip()
    if not text:
        return True
    if text in title_lower:
        return True
    tokens = [t for t in re.findall(r"[a-zа-яё0-9]+", text) if len(t) >= 4]
    if not tokens:
        return False
    return all(t in title_lower for t in tokens)


def format_source_label(metadata: dict | None) -> str:
    if not isinstance(metadata, dict):
        return ""

    title = str(metadata.get("document_title") or "").strip()
    structure_path = metadata.get("structure_path") or []
    title_lower = title.lower()

    extras: list[str] = []
    for item in structure_path:
        text = str(item or "").strip()
        if not text:
            continue
        if text.lower() in _SKIP_STRUCTURE_ITEMS:      # «Введение / Общие положения»
            continue
        if _is_covered_by_title(text, title_lower):    # уже есть в заголовке
            continue
        if text not in extras:
            extras.append(text)

    parts: list[str] = []
    if title:
        parts.append(title)
    parts.extend(extras)
    return ", ".join(parts)


def replace_source_tags(answer: str, sources: dict | None) -> str:
    """Инлайн — только для уникальных источников; повторяющиеся — в конце.

    - источник встречается 1 раз   → ``[Источник: <label>]`` на месте;
    - источник встречается ≥2 раз  → все инлайн-теги удаляются, а сам
      источник один раз попадает в блок ``Источники:`` в конце ответа;
    - id отсутствует в ``sources`` → тег удаляется.
    """
    if not isinstance(answer, str) or "<source>" not in answer:
        return answer
    if not isinstance(sources, dict):
        sources = {}

    # 1. Сколько раз каждый источник упомянут во всём ответе
    mention_count: dict[str, int] = {}
    for match in _SOURCE_TAG_RE.finditer(answer):
        for sid in _parse_source_ids(match.group(1)):
            mention_count[sid] = mention_count.get(sid, 0) + 1

    repeated_ids = {sid for sid, n in mention_count.items() if n >= 2}

    # 2. Инлайн: оставляем только источники с единственным упоминанием
    def _sub_inline(match: re.Match) -> str:
        labels: list[str] = []
        for sid in _parse_source_ids(match.group(1)):
            if sid in repeated_ids:
                continue
            label = format_source_label(sources.get(sid) or {})
            if label and label not in labels:
                labels.append(label)
        if not labels:
            return ""
        return "[Источник: " + "; ".join(labels) + "]"

    cleaned = _SOURCE_TAG_RE.sub(_sub_inline, answer)

    # 3. Блок "Источники:" в конце — только для повторяющихся
    if repeated_ids:
        ordered: list[str] = []
        seen: set[str] = set()
        for match in _SOURCE_TAG_RE.finditer(answer):   # порядок первого упоминания
            for sid in _parse_source_ids(match.group(1)):
                if sid in repeated_ids and sid not in seen:
                    seen.add(sid)
                    ordered.append(sid)

        end_labels: list[str] = []
        for sid in ordered:
            label = format_source_label(sources.get(sid) or {})
            if label and label not in end_labels:
                end_labels.append(label)

        if end_labels:
            # Чистим "пробелы перед точкой/запятой" и "двойные пробелы",
            # оставшиеся после удаления инлайн-тегов.
            cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
            cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
            cleaned = cleaned.rstrip()

            block = "\n\nИсточники:\n" + "\n".join(
                f"- {label}" for label in end_labels
            )
            cleaned = cleaned + block

    return cleaned