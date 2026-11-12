import logging
import time
from typing import Any
from src import config
from src.generation.prompts import templates as prompts
from src.generation.types import RetrievedChunk, SourceMetadata, TraceStep

logger = logging.getLogger(__name__)


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