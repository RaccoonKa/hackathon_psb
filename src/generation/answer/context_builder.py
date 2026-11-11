import logging
import time
from typing import Any
from src import config
from src.generation import prompts
from src.generation.types import RetrievedChunk, SourceMetadata, TraceStep

logger = logging.getLogger(__name__)


def format_single_chunk(source_id: str, chunk: RetrievedChunk) -> str:
    has_provision: bool = (
        chunk.provision_id is not None and bool(chunk.provision_id.strip())
    )
    structure_path_str = config.STRUCTURE_PATH_SEPARATOR.join(
        chunk.structure_path
    )
    rendered = prompts.CONTEXT_ITEM_TEMPLATE.format(
        source_id=source_id,
        score=chunk.score,
        document_title=chunk.document_title,
        structure_path_str=structure_path_str,
        provision_id=chunk.provision_id or "",
        text=chunk.text,
    )
    if not has_provision:
        filtered_lines = [
            line
            for line in rendered.splitlines()
            if not line.startswith(prompts.PROVISION_LINE_PREFIX)
        ]
        return "\n".join(filtered_lines)
    return rendered


def build_source_entry(chunk: RetrievedChunk) -> SourceMetadata:
    return {
        "document_id": chunk.document_id,
        "version_id": chunk.version_id,
        "chunk_id": chunk.chunk_id,
        "document_title": chunk.document_title,
        "provision_id": chunk.provision_id,
        "structure_path": list(chunk.structure_path),
        "score": chunk.score,
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