"""RAG assistant adapter integrating the core retrieval-generation pipeline."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any

from src.generation.answer.context_builder import replace_source_tags
from src.generation.orchestrator import run_agent
from src.retrieval.hybrid import load_canonical_chunks_map

from ..schemas.chat import SourceCitation
from .base import AssistantPort, AssistantReply, ChatTurn

logger = logging.getLogger(__name__)


def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _make_json_safe(val: Any) -> Any:
    if isinstance(val, dict):
        return {str(k): _make_json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_make_json_safe(item) for item in val]
    if isinstance(val, set):
        return [_make_json_safe(item) for item in sorted(val, key=str)]
    if hasattr(val, "value"):
        return val.value
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    return str(val)


class RAGAssistantAdapter(AssistantPort):
    """Adapter bridging the RAG generation core with the application's AssistantPort."""

    def __init__(
        self,
        chunks_map: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._chunks_map = chunks_map

    def _get_chunks_map(self) -> dict[str, dict[str, Any]]:
        if self._chunks_map is None:
            try:
                self._chunks_map = load_canonical_chunks_map()
            except Exception as exc:
                logger.warning("Failed to load canonical chunks map: %s", exc)
                self._chunks_map = {}
        return self._chunks_map

    def preload_chunks(self) -> None:
        """Preload chunks map into memory to avoid disk I/O on requests."""
        self._get_chunks_map()

    async def generate_reply(
        self,
        history: list[ChatTurn],
        event_date: str | None = None,
        selected_doc_ids: set[str] | None = None,
    ) -> AssistantReply:
        user_turns = [turn for turn in history if turn.role == "user"]
        query = user_turns[-1].content if user_turns else ""

        chat_history = [
            {"role": turn.role, "content": turn.content}
            for turn in history[:-1]
        ]

        normalized_doc_ids: set[str] | None = None
        if selected_doc_ids:
            normalized_doc_ids = set(str(doc_id) for doc_id in selected_doc_ids)

        clean_event_date = (
            str(event_date).strip()
            if event_date and str(event_date).strip()
            else None
        )

        try:
            result = await asyncio.to_thread(
                run_agent,
                query,
                event_date=clean_event_date,
                selected_doc_ids=normalized_doc_ids,
                chat_history=chat_history,
            )
        except Exception as exc:
            logger.exception("Failed to execute RAG agent for query: %r", query)
            return AssistantReply(
                content=(
                    "К сожалению, при обработке вашего запроса произошла техническая ошибка. "
                    "Пожалуйста, повторите попытку или переформулируйте вопрос."
                ),
                sources=[],
                refused=True,
                intent=None,
                clarification_needed=False,
                trace=[],
                confidence="low",
                assumptions=[],
                missing_info=[],
                refusal_reason=str(exc),
            )

        raw_answer = result.final_answer or ""
        raw_sources = result.sources or {}
        try:
            content = replace_source_tags(raw_answer, raw_sources)
        except Exception as exc:
            logger.warning("replace_source_tags failed, falling back to raw answer: %s", exc)
            content = raw_answer

        chunks_map = self._get_chunks_map()
        citations: list[SourceCitation] = []

        for sid, sdata in raw_sources.items():
            if not isinstance(sdata, dict):
                continue

            cid = sdata.get("chunk_id")
            chunk = chunks_map.get(str(cid)) if cid else None
            loc = (
                chunk.get("location")
                if chunk and isinstance(chunk.get("location"), dict)
                else {}
            )

            excerpt = (
                (chunk.get("text") or "")
                if chunk
                else str(sdata.get("excerpt") or sdata.get("text") or "")
            )

            raw_char_start = (
                chunk.get("char_start")
                if chunk and chunk.get("char_start") is not None
                else loc.get("char_start")
            ) if chunk else None

            raw_char_end = (
                chunk.get("char_end")
                if chunk and chunk.get("char_end") is not None
                else loc.get("char_end")
            ) if chunk else None

            raw_page = (
                chunk.get("page")
                if chunk and chunk.get("page") is not None
                else chunk.get("page_num")
                if chunk and chunk.get("page_num") is not None
                else loc.get("page")
                if loc.get("page") is not None
                else loc.get("page_num")
            ) if chunk else None

            char_start = _safe_int(raw_char_start)
            char_end = _safe_int(raw_char_end)
            page = _safe_int(raw_page)
            score = _safe_float(sdata.get("score"))

            document_id = str(
                sdata.get("document_id")
                or (chunk.get("document_id") if chunk else "")
                or ""
            )
            version_id = sdata.get("version_id") or (
                chunk.get("version_id") if chunk else None
            )
            document_title = str(
                sdata.get("document_title")
                or (chunk.get("document_title") if chunk else "")
                or ""
            )
            provision_id = sdata.get("provision_id") or (
                chunk.get("structure", {}).get("provision_id")
                if chunk and isinstance(chunk.get("structure"), dict)
                else None
            )
            structure_path = list(
                sdata.get("structure_path")
                or (
                    chunk.get("structure", {}).get("path")
                    if chunk and isinstance(chunk.get("structure"), dict)
                    else []
                )
                or []
            )

            citations.append(
                SourceCitation(
                    source_id=sid,
                    document_id=document_id,
                    version_id=version_id,
                    document_title=document_title,
                    provision_id=provision_id,
                    structure_path=structure_path,
                    excerpt=excerpt,
                    char_start=char_start,
                    char_end=char_end,
                    page=page,
                    score=score,
                )
            )

        raw_trace = [
            (
                dataclasses.asdict(step)
                if dataclasses.is_dataclass(step)
                else (step if isinstance(step, dict) else dict(step))
            )
            for step in (result.trace or [])
        ]
        trace = [_make_json_safe(step) for step in raw_trace]

        confidence = str(result.confidence or "high")
        assumptions = [str(a) for a in (result.assumptions or [])]
        missing_info = [str(m) for m in (result.missing_info or [])]
        refusal_reason = result.refusal_reason
        refused = bool(result.refused)
        intent = (
            result.intent.value
            if hasattr(result.intent, "value")
            else (str(result.intent) if result.intent is not None else None)
        )
        clarification_needed = bool(getattr(result, "clarification_needed", False))

        return AssistantReply(
            content=content,
            sources=citations,
            refused=refused,
            intent=intent,
            clarification_needed=clarification_needed,
            trace=trace,
            confidence=confidence,
            assumptions=assumptions,
            missing_info=missing_info,
            refusal_reason=refusal_reason,
        )
