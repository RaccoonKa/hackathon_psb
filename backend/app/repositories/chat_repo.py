"""SQLAlchemy repository for chat sessions and messages."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, cast

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from ..models.chat import Chat, Message
from ..schemas.chat import ChatRead, MessageRead, SourceCitation

DEFAULT_CHAT_TITLE = "Новый чат"
_AUTO_TITLE_MAX_LEN = 60


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _auto_title_from_content(content: str) -> str:
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    normalized = " ".join(first_line.split())
    if not normalized:
        return DEFAULT_CHAT_TITLE
    if len(normalized) <= _AUTO_TITLE_MAX_LEN:
        return normalized
    truncated = normalized[:_AUTO_TITLE_MAX_LEN].rstrip()
    return f"{truncated}..."


def _to_chat_read(chat: Chat) -> ChatRead:
    return ChatRead(
        id=chat.id,
        title=chat.title,
        created_at=_ensure_utc(chat.created_at),
        updated_at=_ensure_utc(chat.updated_at),
    )


def _to_message_read(message: Message) -> MessageRead:
    sources: list[SourceCitation] = []
    if message.sources_json:
        try:
            raw_sources = json.loads(message.sources_json)
        except json.JSONDecodeError:
            raw_sources = []
        if isinstance(raw_sources, list):
            sources = [SourceCitation.model_validate(item) for item in raw_sources]

    trace: list[dict] = []
    if message.trace_json:
        try:
            raw_trace = json.loads(message.trace_json)
        except json.JSONDecodeError:
            raw_trace = []
        if isinstance(raw_trace, list):
            trace = [item for item in raw_trace if isinstance(item, dict)]

    confidence: str | None = None
    assumptions: list[str] = []
    missing_info: list[str] = []
    refusal_reason: str | None = None
    if message.meta_json:
        try:
            raw_meta = json.loads(message.meta_json)
        except json.JSONDecodeError:
            raw_meta = {}
        if isinstance(raw_meta, dict):
            confidence = raw_meta.get("confidence")
            raw_assumptions = raw_meta.get("assumptions")
            if isinstance(raw_assumptions, list):
                assumptions = [str(x) for x in raw_assumptions]
            raw_missing_info = raw_meta.get("missing_info")
            if isinstance(raw_missing_info, list):
                missing_info = [str(x) for x in raw_missing_info]
            refusal_reason = raw_meta.get("refusal_reason")

    return MessageRead(
        id=message.id,
        chat_id=message.chat_id,
        role=cast(Literal["user", "assistant"], message.role),
        content=message.content,
        sources=sources,
        trace=trace,
        confidence=confidence,
        assumptions=assumptions,
        missing_info=missing_info,
        refusal_reason=refusal_reason,
        intent=message.intent,
        refused=message.refused,
        clarification_needed=message.clarification_needed,
        created_at=_ensure_utc(message.created_at),
    )


class ChatRepository:
    """Synchronous repository encapsulating chat and message persistence."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create_chat(self, title: str | None = None) -> ChatRead:
        """Create and persist a new chat session."""
        clean_title = (title or "").strip()[:255] or DEFAULT_CHAT_TITLE
        now = _utcnow()

        with self._session_factory() as session:
            chat = Chat(
                id=str(uuid.uuid4()),
                title=clean_title,
                created_at=now,
                updated_at=now,
            )
            session.add(chat)
            session.commit()
            session.refresh(chat)
            return _to_chat_read(chat)

    def list_chats(self, query: str | None = None) -> list[ChatRead]:
        """List chats ordered by most recently updated, optionally filtered by title."""
        normalized_query = (query or "").strip().casefold()

        with self._session_factory() as session:
            stmt = select(Chat).order_by(Chat.updated_at.desc(), Chat.created_at.desc())
            chats = list(session.scalars(stmt))
            if normalized_query:
                chats = [
                    chat for chat in chats if normalized_query in chat.title.casefold()
                ]
            return [_to_chat_read(chat) for chat in chats]

    def get_chat(self, chat_id: str) -> ChatRead | None:
        """Return a chat by ID, or None if it does not exist."""
        with self._session_factory() as session:
            chat = session.get(Chat, chat_id)
            if chat is None:
                return None
            return _to_chat_read(chat)

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and its messages; return True if deleted, False if missing."""
        with self._session_factory() as session:
            chat = session.get(Chat, chat_id)
            if chat is None:
                return False
            session.delete(chat)
            session.commit()
            return True

    def get_messages(self, chat_id: str, limit: int | None = None) -> list[MessageRead]:
        """Return messages for `chat_id` in chronological ascending order."""
        if limit is not None and limit <= 0:
            return []

        with self._session_factory() as session:
            if limit is None:
                stmt = (
                    select(Message)
                    .where(Message.chat_id == chat_id)
                    .order_by(Message.created_at.asc(), Message.id.asc())
                )
                rows = list(session.scalars(stmt))
            else:
                stmt = (
                    select(Message)
                    .where(Message.chat_id == chat_id)
                    .order_by(Message.created_at.desc(), Message.id.desc())
                    .limit(limit)
                )
                rows = list(session.scalars(stmt))
                rows.reverse()

            return [_to_message_read(row) for row in rows]

    def add_message(
        self,
        chat_id: str,
        role: Literal["user", "assistant"],
        content: str,
        sources: list[SourceCitation] | None = None,
        intent: str | None = None,
        refused: bool = False,
        clarification_needed: bool = False,
        trace: list[dict] | None = None,
        meta: dict | None = None,
    ) -> MessageRead | None:
        """Append a message to a chat, updating `updated_at` and auto-titling if needed."""
        with self._session_factory() as session:
            chat = session.get(Chat, chat_id)
            if chat is None:
                return None

            now = _utcnow()
            latest_ts = session.scalar(
                select(func.max(Message.created_at)).where(Message.chat_id == chat_id)
            )
            if latest_ts is not None:
                latest_utc = _ensure_utc(latest_ts)
                if now <= latest_utc:
                    now = latest_utc + timedelta(microseconds=1)

            sources_payload = [source.model_dump() for source in (sources or [])]
            trace_json = (
                json.dumps(trace, ensure_ascii=False) if trace is not None else None
            )
            meta_json = (
                json.dumps(meta, ensure_ascii=False) if meta is not None else None
            )
            message = Message(
                id=str(uuid.uuid4()),
                chat_id=chat_id,
                role=role,
                content=content,
                sources_json=json.dumps(sources_payload, ensure_ascii=False),
                trace_json=trace_json,
                meta_json=meta_json,
                intent=intent,
                refused=refused,
                clarification_needed=clarification_needed,
                created_at=now,
            )

            chat.updated_at = now
            if role == "user" and chat.title == DEFAULT_CHAT_TITLE:
                chat.title = _auto_title_from_content(content)

            session.add(message)
            session.commit()
            session.refresh(message)
            return _to_message_read(message)
