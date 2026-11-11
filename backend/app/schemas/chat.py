"""Pydantic schemas for chat sessions, messages, and citations."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MESSAGE_LENGTH = 10_000


class SourceCitation(BaseModel):
    """Structured citation pointing to a source document provision."""

    document_id: str
    version_id: str | None = None
    document_title: str
    provision_id: str | None = None
    structure_path: list[str] = Field(default_factory=list)
    excerpt: str
    char_start: int | None = None
    char_end: int | None = None
    score: float | None = None


class ChatCreate(BaseModel):
    """Request payload for creating a chat session."""

    title: str | None = Field(default=None, max_length=255)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ChatRead(BaseModel):
    """Serialized representation of a chat session."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatListResponse(BaseModel):
    """Paginated/listed collection of chats."""

    items: list[ChatRead]
    total: int


class MessageCreate(BaseModel):
    """Request payload for posting a new user message."""

    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Message content must not be empty")
        if len(stripped) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Message content must not exceed {MAX_MESSAGE_LENGTH} characters"
            )
        return stripped


class MessageRead(BaseModel):
    """Serialized representation of a stored chat message."""

    id: str
    chat_id: str
    role: Literal["user", "assistant"]
    content: str
    sources: list[SourceCitation] = Field(default_factory=list)
    intent: str | None = None
    refused: bool = False
    clarification_needed: bool = False
    created_at: datetime


class ChatMessagesResponse(BaseModel):
    """Message history response for a single chat."""

    chat_id: str
    messages: list[MessageRead]


class SendMessageResponse(BaseModel):
    """Turn response containing both the persisted user and assistant messages."""

    chat_id: str
    user_message: MessageRead
    assistant_message: MessageRead
