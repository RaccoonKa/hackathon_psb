"""Internal port contracts for assistant/LLM backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..schemas.chat import SourceCitation


@dataclass(frozen=True)
class ChatTurn:
    """Single historical conversation turn passed to an assistant backend."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class AssistantReply:
    """Structured reply returned by an assistant backend."""

    content: str
    sources: list[SourceCitation] = field(default_factory=list)
    refused: bool = False
    intent: str | None = None
    clarification_needed: bool = False


class AssistantPort(Protocol):
    """Port interface decoupling services from direct LLM or RAG backends."""

    async def generate_reply(self, history: list[ChatTurn]) -> AssistantReply:
        """Generate an assistant reply from chronological conversation turns."""
        ...
