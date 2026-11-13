"""Business logic for chat sessions and conversation turns."""

from __future__ import annotations

import asyncio

from ..core.exceptions import NotFoundError
from ..integrations.base import AssistantPort, ChatTurn
from ..repositories.chat_repo import ChatRepository
from ..schemas.chat import (
    ChatListResponse,
    ChatMessagesResponse,
    ChatRead,
    SendMessageResponse,
)


class ChatService:
    """Service coordinating chat persistence and assistant reply generation."""

    def __init__(
        self,
        chat_repo: ChatRepository,
        assistant: AssistantPort,
        history_limit: int = 20,
    ) -> None:
        self._chat_repo = chat_repo
        self._assistant = assistant
        self._history_limit = history_limit

    async def create_chat(self, title: str | None = None) -> ChatRead:
        """Create a new chat session."""
        return await asyncio.to_thread(self._chat_repo.create_chat, title)

    async def list_chats(self, query: str | None = None) -> ChatListResponse:
        """List existing chats, optionally filtered by title substring."""
        items = await asyncio.to_thread(self._chat_repo.list_chats, query)
        return ChatListResponse(items=items, total=len(items))

    async def get_messages(self, chat_id: str) -> ChatMessagesResponse:
        """Return full chronological message history for an existing chat."""
        chat = await asyncio.to_thread(self._chat_repo.get_chat, chat_id)
        if chat is None:
            raise NotFoundError(f"Chat '{chat_id}' not found")
        messages = await asyncio.to_thread(self._chat_repo.get_messages, chat_id)
        return ChatMessagesResponse(chat_id=chat_id, messages=messages)

    async def delete_chat(self, chat_id: str) -> None:
        """Delete an existing chat session or raise `NotFoundError`."""
        deleted = await asyncio.to_thread(self._chat_repo.delete_chat, chat_id)
        if not deleted:
            raise NotFoundError(f"Chat '{chat_id}' not found")

    async def send_message(self, chat_id: str, content: str) -> SendMessageResponse:
        """Persist a user message, generate an assistant reply, and persist the reply."""
        chat = await asyncio.to_thread(self._chat_repo.get_chat, chat_id)
        if chat is None:
            raise NotFoundError(f"Chat '{chat_id}' not found")

        user_message = await asyncio.to_thread(
            self._chat_repo.add_message,
            chat_id,
            "user",
            content,
        )
        if user_message is None:
            raise NotFoundError(f"Chat '{chat_id}' not found")

        recent_messages = await asyncio.to_thread(
            self._chat_repo.get_messages,
            chat_id,
            limit=self._history_limit,
        )
        history = [
            ChatTurn(role=msg.role, content=msg.content) for msg in recent_messages
        ]

        reply = await self._assistant.generate_reply(history)

        assistant_message = await asyncio.to_thread(
            self._chat_repo.add_message,
            chat_id,
            "assistant",
            reply.content,
            sources=reply.sources,
            intent=reply.intent,
            refused=reply.refused,
            clarification_needed=reply.clarification_needed,
        )
        if assistant_message is None:
            raise NotFoundError(f"Chat '{chat_id}' not found")

        return SendMessageResponse(
            chat_id=chat_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )
