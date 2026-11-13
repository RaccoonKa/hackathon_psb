"""REST routes for chat sessions and message history."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query, Response, status

from ...schemas.chat import (
    ChatCreate,
    ChatListResponse,
    ChatMessagesResponse,
    ChatRead,
    MessageCreate,
    SendMessageResponse,
)
from ...services.chat_service import ChatService
from ..deps import get_chat_service

router = APIRouter(prefix="/chats", tags=["chats"])


@router.get("", response_model=ChatListResponse)
async def list_chats(
    q: str | None = Query(default=None, description="Case-insensitive title filter"),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatListResponse:
    """List chat sessions ordered by recent activity."""
    return await chat_service.list_chats(query=q)


@router.post("", response_model=ChatRead, status_code=status.HTTP_201_CREATED)
async def create_chat(
    payload: ChatCreate | None = Body(default=None),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatRead:
    """Create a new chat session with an optional title."""
    title = payload.title if payload is not None else None
    return await chat_service.create_chat(title=title)


@router.get("/{chat_id}/messages", response_model=ChatMessagesResponse)
async def get_chat_messages(
    chat_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatMessagesResponse:
    """Fetch chronological message history for a chat session."""
    return await chat_service.get_messages(chat_id)


@router.post("/{chat_id}/messages", response_model=SendMessageResponse)
async def send_chat_message(
    chat_id: str,
    payload: MessageCreate,
    chat_service: ChatService = Depends(get_chat_service),
) -> SendMessageResponse:
    """Submit a user message and return both user and assistant messages."""
    return await chat_service.send_message(chat_id, payload.content)


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: str,
    chat_service: ChatService = Depends(get_chat_service),
) -> Response:
    """Delete a chat session and all its messages."""
    await chat_service.delete_chat(chat_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
