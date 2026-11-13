"""External integration ports and concrete backend implementations."""

from .base import AssistantPort, AssistantReply, ChatTurn
from .llm_client import DirectLLMBackend

__all__ = [
    "AssistantPort",
    "AssistantReply",
    "ChatTurn",
    "DirectLLMBackend",
]
