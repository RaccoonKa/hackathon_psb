"""Direct OpenAI-compatible HTTP LLM backend implementing `AssistantPort`."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from ..core.config import Settings
from ..core.exceptions import LLMNotConfiguredError, UpstreamLLMError
from .base import AssistantReply, ChatTurn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — юридический ИИ-ассистент. Отвечай точно, понятно и вежливо на русском языке. "
    "Опирайся только на достоверные правовые знания и контекст текущего диалога. "
    "Не выдумывай несуществующие нормативные акты, статьи, пункты, реквизиты документов или цитаты. "
    "Если для точного ответа не хватает сведений или требуются конкретные документы из базы знаний, "
    "прямо сообщи об этом пользователю и предложи уточнить вопрос. "
    "Формулируй ответ структурированно и по существу."
)

_MAX_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 0.5


def _extract_reply_content(payload: Any) -> str:
    """Extract non-empty assistant message text from an OpenAI-compatible response."""
    if not isinstance(payload, dict):
        raise UpstreamLLMError("Upstream LLM request failed", status_code=502)

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamLLMError("Upstream LLM request failed", status_code=502)

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise UpstreamLLMError("Upstream LLM request failed", status_code=502)

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise UpstreamLLMError("Upstream LLM request failed", status_code=502)

    raw_content = message.get("content")
    if isinstance(raw_content, str):
        content = raw_content.strip()
    elif isinstance(raw_content, list):
        parts = [
            part.get("text", "")
            for part in raw_content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        content = "".join(parts).strip()
    else:
        content = ""

    if not content:
        raise UpstreamLLMError("Upstream LLM request failed", status_code=502)
    return content


class DirectLLMBackend:
    """OpenAI-compatible `/chat/completions` client implementing `AssistantPort`."""

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings) -> None:
        self._http_client = http_client
        self._settings = settings

    async def generate_reply(self, history: list[ChatTurn]) -> AssistantReply:
        """Send conversation history to the configured LLM endpoint and parse the reply."""
        if not self._settings.llm_configured:
            raise LLMNotConfiguredError(
                "LLM API is not configured (missing LLM_API_KEY or LLM_BASE_URL)"
            )

        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._settings.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *[{"role": turn.role, "content": turn.content} for turn in history],
            ],
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._http_client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self._settings.llm_timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                logger.warning(
                    "Upstream LLM request timed out (attempt %d/%d, error=%s)",
                    attempt,
                    _MAX_ATTEMPTS,
                    type(exc).__name__,
                )
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                raise UpstreamLLMError(
                    "Upstream LLM request timed out",
                    status_code=504,
                ) from None
            except httpx.TransportError as exc:
                logger.warning(
                    "Upstream LLM transport failure (attempt %d/%d, error=%s)",
                    attempt,
                    _MAX_ATTEMPTS,
                    type(exc).__name__,
                )
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                raise UpstreamLLMError(
                    "Upstream LLM request failed",
                    status_code=502,
                ) from None
            except httpx.HTTPError as exc:
                logger.warning(
                    "Upstream LLM HTTP failure (error=%s)",
                    type(exc).__name__,
                )
                raise UpstreamLLMError(
                    "Upstream LLM request failed",
                    status_code=502,
                ) from None

            if response.status_code == 429 or response.status_code >= 500:
                logger.warning(
                    "Upstream LLM returned retryable status %d (attempt %d/%d)",
                    response.status_code,
                    attempt,
                    _MAX_ATTEMPTS,
                )
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)
                    continue
                raise UpstreamLLMError(
                    "Upstream LLM request failed",
                    status_code=502,
                )

            if response.status_code >= 400:
                logger.warning(
                    "Upstream LLM returned client error status %d",
                    response.status_code,
                )
                raise UpstreamLLMError(
                    "Upstream LLM request failed",
                    status_code=502,
                )

            try:
                data = response.json()
            except ValueError:
                logger.warning("Upstream LLM returned invalid JSON payload")
                raise UpstreamLLMError(
                    "Upstream LLM request failed",
                    status_code=502,
                ) from None

            content = _extract_reply_content(data)
            return AssistantReply(content=content)

        raise UpstreamLLMError("Upstream LLM request failed", status_code=502)
