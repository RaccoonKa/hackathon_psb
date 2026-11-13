from functools import lru_cache
import logging
import re
from langchain_openai import ChatOpenAI
import openai
from rag_core import config
from rag_core.generation.errors import LLMAPIError, LLMTimeoutError
from rag_core.generation.stubs import stub_llm_complete

logger = logging.getLogger(__name__)

_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


@lru_cache(maxsize=16)
def _build_chat_model(
    model: str, temperature: float, max_tokens: int, timeout: float
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        max_retries=config.MAX_RETRIES,
        timeout=timeout,
    )


def _strip_reasoning_tokens(text: str) -> str:
    cleaned = _THINK_TAG_PATTERN.sub("", text)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def _extract_token_usage(response: object) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        if prompt_tokens is not None and completion_tokens is not None:
            return int(prompt_tokens), int(completion_tokens)

    resp_meta = getattr(response, "response_metadata", None)
    if isinstance(resp_meta, dict):
        token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
        if isinstance(token_usage, dict):
            prompt_tokens = token_usage.get("prompt_tokens")
            completion_tokens = token_usage.get("completion_tokens")
            if prompt_tokens is not None and completion_tokens is not None:
                return int(prompt_tokens), int(completion_tokens)

    return config.STUB_PROMPT_TOKENS, config.STUB_COMPLETION_TOKENS


def _invoke_real_llm(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> tuple[str, int, int]:
    llm = _build_chat_model(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    try:
        response = llm.invoke(messages)
    except openai.APITimeoutError as exc:
        raise LLMTimeoutError(f"LLM request timed out after {timeout}s: {exc}") from exc
    except openai.APIError as exc:
        raise LLMAPIError(f"LLM API error: {exc}") from exc
    except TimeoutError as exc:
        raise LLMTimeoutError(f"LLM request timed out: {exc}") from exc
    except Exception as exc:
        if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
            raise LLMTimeoutError(f"LLM timeout: {exc}") from exc
        raise LLMAPIError(f"Unexpected LLM transport error: {exc}") from exc

    raw_content = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    content = _strip_reasoning_tokens(raw_content)
    prompt_tokens, completion_tokens = _extract_token_usage(response)
    return content, prompt_tokens, completion_tokens


def llm_complete_with_usage(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float | None = None,
) -> tuple[str, int, int]:
    effective_timeout = (
        float(timeout)
        if timeout is not None
        else float(config.LLM_REQUEST_TIMEOUT_S)
    )
    if config.USE_STUB_LLM or getattr(config, "LLM_MODE", "real") == config.LLM_MODE_STUB:
        text = stub_llm_complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return text, config.STUB_PROMPT_TOKENS, config.STUB_COMPLETION_TOKENS
    return _invoke_real_llm(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=effective_timeout,
    )


def llm_complete(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float | None = None,
) -> str:
    content, _, _ = llm_complete_with_usage(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return content