import logging
from src import config
from src.generation.stubs import stub_llm_complete

logger = logging.getLogger(__name__)


def _build_chat_model(model: str, temperature: float, max_tokens: int):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        max_retries=config.MAX_RETRIES,
    )


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
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> tuple[str, int, int]:
    llm = _build_chat_model(
        model=model, temperature=temperature, max_tokens=max_tokens
    )
    response = llm.invoke(messages)
    content = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    prompt_tokens, completion_tokens = _extract_token_usage(response)
    return content, prompt_tokens, completion_tokens


def llm_complete_with_usage(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, int, int]:
    if config.USE_STUB_LLM:
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
    )


def llm_complete(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    content, _, _ = llm_complete_with_usage(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return content