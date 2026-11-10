import logging
from langchain_openai import ChatOpenAI
from src import config
from src.generation.stubs import stub_llm_complete

logger = logging.getLogger(__name__)


def _build_chat_model(
    model: str, temperature: float, max_tokens: int
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        max_retries=config.MAX_RETRIES,
    )


def _invoke_real_llm(
    messages: list[dict], model: str, temperature: float, max_tokens: int
) -> str:
    llm = _build_chat_model(
        model=model, temperature=temperature, max_tokens=max_tokens
    )
    response = llm.invoke(messages)
    return response.content


def llm_complete(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if config.USE_STUB_LLM or config.LLM_MODE == config.LLM_MODE_STUB:
        return stub_llm_complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _invoke_real_llm(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )