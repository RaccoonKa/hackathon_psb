from unittest.mock import MagicMock, patch

from rag_core import config
from rag_core.generation.llm import client as lc_mod
from rag_core.generation.llm.client import (
    _build_chat_model,
    _extract_token_usage,
    _invoke_real_llm,
    llm_complete,
    llm_complete_with_usage,
)


def _clear_chat_model_cache():
    _build_chat_model.cache_clear()


def test_build_chat_model_params():
    _clear_chat_model_cache()
    with patch.object(lc_mod, "ChatOpenAI") as mock:
        _build_chat_model("m", 0.5, 100, 30.0)
    kwargs = mock.call_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_completion_tokens"] == 100
    assert kwargs["max_retries"] == config.MAX_RETRIES
    assert kwargs["timeout"] == 30.0


def test_build_chat_model_uses_config_keys():
    _clear_chat_model_cache()
    with patch.object(lc_mod, "ChatOpenAI") as mock:
        _build_chat_model("m-cfg", 0.0, 100, 10.0)
    kwargs = mock.call_args.kwargs
    assert kwargs["api_key"] == config.LLM_API_KEY
    assert kwargs["base_url"] == config.LLM_BASE_URL


def test_build_chat_model_cached():
    _clear_chat_model_cache()
    with patch.object(lc_mod, "ChatOpenAI") as mock:
        _build_chat_model("m-cache", 0.1, 10, 5.0)
        _build_chat_model("m-cache", 0.1, 10, 5.0)
    assert mock.call_count == 1


def test_extract_token_usage_from_usage_metadata():
    resp = MagicMock()
    resp.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    resp.response_metadata = None
    p, c = _extract_token_usage(resp)
    assert p == 100
    assert c == 50


def test_extract_token_usage_from_response_metadata_token_usage():
    resp = MagicMock()
    resp.usage_metadata = None
    resp.response_metadata = {
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }
    p, c = _extract_token_usage(resp)
    assert p == 10
    assert c == 5


def test_extract_token_usage_from_response_metadata_usage():
    resp = MagicMock()
    resp.usage_metadata = None
    resp.response_metadata = {
        "usage": {"prompt_tokens": 20, "completion_tokens": 8}
    }
    p, c = _extract_token_usage(resp)
    assert p == 20
    assert c == 8


def test_extract_token_usage_fallback():
    resp = MagicMock()
    resp.usage_metadata = None
    resp.response_metadata = None
    p, c = _extract_token_usage(resp)
    assert p == config.STUB_PROMPT_TOKENS
    assert c == config.STUB_COMPLETION_TOKENS


def test_extract_token_usage_partial_returns_fallback():
    resp = MagicMock()
    resp.usage_metadata = {"input_tokens": 100}
    resp.response_metadata = None
    p, c = _extract_token_usage(resp)
    assert p == config.STUB_PROMPT_TOKENS
    assert c == config.STUB_COMPLETION_TOKENS


def test_extract_token_usage_zero_tokens():
    resp = MagicMock()
    resp.usage_metadata = {"input_tokens": 0, "output_tokens": 0}
    resp.response_metadata = None
    p, c = _extract_token_usage(resp)
    assert p == 0
    assert c == 0


def test_invoke_real_llm_string_content():
    fake_llm = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = "Ответ"
    fake_resp.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
    fake_resp.response_metadata = None
    fake_llm.invoke.return_value = fake_resp

    with patch.object(lc_mod, "_build_chat_model", return_value=fake_llm):
        content, p, c = _invoke_real_llm(
            messages=[{"role": "user", "content": "hi"}],
            model="m", temperature=0.0, max_tokens=100, timeout=30.0,
        )
    assert content == "Ответ"
    assert p == 10
    assert c == 5


def test_invoke_real_llm_list_content():
    fake_llm = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = [{"type": "text", "text": "Ответ"}]
    fake_resp.usage_metadata = None
    fake_resp.response_metadata = None
    fake_llm.invoke.return_value = fake_resp

    with patch.object(lc_mod, "_build_chat_model", return_value=fake_llm):
        content, _, _ = _invoke_real_llm(
            messages=[], model="m", temperature=0.0, max_tokens=100, timeout=30.0,
        )
    assert isinstance(content, str)
    assert content


def test_invoke_real_llm_strips_think_tokens():
    fake_llm = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = "<think>рассуждение</think>Ответ"
    fake_resp.usage_metadata = {"input_tokens": 1, "output_tokens": 1}
    fake_resp.response_metadata = None
    fake_llm.invoke.return_value = fake_resp

    with patch.object(lc_mod, "_build_chat_model", return_value=fake_llm):
        content, _, _ = _invoke_real_llm(
            messages=[], model="m", temperature=0.0, max_tokens=100, timeout=30.0,
        )
    assert content == "Ответ"


def test_llm_complete_with_usage_real():
    fake_llm = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = "Ответ"
    fake_resp.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    fake_resp.response_metadata = None
    fake_llm.invoke.return_value = fake_resp

    with patch.object(lc_mod, "_build_chat_model", return_value=fake_llm):
        text, p, c = llm_complete_with_usage(
            messages=[{"role": "user", "content": "hi"}],
            model="m", temperature=0.0, max_tokens=100,
        )
    assert text == "Ответ"
    assert p == 100
    assert c == 50


def test_llm_complete_returns_text_only():
    fake_llm = MagicMock()
    fake_resp = MagicMock()
    fake_resp.content = "Ответ"
    fake_resp.usage_metadata = None
    fake_resp.response_metadata = None
    fake_llm.invoke.return_value = fake_resp

    with patch.object(lc_mod, "_build_chat_model", return_value=fake_llm):
        text = llm_complete(
            messages=[{"role": "user", "content": "hi"}],
            model="m", temperature=0.0, max_tokens=100,
        )
    assert text == "Ответ"


def test_llm_complete_with_usage_stub_path():
    with patch.object(lc_mod, "stub_llm_complete", return_value="stub answer"):
        with patch.object(lc_mod.config, "USE_STUB_LLM", True):
            text, p, c = llm_complete_with_usage(
                messages=[{"role": "user", "content": "hi"}],
                model="m", temperature=0.0, max_tokens=100,
            )
    assert text == "stub answer"
    assert p == config.STUB_PROMPT_TOKENS
    assert c == config.STUB_COMPLETION_TOKENS
