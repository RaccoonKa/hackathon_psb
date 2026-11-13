"""Регрессионные тесты провайдеров эмбеддингов (пункты 9–10).

Локальная модель замокана как падающая при загрузке; API-клиент — мок,
возвращающий вектор корректной размерности. Проверяется fallback-диагностика,
кэширование неудачи локальной загрузки и явная ошибка при недоступности
обоих провайдеров (без тихого hash-fallback).
"""

import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag_core.retrieval import vector_store as vs_mod
from rag_core.retrieval.vector_store import EmbeddingConfigError, compute_embeddings


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorchModule:
    cuda = _FakeCuda()


def _install_fake_torch():
    saved = sys.modules.get("torch")
    sys.modules["torch"] = _FakeTorchModule()
    return saved


def _restore_module(name: str, saved) -> None:
    if saved is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = saved


def _install_failing_transformers(counter: dict):
    saved = sys.modules.get("transformers")

    def fail_from_pretrained(path):
        counter["calls"] = counter.get("calls", 0) + 1
        raise RuntimeError("synthetic: local bge-m3 is unavailable")

    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=fail_from_pretrained)
    fake.AutoModel = types.SimpleNamespace(from_pretrained=fail_from_pretrained)
    sys.modules["transformers"] = fake
    return saved


def _api_client(dimension: int = 1024):
    item = MagicMock()
    item.embedding = [0.1] * dimension
    item.index = 0
    response = MagicMock()
    response.data = [item]
    client = MagicMock()
    client.embeddings.create.return_value = response
    return client


def _prepare_state() -> Path:
    vs_mod.reset_local_model_cache()
    vs_mod._api_client = None
    return Path(tempfile.mkdtemp(prefix="fake_bge_m3_"))


def _cleanup_state(fake_dir: Path) -> None:
    vs_mod.reset_local_model_cache()
    vs_mod._api_client = None
    shutil.rmtree(fake_dir, ignore_errors=True)


# --- 9. API fallback с кэшем неудачи локальной модели ----------------------


def test_api_fallback_when_local_model_fails_and_failure_is_cached():
    fake_dir = _prepare_state()
    counter: dict = {}
    saved_torch = _install_fake_torch()
    saved_transformers = _install_failing_transformers(counter)
    client = _api_client(1024)
    try:
        with patch.object(vs_mod, "LOCAL_MODEL_PATH", fake_dir), \
             patch.object(vs_mod, "USE_SUBMIT", False), \
             patch.object(vs_mod, "USE_LOCAL_EMBEDDING_MODEL", True), \
             patch.object(vs_mod, "ALLOW_HASH_FALLBACK", True), \
             patch.object(vs_mod, "EMBEDDING_DIMENSION", 0), \
             patch.object(vs_mod, "_get_api_client", return_value=client):
            first = compute_embeddings(["первый текст"])
            second = compute_embeddings(["второй текст"])
            diagnostics = vs_mod.get_last_embedding_diagnostics()

            # Неудача локальной загрузки не повторяется при втором вызове.
            assert counter["calls"] == 1
            assert client.embeddings.create.call_count == 2

            assert len(first[0]) == 1024
            assert len(second[0]) == 1024
            assert diagnostics["provider"] == "api"
            assert diagnostics["success"] is True
            assert diagnostics["fallback_used"] is True
            assert diagnostics["dimension"] == 1024
            assert diagnostics["error_type"] is None

            # Повторная попытка возможна только после явного сброса кэша.
            vs_mod.reset_local_model_cache()
            compute_embeddings(["третий текст"])
            assert counter["calls"] == 2
    finally:
        _restore_module("torch", saved_torch)
        _restore_module("transformers", saved_transformers)
        _cleanup_state(fake_dir)


def test_local_model_status_reports_cached_failure():
    fake_dir = _prepare_state()
    counter: dict = {}
    saved_torch = _install_fake_torch()
    saved_transformers = _install_failing_transformers(counter)
    try:
        with patch.object(vs_mod, "LOCAL_MODEL_PATH", fake_dir), \
             patch.object(vs_mod, "USE_SUBMIT", False), \
             patch.object(vs_mod, "USE_LOCAL_EMBEDDING_MODEL", True):
            vs_mod._load_local_model()
            status = vs_mod.get_local_model_status()
            assert status["available"] is False
            assert status["error_type"] == "RuntimeError"
            assert "synthetic" in (status["error_message"] or "")
            # Повторный вызов не грузит модель заново.
            vs_mod._load_local_model()
            assert counter["calls"] == 1
    finally:
        _restore_module("torch", saved_torch)
        _restore_module("transformers", saved_transformers)
        _cleanup_state(fake_dir)


# --- 10. Оба провайдера недоступны ----------------------------------------


def test_both_providers_unavailable_raises_explicit_error():
    fake_dir = _prepare_state()
    counter: dict = {}
    saved_torch = _install_fake_torch()
    saved_transformers = _install_failing_transformers(counter)
    try:
        with patch.object(vs_mod, "LOCAL_MODEL_PATH", fake_dir), \
             patch.object(vs_mod, "USE_SUBMIT", False), \
             patch.object(vs_mod, "USE_LOCAL_EMBEDDING_MODEL", True), \
             patch.object(vs_mod, "ALLOW_HASH_FALLBACK", False), \
             patch.object(vs_mod, "EMBEDDING_DIMENSION", 0), \
             patch.object(vs_mod, "EMBEDDING_API_BASE_URL", ""), \
             patch.object(vs_mod, "EMBEDDING_API_KEY", ""), \
             patch.object(
                 vs_mod,
                 "_fallback_embedding",
                 side_effect=AssertionError(
                     "hash-fallback запрещён и не должен вызываться"
                 ),
             ):
            try:
                compute_embeddings(["текст без провайдера"])
            except EmbeddingConfigError:
                pass
            else:
                raise AssertionError(
                    "Ожидалась EmbeddingConfigError при недоступности "
                    "обоих провайдеров"
                )

            diagnostics = vs_mod.get_last_embedding_diagnostics()
            assert counter["calls"] == 1
            assert diagnostics["provider"] == "api"
            assert diagnostics["success"] is False
            assert diagnostics["fallback_used"] is False
            assert diagnostics["error_type"] == "EmbeddingConfigError"
    finally:
        _restore_module("torch", saved_torch)
        _restore_module("transformers", saved_transformers)
        _cleanup_state(fake_dir)
