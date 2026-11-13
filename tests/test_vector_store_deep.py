import sys
import types
from unittest.mock import patch

from rag_core.retrieval import vector_store as vs_mod
from rag_core.retrieval.vector_store import _load_local_model, compute_embeddings


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeTensor:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, key):
        if isinstance(key, tuple) and key == (slice(None), 0):
            return _FakeTensor([[r[0]] for r in self._rows])
        return self

    def cpu(self):
        return self

    def to(self, device):
        return self

    def tolist(self):
        return self._rows


class _FakeOutputs:
    def __init__(self, rows):
        self.last_hidden_state = _FakeTensor(rows)


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.call_count = 0

    def __call__(self, texts, **kwargs):
        self.call_count += 1
        return _FakeBatchEncoding({"input_ids": [[1, 2, 3]] * len(texts)})


class _FakeModel:
    def __init__(self):
        self.call_count = 0

    def __call__(self, **kwargs):
        self.call_count += 1
        n = len(kwargs.get("input_ids", []))
        return _FakeOutputs([[0.5, 0.5] for _ in range(n)])


class _FakeLoadedModel:
    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeTorchModule:
    class cuda:
        @staticmethod
        def is_available():
            return False

    class nn:
        class functional:
            @staticmethod
            def normalize(t, p, dim):
                return t

    @staticmethod
    def no_grad():
        return _NullCtx()


def _install_fake_torch():
    saved = sys.modules.get("torch")
    sys.modules["torch"] = _FakeTorchModule()
    return saved


def _restore_torch(saved):
    if saved is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = saved


def _install_fake_transformers(tok_from, model_from):
    saved = sys.modules.get("transformers")
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=tok_from)
    fake.AutoModel = types.SimpleNamespace(from_pretrained=model_from)
    sys.modules["transformers"] = fake
    return saved


def _restore_transformers(saved):
    if saved is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = saved


def _reset_local_cache():
    # Публичный сброс также очищает кэш неудачи загрузки (_local_model_error).
    vs_mod.reset_local_model_cache()
    vs_mod._local_tokenizer = None
    vs_mod._local_model = None
    vs_mod._local_device = None


def test_load_local_model_returns_none_when_path_missing():
    _reset_local_cache()
    with patch.object(vs_mod, "LOCAL_MODEL_PATH") as mock_path:
        mock_path.exists.return_value = False
        tok, model, dev = _load_local_model()
    assert tok is None
    assert model is None
    assert dev is None


def test_load_local_model_returns_none_when_import_fails():
    _reset_local_cache()
    saved_torch = sys.modules.pop("torch", None)
    saved_transformers = sys.modules.pop("transformers", None)
    try:
        with patch.object(vs_mod, "LOCAL_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch.dict(sys.modules, {"torch": None, "transformers": None}):
                tok, model, dev = _load_local_model()
        assert tok is None
        assert model is None
        assert dev is None
    finally:
        if saved_torch is not None:
            sys.modules["torch"] = saved_torch
        if saved_transformers is not None:
            sys.modules["transformers"] = saved_transformers
        _reset_local_cache()


def test_load_local_model_loads_and_returns_triple():
    _reset_local_cache()
    calls = {"tok": 0, "model": 0}

    def tok_from(path):
        calls["tok"] += 1
        return _FakeTokenizer()

    def model_from(path):
        calls["model"] += 1
        return _FakeLoadedModel()

    saved_torch = _install_fake_torch()
    saved_trans = _install_fake_transformers(tok_from, model_from)
    try:
        with patch.object(vs_mod, "LOCAL_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            tok, model, dev = _load_local_model()
        assert tok is not None
        assert model is not None
        assert dev in ("cpu", "cuda")
        assert calls == {"tok": 1, "model": 1}
    finally:
        _restore_torch(saved_torch)
        _restore_transformers(saved_trans)
        _reset_local_cache()


def test_load_local_model_caches_across_calls():
    _reset_local_cache()
    calls = {"tok": 0, "model": 0}

    def tok_from(path):
        calls["tok"] += 1
        return _FakeTokenizer()

    def model_from(path):
        calls["model"] += 1
        return _FakeLoadedModel()

    saved_torch = _install_fake_torch()
    saved_trans = _install_fake_transformers(tok_from, model_from)
    try:
        with patch.object(vs_mod, "LOCAL_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            tok1, model1, dev1 = _load_local_model()
            tok2, model2, dev2 = _load_local_model()
            tok3, model3, dev3 = _load_local_model()
        assert tok1 is tok2 is tok3
        assert model1 is model2 is model3
        assert dev1 == dev2 == dev3
        assert calls == {"tok": 1, "model": 1}
    finally:
        _restore_torch(saved_torch)
        _restore_transformers(saved_trans)
        _reset_local_cache()


def test_compute_embeddings_uses_local_model():
    _reset_local_cache()
    fake_tok = _FakeTokenizer()
    fake_model = _FakeModel()

    saved_torch = _install_fake_torch()
    try:
        with patch.object(vs_mod, "_load_local_model", return_value=(fake_tok, fake_model, "cpu")), \
             patch.object(vs_mod, "_get_api_client") as mock_get_client:
            mock_get_client.return_value.embeddings.create.side_effect = AssertionError(
                "Не должен вызывать API"
            )
            result = compute_embeddings(["a", "b"], batch_size=10)
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)
        assert fake_tok.call_count == 1
        assert fake_model.call_count == 1
    finally:
        _restore_torch(saved_torch)
        _reset_local_cache()


def test_compute_embeddings_local_batches_by_batch_size():
    _reset_local_cache()
    fake_tok = _FakeTokenizer()
    fake_model = _FakeModel()

    saved_torch = _install_fake_torch()
    try:
        with patch.object(vs_mod, "_load_local_model", return_value=(fake_tok, fake_model, "cpu")):
            result = compute_embeddings(["a", "b", "c", "d", "e"], batch_size=2)
        assert len(result) == 5
        assert fake_tok.call_count == 3
        assert fake_model.call_count == 3
    finally:
        _restore_torch(saved_torch)
        _reset_local_cache()


def test_compute_embeddings_local_empty_input():
    _reset_local_cache()
    fake_tok = _FakeTokenizer()
    fake_model = _FakeModel()

    saved_torch = _install_fake_torch()
    try:
        with patch.object(vs_mod, "_load_local_model", return_value=(fake_tok, fake_model, "cpu")):
            result = compute_embeddings([])
        assert result == []
        assert fake_tok.call_count == 0
        assert fake_model.call_count == 0
    finally:
        _restore_torch(saved_torch)
        _reset_local_cache()