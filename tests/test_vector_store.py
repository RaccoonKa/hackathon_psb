from unittest.mock import MagicMock, patch

from src.retrieval import vector_store as vs_mod
from src.retrieval.vector_store import (
    _fallback_embedding,
    compute_embeddings,
    get_query_embedding,
    prepare_flat_metadata,
)
from tests._common import make_chunk_dict


def test_fallback_embedding_deterministic():
    a = _fallback_embedding("привет")
    b = _fallback_embedding("привет")
    assert a == b


def test_fallback_embedding_differs_for_different_text():
    a = _fallback_embedding("привет")
    b = _fallback_embedding("другой")
    assert a != b


def test_fallback_embedding_dim():
    e = _fallback_embedding("привет")
    assert len(e) == 1024


def test_fallback_embedding_custom_dim():
    e = _fallback_embedding("привет", dim=64)
    assert len(e) == 64


def test_fallback_embedding_returns_floats():
    e = _fallback_embedding("x")
    for v in e[:5]:
        assert isinstance(v, float)


def test_prepare_flat_metadata_keeps_valid_fields():
    chunk = make_chunk_dict(
        document_id="d1", version_id="v1", chunk_index=0,
        document_title="T", document_type="contract",
        scope="global", effective_dates_known=True,
        effective_from="2024-01-01",
    )
    meta = prepare_flat_metadata(chunk)
    assert meta["document_id"] == "d1"
    assert meta["version_id"] == "v1"
    assert meta["document_type"] == "contract"
    assert meta["effective_from"] == "2024-01-01"
    assert meta["provision_id"] == "1.1"


def test_prepare_flat_metadata_drops_none():
    chunk = make_chunk_dict(chat_id=None, effective_to=None, amends_document_id=None)
    meta = prepare_flat_metadata(chunk)
    assert "chat_id" not in meta
    assert "effective_to" not in meta
    assert "amends_document_id" not in meta


def test_prepare_flat_metadata_preserves_bool():
    chunk = make_chunk_dict(effective_dates_known=False)
    meta = prepare_flat_metadata(chunk)
    assert meta["effective_dates_known"] is False


def test_prepare_flat_metadata_preserves_int():
    chunk = make_chunk_dict(chunk_index=42)
    meta = prepare_flat_metadata(chunk)
    assert meta["chunk_index"] == 42
    assert isinstance(meta["chunk_index"], int)


def test_prepare_flat_metadata_converts_non_primitive():
    chunk = make_chunk_dict(document_id="d")
    meta = prepare_flat_metadata(chunk)
    assert isinstance(meta["document_id"], str)


def test_prepare_flat_metadata_no_provision():
    chunk = make_chunk_dict()
    chunk["structure"] = {"provision_id": None, "path": []}
    meta = prepare_flat_metadata(chunk)
    assert "provision_id" not in meta


def test_prepare_flat_metadata_no_structure_key():
    chunk = make_chunk_dict()
    del chunk["structure"]
    meta = prepare_flat_metadata(chunk)
    assert "provision_id" not in meta


def test_compute_embeddings_fallback():
    with patch.object(vs_mod, "_load_local_model", return_value=(None, None, None)), \
         patch.object(vs_mod, "USE_SUBMIT", False), \
         patch.object(vs_mod, "ALLOW_HASH_FALLBACK", True), \
         patch.object(vs_mod, "_get_api_client") as mock_get_client:
        mock_get_client.return_value.embeddings.create.side_effect = Exception("no api")
        result = compute_embeddings(["a", "b"])
    assert len(result) == 2
    assert len(result[0]) == 1024


def test_compute_embeddings_uses_api():
    first = MagicMock(embedding=[0.1] * 1024)
    second = MagicMock(embedding=[0.2] * 1024)
    first.index = 0
    second.index = 1
    mock_data = [first, second]
    mock_resp = MagicMock(data=mock_data)
    with patch.object(vs_mod, "_load_local_model", return_value=(None, None, None)), \
         patch.object(vs_mod, "_get_api_client") as mock_get_client:
        mock_get_client.return_value.embeddings.create.return_value = mock_resp
        result = compute_embeddings(["a", "b"])
    assert len(result) == 2
    assert result[0][0] == 0.1
    assert result[1][0] == 0.2


def test_get_query_embedding_returns_vector():
    with patch.object(vs_mod, "compute_embeddings", return_value=[[0.5] * 1024]):
        vec = get_query_embedding("q")
    assert len(vec) == 1024
    assert vec[0] == 0.5


def test_get_query_embedding_calls_with_single():
    with patch.object(vs_mod, "compute_embeddings", return_value=[[0.1]]) as mock_compute:
        get_query_embedding("hello")
    mock_compute.assert_called_once_with(["hello"])