import os
import sys
from pathlib import Path

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("RERANKER_API_KEY", "test-key")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core import config as _cfg

_cfg.USE_STUB_LLM = False
_cfg.USE_STUB_RAG = False
_cfg.USE_STUB_RERANKER = False

from rag_core.generation.types import RetrievedChunk

RAW_DIR = ROOT / "data" / "raw"
_CACHE = {}


def raw_files():
    if "files" not in _CACHE:
        _CACHE["files"] = sorted(RAW_DIR.glob("**/*.txt")) if RAW_DIR.exists() else []
    return _CACHE["files"]


def require_data():
    if not raw_files():
        from unittest import SkipTest
        raise SkipTest(f"Нет исходных данных в {RAW_DIR}")


def all_docs():
    if "docs" not in _CACHE:
        require_data()
        from rag_core.indexing.parser import load_documents_with_metadata
        _CACHE["docs"] = load_documents_with_metadata(RAW_DIR)
    return _CACHE["docs"]


def doc_by_filename(fragment):
    for d in all_docs():
        if fragment in d["metadata"]["source_filename"]:
            return d
    return None


def make_chunk_dict(**kwargs):
    base = {
        "chunk_id": "chunk_1",
        "text": "Текст",
        "embedding_text": "Документ. Текст",
        "document_id": "doc_1",
        "version_id": "doc_1_v1",
        "chunk_index": 0,
        "document_title": "Документ",
        "document_type": "contract",
        "scope": "global",
        "chat_id": None,
        "structure": {"provision_id": "1.1", "path": ["Раздел 1"]},
        "location": {
            "start_line": 1, "end_line": 1, "char_start": 0, "char_end": 6, "page": None,
        },
        "score": 0.5,
        "effective_from": None,
        "effective_to": None,
        "effective_dates_known": True,
        "amends_document_id": None,
        "source_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "chunker_version": "structural-1.0",
    }
    base.update(kwargs)
    return base


def make_retrieved_chunk(**kwargs):
    data = {
        "chunk_id": "chunk_1",
        "text": "Текст",
        "document_id": "doc_1",
        "version_id": "doc_1_v1",
        "document_title": "Документ",
        "provision_id": "1.1",
        "structure_path": ["Раздел 1"],
        "score": 0.5,
        "effective_from": None,
        "effective_to": None,
        "effective_dates_known": True,
        "amends_document_id": None,
        # Назначение/статус источника (P1): нейтральные значения по умолчанию
        # сохраняют обратную совместимость существующих вызовов.
        "document_type": "contract",
        "source_kind": None,
        "version_status": None,
        "validity_uncertain": False,
        "effective_date_mode": None,
    }
    data.update(kwargs)
    return RetrievedChunk(**data)


def make_query_spec(**kwargs):
    from rag_core.generation.types import Ambiguity, QuerySpec, TaskType

    data = {
        "intent": "semantic",
        "task_type": TaskType.EXPLAIN,
        "entities_raw": [],
        "entities_resolved": [],
        "article_refs": [],
        "dates": [],
        "event_date": None,
        "premises": [],
        "required_slots": ["general"],
        "assumptions": [],
        "ambiguity": Ambiguity.LOW,
        "needs_clarification": False,
        "decomposition": [],
        "rewritten_queries_bm25": ["запрос"],
        "rewritten_queries_dense": ["запрос"],
        "history_resolved": False,
        "degradation_reason": None,
        "required_source_kinds": [],
        "date_precision": "none",
    }
    data.update(kwargs)
    return QuerySpec(**data)


def make_retrieval_task(**kwargs):
    from rag_core.generation.types import RetrievalTask

    data = {
        "query": "запрос",
        "mode": "hybrid",
        "purpose": "test_hybrid",
        "selected_doc_ids": None,
        "boost_doc_ids": None,
        "event_date": None,
        "top_k": 5,
        "intent": None,
        "source_kinds": None,
    }
    data.update(kwargs)
    return RetrievalTask(**data)


def make_policy_verdict(**kwargs):
    from rag_core.generation.types import PolicyVerdict

    data = {
        "premise_mismatch": False,
        "premise_details": [],
        "injection_flagged": False,
        "refusal_forbidden": False,
        "premise_statuses": [],
    }
    data.update(kwargs)
    return PolicyVerdict(**data)


def make_evidence_bundle(**kwargs):
    from rag_core.generation.types import EvidenceBundle

    data = {
        "chunks": [],
        "slot_map": {},
        "contradictions": [],
        "version_issues": [],
        "missing_source_kinds": [],
        "generation_warnings": [],
    }
    data.update(kwargs)
    return EvidenceBundle(**data)


def make_verification_report(**kwargs):
    from rag_core.generation.types import VerificationReport, VerificationVerdict

    data = {
        "batch_status": [],
        "citation_errors": [],
        "injection_signals": [],
        "supported_locked": [],
        "final_verdict": VerificationVerdict.ACCEPT,
        "answer_to_question": None,
        "verifier_unavailable": False,
        "auto_accept_blocked": False,
        "claims_synthesized": False,
        "source_role_issues": [],
    }
    data.update(kwargs)
    return VerificationReport(**data)


def make_premise_status_entry(**kwargs):
    from rag_core.generation.types import PremiseStatus, PremiseStatusEntry

    data = {
        "text": "предпосылка",
        "kind": "fact",
        "value": None,
        "status": PremiseStatus.UNKNOWN,
        "reason": None,
        "evidence_source_id": None,
        "conflicting_value": None,
    }
    data.update(kwargs)
    return PremiseStatusEntry(**data)


def make_bm25_stats(
    results,
    *,
    candidates_before=None,
    date_allowed=None,
    allowed=None,
):
    """Диагностика в формате ``search_with_stats`` для моков BM25."""
    before = len(results) if candidates_before is None else candidates_before
    after_date = (
        before
        if date_allowed is None
        else sum(1 for cid, _score in results if cid in date_allowed)
    )
    after_document = (
        before
        if allowed is None
        else sum(1 for cid, _score in results if cid in allowed)
    )
    return {
        "candidates_before_filters": before,
        "candidates_after_date_filter": after_date,
        "candidates_after_document_filter": after_document,
        "returned_count": len(results),
    }


def filler_corpus(n=10, term="срок", base_prefix="doc"):
    chunks = []
    for i in range(n):
        chunks.append(make_chunk_dict(
            chunk_id=f"{base_prefix}{i}",
            text=f"фоновый текст {i}",
            embedding_text=f"фоновый документ {i} общий текст",
        ))
    chunks.append(make_chunk_dict(
        chunk_id="target",
        text=f"целевой текст с {term}",
        embedding_text=f"целевой документ {term} уникальный",
    ))
    return chunks