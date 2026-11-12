from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import Any
from rapidfuzz import fuzz
from src import config
from src.generation.types import DocRef

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path("data/manifests/indexing_manifest.json")
_FALLBACK_CHUNKS_PATH = Path("data/chunks.jsonl")


def _normalize_text(text: str) -> str:
    cleaned = text.lower().replace("ё", "е")
    cleaned = re.sub(r"[«»\"'()]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@lru_cache(maxsize=1)
def _load_document_catalog() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}

    chunks_paths: list[Path] = []
    if hasattr(config, "PROCESSED_DATA_DIR"):
        chunks_paths.append(Path(config.PROCESSED_DATA_DIR) / "chunks.jsonl")
    chunks_paths.append(_FALLBACK_CHUNKS_PATH)

    for path in chunks_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ch = json.loads(line)
                        except Exception:
                            continue
                        doc_id = ch.get("document_id")
                        if not doc_id:
                            continue
                        entry = catalog.setdefault(
                            doc_id,
                            {
                                "document_id": doc_id,
                                "title": ch.get("document_title") or doc_id,
                                "has_effective_from": False,
                                "amends_document_id": ch.get("amends_document_id"),
                            },
                        )
                        if ch.get("effective_from") is not None:
                            entry["has_effective_from"] = True
                break
            except Exception as exc:
                logger.warning("Failed reading chunks catalog from %s: %s", path, exc)

    if _MANIFEST_PATH.exists():
        try:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            docs_list = (
                manifest_data.get("documents", [])
                if isinstance(manifest_data, dict)
                else manifest_data
            )
            if isinstance(docs_list, list):
                for doc in docs_list:
                    if not isinstance(doc, dict):
                        continue
                    doc_id = doc.get("document_id") or doc.get("id")
                    if not doc_id:
                        continue
                    entry = catalog.setdefault(
                        doc_id,
                        {
                            "document_id": doc_id,
                            "title": doc.get("document_title") or doc.get("title") or doc_id,
                            "has_effective_from": bool(doc.get("effective_from")),
                            "amends_document_id": doc.get("amends_document_id"),
                        },
                    )
                    if doc.get("effective_from"):
                        entry["has_effective_from"] = True
        except Exception as exc:
            logger.warning("Failed reading indexing_manifest.json: %s", exc)

    return catalog


def resolve_single_entity(raw_entity: str) -> DocRef:
    catalog = _load_document_catalog()
    if not catalog or not raw_entity.strip():
        return DocRef(raw=raw_entity, document_id=None, candidates=[], resolved=False)

    norm_entity = _normalize_text(raw_entity)
    exact_matches: list[str] = []
    fuzzy_candidates: list[tuple[str, float]] = []

    for doc_id, meta in catalog.items():
        norm_id = _normalize_text(doc_id)
        norm_title = _normalize_text(str(meta.get("title", "")))

        if norm_entity == norm_id or norm_entity == norm_title:
            exact_matches.append(doc_id)
            continue

        if len(norm_entity) >= 3 and (norm_entity in norm_title or norm_entity in norm_id):
            exact_matches.append(doc_id)
            continue

        score = max(
            fuzz.partial_ratio(norm_entity, norm_title),
            fuzz.token_set_ratio(norm_entity, norm_title),
        )
        if score >= 78.0:
            fuzzy_candidates.append((doc_id, score))

    if len(exact_matches) == 1:
        return DocRef(
            raw=raw_entity,
            document_id=exact_matches[0],
            candidates=exact_matches,
            resolved=True,
        )
    if len(exact_matches) > 1:
        return DocRef(
            raw=raw_entity,
            document_id=None,
            candidates=sorted(exact_matches),
            resolved=False,
        )

    fuzzy_candidates.sort(key=lambda item: item[1], reverse=True)
    matched_ids = [doc_id for doc_id, _ in fuzzy_candidates[:5]]

    if len(matched_ids) == 1 and fuzzy_candidates[0][1] >= 88.0:
        return DocRef(
            raw=raw_entity,
            document_id=matched_ids[0],
            candidates=matched_ids,
            resolved=True,
        )

    return DocRef(
        raw=raw_entity,
        document_id=None,
        candidates=matched_ids,
        resolved=False,
    )


def resolve_entities(entities_raw: list[str]) -> list[DocRef]:
    resolved_list: list[DocRef] = []
    seen: set[str] = set()
    for raw in entities_raw:
        cleaned = raw.strip()
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        resolved_list.append(resolve_single_entity(cleaned))
    return resolved_list


def extract_doc_routing_sets(
    entities_resolved: list[DocRef],
    user_selected_doc_ids: set[str] | None = None,
) -> tuple[set[str] | None, set[str] | None]:
    selected: set[str] = set(user_selected_doc_ids) if user_selected_doc_ids else set()
    boost: set[str] = set()

    for ref in entities_resolved:
        if ref.resolved and ref.document_id:
            selected.add(ref.document_id)
        elif len(ref.candidates) >= 2:
            boost.update(ref.candidates)
        elif len(ref.candidates) == 1:
            boost.add(ref.candidates[0])

    return (selected if selected else None), (boost if boost else None)


def has_effective_dates_in_corpus(entities_resolved: list[DocRef]) -> bool:
    catalog = _load_document_catalog()
    if not catalog:
        return False

    target_ids = [r.document_id for r in entities_resolved if r.document_id]
    if target_ids:
        return any(
            catalog.get(doc_id, {}).get("has_effective_from", False)
            for doc_id in target_ids
        )
    return any(meta.get("has_effective_from", False) for meta in catalog.values())