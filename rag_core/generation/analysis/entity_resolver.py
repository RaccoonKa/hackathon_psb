from functools import lru_cache
import json
import logging
from pathlib import Path
import re
from typing import Any
from rapidfuzz import fuzz
from rag_core import config
from rag_core.generation.types import DocRef

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path("data/manifests/indexing_manifest.json")
_FALLBACK_CHUNKS_PATH = Path("data/chunks.jsonl")

# Организационно-правовые формы и их сокращения: не несут смысла при
# сопоставлении названия контрагента.
_LEGAL_FORM_TOKENS = {
    "ооо",
    "оао",
    "зао",
    "пао",
    "ао",
    "ип",
    "нко",
    "анo",
    "фгуп",
    "гуп",
    "муп",
    "тоо",
    "llc",
    "ltd",
    "inc",
    "gmbh",
    "co",
}
_LEGAL_FORM_RE = re.compile(
    r"\b(?:ооо|оао|зао|пао|ао|ип|нко|фгуп|гуп|муп|тоо|llc|ltd|inc)\b",
    re.IGNORECASE,
)
_DOC_NUMBER_RE = re.compile(
    r"(?:№|N|No\.?)\s*([A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9\-\/_]*)"
)
_NON_ALNUM_RE = re.compile(r"[^0-9a-zа-я]+")

# Ограничения на объём «содержимого» документа для поиска контрагента.
_HAYSTACK_MAX_CHARS = 4000
_HAYSTACK_FRAGMENT_CHARS = 1500
_HEAD_CHUNKS = 3
_MIN_CORE_LEN = 5
_FUZZY_THRESHOLD = 85.0
_FUZZY_ACCEPT = 90.0


def _normalize_text(text: str) -> str:
    """Нормализация: регистр, ё, кавычки, пунктуация, пробелы."""
    cleaned = (text or "").lower().replace("ё", "е")
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _entity_core(raw_entity: str) -> str:
    """Ядро названия без организационно-правовой формы и кавычек."""
    tokens = _normalize_text(raw_entity).split()
    core = [token for token in tokens if token not in _LEGAL_FORM_TOKENS]
    return " ".join(core)


def _extract_doc_number(title: str) -> str:
    match = _DOC_NUMBER_RE.search(title or "")
    return _normalize_text(match.group(1)) if match else ""


def _new_entry(
    doc_id: str,
    title: str,
    amends_document_id: Any = None,
) -> dict[str, Any]:
    return {
        "document_id": doc_id,
        "title": title or doc_id,
        "has_effective_from": False,
        "amends_document_id": amends_document_id,
        "aliases": [],
        "_parts": [],
        "_parts_len": 0,
    }


def _finalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Добавляет нормализованные поля и убирает служебные."""
    fragments = entry.pop("_parts", [])
    entry.pop("_parts_len", None)
    haystack = _normalize_text(" ".join(fragments))
    entry["haystack"] = haystack[:6000]
    entry["aliases"] = sorted({a for a in entry.get("aliases", []) if a})
    entry["aliases_norm"] = [
        _normalize_text(alias) for alias in entry["aliases"]
    ]
    entry["title_norm"] = _normalize_text(str(entry.get("title", "")))
    entry["id_norm"] = _normalize_text(str(entry.get("document_id", "")))
    entry["number_norm"] = _extract_doc_number(str(entry.get("title", "")))
    return entry


def _append_haystack_fragment(entry: dict[str, Any], text: str) -> None:
    used = entry.get("_parts_len", 0)
    if used >= _HAYSTACK_MAX_CHARS:
        return
    fragment = (text or "").strip()[:_HAYSTACK_FRAGMENT_CHARS]
    if not fragment:
        return
    entry["_parts"].append(fragment)
    entry["_parts_len"] = used + len(fragment)


def _load_catalog_from_chunks() -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    if hasattr(config, "PROCESSED_DATA_DIR"):
        paths.append(Path(config.PROCESSED_DATA_DIR) / "chunks.jsonl")
    paths.append(_FALLBACK_CHUNKS_PATH)

    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    doc_id = chunk.get("document_id")
                    if not doc_id:
                        continue
                    doc_id = str(doc_id)
                    title = str(
                        chunk.get("document_title") or doc_id
                    )
                    entry = catalog.get(doc_id)
                    if entry is None:
                        entry = _new_entry(
                            doc_id, title, chunk.get("amends_document_id")
                        )
                        catalog[doc_id] = entry
                    if chunk.get("document_title"):
                        entry["title"] = str(chunk["document_title"])
                    if chunk.get("effective_from") is not None:
                        entry["has_effective_from"] = True
                    if (
                        entry.get("amends_document_id") is None
                        and chunk.get("amends_document_id")
                    ):
                        entry["amends_document_id"] = chunk.get(
                            "amends_document_id"
                        )
                    version_id = chunk.get("version_id")
                    if version_id:
                        entry["aliases"].append(str(version_id))
                    try:
                        chunk_index = int(chunk.get("chunk_index") or 0)
                    except (TypeError, ValueError):
                        chunk_index = 0
                    text = str(chunk.get("text") or "")
                    if chunk_index < _HEAD_CHUNKS or _LEGAL_FORM_RE.search(text):
                        _append_haystack_fragment(entry, text)
        except Exception as exc:  # noqa: BLE001 - каталог не критичен
            logger.warning(
                "Failed reading chunks catalog from %s: %s", path, exc
            )
            continue
        # Используем первый доступный источник чанков.
        if catalog:
            break
    return catalog


def _manifest_paths() -> list[Path]:
    paths: list[Path] = []
    if hasattr(config, "MANIFESTS_DIR"):
        paths.append(Path(config.MANIFESTS_DIR) / "indexing_manifest.json")
    paths.append(_MANIFEST_PATH)
    return paths


def _enrich_from_manifest(catalog: dict[str, dict[str, Any]]) -> None:
    manifest_path = next(
        (path for path in _manifest_paths() if path.exists()), None
    )
    if manifest_path is None:
        return
    chunks_doc_ids = set(catalog)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest_data = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed reading indexing_manifest.json: %s", exc)
        return

    docs_list = (
        manifest_data.get("documents", [])
        if isinstance(manifest_data, dict)
        else manifest_data
    )
    if not isinstance(docs_list, list):
        return

    for doc in docs_list:
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("document_id") or doc.get("id")
        if not doc_id:
            continue
        doc_id = str(doc_id)
        entry = catalog.get(doc_id)
        if entry is None:
            # Не добавляем документы, которых нет в chunks.jsonl: их
            # document_id не сможет отфильтровать чанки при поиске.
            if chunks_doc_ids:
                continue
            entry = _new_entry(
                doc_id,
                str(doc.get("document_title") or doc.get("title") or doc_id),
                doc.get("amends_document_id"),
            )
            catalog[doc_id] = entry
        if doc.get("effective_from"):
            entry["has_effective_from"] = True
        if entry.get("title") in (None, "", doc_id):
            entry["title"] = str(
                doc.get("document_title") or doc.get("title") or doc_id
            )


@lru_cache(maxsize=1)
def _load_document_catalog() -> dict[str, dict[str, Any]]:
    catalog = _load_catalog_from_chunks()
    _enrich_from_manifest(catalog)
    return {
        doc_id: _finalize_entry(entry)
        for doc_id, entry in catalog.items()
    }


def clear_document_catalog_cache() -> None:
    """Сброс кэша каталога (например, после переиндексации)."""
    _load_document_catalog.cache_clear()


def _match_documents(
    norm_entity: str,
    entity_core: str,
    catalog: dict[str, dict[str, Any]],
) -> tuple[list[str], list[tuple[str, float]]]:
    exact: list[str] = []
    fuzzy: list[tuple[str, float]] = []
    seen_exact: set[str] = set()

    for doc_id, meta in catalog.items():
        id_norm = meta.get("id_norm") or _normalize_text(doc_id)
        title_norm = meta.get("title_norm") or _normalize_text(
            str(meta.get("title", ""))
        )
        haystack = meta.get("haystack") or ""
        candidates = {id_norm, title_norm}
        number_norm = meta.get("number_norm")
        if number_norm:
            candidates.add(number_norm)
        for alias in meta.get("aliases_norm") or []:
            if alias:
                candidates.add(alias)
        candidates.discard("")

        # Word-boundary проверка: "альфа трейд" не должно матчиться
        # внутри "альфа трейдинг". (?<!...) и (?!...) запрещают совпадение,
        # если слева/справа есть буква или цифра.
        core_boundary_re = (
            re.compile(
                rf"(?<![0-9a-zа-я]){re.escape(entity_core)}(?![0-9a-zа-я])"
            )
            if entity_core and len(entity_core) >= _MIN_CORE_LEN
            else None
        )

        matched = False
        if norm_entity and norm_entity in candidates:
            matched = True
        elif entity_core and entity_core in candidates:
            matched = True
        elif core_boundary_re is not None and (
            core_boundary_re.search(title_norm)
            or core_boundary_re.search(haystack)
        ):
            # Название контрагента встречается в содержимом документа
            # (преамбула/реквизиты), хотя отсутствует в заголовке.
            matched = True

        if matched:
            if doc_id not in seen_exact:
                seen_exact.add(doc_id)
                exact.append(doc_id)
            continue

        targets = [title_norm]
        if haystack:
            targets.append(haystack[:1500])
        targets.extend(meta.get("aliases_norm") or [])
        probe = entity_core or norm_entity
        score = 0.0
        for target in targets:
            if not target:
                continue
            score = max(
                score,
                fuzz.partial_ratio(probe, target),
                fuzz.token_set_ratio(probe, target),
            )
        if score >= _FUZZY_THRESHOLD:
            fuzzy.append((doc_id, float(score)))

    return exact, fuzzy


def resolve_single_entity(raw_entity: str) -> DocRef:
    catalog = _load_document_catalog()
    if not catalog or not raw_entity.strip():
        return DocRef(
            raw=raw_entity, document_id=None, candidates=[], resolved=False
        )

    norm_entity = _normalize_text(raw_entity)
    entity_core = _entity_core(raw_entity)
    if not norm_entity:
        return DocRef(
            raw=raw_entity, document_id=None, candidates=[], resolved=False
        )

    exact_matches, fuzzy_candidates = _match_documents(
        norm_entity, entity_core, catalog
    )

    if len(exact_matches) == 1:
        return DocRef(
            raw=raw_entity,
            document_id=exact_matches[0],
            candidates=exact_matches,
            resolved=True,
        )
    if len(exact_matches) > 1:
        # Неоднозначность → не сужаем поиск, кандидаты идут в boost.
        logger.info(
            "Entity '%s' is ambiguous (%d documents); broad search",
            raw_entity,
            len(exact_matches),
        )
        return DocRef(
            raw=raw_entity,
            document_id=None,
            candidates=sorted(exact_matches),
            resolved=False,
        )

    fuzzy_candidates.sort(key=lambda item: item[1], reverse=True)
    matched_ids = [doc_id for doc_id, _ in fuzzy_candidates[:5]]

    if len(matched_ids) == 1 and fuzzy_candidates[0][1] >= _FUZZY_ACCEPT:
        return DocRef(
            raw=raw_entity,
            document_id=matched_ids[0],
            candidates=matched_ids,
            resolved=True,
        )

    if not matched_ids:
        logger.info(
            "Entity '%s' not resolved; fallback to broad search", raw_entity
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
    """Маршрутизация поиска.

    Пустой результат резолва никогда не означает «искать нечего»:
    возвращается ``None`` (широкий поиск). Явная причина пишется в лог.
    """
    selected: set[str] = set(user_selected_doc_ids) if user_selected_doc_ids else set()
    boost: set[str] = set()

    for ref in entities_resolved:
        if ref.resolved and ref.document_id:
            selected.add(ref.document_id)
        elif ref.candidates:
            boost.update(ref.candidates)
        else:
            logger.info(
                "Entity '%s' unresolved: selected_doc_ids не сужаются "
                "(широкий поиск)",
                ref.raw,
            )

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
    return any(
        meta.get("has_effective_from", False) for meta in catalog.values()
    )
