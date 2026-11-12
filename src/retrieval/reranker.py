"""Два backend-а reranker-а: локальный Transformers (LOCAL) и HTTP API (SUBMIT).

Выбор backend-а приходит из src.config.RERANKER_BACKEND. USE_SUBMIT здесь
не читается и не проверяется: переключение режимов централизовано в config.py.
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from src import config

logger = logging.getLogger(__name__)

# Модель тяжёлая: загружаем один раз на процесс, только при первом rerank.
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None
_LOCAL_TRUE_TOKEN_ID: Optional[int] = None
_LOCAL_FALSE_TOKEN_ID: Optional[int] = None
_LOCAL_PREFIX_TOKENS: Optional[List[int]] = None
_LOCAL_SUFFIX_TOKENS: Optional[List[int]] = None
_MODEL_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()

# Специализированная инструкция для поиска положений в юридических документах.
_RERANKER_INSTRUCTION = (
    "Найди фрагменты юридических документов, которые непосредственно отвечают "
    "на вопрос. Учитывай стороны и реквизиты документов, номера пунктов, "
    "сроки, суммы, права, обязанности, условия и исключения."
)

_PREFIX = (
    "<|im_start|>system\n"
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def _load_local_model() -> Tuple[Any, Any, int, int, List[int], List[int]]:
    """Ленивая загрузка Qwen3-Reranker-8B с учётом доступного устройства."""
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    global _LOCAL_TRUE_TOKEN_ID, _LOCAL_FALSE_TOKEN_ID
    global _LOCAL_PREFIX_TOKENS, _LOCAL_SUFFIX_TOKENS

    if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None:
        return (
            _LOCAL_TOKENIZER,
            _LOCAL_MODEL,
            _LOCAL_TRUE_TOKEN_ID,
            _LOCAL_FALSE_TOKEN_ID,
            _LOCAL_PREFIX_TOKENS,
            _LOCAL_SUFFIX_TOKENS,
        )

    with _MODEL_LOAD_LOCK:
        if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None:
            return (
                _LOCAL_TOKENIZER,
                _LOCAL_MODEL,
                _LOCAL_TRUE_TOKEN_ID,
                _LOCAL_FALSE_TOKEN_ID,
                _LOCAL_PREFIX_TOKENS,
                _LOCAL_SUFFIX_TOKENS,
            )

        # Импорты намеренно локальные: в SUBMIT этот код не исполняется,
        # поэтому torch/transformers не нужны в submit/requirements.txt.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = config.RERANKER_LOCAL_PATH
        if not model_path.is_dir():
            raise FileNotFoundError(
                "Каталог локального Qwen3-Reranker-8B не найден: "
                f"{model_path}. Проверьте RERANKER_LOCAL_PATH."
            )
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"В каталоге модели отсутствует config.json: {model_path}"
            )

        logger.info("Загрузка локального Qwen3-Reranker-8B из %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            padding_side="left",
        )
        tokenizer.padding_side = "left"

        load_kwargs: Dict[str, Any] = {"local_files_only": True}
        if torch.cuda.is_available():
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            use_bf16 = bool(is_bf16_supported and is_bf16_supported())
            load_kwargs["torch_dtype"] = (
                torch.bfloat16 if use_bf16 else torch.float16
            )
            # accelerate требуется только для автоматического размещения на CUDA.
            load_kwargs["device_map"] = "auto"
        else:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                # На Apple Silicon используем FP16 и MPS.
                load_kwargs["torch_dtype"] = torch.float16
            # На CPU оставляем dtype из config.json модели (обычно BF16),
            # чтобы не раздувать 8B-веса до FP32 без необходимости.

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **load_kwargs,
        )

        if not torch.cuda.is_available():
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                model = model.to("mps")
            else:
                model = model.to("cpu")

        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        if token_false_id is None or token_true_id is None:
            raise RuntimeError("Не удалось получить yes/no token IDs Qwen3-Reranker")
        if token_false_id == token_true_id:
            raise RuntimeError("Токены yes и no Qwen3-Reranker имеют одинаковый ID")

        prefix_tokens = tokenizer.encode(_PREFIX, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(_SUFFIX, add_special_tokens=False)

        _LOCAL_TOKENIZER = tokenizer
        _LOCAL_MODEL = model
        _LOCAL_TRUE_TOKEN_ID = int(token_true_id)
        _LOCAL_FALSE_TOKEN_ID = int(token_false_id)
        _LOCAL_PREFIX_TOKENS = prefix_tokens
        _LOCAL_SUFFIX_TOKENS = suffix_tokens

        input_device = model.get_input_embeddings().weight.device
        logger.info(
            "Локальный Qwen3-Reranker-8B загружен: device=%s, dtype=%s",
            input_device,
            next(model.parameters()).dtype,
        )

    return (
        _LOCAL_TOKENIZER,
        _LOCAL_MODEL,
        _LOCAL_TRUE_TOKEN_ID,
        _LOCAL_FALSE_TOKEN_ID,
        _LOCAL_PREFIX_TOKENS,
        _LOCAL_SUFFIX_TOKENS,
    )


def _rerank_with_transformers(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Считает Qwen yes/no relevance score для каждой пары query-document."""
    import torch

    (
        tokenizer,
        model,
        token_true_id,
        token_false_id,
        prefix_tokens,
        suffix_tokens,
    ) = _load_local_model()

    max_content_length = (
        config.RERANKER_MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens)
    )
    if max_content_length <= 0:
        raise ValueError("RERANKER_MAX_LENGTH слишком мал для шаблона Qwen")

    scored: List[Dict[str, Any]] = []
    batch_size = max(1, config.RERANKER_BATCH_SIZE)

    # Один процесс использует одну копию 8B-модели; сериализуем inference,
    # чтобы параллельные вопросы не создавали несколько пиковых batch в VRAM.
    with _INFERENCE_LOCK:
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            pairs = [
                "<Instruct>: "
                + _RERANKER_INSTRUCTION
                + "\n<Query>: "
                + query
                + "\n<Document>: "
                + (candidate.get("embedding_text") or candidate["text"])
                for candidate in batch
            ]

            encoded = tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_content_length,
            )
            input_ids = [
                prefix_tokens + token_ids + suffix_tokens
                for token_ids in encoded["input_ids"]
            ]
            inputs = tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
                max_length=config.RERANKER_MAX_LENGTH,
            )
            input_device = model.get_input_embeddings().weight.device
            inputs = {key: value.to(input_device) for key, value in inputs.items()}

            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits[:, -1, :]
                yes_logits = logits[:, token_true_id]
                no_logits = logits[:, token_false_id]
                two_class_logits = torch.stack([no_logits, yes_logits], dim=1)
                scores = torch.log_softmax(two_class_logits, dim=1)[:, 1].exp()

            for candidate, score in zip(batch, scores.detach().cpu().tolist()):
                item = dict(candidate)
                # Совместимо с прежним API-backend: downstream получает rerank
                # score в стандартном поле score (в т.ч. threshold_filter).
                item["score"] = float(score)
                scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _rerank_with_api(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Существующий HTTP backend для платформенного SUBMIT без изменений API."""
    import requests

    url = f"{config.RERANKER_API_BASE_URL.rstrip('/')}/rerank"
    headers = {
        "Authorization": f"Bearer {config.RERANKER_API_KEY}",
        "Content-Type": "application/json",
    }
    doc_texts = [
        candidate.get("embedding_text") or candidate["text"]
        for candidate in candidates
    ]
    payload = {
        "model": config.RERANKER_MODEL,
        "query": query,
        "documents": doc_texts,
        "top_n": min(top_k, len(candidates)),
    }
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=config.RERANKER_TIMEOUT,
    )
    if response.status_code != 200:
        logger.warning(
            "Реранкер вернул HTTP %s; сохранён исходный порядок",
            response.status_code,
        )
        return []

    data = response.json()
    reranked: List[Dict[str, Any]] = []
    for item in data.get("results", []):
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < len(candidates):
            candidate_copy = dict(candidates[index])
            raw_score = item.get("relevance_score", item.get("score"))
            if raw_score is not None:
                candidate_copy["score"] = float(raw_score)
            reranked.append(candidate_copy)

    return reranked[:top_k]


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Ранжирует кандидатов выбранным в config backend-ом.

    Если локальная модель или HTTP API недоступны, сохраняет исходный
    порядок Weighted RRF — такое fallback-поведение уже было в проекте.
    """
    if not candidates:
        return []

    if len(candidates) <= 1:
        return candidates[:top_k]

    try:
        if config.RERANKER_BACKEND == "transformers":
            reranked = _rerank_with_transformers(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        elif config.RERANKER_BACKEND == "api":
            reranked = _rerank_with_api(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )
        else:
            raise ValueError(
                "Неизвестный RERANKER_BACKEND: "
                f"{config.RERANKER_BACKEND!r}"
            )

        if reranked:
            return reranked[:top_k]
    except Exception as exc:  # noqa: BLE001 - reranker необязателен
        logger.warning(
            "Реранкер (%s) недоступен, сохранён исходный RRF-порядок: %s",
            config.RERANKER_BACKEND,
            exc,
            exc_info=True,
        )

    return candidates[:top_k]
