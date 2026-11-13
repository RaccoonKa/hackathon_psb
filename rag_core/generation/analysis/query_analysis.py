import json
import logging
import re
import time
from typing import Callable, Optional
from pydantic import ValidationError
from rag_core import config
from rag_core.generation.analysis.entity_resolver import (
    has_effective_dates_in_corpus,
    resolve_entities,
)
from rag_core.generation.errors import LLMError, LLMFormatError
from rag_core.generation.llm.client import llm_complete_with_usage
from rag_core.generation.prompts import history as history_prompts
from rag_core.generation.prompts import query_analysis as qa_prompts
from rag_core.generation.types import (
    Ambiguity,
    Premise,
    QueryAnalysisSchema,
    QuerySpec,
    SubQuestion,
    TaskType,
    TraceStep,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CONTRACT_NUM_RE = re.compile(r"(?:№\s*|No\s*|договор[ауеом]?\s+)([А-Яа-яA-Za-z0-9\-\/]+)", re.IGNORECASE)
_QUOTED_ENTITY_RE = re.compile(r"[«\"]([^»\"]+)[»\"]")
_ARTICLE_EXTRACT_RE = re.compile(
    r"(?:стать[яеиюй]|ст\.|пункт[ауеом]?|п\.|част[ьи]|ч\.)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_EXPLICIT_POINT_OR_QUOTE_RE = re.compile(
    r"(?:\bстать[яеиюй]\s*\d|\bст\.\s*\d|\bпункт\w*\s*\d|\bп\.\s*\d|\bраздел\w*\s*\d|\bч\.\s*\d|дословн\w+\s+цитат|процитируй)",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_RU_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(20\d{2})\b")
_TEXT_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(январ\w*|феврал\w*|март\w*|апрел\w*|ма[яе]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\s+(20\d{2})(?:\s*(?:года|г\.?))?\b",
    re.IGNORECASE,
)
_RU_MONTH_MAP: dict[str, str] = {
    "январ": "01",
    "феврал": "02",
    "март": "03",
    "апрел": "04",
    "ма": "05",
    "июн": "06",
    "июл": "07",
    "август": "08",
    "сентябр": "09",
    "октябр": "10",
    "ноябр": "11",
    "декабр": "12",
}

_NON_AMOUNT_RENT_MENTIONS_RE = re.compile(
    r"(?:просрочк\w*|срок\w*\s+(?:внесени\w*|оплат\w*|уплат\w*)|дата\s+(?:внесени\w*|оплат\w*)|до\s+какого\s+числа\s+[^\?]*?внес\w*|пен[яиюей]\s+за\s+[^\?]*?)\s+арендн\w+\s+плат\w*",
    re.IGNORECASE,
)

_SLOT_DETECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("security_deposit", re.compile(r"обеспечительн\w+\s+платеж|гарантийн\w+\s+взнос|депозит\w*", re.IGNORECASE)),
    ("penalty", re.compile(r"\bпен[яиюей]\b|неустойк\w+|штраф\w*", re.IGNORECASE)),
    ("accrual_period", re.compile(r"за\s+какой\s+период|период\s+начислен|с\s+какого\s+момента\s+начисля", re.IGNORECASE)),
    ("liability_cap", re.compile(r"ограничен\w*\s+(?:ли\s+)?ответственност|чем\s+(?:именно\s+)?ограничен|предел\w*\s+ответственност", re.IGNORECASE)),
    ("payment_deadline", re.compile(r"до\s+какого\s+числа|срок\w*\s+(?:оплат\w*|внесени\w*|расчет\w*)|когда\s+(?:нужно\s+)?(?:оплач|вноси)", re.IGNORECASE)),
    ("probation_period", re.compile(r"испытательн\w+\s+срок|срок\w*\s+испытани", re.IGNORECASE)),
    ("overtime_limit", re.compile(r"сколько\s+часов\s+сверхурочн|продолжительност\w+\s+сверхурочн|120\s+часов", re.IGNORECASE)),
    ("overtime_pay", re.compile(r"оплач\w+\s+сверхурочн|оплат\w+\s+сверхурочн", re.IGNORECASE)),
    ("approval_process", re.compile(r"согласован\w+|одобрени\w+\s+генеральн|визирован|кто\s+должен\s+запустить\s+согласован", re.IGNORECASE)),
    ("termination", re.compile(r"расторжен\w+|прекращен\w+\s+договор|односторонн\w+\s+отказ", re.IGNORECASE)),
    ("tax_rate", re.compile(r"налог\w+\s+ставк|ндс|налог\s+на\s+прибыл", re.IGNORECASE)),
    ("delivery_terms", re.compile(r"адрес\w*\s+доставк|доставк\w+\s+и\s+упаковк|услови\w+\s+доставк", re.IGNORECASE)),
    ("acceptance_terms", re.compile(r"провери\w+\s+качеств|приемк\w+\s+товар|обнаружен\w+\s+недостатк|приложи\w+\s+к\s+претензи", re.IGNORECASE)),
    ("warranty_period", re.compile(r"гарантийн\w+\s+срок|на\s+сколько\s+гаранти|гаранти\w+\s+по\s+договор", re.IGNORECASE)),
    ("prepayment", re.compile(r"\bаванс\w*|\bпредоплат\w*", re.IGNORECASE)),
    ("return_procedure", re.compile(r"сдать\s+ключ|возврат\w*\s+помещен|оформля\w+\s+акт", re.IGNORECASE)),
    ("document_handover", re.compile(r"оригинал\w*\s+договор|переда\w+\s+в\s+юр\w*отдел|из\s+архива", re.IGNORECASE)),
    ("consent_withdrawal", re.compile(r"отозвал\w*\s+согласи|персональн\w+\s+данн|удалить\s+вообще\s+вс[её]", re.IGNORECASE)),
    ("seasonal_work", re.compile(r"сезонн\w+\s+работ|срочн\w+\s+трудов\w+\s+договор", re.IGNORECASE)),
]

_SLOT_QUERY_TEMPLATES: dict[str, str] = {
    "security_deposit": "размер обеспечительного платежа при подписании договора месячной арендной платы пункт 3.5",
    "penalty": "размер пеней неустойки процент за просрочку исполнения обязательств пункт 5.1",
    "accrual_period": "период начисления пеней за каждый день просрочки исполнения обязательств",
    "liability_cap": "ограничение ответственности сторон предел ответственности реальный ущерб упущенная выгода",
    "rent_amount": "размер арендной платы составляет рублей в месяц включая НДС пункт 3.1",
    "payment_deadline": "срок внесения арендной платы оплаты не позднее 5 числа текущего месяца порядок расчетов пункт 3.2",
    "probation_period": "максимальный срок испытания при приеме на работу главный бухгалтер",
    "overtime_limit": "продолжительность сверхурочной работы лимит часов в год",
    "overtime_pay": "размер оплаты сверхурочной работы первые два часа последующие часы",
    "approval_process": "порядок согласования договоров одобрение генеральным директором сумма порог",
    "termination": "порядок расторжения прекращения договора уведомление",
    "tax_rate": "налоговая ставка НДС налог на прибыль",
    "delivery_terms": "адрес доставки стоимость упаковки и доставки в цену товара",
    "acceptance_terms": "срок приемки товара по качеству претензия акт о выявленных недостатках",
    "warranty_period": "гарантийный срок на товар гарантийный талон производителя",
    "prepayment": "условия предоплаты оплата авансом счет поставщика",
    "return_procedure": "возврат помещения при прекращении договора аренды акт возврата передача ключей",
    "document_handover": "выдача и передача оригиналов договоров в юридический отдел срок рабочих дней",
    "consent_withdrawal": "отзыв согласия на обработку персональных данных продолжение обработки без согласия",
    "seasonal_work": "срочный трудовой договор для выполнения сезонных работ",
}

_COMPOUND_SPLIT_RE = re.compile(
    r"(?:\?\s*(?:Если\s+да,?\s*то\s+|А\s+)|\s+и\s+(?:за\s+какой|до\s+какого|в\s+каком|какой|сколько|как\s+|чем\s+|срок\s+))",
    re.IGNORECASE,
)

_PREMISE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("amount", re.compile(r"(\d[\d\s]{2,12}\s*(?:руб(?:лей|\.)?|млн\s*руб(?:лей|\.)?))", re.IGNORECASE)),
    ("term", re.compile(r"((?:на\s+)?(?:\d+|один|два|три|четыре|пять|шесть|семь|восемь|десять)\s+(?:рабочих|календарных)\s+дн(?:я|ей))", re.IGNORECASE)),
]


def _asks_rent_amount(query: str) -> bool:
    stripped = _NON_AMOUNT_RENT_MENTIONS_RE.sub(" ", query.lower())
    return bool(
        re.search(
            r"(?:какая|каков\w*|размер\w*|стоимост\w*|сколько|почему)\s+арендн\w+\s+плат|арендн\w+\s+плат\w*\s+\d",
            stripped,
        )
    )


_RESOLVED_PREFIX_RE = re.compile(
    r"^(?:самодостаточный\s+поисковый\s+запрос|переформулированный\s+запрос|запрос|вопрос)\s*:\s*",
    re.IGNORECASE,
)


_INJECTION_TOKENS_RE = re.compile(
    r"(?:<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<system>|</system>|<\|endoftext\|>|---\s*system)",
    re.IGNORECASE,
)

_ANAPHORA_BACKWARD_RE = re.compile(
    r"(?:\b(?:по\s+нему|по\s+ней|по\s+ним|к\s+нему|к\s+ней|к\s+ним|в\s+нем|в\s+ней|в\s+них|о\s+нем|о\s+ней|о\s+них|с\s+ним|с\s+ней|с\s+ними)\b"
    r"|\b(?:этот|эта|это|эти|этого|этой|этому|этим|этом|этих|этими)\b"
    r"|\b(?:тот|та|то|те|того|той|тому|тем|том|тех|теми)\s+(?:же\s+)?(?:договор|соглашен|пункт|акт|документ)\b"
    r"|\b(?:ранее|предыдущ\w+|выше\w+|тогда|в\s+тот\s+момент|в\s+то\s+время)\b"
    r"|\b(?:он|она|оно|они|его|е[её]|их|ему|ей|им|нем|ней|них|нему|него|нее|не[её]|нею|ними)\b"
    r"|^(?:а|но|и)\s+(?:в\s+какой|какой|какая|какое|какие|сколько|до\s+какого|каков\w*)\b)",
    re.IGNORECASE,
)

_TRUE_ANAPHORA_PRONOUNS_RE = re.compile(
    r"(?:\b(?:по\s+нему|по\s+ней|по\s+ним|к\s+нему|к\s+ней|к\s+ним|в\s+нем|в\s+ней|в\s+них|о\s+нем|о\s+ней|о\s+них|с\s+ним|с\s+ней|с\s+ними)\b"
    r"|\b(?:этот|эта|это|эти|этого|этой|этому|этим|этом|этих|этими)\s+(?:договор|соглашен|пункт|акт|документ)\b"
    r"|\b(?:тот|та|то|те|того|той|тому|тем|том|тех|теми)\s+(?:же\s+)?(?:договор|соглашен|пункт|акт|документ)\b"
    r"|\b(?:ранее|предыдущ\w+|выше\w+|тогда|в\s+тот\s+момент|в\s+то\s+время)\b)",
    re.IGNORECASE,
)

_EXPLICIT_LAW_OR_DOC_RE = re.compile(
    r"(?:\b(?:коап|гк|нк|тк|ук)\s+рф\b|\bфз\b|\bфедеральн\w+\s+закон\b|\bзакон\s+(?:рф|о\b)|договор\w*\s+(?:№|номер|[«\"А-Яа-я0-9\-\/])|приказ\w*\s+(?:№|номер))",
    re.IGNORECASE,
)


def extract_dates_from_text(text: str) -> list[str]:
    """Извлекает даты в формате YYYY-MM-DD из текста (ISO, DD.MM.YYYY или текстовый месяц)."""
    dates: list[str] = []
    if not text:
        return dates
    for match in _ISO_DATE_RE.findall(text):
        if match not in dates:
            dates.append(match)
    for day, month, year in _RU_DATE_RE.findall(text):
        try:
            iso = f"{year}-{int(month):02d}-{int(day):02d}"
            if iso not in dates:
                dates.append(iso)
        except ValueError:
            pass
    for match in _TEXT_DATE_RE.finditer(text):
        day_str, month_str, year_str = match.group(1), match.group(2).lower(), match.group(3)
        month_num = None
        for prefix, num in _RU_MONTH_MAP.items():
            if month_str.startswith(prefix):
                month_num = num
                break
        if month_num:
            try:
                iso = f"{year_str}-{int(month_num):02d}-{int(day_str):02d}"
                if iso not in dates:
                    dates.append(iso)
            except ValueError:
                pass
    return dates


def extract_date_from_history(history: list[dict] | None) -> str | None:
    """Извлекает дату события из последних реплик диалога (от новых к старым)."""
    if not history:
        return None
    # Сначала проверяем реплики пользователя от самых свежих к старым
    for turn in reversed(history):
        if not isinstance(turn, dict):
            continue
        if str(turn.get("role") or "").lower() == "user":
            content = str(turn.get("content") or "")
            dates = extract_dates_from_text(content)
            if dates:
                return dates[0]
    # Если в репликах пользователя нет, проверяем все реплики (включая ассистента)
    for turn in reversed(history):
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content") or "")
        dates = extract_dates_from_text(content)
        if dates:
            return dates[0]
    return None


def is_independent_query(query: str) -> bool:
    """Проверяет, является ли запрос полностью самостоятельным (смена темы / самодостаточный вопрос)."""
    q = query.strip()
    if not q:
        return False
    # Если вопрос явно ссылается на закон или конкретный документ и не содержит анафор к прошлому:
    if _EXPLICIT_LAW_OR_DOC_RE.search(q):
        if not _TRUE_ANAPHORA_PRONOUNS_RE.search(q):
            return True
    # Если есть явные анафоры (местоимения, неполные отсылки назад) — запрос зависим
    if _ANAPHORA_BACKWARD_RE.search(q):
        return False
    return False


def _format_history_turns(history: list[dict]) -> list[str]:
    max_turns = max(1, int(getattr(config, "MAX_HISTORY_TURNS", 6)))
    max_chars = max(100, int(getattr(config, "HISTORY_TURN_MAX_CHARS", 1200)))
    last_turns = history[-max_turns:]
    formatted: list[str] = []
    for turn in last_turns:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        # Защита от prompt injection control tokens
        content = _INJECTION_TOKENS_RE.sub("", content).strip()
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        # Заменяем переносы строк пробелами для защиты структуры промпта
        content = " ".join(content.split())
        role_raw = str(turn.get("role") or "user").strip().lower()
        role_label = "Пользователь" if role_raw == "user" else "Ассистент"
        formatted.append(f"{role_label}: {content}")
    return formatted


def _resolve_history_with_usage(
    query: str,
    history: list[dict],
    timeout: float | None = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
) -> tuple[str, bool, int, int]:
    if not history:
        return query, False, 0, 0

    # Детекция смены темы / самостоятельного вопроса без анафор
    if is_independent_query(query):
        return query, False, 0, 0

    context_parts = _format_history_turns(history)
    if not context_parts:
        return query, False, 0, 0

    fallback_query = f"{query} (Контекст диалога: {' | '.join(context_parts)})"

    if config.USE_STUB_LLM or config.LLM_MODE == config.LLM_MODE_STUB:
        return fallback_query, True, 0, 0

    history_text = "\n".join(context_parts)
    messages = [
        {"role": config.ROLE_SYSTEM, "content": history_prompts.HISTORY_RESOLUTION_SYSTEM},
        {
            "role": config.ROLE_USER,
            "content": history_prompts.HISTORY_RESOLUTION_USER.format(
                history_text=history_text,
                query=query,
            ),
        },
    ]

    try:
        raw_output, p_tok, c_tok = llm_complete_with_usage(
            messages=messages,
            model=config.LLM_MODEL,
            temperature=history_prompts.HISTORY_RESOLUTION_TEMPERATURE,
            max_tokens=history_prompts.HISTORY_RESOLUTION_MAX_TOKENS,
            timeout=timeout,
        )
        if on_llm_success:
            on_llm_success()

        resolved = _RESOLVED_PREFIX_RE.sub("", raw_output.strip()).strip().strip("«»\"'")
        max_reasonable_len = max(len(query) * 4, 600)
        if not resolved or len(resolved) > max_reasonable_len:
            logger.warning(
                "Coreference resolution returned invalid length (%d), using fallback",
                len(resolved),
            )
            return fallback_query, True, p_tok, c_tok

        return resolved, True, p_tok, c_tok
    except LLMError as exc:
        logger.warning("Coreference resolution failed (%s), using fallback", exc)
        if on_failure:
            on_failure(exc)
        return fallback_query, True, 0, 0


def resolve_history(
    query: str,
    history: list[dict],
    timeout: float | None = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
) -> tuple[str, bool]:
    resolved, used, _, _ = _resolve_history_with_usage(
        query=query,
        history=history,
        timeout=timeout,
        on_failure=on_failure,
        on_llm_success=on_llm_success,
    )
    return resolved, used


def _extract_json_dict(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = _JSON_BLOCK_RE.search(raw_text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            raise LLMFormatError(f"Invalid JSON object in LLM output: {exc}") from exc

    raise LLMFormatError("No JSON object found in QueryAnalysis LLM output")


def _infer_slots_by_regex(query: str) -> list[str]:
    slots = [
        slot_name
        for slot_name, pattern in _SLOT_DETECTION_PATTERNS
        if pattern.search(query)
    ]
    if _asks_rent_amount(query) and "rent_amount" not in slots:
        slots.insert(0, "rent_amount")
    return slots if slots else ["general"]


def _sanitize_intent(query: str, parsed_intent: str, article_refs: list[str]) -> str:
    if parsed_intent == "exact_lookup":
        if not article_refs and not _EXPLICIT_POINT_OR_QUOTE_RE.search(query):
            return "semantic"
    return parsed_intent


def _sanitize_required_slots(query: str, llm_slots: list[str]) -> list[str]:
    regex_slots = _infer_slots_by_regex(query)
    q_lower = query.lower()

    asks_rent = _asks_rent_amount(query)
    asks_payment_deadline = bool(
        re.search(
            r"до\s+какого\s+числа|срок\w*\s+(?:оплат\w*|внесени\w*|расчет\w*)|когда\s+(?:нужно\s+)?(?:оплач|вноси)",
            q_lower,
        )
    )
    asks_penalty = bool(re.search(r"\bпен[яиюей]\b|неустойк\w+|штраф\w*", q_lower))
    asks_prepayment = bool(re.search(r"\bаванс\w*|\bпредоплат\w*", q_lower))
    asks_security_deposit = bool(re.search(r"обеспечительн\w+\s+платеж|депозит\w*", q_lower))

    cleaned: list[str] = []
    for slot in llm_slots:
        s = slot.strip()
        if not s:
            continue
        if s == "prepayment" and asks_security_deposit and not asks_prepayment:
            s = "security_deposit"
        if s == "rent_amount" and not asks_rent:
            continue
        if s == "payment_deadline" and not asks_payment_deadline:
            continue
        if s == "penalty" and not asks_penalty:
            continue
        if s == "prepayment" and not asks_prepayment:
            continue
        if s not in cleaned:
            cleaned.append(s)

    for r_slot in regex_slots:
        if r_slot != "general" and r_slot not in cleaned:
            cleaned.append(r_slot)

    return cleaned if cleaned else ["general"]


def _ensure_slot_queries(
    query: str,
    entities_raw: list[str],
    required_slots: list[str],
    bm25_queries: list[str],
    dense_queries: list[str],
) -> tuple[list[str], list[str]]:
    exact_tokens = _CONTRACT_NUM_RE.findall(query)
    entity_prefix = " ".join(entities_raw or exact_tokens).strip()

    bad_prepayment_noise = (
        "security_deposit" in required_slots and "prepayment" not in required_slots
    )
    out_bm25 = [
        q.strip()
        for q in bm25_queries
        if q.strip() and not (bad_prepayment_noise and "аванс" in q.lower())
    ]
    out_dense = [
        q.strip()
        for q in dense_queries
        if q.strip() and not (bad_prepayment_noise and "аванс" in q.lower())
    ]
    out_bm25 = list(dict.fromkeys(out_bm25))
    out_dense = list(dict.fromkeys(out_dense))

    if exact_tokens and out_bm25:
        combined_bm25 = " ".join(out_bm25).lower()
        missing_tokens = [t for t in exact_tokens if t.lower() not in combined_bm25]
        if missing_tokens:
            out_bm25[0] = f"{out_bm25[0]} {' '.join(missing_tokens)}".strip()

    for slot in required_slots:
        if slot == "general":
            continue
        slot_terms = _SLOT_QUERY_TEMPLATES.get(slot, slot.replace("_", " "))
        key_tokens = [w for w in slot_terms.split() if len(w) >= 4][:3]

        dense_joined = " ".join(out_dense).lower()
        if not any(tok[:5] in dense_joined for tok in key_tokens):
            slot_q = f"{entity_prefix} {slot_terms}".strip()
            out_dense.append(slot_q)

        bm25_joined = " ".join(out_bm25).lower()
        if not any(tok[:5] in bm25_joined for tok in key_tokens):
            slot_bm25 = f"{entity_prefix} {slot_terms}".strip()
            out_bm25.append(slot_bm25)

    return out_bm25 or [query], out_dense or [query]


def _ensure_decomposition(
    query: str,
    entities_raw: list[str],
    required_slots: list[str],
    decomposition: list[SubQuestion],
) -> list[SubQuestion]:
    if decomposition:
        return decomposition

    target_doc = entities_raw[0] if len(entities_raw) == 1 else None
    real_slots = [s for s in required_slots if s != "general"]

    if len(real_slots) >= 2:
        built: list[SubQuestion] = []
        for slot in real_slots:
            desc = _SLOT_QUERY_TEMPLATES.get(slot, slot)
            q_text = f"{target_doc + ': ' if target_doc else ''}{desc}"
            built.append(
                SubQuestion(text=q_text.strip(), target_doc_raw=target_doc, aspect=slot)
            )
        return built

    if _COMPOUND_SPLIT_RE.search(query):
        parts = [
            p.strip(" ?.")
            for p in re.split(
                r"\?\s*|\s+и\s+(?=за\s+какой|до\s+какого|чем\s+|срок\s+)", query
            )
            if p.strip(" ?.")
        ]
        if len(parts) >= 2:
            aspect = real_slots[0] if real_slots else None
            return [
                SubQuestion(
                    text=f"{p} ({target_doc})" if target_doc and target_doc not in p else p,
                    target_doc_raw=target_doc,
                    aspect=aspect,
                )
                for p in parts
            ]

    return []


def _augment_premises(query: str, premises: list[Premise]) -> list[Premise]:
    if premises:
        return premises
    extracted: list[Premise] = []
    for kind, pattern in _PREMISE_PATTERNS:
        for match in pattern.finditer(query):
            val = match.group(1).strip()
            num_match = re.search(r"\d[\d\s]*", val)
            clean_val = num_match.group(0).strip() if num_match else val
            extracted.append(
                Premise(text=match.group(0).strip(), kind=kind, value=clean_val, verified=False)
            )
    return extracted


def _build_regex_fallback_spec(
    query: str,
    event_date: str | None,
    history_resolved: bool,
    chat_history: list[dict] | None = None,
) -> QuerySpec:
    entities_raw: list[str] = []
    for m in _QUOTED_ENTITY_RE.findall(query):
        entities_raw.append(m.strip())
    for m in _CONTRACT_NUM_RE.findall(query):
        entities_raw.append(m.strip())

    article_refs = [m.strip(".") for m in _ARTICLE_EXTRACT_RE.findall(query) if m.strip(".")]

    dates: list[str] = extract_dates_from_text(query)
    resolved_event_date = event_date or (dates[0] if dates else None)
    assumptions: list[str] = []
    if not resolved_event_date and chat_history:
        hist_date = extract_date_from_history(chat_history)
        if hist_date:
            resolved_event_date = hist_date
            if hist_date not in dates:
                dates.insert(0, hist_date)
    if not event_date and resolved_event_date and (history_resolved or chat_history):
        assumptions.append(f"дата события ({resolved_event_date}) определена из истории диалога")

    entities_resolved = resolve_entities(entities_raw)
    required_slots = _infer_slots_by_regex(query)
    bm25_q, dense_q = _ensure_slot_queries(
        query, entities_raw, required_slots, [query], [query]
    )
    decomposition = _ensure_decomposition(query, entities_raw, required_slots, [])
    premises = _augment_premises(query, [])

    return QuerySpec(
        intent="semantic",
        task_type=TaskType.EXPLAIN,
        entities_raw=entities_raw,
        entities_resolved=entities_resolved,
        article_refs=article_refs,
        dates=dates,
        event_date=resolved_event_date,
        premises=premises,
        required_slots=required_slots,
        assumptions=assumptions,
        ambiguity=Ambiguity.LOW,
        needs_clarification=False,
        decomposition=decomposition,
        rewritten_queries_bm25=bm25_q,
        rewritten_queries_dense=dense_q,
        history_resolved=history_resolved,
        degradation_reason="regex_fallback",
    )


def _apply_date_version_degradation(spec: QuerySpec) -> QuerySpec:
    if spec.intent == "date_version" and not has_effective_dates_in_corpus(
        spec.entities_resolved
    ):
        spec.intent = "semantic"
        spec.degradation_reason = "no_effective_dates"
        msg = "версия не определена, нужна дата"
        if msg not in spec.assumptions:
            spec.assumptions.append(msg)
    return spec


def analyze_query(
    query: str,
    event_date: str | None = None,
    chat_history: list[dict] | None = None,
    on_failure: Optional[Callable[[Exception], None]] = None,
    on_llm_success: Optional[Callable[[], None]] = None,
    timeout: float | None = None,
) -> tuple[QuerySpec, TraceStep, bool]:
    start_time = time.perf_counter()
    used_history = False
    effective_query = query
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if config.ENABLE_MEMORY and chat_history:
        (
            effective_query,
            used_history,
            hist_p_tok,
            hist_c_tok,
        ) = _resolve_history_with_usage(
            query=query,
            history=chat_history,
            timeout=timeout,
            on_failure=on_failure,
            on_llm_success=on_llm_success,
        )
        total_prompt_tokens += hist_p_tok
        total_completion_tokens += hist_c_tok

    max_attempts = max(1, int(config.MAX_ATTEMPTS_QUERY_ANALYSIS))
    retry_feedback = ""
    last_format_error: Optional[Exception] = None
    last_api_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        messages = [
            {"role": config.ROLE_SYSTEM, "content": qa_prompts.QUERY_ANALYSIS_SYSTEM},
            {
                "role": config.ROLE_USER,
                "content": qa_prompts.QUERY_ANALYSIS_USER.format(
                    query=effective_query,
                    event_date=event_date or "не указана",
                    retry_feedback=retry_feedback,
                ),
            },
        ]
        try:
            raw_output, p_tok, c_tok = llm_complete_with_usage(
                messages=messages,
                model=config.LLM_MODEL,
                temperature=qa_prompts.QUERY_ANALYSIS_TEMPERATURE,
                max_tokens=qa_prompts.QUERY_ANALYSIS_MAX_TOKENS,
                timeout=timeout,
            )
            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            if on_llm_success:
                on_llm_success()

            data = _extract_json_dict(raw_output)
            try:
                parsed = QueryAnalysisSchema.model_validate(data)
            except ValidationError as val_exc:
                raise LLMFormatError(f"QueryAnalysis validation error: {val_exc}") from val_exc

            sanitized_intent = _sanitize_intent(
                effective_query, parsed.intent, list(parsed.article_refs)
            )
            required_slots = _sanitize_required_slots(
                effective_query, list(parsed.required_slots)
            )
            bm25_queries, dense_queries = _ensure_slot_queries(
                effective_query,
                list(parsed.entities_raw),
                required_slots,
                list(parsed.rewritten_queries_bm25),
                list(parsed.rewritten_queries_dense),
            )
            raw_decomp = [
                SubQuestion(
                    text=sq.text,
                    target_doc_raw=sq.target_doc_raw,
                    aspect=sq.aspect,
                )
                for sq in parsed.decomposition
            ]
            decomposition = _ensure_decomposition(
                effective_query,
                list(parsed.entities_raw),
                required_slots,
                raw_decomp,
            )
            raw_premises = [
                Premise(text=p.text, kind=p.kind, value=p.value, verified=False)
                for p in parsed.premises
            ]
            premises = _augment_premises(effective_query, raw_premises)

            entities_resolved = resolve_entities(parsed.entities_raw)
            effective_event_date = event_date or parsed.event_date or (
                parsed.dates[0] if parsed.dates else None
            )
            spec_assumptions = list(parsed.assumptions)
            if not effective_event_date:
                effective_dates = extract_dates_from_text(effective_query)
                if effective_dates:
                    effective_event_date = effective_dates[0]
                elif chat_history:
                    effective_event_date = extract_date_from_history(chat_history)
                    if effective_event_date:
                        msg = f"дата события ({effective_event_date}) определена из истории диалога"
                        if msg not in spec_assumptions:
                            spec_assumptions.append(msg)

            spec_dates = list(parsed.dates)
            if effective_event_date and effective_event_date not in spec_dates:
                spec_dates.insert(0, effective_event_date)

            spec = QuerySpec(
                intent=sanitized_intent,
                task_type=parsed.task_type,
                entities_raw=list(parsed.entities_raw),
                entities_resolved=entities_resolved,
                article_refs=list(parsed.article_refs),
                dates=spec_dates,
                event_date=effective_event_date,
                premises=premises,
                required_slots=required_slots,
                assumptions=spec_assumptions,
                ambiguity=parsed.ambiguity,
                needs_clarification=False,
                decomposition=decomposition,
                rewritten_queries_bm25=bm25_queries,
                rewritten_queries_dense=dense_queries,
                history_resolved=used_history,
                degradation_reason=None,
            )
            spec = _apply_date_version_degradation(spec)

            latency_ms = round(
                (time.perf_counter() - start_time) * config.MS_PER_SECOND,
                config.LATENCY_ROUND_DIGITS,
            )
            step = TraceStep(
                step="query_analysis",
                input={
                    "query": effective_query,
                    "event_date": event_date,
                    "attempts": attempt,
                },
                output={
                    "intent": spec.intent,
                    "task_type": spec.task_type.value,
                    "required_slots": spec.required_slots,
                    "decomposition_count": len(spec.decomposition),
                    "premises_count": len(spec.premises),
                    "rewritten_bm25": spec.rewritten_queries_bm25,
                    "rewritten_dense": spec.rewritten_queries_dense,
                    "event_date": spec.event_date,
                    "degradation_reason": spec.degradation_reason,
                },
                latency_ms=latency_ms,
                model=config.LLM_MODEL,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                stage_idx=1,
                degradation=spec.degradation_reason,
            )
            return spec, step, used_history

        except LLMFormatError as exc:
            last_format_error = exc
            if on_failure:
                on_failure(exc)
            retry_feedback = (
                f"\nПредыдущий ответ содержал ошибку формата JSON: {exc}. "
                "Верни строго валидный JSON по схеме."
            )
        except LLMError as exc:
            last_api_error = exc
            if on_failure:
                on_failure(exc)

    if last_api_error is not None and last_format_error is None:
        raise last_api_error

    fallback_spec = _build_regex_fallback_spec(
        query=effective_query,
        event_date=event_date,
        history_resolved=used_history,
        chat_history=chat_history,
    )
    fallback_spec = _apply_date_version_degradation(fallback_spec)
    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step="query_analysis",
        input={
            "query": effective_query,
            "event_date": event_date,
            "attempts": max_attempts,
        },
        output={
            "intent": fallback_spec.intent,
            "task_type": fallback_spec.task_type.value,
            "required_slots": fallback_spec.required_slots,
            "decomposition_count": len(fallback_spec.decomposition),
            "premises_count": len(fallback_spec.premises),
            "event_date": fallback_spec.event_date,
            "degradation_reason": fallback_spec.degradation_reason,
        },
        latency_ms=latency_ms,
        model=config.LLM_MODEL,
        prompt_tokens=total_prompt_tokens or None,
        completion_tokens=total_completion_tokens or None,
        stage_idx=1,
        degradation=fallback_spec.degradation_reason,
    )
    return fallback_spec, step, used_history