import logging
import re
import time
from src import config
from src.generation.types import PreFilterResult, TraceStep

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS = [
    re.compile(r"игнорируй\s+(?:все\s+)?предыдущие", re.IGNORECASE),
    re.compile(r"забудь\s+(?:все\s+)?(?:инструкции|правила)", re.IGNORECASE),
    re.compile(r"ты\s+теперь\s+(?:не\s+юрист|хакер|свободный|пират|дан\b)", re.IGNORECASE),
    re.compile(r"системная\s+команда\s*:", re.IGNORECASE),
    re.compile(r"внимание\s+для\s+ии\b", re.IGNORECASE),
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
]

_ILLEGAL_PATTERNS = [
    re.compile(r"как\s+(?:подделать|сфальсифицировать)\s+(?:договор|подпись|печать|документ)", re.IGNORECASE),
    re.compile(r"как\s+(?:дать|передать)\s+взятку", re.IGNORECASE),
    re.compile(r"схем[аыу]\s+обналичивания", re.IGNORECASE),
    re.compile(r"как\s+обойти\s+закон", re.IGNORECASE),
    re.compile(r"как\s+(?:сделать|изготовить)\s+(?:бомбу|взрывчатку|наркотик|яд|оружие)", re.IGNORECASE),
    re.compile(r"как\s+взломать\s+(?:банк|сервер|аккаунт|систему)", re.IGNORECASE),
]

_ABOUT_ASSISTANT_PATTERNS = [
    re.compile(r"^\s*(?:что\s+ты\s+умеешь|кто\s+ты(?:\s+такой)?|твои\s+возможности|расскажи\s+о\s+себе|чем\s+ты\s+можешь\s+помочь|как\s+ты\s+работаешь)\s*[?!.]*\s*$", re.IGNORECASE),
]

_OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\b(?:рецепт\s+(?:пиццы|борща|торта|блинов)|какая\s+погода|расскажи\s+анекдот|напиши\s+стих|кто\s+выиграл\s+матч)\b", re.IGNORECASE),
]

_LEGAL_DOMAIN_MARKERS = re.compile(
    r"(?:договор|соглашен|стать[яеиюй]|\bст\.|\bп\.|пункт|кодекс|\bтк\b|\bнк\b|\bгк\b|\bфз\b|"
    r"закон|регламент|аренд|неустойк|штраф|пен[яи]|ответственност|работник|работодател|"
    r"испытательн|сверхурочн|налог|ндс|контрагент|расторжен|обязательств|срок|оплат|право)",
    re.IGNORECASE,
)


def run_prefilter(query: str) -> tuple[PreFilterResult, TraceStep]:
    start_time = time.perf_counter()
    normalized = query.strip()

    result = PreFilterResult(triggered=False, template_key=None, reason=None)

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(normalized):
            result = PreFilterResult(
                triggered=True,
                template_key="RESPONSE_PROMPT_INJECTION",
                reason="prompt_injection",
            )
            break

    if not result.triggered:
        for pattern in _ILLEGAL_PATTERNS:
            if pattern.search(normalized):
                result = PreFilterResult(
                    triggered=True,
                    template_key="RESPONSE_ILLEGAL",
                    reason="illegal",
                )
                break

    has_legal_marker = bool(_LEGAL_DOMAIN_MARKERS.search(normalized))

    if not result.triggered and not has_legal_marker:
        for pattern in _ABOUT_ASSISTANT_PATTERNS:
            if pattern.search(normalized):
                result = PreFilterResult(
                    triggered=True,
                    template_key="RESPONSE_ABOUT_ASSISTANT",
                    reason="about_assistant",
                )
                break

    if not result.triggered and not has_legal_marker:
        for pattern in _OUT_OF_SCOPE_PATTERNS:
            if pattern.search(normalized):
                result = PreFilterResult(
                    triggered=True,
                    template_key="RESPONSE_OUT_OF_SCOPE",
                    reason="out_of_scope",
                )
                break

    latency_ms = round(
        (time.perf_counter() - start_time) * config.MS_PER_SECOND,
        config.LATENCY_ROUND_DIGITS,
    )
    step = TraceStep(
        step="prefilter",
        input={"query": query},
        output={
            "triggered": result.triggered,
            "template_key": result.template_key,
            "reason": result.reason,
        },
        latency_ms=latency_ms,
        stage_idx=0,
    )
    return result, step