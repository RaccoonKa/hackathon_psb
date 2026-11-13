from dataclasses import dataclass, field, is_dataclass
from datetime import date
from enum import Enum
import re
from typing import Any, Optional, TypedDict
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Intent(str, Enum):
    EXACT_LOOKUP = "exact_lookup"
    SEMANTIC_QUERY = "semantic_query"
    SEMANTIC = "semantic"
    MIXED_QUERY = "mixed_query"
    MIXED = "mixed"
    COMPARE = "compare"
    DATE_VERSION = "date_version"
    CHECKLIST = "checklist"
    OUT_OF_SCOPE = "out_of_scope"
    ABOUT_ASSISTANT = "about_assistant"
    ILLEGAL = "illegal"
    PROMPT_INJECTION = "prompt_injection"


class TaskType(str, Enum):
    LOOKUP = "lookup"
    EXPLAIN = "explain"
    COMPARE = "compare"
    CHECKLIST = "checklist"
    DATE_CHECK = "date_check"
    PREMISE_CHECK = "premise_check"


class Ambiguity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_IN_SOURCE = "not_in_source"


class VerificationVerdict(str, Enum):
    ACCEPT = "accept"
    REGENERATE = "regenerate"
    PARTIAL = "partial"
    REFUSE = "refuse"


class SourceMetadata(TypedDict, total=False):
    document_id: str
    version_id: str
    chunk_id: str
    document_title: str
    provision_id: Optional[str]
    structure_path: list[str]
    text: str
    location: dict[str, Any]
    score: float
    effective_from: Optional[str]
    effective_to: Optional[str]
    effective_dates_known: bool
    amends_document_id: Optional[str]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    document_id: str
    version_id: str
    document_title: str
    provision_id: Optional[str]
    structure_path: list[str]
    score: float
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    effective_dates_known: bool = True
    amends_document_id: Optional[str] = None
    untrusted: bool = False


@dataclass
class DocRef:
    raw: str
    document_id: str | None
    candidates: list[str]
    resolved: bool


@dataclass
class Premise:
    text: str
    kind: str
    value: str | None
    verified: bool = False


@dataclass
class SubQuestion:
    text: str
    target_doc_raw: str | None
    aspect: str | None


@dataclass
class QuerySpec:
    intent: str
    task_type: TaskType
    entities_raw: list[str]
    entities_resolved: list[DocRef]
    article_refs: list[str]
    dates: list[str]
    event_date: str | None
    premises: list[Premise]
    required_slots: list[str]
    assumptions: list[str]
    ambiguity: Ambiguity
    needs_clarification: bool
    decomposition: list[SubQuestion]
    rewritten_queries_bm25: list[str]
    rewritten_queries_dense: list[str]
    history_resolved: bool
    degradation_reason: str | None


@dataclass
class RetrievalTask:
    query: str
    mode: str
    purpose: str
    selected_doc_ids: set[str] | None
    boost_doc_ids: set[str] | None
    event_date: str | None
    top_k: int
    # H-2A: явный hybrid-intent задачи. ``None`` сохраняет обратную
    # совместимость — мост тогда выводит intent из ``mode``.
    intent: str | None = None


@dataclass
class EvidenceBundle:
    chunks: list[RetrievedChunk]
    slot_map: dict[str, list[str]]
    contradictions: list[dict[str, Any]]
    version_issues: list[dict[str, Any]]


@dataclass
class EvidenceAssessment:
    covered_slots: list[str]
    missing_slots: list[str]
    core_slot_covered: bool
    sufficiency: Sufficiency
    degraded_intents: list[str]
    reason: str


@dataclass
class Claim:
    text: str
    source_ids: list[str]
    slot: str | None
    original_idx: int | None = None


@dataclass
class DraftAnswer:
    answer_text: str
    claims: list[Claim]
    assumptions: list[str]
    missing_info: list[str]
    refusal_reason: str | None


@dataclass
class ClaimVerdict:
    claim_idx: int
    status: ClaimStatus
    quote: str
    source_id: str | None


@dataclass
class VerificationReport:
    batch_status: list[ClaimVerdict]
    citation_errors: list[str]
    injection_signals: list[str]
    supported_locked: list[int]
    final_verdict: VerificationVerdict


@dataclass
class PolicyVerdict:
    premise_mismatch: bool
    premise_details: list[str]
    injection_flagged: bool
    refusal_forbidden: bool


@dataclass
class PreFilterResult:
    triggered: bool
    template_key: str | None
    reason: str | None


@dataclass
class BailoutContext:
    query: str
    failures_total: int
    last_stage: str
    last_error: str | None
    partial_spec: QuerySpec | None
    llm_was_available: bool


@dataclass
class TraceStep:
    step: str
    input: dict[str, Any]
    output: dict[str, Any]
    latency_ms: float
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)
    stage_idx: Optional[int] = None
    failures_global: int = 0
    failures_step: int = 0
    degradation: Optional[str] = None

def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_name: _to_json_compatible(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(v) for v in value]
    return value


@dataclass
class AgentResult:
    final_answer: str
    sources: dict[str, dict[str, Any]]
    intent: str
    refused: bool
    trace: list[TraceStep]
    refusal_reason: Optional[str] = None
    confidence: str = "high"
    assumptions: list[str] = field(default_factory=list)
    missing_info: list[str] = field(default_factory=list)
    verification: Optional[VerificationReport] = None
    query_spec: Optional[QuerySpec] = None
    degradation_reasons: list[str] = field(default_factory=list)
    used_history: bool = False

    def to_dict(self, include_trace: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "final_answer": self.final_answer,
            "sources": _to_json_compatible(self.sources),
            "intent": self.intent,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
            "missing_info": list(self.missing_info),
            "used_history": self.used_history,
            "degradation_reasons": list(self.degradation_reasons),
            "verification": _to_json_compatible(self.verification),
            "query_spec": _to_json_compatible(self.query_spec),
        }
        if include_trace:
            payload["trace"] = _to_json_compatible(self.trace)
        return payload


_ARTICLE_REF_RE = re.compile(r"^[\d\.]+$")
_ALLOWED_ANALYSIS_INTENTS = {
    "exact_lookup",
    "semantic",
    "semantic_query",
    "mixed",
    "mixed_query",
    "compare",
    "date_version",
    "checklist",
}
_NULL_LIKE_STRINGS = {"null", "none", "nil", "false", "нет", "не применимо", "n/a", ""}


class PremiseSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    kind: str = "fact"
    value: Optional[str] = None


class SubQuestionSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    target_doc_raw: Optional[str] = None
    aspect: Optional[str] = None


class QueryAnalysisSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: str
    task_type: TaskType
    entities_raw: list[str] = Field(default_factory=list)
    article_refs: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    event_date: Optional[str] = None
    premises: list[PremiseSchema] = Field(default_factory=list)
    required_slots: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ambiguity: Ambiguity = Ambiguity.LOW
    needs_clarification: bool = False
    decomposition: list[SubQuestionSchema] = Field(default_factory=list)
    rewritten_queries_bm25: list[str]
    rewritten_queries_dense: list[str]

    @field_validator("intent")
    @classmethod
    def _validate_intent(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if cleaned not in _ALLOWED_ANALYSIS_INTENTS:
            raise ValueError(f"Unsupported intent: {v}")
        if cleaned == "semantic_query":
            return "semantic"
        if cleaned == "mixed_query":
            return "mixed"
        return cleaned

    @field_validator("dates")
    @classmethod
    def _validate_dates(cls, v: list[str]) -> list[str]:
        validated: list[str] = []
        for item in v:
            s = item.strip()
            date.fromisoformat(s)
            validated.append(s)
        return validated

    @field_validator("event_date")
    @classmethod
    def _validate_event_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            return None
        s = v.strip()
        date.fromisoformat(s)
        return s

    @field_validator("article_refs")
    @classmethod
    def _validate_article_refs(cls, v: list[str]) -> list[str]:
        cleaned_refs: list[str] = []
        for ref in v:
            s = ref.strip().strip(".")
            if not s or not _ARTICLE_REF_RE.match(s):
                raise ValueError(f"Invalid article_ref: {ref}")
            cleaned_refs.append(s)
        return cleaned_refs

    @field_validator("rewritten_queries_bm25", "rewritten_queries_dense")
    @classmethod
    def _validate_non_empty_queries(cls, v: list[str]) -> list[str]:
        non_empty = [q.strip() for q in v if q and q.strip()]
        if not non_empty:
            raise ValueError("Rewritten queries list must not be empty")
        return non_empty

    @field_validator("needs_clarification", mode="after")
    @classmethod
    def _force_no_clarification(cls, _v: bool) -> bool:
        return False


class ContradictionResolutionItemSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    slot: str
    is_real_contradiction: bool
    preferred_chunk_id: Optional[str] = None
    explanation: str = ""


class ContradictionResolutionSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resolutions: list[ContradictionResolutionItemSchema] = Field(default_factory=list)


class ClaimSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_idx: Optional[int] = None
    text: str
    source_ids: list[str] = Field(default_factory=list)
    slot: Optional[str] = None


class DraftAnswerSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer_text: str
    claims: list[ClaimSchema] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    refusal_reason: Optional[str] = None

    @field_validator("refusal_reason", mode="before")
    @classmethod
    def _sanitize_refusal_reason(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() in _NULL_LIKE_STRINGS:
            return None
        return s


class ClaimVerdictSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_idx: int
    status: ClaimStatus
    quote: str = ""
    source_id: Optional[str] = None


class VerificationBatchSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdicts: list[ClaimVerdictSchema] = Field(default_factory=list)