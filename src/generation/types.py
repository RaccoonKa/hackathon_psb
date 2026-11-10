from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TypedDict


class Intent(str, Enum):
    EXACT_LOOKUP = "exact_lookup"
    SEMANTIC_QUERY = "semantic_query"
    MIXED_QUERY = "mixed_query"
    OUT_OF_SCOPE = "out_of_scope"
    ABOUT_ASSISTANT = "about_assistant"


class SourceMetadata(TypedDict):
    document_id: str
    version_id: str
    chunk_id: str
    document_title: str
    provision_id: Optional[str]
    structure_path: list[str]
    score: float


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


@dataclass
class AgentResult:
    final_answer: str
    sources: dict[str, dict[str, Any]]
    intent: str
    refused: bool
    trace: list[TraceStep]