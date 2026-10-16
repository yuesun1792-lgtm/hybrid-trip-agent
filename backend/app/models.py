from __future__ import annotations

from typing import Any, Literal
from hashlib import sha256

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.field_validation import normalize_trip_fields
from backend.app.place_models import PlaceResolutionSummary
from backend.app.planning_models import Activity, Day, TripConstraints, ValidationIssue, RevisionRequest, new_id


class TripRequest(BaseModel):
    destination: str | None = None
    candidate_destinations: list[str] = Field(default_factory=list)
    requested_output: Literal["options", "itinerary"] = "itinerary"
    response_language: str = "en"
    origin_city: str
    date_range: str
    trip_length: int = Field(gt=0)
    budget: int = Field(gt=0)
    interests: list[str]
    transport_preference: str
    physical_level: str
    home_currency: str = "CNY"
    taste_profile: str = ""


class ToolInsight(BaseModel):
    transport_hours: float
    transport_mode: str
    weather_risk: str   # "low" | "medium" | "high"
    weather_note: str
    estimated_total_cost: int


class Citation(BaseModel):
    evidence_ids: list[str] = Field(default_factory=list)
    title: str
    url: str
    section: str = ""
    snippet: str = ""


class DestinationSuggestion(BaseModel):
    name: str
    country: str
    score: int
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    pros: list[str]
    cons: list[str]
    estimated_cost: int | None
    tool_signals: ToolInsight | None = None
    estimated_cost_home: float | None = None   # cost in traveller's home currency
    home_currency: str | None = None           # ISO code echoed back
    citations: list[Citation] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    evidence_id: str = ""
    corpus_version: str = "wikivoyage-v1"
    text: str
    destination: str = ""
    section: str = ""
    url: str = ""
    score: float | None = None


    @field_validator("destination")
    @classmethod
    def canonical_evidence_city(cls, value):
        from backend.app.destination_names import canonical_destination
        return canonical_destination(value)

    @model_validator(mode="after")
    def assign_evidence_id(self):
        if not self.evidence_id:
            content = f"{self.corpus_version}|{self.url}|{self.destination}|{self.section}|{self.text}"
            self.evidence_id = "ev_" + sha256(content.encode()).hexdigest()[:20]
        return self


class CandidateDestination(BaseModel):
    name: str
    country: str = ""
    source: str = "unknown"
    score: float | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    filtered_reason: str | None = None
    evidence: list[EvidenceChunk] = Field(default_factory=list)


# Public compatibility alias; the canonical day model lives with planning contracts.
ItineraryDay = Day


class DestinationAnalysis(BaseModel):
    destination: str
    depth: Literal["overview", "detailed"]
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    estimated_cost: int | None = None
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    activity_themes: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    tool_signals: ToolInsight | None = None


class PlanningState(BaseModel):
    request: TripRequest
    conversation_id: str | None = None
    stage: str = "ready_to_plan"
    missing_fields: list[str] = Field(default_factory=list)
    rejected_destinations: list[str] = Field(default_factory=list)
    candidate_destinations: list[CandidateDestination] = Field(default_factory=list)
    filtered_candidates: list[CandidateDestination] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    candidate_analyses: list[DestinationAnalysis] = Field(default_factory=list)


class TripPlan(BaseModel):
    place_resolution: PlaceResolutionSummary = Field(default_factory=PlaceResolutionSummary)
    display_language: str = "en"
    plan_id: str = Field(default_factory=new_id)
    version: int = Field(default=1, gt=0)
    parent_version: int | None = None
    constraints: TripConstraints | None = None
    requested_output: Literal["options", "itinerary"] = "itinerary"
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    candidate_analyses: list[DestinationAnalysis] = Field(default_factory=list)
    selected_destinations: list[str] = Field(default_factory=list)
    ranked_destinations: list[DestinationSuggestion]
    itinerary: list[ItineraryDay]
    risks: list[str]
    final_recommendation: str
    # Optional LLM-generated explanation; None when no API key is configured
    explanation: str | None = None
    planner_mode: str = "unknown"
    fallback_reason: str | None = None
    tradeoff_summary: list[str] = Field(default_factory=list)
    filtered_candidates: list[CandidateDestination] = Field(default_factory=list)


class StrategyDecision(BaseModel):
    requested_output: Literal["options", "itinerary"] | None = None
    intent: Literal["plan_trip", "recommend_destination", "plan_fixed_destination", "revise_plan", "clarify"]
    next_action: Literal[
        "ask_user",
        "run_planning",
        "run_destination_discovery",
        "run_fixed_itinerary",
        "revise_existing_plan",
    ]
    known_fields: dict[str, Any] = Field(default_factory=dict)

    @field_validator("known_fields")
    @classmethod
    def normalize_known_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        return normalize_trip_fields(value)

    missing_fields: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    reason: str = ""
    router_mode: Literal["llm", "rule"] = "rule"
    router_fallback_reason: str | None = None
    llm_usage: LLMUsage | None = None


class ExecutionTraceStep(BaseModel):
    stage: str
    status: Literal["ok", "fallback", "error", "skipped"] = "ok"
    detail: str = ""


class LLMUsage(BaseModel):
    provider: str = "anthropic"
    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    estimated_cost_usd: float | None = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    expected_plan_version: int | None = Field(default=None, gt=0)
    revision: RevisionRequest | None = None
    requested_output: Literal["options", "itinerary"] | None = None
    response_language: str | None = Field(default=None, pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
    extracted_fields: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extracted_fields")
    @classmethod
    def normalize_extracted_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        return normalize_trip_fields(value)



class ChatResponse(BaseModel):
    response_language: str = "en"
    planning_task: dict[str, Any] | None = None
    run_id: str | None = None
    trajectory: dict[str, Any] | None = None
    conversation_id: str
    assistant_message: str
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    trace: list[ExecutionTraceStep] = Field(default_factory=list)
    llm_usage: list[LLMUsage] = Field(default_factory=list)
    state: PlanningState | None = None
    plan: TripPlan | None = None
