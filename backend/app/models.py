from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TripRequest(BaseModel):
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
    estimated_cost: int
    tool_signals: ToolInsight | None = None
    estimated_cost_home: float | None = None   # cost in traveller's home currency
    home_currency: str | None = None           # ISO code echoed back
    citations: list[Citation] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    text: str
    destination: str = ""
    section: str = ""
    url: str = ""
    score: float | None = None


class CandidateDestination(BaseModel):
    name: str
    country: str = ""
    source: str = "unknown"
    score: float | None = None
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    filtered_reason: str | None = None
    evidence: list[EvidenceChunk] = Field(default_factory=list)


class ItineraryDay(BaseModel):
    day: int
    destination: str
    plan: str


class PlanningState(BaseModel):
    request: TripRequest
    conversation_id: str | None = None
    stage: str = "ready_to_plan"
    missing_fields: list[str] = Field(default_factory=list)
    rejected_destinations: list[str] = Field(default_factory=list)
    candidate_destinations: list[CandidateDestination] = Field(default_factory=list)
    filtered_candidates: list[CandidateDestination] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)


class TripPlan(BaseModel):
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
    intent: Literal["recommend_destination", "plan_fixed_destination", "revise_plan", "clarify"]
    next_action: Literal[
        "ask_user",
        "run_destination_discovery",
        "run_fixed_itinerary",
        "revise_existing_plan",
    ]
    known_fields: dict[str, Any] = Field(default_factory=dict)
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
    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    estimated_cost_usd: float = 0.0


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    extracted_fields: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    conversation_id: str
    assistant_message: str
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    trace: list[ExecutionTraceStep] = Field(default_factory=list)
    llm_usage: list[LLMUsage] = Field(default_factory=list)
    state: PlanningState | None = None
    plan: TripPlan | None = None
