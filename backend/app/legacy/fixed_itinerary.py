"""Compatibility adapter for callers of the former fixed-destination planner."""
from pydantic import BaseModel
from backend.app.domain.models import LLMUsage, TripPlan
from backend.app.domain.destination_names import COUNTRY_BY_DESTINATION


class FixedItineraryResult(BaseModel):
    plan: TripPlan
    evidence_count: int
    llm_used: bool = False
    fallback_reason: str | None = None
    llm_usage: LLMUsage | None = None


def build_fixed_destination_plan(fields: dict, queries: list[str] | None = None) -> FixedItineraryResult:
    from backend.app.planning.task_builder import task_from_fields
    from backend.app.planning.unified_planner import execute_task
    task = task_from_fields(fields, queries=queries)
    plan = execute_task(task)
    return FixedItineraryResult(plan=plan, evidence_count=len(plan.evidence),
        llm_used=plan.planner_mode == "unified_itinerary_llm", fallback_reason=plan.fallback_reason)
