"""Compatibility adapter for historical ranked-state callers; generation is shared."""
from backend.app.domain.models import PlanningState, TripPlan


def build_trip_plan_from_ranked_state(state: PlanningState, generated_copy: dict[str, object] | None = None,
                                      planner_mode: str = "rag_ranked") -> TripPlan:
    from backend.app.planning.planning_engine import plan_from_state
    # Existing entry point is retained; selected destinations now get structured activities.
    return plan_from_state(state)
