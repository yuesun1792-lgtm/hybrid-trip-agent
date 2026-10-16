"""Public adapters for the unified planning scheduler and existing debug endpoints."""
from backend.app.domain.models import PlanningState, TripPlan, TripRequest, CandidateDestination
from backend.app.planning.task_builder import task_from_request
from backend.app.planning.unified_planner import execute_task


def state_from_request(request: TripRequest, conversation_id: str | None = None) -> PlanningState:
    candidates = request.candidate_destinations or ([request.destination] if request.destination else [])
    return PlanningState(request=request, conversation_id=conversation_id, stage="ready_to_plan",
                         candidate_destinations=[CandidateDestination(name=name, source="user") for name in candidates])


def discover_candidates_for_state(state: PlanningState, queries: list[str] | None = None) -> PlanningState:
    from backend.app.planning.unified_planner import resolve_candidates
    task = task_from_request(state.request)
    task.rejected_destinations = state.rejected_destinations
    candidates = resolve_candidates(task, allow_research=False)
    return state.model_copy(update={"stage": "candidates_discovered", "candidate_destinations": candidates,
                                    "evidence": [chunk for candidate in candidates for chunk in candidate.evidence]})


def rank_candidates_for_state(state: PlanningState) -> PlanningState:
    from backend.app.planning.destination_analysis import analyze_destination
    from backend.app.planning.unified_planner import compare_candidates
    from backend.app.domain.destination_names import COUNTRY_BY_DESTINATION
    task = task_from_request(state.request)
    analyses = [analyze_destination(c.name, task.constraints, "overview", c.evidence)
                for c in state.candidate_destinations if c.name not in state.rejected_destinations]
    ranked, filtered = compare_candidates(analyses, task.constraints, bool(task.constraints.destination))
    candidates = [CandidateDestination(name=a.destination, country=COUNTRY_BY_DESTINATION.get(a.destination, "Unknown"),
                                       score=a.score, score_breakdown=a.score_breakdown, evidence=a.evidence) for a in ranked]
    return state.model_copy(update={"stage": "candidates_ranked" if candidates else "candidate_ranking_empty",
        "candidate_destinations": candidates, "candidate_analyses": analyses,
        "filtered_candidates": [CandidateDestination(name=a.destination, evidence=a.evidence,
                                                      filtered_reason="Estimated cost exceeds budget") for a in filtered]})


def plan_from_state(state: PlanningState) -> TripPlan:
    task = task_from_request(state.request)
    task.rejected_destinations = state.rejected_destinations
    return execute_task(task, seed_candidates=state.candidate_destinations)


# Compatibility entry points use the same core and do not implement separate planners.
def rag_plan_from_state(state: PlanningState) -> TripPlan:
    return plan_from_state(state)


def deterministic_plan_from_state(state: PlanningState, fallback_reason: str) -> TripPlan:
    plan = plan_from_state(state)
    return plan.model_copy(update={"fallback_reason": fallback_reason})
