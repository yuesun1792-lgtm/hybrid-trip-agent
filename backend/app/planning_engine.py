from backend.app.config import (
    llm_rag_enhancements_enabled,
    research_max_candidates,
    research_max_rounds,
)
from backend.app.explainer import generate_explanation
from backend.app.models import PlanningState, TripPlan, TripRequest
from backend.app.planner import create_trip_plan
from backend.app.candidate_discovery import discover_candidate_destinations
from backend.app.candidate_ranker import rank_candidate_destinations
from backend.app.rag_plan_builder import build_trip_plan_from_ranked_state
from backend.app.rag_enhancements import decide_research, rerank_candidates, write_candidate_copy


def state_from_request(
    request: TripRequest,
    conversation_id: str | None = None,
) -> PlanningState:
    """Normalize any input surface into the planning state used by the engine."""
    return PlanningState(
        request=request,
        conversation_id=conversation_id,
        stage="ready_to_plan",
    )


def deterministic_plan_from_state(
    state: PlanningState,
    fallback_reason: str,
) -> TripPlan:
    """Run the legacy deterministic planner through the state-based engine boundary."""
    plan = create_trip_plan(state.request)
    explanation = generate_explanation(plan, state.request)
    return plan.model_copy(
        update={
            "explanation": explanation,
            "planner_mode": "deterministic_fallback",
            "fallback_reason": fallback_reason,
        }
    )


def discover_candidates_for_state(
    state: PlanningState,
    queries: list[str] | None = None,
) -> PlanningState:
    """Run V5.0 RAG candidate discovery for a normalized planning state."""
    return discover_candidate_destinations(state, queries=queries)


def rank_candidates_for_state(state: PlanningState) -> PlanningState:
    """Run V5.1 constraint-aware ranking for discovered candidates."""
    return rank_candidate_destinations(state)


def rag_plan_from_state(state: PlanningState) -> TripPlan:
    """Build a constrained RAG plan, optionally enhanced by bounded LLM research."""
    llm_enabled = llm_rag_enhancements_enabled()
    llm_used = False
    discovered = state

    if llm_enabled:
        searched_queries: list[str] = []
        for round_number in range(1, research_max_rounds() + 1):
            try:
                decision = decide_research(discovered, searched_queries, round_number)
                llm_used = True
            except Exception:
                decision = None

            if decision is None or decision.action == "stop":
                if not discovered.candidate_destinations:
                    discovered = discover_candidate_destinations(
                        discovered,
                        max_candidates=research_max_candidates(),
                    )
                break

            searched_queries.extend(decision.queries)
            discovered = discover_candidate_destinations(
                discovered,
                queries=decision.queries,
                max_candidates=research_max_candidates(),
            )
    else:
        discovered = discover_candidates_for_state(state)

    ranked = rank_candidates_for_state(discovered)
    if not ranked.candidate_destinations:
        raise ValueError("No ranked candidates available")

    generated_copy = None
    if llm_enabled:
        try:
            reordered = rerank_candidates(ranked.candidate_destinations, ranked)
            ranked = ranked.model_copy(update={"candidate_destinations": reordered})
            llm_used = True
        except Exception:
            pass
        try:
            generated_copy = write_candidate_copy(ranked.candidate_destinations[:3], ranked)
            llm_used = True
        except Exception:
            generated_copy = None

    return build_trip_plan_from_ranked_state(
        ranked,
        generated_copy=generated_copy,
        planner_mode="rag_ranked_llm" if llm_used else "rag_ranked",
    )


def plan_from_state(state: PlanningState) -> TripPlan:
    """Public planning boundary shared by every input surface."""
    try:
        return rag_plan_from_state(state)
    except Exception as exc:
        return deterministic_plan_from_state(
            state,
            f"rag_plan_error: {type(exc).__name__}",
        )
