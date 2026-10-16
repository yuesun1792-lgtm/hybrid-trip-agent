from backend.app.domain.models import CandidateDestination, PlanningState
from backend.app.planning.costs import estimate_trip_cost, load_daily_costs
from backend.app.tool_calling.tools import estimate_transport_time


def _load_costs() -> dict[str, int]:
    return load_daily_costs()


def _contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(term.lower() in lower for term in terms if term)


def _evidence_text(candidate: CandidateDestination) -> str:
    return "\n".join(chunk.text for chunk in candidate.evidence)


def score_candidate(
    candidate: CandidateDestination,
    state: PlanningState,
    costs: dict[str, int] | None = None,
) -> CandidateDestination:
    """Score one RAG candidate against hard constraints and soft preferences."""
    request = state.request
    costs = costs or _load_costs()
    evidence_text = _evidence_text(candidate)

    retrieval_score = min(max(candidate.score or 0.0, 0.0), 1.0) * 20

    estimated_cost = estimate_trip_cost(candidate.name, request.trip_length, costs)
    if estimated_cost is None:
        budget_score = -5
    elif estimated_cost <= request.budget:
        budget_score = 25
    elif estimated_cost <= request.budget * 1.10:
        budget_score = -15
    else:
        budget_score = -60

    transport = estimate_transport_time(request.origin_city, candidate.name)
    if transport["mode"] == request.transport_preference:
        transport_score = 20
    else:
        transport_score = -15

    interest_hits = sum(1 for interest in request.interests if _contains_any(evidence_text, [interest]))
    interest_score = min(interest_hits * 8, 24)

    taste_score = 0
    if request.taste_profile:
        taste_terms = [
            term.strip(" .,;:!?").lower()
            for term in request.taste_profile.split()
            if len(term.strip(" .,;:!?")) > 3
        ]
        taste_hits = sum(1 for term in taste_terms if term in evidence_text.lower())
        taste_score = min(taste_hits * 3, 12)

    season_terms = [request.date_range[:3], "summer", "winter", "spring", "autumn", "fall"]
    season_score = 5 if _contains_any(evidence_text, season_terms) else 0

    breakdown = {
        "retrieval": round(retrieval_score, 2),
        "budget": float(budget_score),
        "transport": float(transport_score),
        "interests": float(interest_score),
        "taste": float(taste_score),
        "season": float(season_score),
    }
    total = round(sum(breakdown.values()), 2)

    rationale_parts = [
        f"transport {transport['mode']} from {request.origin_city}",
        f"estimated €{estimated_cost}" if estimated_cost else "budget uncertain",
        f"{interest_hits} interest evidence hits",
    ]
    return candidate.model_copy(
        update={
            "score": total,
            "score_breakdown": breakdown,
            "rationale": "; ".join(rationale_parts),
        }
    )


def rank_candidate_destinations(state: PlanningState) -> PlanningState:
    """Rank discovered candidates and preserve score breakdowns for debugging/UI."""
    costs = _load_costs()
    ranked = [
        score_candidate(candidate, state, costs)
        for candidate in state.candidate_destinations
        if candidate.name not in state.rejected_destinations
    ]
    within_budget = []
    filtered = []
    for candidate in ranked:
        estimated_cost = estimate_trip_cost(candidate.name, state.request.trip_length, costs)
        if estimated_cost is not None and estimated_cost <= state.request.budget:
            within_budget.append(candidate)
        elif estimated_cost is not None:
            filtered.append(
                candidate.model_copy(
                    update={
                        "filtered_reason": (
                            f"Estimated €{estimated_cost} for {state.request.trip_length} days "
                            f"exceeds €{state.request.budget} budget"
                        )
                    }
                )
            )
    if within_budget:
        ranked = within_budget
    ranked.sort(key=lambda candidate: candidate.score or 0.0, reverse=True)
    return state.model_copy(
        update={
            "stage": "candidates_ranked" if ranked else "candidate_ranking_empty",
            "candidate_destinations": ranked,
            "filtered_candidates": filtered if within_budget else [],
        }
    )
