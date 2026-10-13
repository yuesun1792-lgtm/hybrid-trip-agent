from backend.app.costs import estimate_trip_cost, load_daily_costs
from backend.app.models import (
    CandidateDestination,
    Citation,
    DestinationSuggestion,
    ItineraryDay,
    PlanningState,
    TripPlan,
)
from backend.app.tools import gather_tool_insights


def _load_costs() -> dict[str, int]:
    return load_daily_costs()


def _country_from_url(candidate: CandidateDestination) -> str:
    # Country is not always available in Wikivoyage chunk metadata yet.
    return candidate.country or "Unknown"


def _short_evidence(chunk_text: str) -> str:
    body = chunk_text.split("\n", 1)[-1].strip()
    sentence = body.split(". ")[0].strip()
    return sentence[:160] if sentence else "Supported by retrieved travel knowledge"


def _citations_from_candidate(candidate: CandidateDestination) -> list[Citation]:
    citations = []
    seen = set()
    for chunk in candidate.evidence:
        if not chunk.url:
            continue
        key = (chunk.url, chunk.section)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                title=f"{candidate.name} - {chunk.section or 'Wikivoyage'}",
                url=chunk.url,
                section=chunk.section,
                snippet=_short_evidence(chunk.text),
            )
        )
    return citations[:3]


def _suggestion_from_candidate(
    candidate: CandidateDestination,
    state: PlanningState,
    costs: dict[str, int],
    generated_copy: object | None = None,
) -> DestinationSuggestion:
    request = state.request
    evidence_points = [_short_evidence(chunk.text) for chunk in candidate.evidence[:2]]
    pros = evidence_points or ["Supported by retrieved travel knowledge"]
    if candidate.rationale:
        pros.append(candidate.rationale)

    cons = []
    if candidate.score_breakdown.get("transport", 0) < 0:
        cons.append(f"Transport may not match your {request.transport_preference} preference")
    if candidate.score_breakdown.get("budget", 0) < 0:
        cons.append("Estimated cost may exceed your budget")
    if not cons:
        cons.append("Details are based on retrieved guide text and mock cost/transport data")

    if generated_copy is not None:
        generated_pros = getattr(generated_copy, "pros", [])
        generated_cons = getattr(generated_copy, "cons", [])
        if generated_pros:
            pros = generated_pros + pros
        if generated_cons:
            cons = cons + generated_cons

    return DestinationSuggestion(
        name=candidate.name,
        country=_country_from_url(candidate),
        score=int(round(candidate.score or 0)),
        score_breakdown=candidate.score_breakdown,
        pros=pros[:3],
        cons=cons[:3],
        estimated_cost=estimate_trip_cost(candidate.name, request.trip_length, costs) or 0,
        tool_signals=gather_tool_insights(
            request.origin_city,
            candidate.name,
            request.date_range,
            request.trip_length,
        ),
        citations=_citations_from_candidate(candidate),
    )


def _top_positive_factors(candidate: CandidateDestination) -> list[str]:
    labels = {
        "retrieval": "retrieved evidence",
        "budget": "budget fit",
        "transport": "transport fit",
        "interests": "interest match",
        "taste": "taste profile",
        "season": "season fit",
    }
    positives = [
        (labels.get(key, key), value)
        for key, value in candidate.score_breakdown.items()
        if value > 0
    ]
    positives.sort(key=lambda item: item[1], reverse=True)
    return [label for label, _ in positives[:3]]


def _tradeoff_summary(state: PlanningState, suggestions: list[DestinationSuggestion]) -> list[str]:
    if not state.candidate_destinations or not suggestions:
        return []
    top_candidate = state.candidate_destinations[0]
    factors = _top_positive_factors(top_candidate)
    summary = []
    if factors:
        summary.append(
            f"{top_candidate.name} ranks first mainly because of "
            f"{', '.join(factors)}."
        )
    if len(state.candidate_destinations) > 1:
        runner_up = state.candidate_destinations[1]
        gap = round((top_candidate.score or 0) - (runner_up.score or 0), 2)
        summary.append(
            f"It beat {runner_up.name} by {gap} points after budget, transport, "
            "interest, season, and retrieval signals were combined."
        )
    if state.filtered_candidates:
        filtered = state.filtered_candidates[0]
        summary.append(
            f"{filtered.name} was filtered out: {filtered.filtered_reason}."
        )
    summary.append(
        "Raw RAG similarity is used for discovery, but budget and transport are treated as stronger ranking constraints."
    )
    return summary[:4]


def build_trip_plan_from_ranked_state(
    state: PlanningState,
    generated_copy: dict[str, object] | None = None,
    planner_mode: str = "rag_ranked",
) -> TripPlan:
    """Convert ranked RAG candidates into the existing TripPlan response schema."""
    ranked_candidates = state.candidate_destinations[:3]
    if not ranked_candidates:
        raise ValueError("No ranked candidates available")

    costs = _load_costs()
    suggestions = [
        _suggestion_from_candidate(
            candidate,
            state,
            costs,
            (generated_copy or {}).get(candidate.name),
        )
        for candidate in ranked_candidates
    ]
    top = suggestions[0]
    interests = ", ".join(state.request.interests) or "local highlights"
    itinerary = [
        ItineraryDay(
            day=day,
            destination=top.name,
            plan=f"Use retrieved local guidance to explore {top.name}, with a focus on {interests}.",
        )
        for day in range(1, state.request.trip_length + 1)
    ]
    risks = [
        "Candidate discovery is based on the local Wikivoyage RAG index.",
        "Costs and transport signals are still mock estimates.",
        "Ranking uses transparent heuristics; verify logistics before booking.",
    ]
    explanation = (
        f"{top.name} ranks highest because it best balances retrieved travel evidence, "
        f"your {state.request.transport_preference} preference, €{state.request.budget} budget, "
        f"and {state.request.trip_length}-day trip length."
    )
    return TripPlan(
        ranked_destinations=suggestions,
        itinerary=itinerary,
        risks=risks,
        final_recommendation=f"Choose {top.name}, {top.country} for the best match.",
        explanation=explanation,
        planner_mode=planner_mode,
        fallback_reason=None,
        tradeoff_summary=_tradeoff_summary(state, suggestions),
        filtered_candidates=state.filtered_candidates,
    )
