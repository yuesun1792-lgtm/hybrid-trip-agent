import json

from backend.app.config import DESTINATION_NOTES_PATH
from backend.app.domain.models import DestinationSuggestion, ItineraryDay, TripPlan, TripRequest, ToolInsight
from backend.app.tool_calling.tools import gather_tool_insights


def load_destination_notes() -> list[dict]:
    with DESTINATION_NOTES_PATH.open() as file:
        return json.load(file)


def score_destination(
    destination: dict,
    request: TripRequest,
    tool_insight: ToolInsight | None = None,
) -> int:
    score = 0

    # Static rules (unchanged from V0)
    matching_interests = set(request.interests) & set(destination["interests"])
    score += len(matching_interests) * 3

    if destination["estimated_cost"] <= request.budget:
        score += 2
    else:
        score -= 2

    if request.transport_preference in destination["transport_options"]:
        score += 1

    if request.physical_level == destination["physical_level"]:
        score += 1

    # Tool-signal adjustments: small, deterministic, additive
    if tool_insight is not None:
        if tool_insight.transport_mode == request.transport_preference:
            score += 2
        else:
            score -= 3

        if tool_insight.transport_mode == request.transport_preference and tool_insight.transport_hours <= 3:
            score += 1
        elif tool_insight.transport_hours > 6:
            score -= 1

        if tool_insight.weather_risk == "low":
            score += 1
        elif tool_insight.weather_risk == "high":
            score -= 1

        if tool_insight.estimated_total_cost <= request.budget:
            score += 1

    return score


def create_trip_plan(request: TripRequest) -> TripPlan:
    destinations = load_destination_notes()

    # Gather tool insights for every candidate once, reuse for scoring and response
    insights: dict[str, ToolInsight] = {
        d["name"]: gather_tool_insights(
            request.origin_city, d["name"], request.date_range, request.trip_length
        )
        for d in destinations
    }

    ranked = sorted(
        destinations,
        key=lambda d: score_destination(d, request, insights[d["name"]]),
        reverse=True,
    )[:3]

    suggestions = [
        DestinationSuggestion(
            name=d["name"],
            country=d["country"],
            score=score_destination(d, request, insights[d["name"]]),
            pros=d["pros"],
            cons=d["cons"],
            estimated_cost=d["estimated_cost"],
            tool_signals=insights[d["name"]],
        )
        for d in ranked
    ]

    top_destination = suggestions[0]
    itinerary = [
        ItineraryDay(
            day=day,
            destination=top_destination.name,
            plan=f"Explore {top_destination.name} with a focus on {', '.join(request.interests) or 'local highlights'}.",
        )
        for day in range(1, request.trip_length + 1)
    ]

    risks = [
        "Costs are rough mock estimates.",
        "Transport times are mock values, not real schedules.",
        "Weather risk is based on historical patterns, not a live forecast.",
    ]

    return TripPlan(
        ranked_destinations=suggestions,
        itinerary=itinerary,
        risks=risks,
        final_recommendation=f"Choose {top_destination.name}, {top_destination.country} for the best match.",
    )
