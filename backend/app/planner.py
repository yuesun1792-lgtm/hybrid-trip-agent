import json

from backend.app.config import DESTINATION_NOTES_PATH
from backend.app.models import DestinationSuggestion, ItineraryDay, TripPlan, TripRequest


def load_destination_notes() -> list[dict]:
    with DESTINATION_NOTES_PATH.open() as file:
        return json.load(file)


def score_destination(destination: dict, request: TripRequest) -> int:
    score = 0

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

    return score


def create_trip_plan(request: TripRequest) -> TripPlan:
    destinations = load_destination_notes()

    ranked = sorted(
        destinations,
        key=lambda destination: score_destination(destination, request),
        reverse=True,
    )[:3]

    suggestions = [
        DestinationSuggestion(
            name=destination["name"],
            country=destination["country"],
            score=score_destination(destination, request),
            pros=destination["pros"],
            cons=destination["cons"],
            estimated_cost=destination["estimated_cost"],
        )
        for destination in ranked
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
        "Transport times are not checked against real schedules.",
        "Weather and local events are not considered in V0.",
    ]

    return TripPlan(
        ranked_destinations=suggestions,
        itinerary=itinerary,
        risks=risks,
        final_recommendation=f"Choose {top_destination.name}, {top_destination.country} for the best V0 match.",
    )

