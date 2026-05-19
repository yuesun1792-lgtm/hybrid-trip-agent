from backend.app.models import TripRequest
from backend.app.planner import create_trip_plan


def test_create_trip_plan_returns_three_destinations_and_itinerary_days():
    request = TripRequest(
        origin_city="Munich",
        date_range="June 10-15",
        trip_length=4,
        budget=700,
        interests=["food", "history"],
        transport_preference="train",
        physical_level="medium",
    )

    plan = create_trip_plan(request)

    assert len(plan.ranked_destinations) == 3
    assert len(plan.itinerary) == 4
    assert plan.final_recommendation
