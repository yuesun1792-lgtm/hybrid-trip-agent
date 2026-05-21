import os

import pytest

from backend.app.explainer import _fallback_explanation, generate_explanation
from backend.app.models import TripRequest
from backend.app.planner import create_trip_plan


def _make_plan():
    request = TripRequest(
        origin_city="Munich",
        date_range="June 10-15",
        trip_length=4,
        budget=700,
        interests=["food", "history"],
        transport_preference="train",
        physical_level="medium",
    )
    return create_trip_plan(request)


def test_create_trip_plan_returns_three_destinations_and_itinerary_days():
    plan = _make_plan()
    assert len(plan.ranked_destinations) == 3
    assert len(plan.itinerary) == 4
    assert plan.final_recommendation


def test_fallback_explanation_contains_top_destination():
    plan = _make_plan()
    explanation = _fallback_explanation(plan)
    top_dest = plan.ranked_destinations[0].name
    assert top_dest in explanation


def test_fallback_explanation_contains_budget():
    plan = _make_plan()
    explanation = _fallback_explanation(plan)
    assert str(plan.ranked_destinations[0].estimated_cost) in explanation


def test_generate_explanation_uses_fallback_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan = _make_plan()
    explanation = generate_explanation(plan)
    # Fallback must be deterministic and mention the top destination
    assert plan.ranked_destinations[0].name in explanation
