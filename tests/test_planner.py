import json
import os
from unittest.mock import MagicMock, patch

import pytest

from backend.app.explainer import _fallback_explanation, generate_explanation
from backend.app.models import ToolInsight, TripRequest
from backend.app.planner import create_trip_plan, score_destination
from backend.app.tools import (
    check_budget_fit,
    dispatch_tool,
    get_destination_details,
    list_destinations,
    score_destinations,
)


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


def test_tool_signals_attached_to_all_ranked_destinations():
    plan = _make_plan()
    for dest in plan.ranked_destinations:
        assert dest.tool_signals is not None, f"{dest.name} is missing tool_signals"
        assert dest.tool_signals.transport_hours > 0
        assert dest.tool_signals.weather_risk in ("low", "medium", "high")
        assert dest.tool_signals.estimated_total_cost > 0


# ---------------------------------------------------------------------------
# V2: tools.py unit tests
# ---------------------------------------------------------------------------

def test_get_destination_details_known():
    result = get_destination_details("Lisbon")
    assert result["name"] == "Lisbon"
    assert "estimated_cost" in result
    assert "interests" in result


def test_get_destination_details_case_insensitive():
    result = get_destination_details("lisbon")
    assert result["name"] == "Lisbon"


def test_get_destination_details_unknown():
    result = get_destination_details("Atlantis")
    assert "error" in result
    assert result["name"] == "Atlantis"


def test_check_budget_fit_within_budget():
    result = check_budget_fit("Prague", 700)
    assert result["fits"] is True
    assert result["delta"] > 0
    assert result["estimated_cost"] == 520


def test_check_budget_fit_over_budget():
    result = check_budget_fit("Amsterdam", 700)
    assert result["fits"] is False
    assert result["delta"] < 0


def test_check_budget_fit_unknown_destination():
    result = check_budget_fit("Atlantis", 700)
    assert "error" in result


def test_dispatch_tool_get_destination_details():
    result = dispatch_tool("get_destination_details", {"name": "Prague"})
    assert result["name"] == "Prague"


def test_dispatch_tool_check_budget_fit():
    result = dispatch_tool("check_budget_fit", {"name": "Lisbon", "budget": 700})
    assert "fits" in result


def test_dispatch_tool_unknown():
    result = dispatch_tool("fly_me_to_the_moon", {})
    assert "error" in result


# ---------------------------------------------------------------------------
# V3: list_destinations and score_destinations
# ---------------------------------------------------------------------------

def test_list_destinations_returns_all_destinations():
    results = list_destinations()
    assert len(results) >= 4
    names = [d["name"] for d in results]
    assert "Lisbon" in names
    assert "Prague" in names


def test_list_destinations_entries_have_required_keys():
    for entry in list_destinations():
        assert "name" in entry
        assert "country" in entry
        assert "summary" in entry


def test_score_destinations_returns_ranked_list():
    request = {
        "origin_city": "Munich",
        "date_range": "June 10-15",
        "trip_length": 4,
        "budget": 700,
        "interests": ["food", "history"],
        "transport_preference": "train",
        "physical_level": "medium",
    }
    results = score_destinations(request)
    assert len(results) >= 4
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be ranked best-first"


def test_score_destinations_entries_have_required_keys():
    request = {
        "origin_city": "Munich",
        "date_range": "June 10-15",
        "trip_length": 4,
        "budget": 700,
        "interests": ["food"],
        "transport_preference": "train",
        "physical_level": "medium",
    }
    for entry in score_destinations(request):
        assert "name" in entry
        assert "score" in entry
        assert "estimated_cost" in entry


def test_score_destinations_invalid_request_returns_error():
    result = score_destinations({"bad_field": "x"})
    assert len(result) == 1
    assert "error" in result[0]


def test_dispatch_tool_list_destinations():
    result = dispatch_tool("list_destinations", {})
    assert "destinations" in result
    assert len(result["destinations"]) >= 1


def test_dispatch_tool_score_destinations():
    result = dispatch_tool("score_destinations", {
        "origin_city": "Munich",
        "date_range": "June 10-15",
        "trip_length": 4,
        "budget": 700,
        "interests": ["food"],
        "transport_preference": "train",
        "physical_level": "medium",
    })
    assert "ranked" in result
    assert len(result["ranked"]) >= 1


# ---------------------------------------------------------------------------
# V3: mocked create_plan loop
# ---------------------------------------------------------------------------

_VALID_SUBMIT_INPUT = {
    "ranked_destinations": [
        {"name": "Lisbon",    "country": "Portugal", "score": 10, "pros": ["Great food"], "cons": ["Hilly"],    "estimated_cost": 650},
        {"name": "Prague",    "country": "Czechia",  "score":  8, "pros": ["Cheap"],      "cons": ["Touristy"], "estimated_cost": 520},
        {"name": "Ljubljana", "country": "Slovenia", "score":  6, "pros": ["Relaxed"],    "cons": ["Small"],    "estimated_cost": 580},
    ],
    "itinerary": [
        {"day": 1, "destination": "Lisbon", "plan": "Explore the old town."},
        {"day": 2, "destination": "Lisbon", "plan": "Visit Alfama district."},
        {"day": 3, "destination": "Lisbon", "plan": "Day trip to Sintra."},
        {"day": 4, "destination": "Lisbon", "plan": "Free day."},
    ],
    "risks": ["Costs are estimates.", "Weather may vary."],
    "final_recommendation": "Choose Lisbon for the best match.",
    "explanation": "Lisbon suits your food and history interests within budget.",
}


def _make_tool_use_block(name, tool_id, input_data):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


def _make_response(stop_reason, blocks):
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = blocks
    return r


def test_create_plan_mocked_loop_succeeds(monkeypatch):
    """Stub: list_destinations → submit_trip_plan. Assert valid TripPlan returned."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    list_block = _make_tool_use_block("list_destinations", "tu_1", {})
    submit_block = _make_tool_use_block("submit_trip_plan", "tu_2", _VALID_SUBMIT_INPUT)

    mock_create = MagicMock(side_effect=[
        _make_response("tool_use", [list_block]),
        _make_response("tool_use", [submit_block]),
    ])
    mock_client = MagicMock()
    mock_client.messages.create = mock_create

    import backend.app.agent as agent_module
    request = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food", "history"],
        transport_preference="train", physical_level="medium",
    )
    with patch.object(agent_module, "anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value = mock_client
        plan = agent_module.create_plan(request)

    assert len(plan.ranked_destinations) == 3
    assert plan.ranked_destinations[0].name == "Lisbon"
    assert plan.explanation == "Lisbon suits your food and history interests within budget."
    assert mock_create.call_count == 2


def test_create_plan_fallback_when_no_submit(monkeypatch):
    """Stub always returns end_turn; after nudge still end_turn → deterministic fallback."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    end_turn_response = _make_response("end_turn", [])
    mock_create = MagicMock(return_value=end_turn_response)
    mock_client = MagicMock()
    mock_client.messages.create = mock_create

    import backend.app.agent as agent_module
    request = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food", "history"],
        transport_preference="train", physical_level="medium",
    )
    with patch.object(agent_module, "anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value = mock_client
        plan = agent_module.create_plan(request)

    # Fallback must return a valid TripPlan from the deterministic planner
    assert len(plan.ranked_destinations) == 3
    assert plan.explanation is not None
    assert plan.ranked_destinations[0].name in plan.explanation


def test_tool_signals_influence_score_deterministically():
    # A generic destination dict that mirrors the JSON structure
    destination = {
        "name": "Lisbon",
        "country": "Portugal",
        "interests": ["food", "history"],
        "transport_options": ["flight", "train"],
        "physical_level": "medium",
        "estimated_cost": 650,
    }
    request = TripRequest(
        origin_city="Munich",
        date_range="June 10-15",
        trip_length=4,
        budget=700,
        interests=["food", "history"],
        transport_preference="train",
        physical_level="medium",
    )

    base_score = score_destination(destination, request)

    # All three bonuses: short travel (+1), low weather risk (+1), cost within budget (+1)
    good_insight = ToolInsight(
        transport_hours=2.0,
        transport_mode="flight",
        weather_risk="low",
        weather_note="Sunny",
        estimated_total_cost=600,
    )
    assert score_destination(destination, request, good_insight) == base_score + 3

    # Both penalties: long travel (-1), high weather risk (-1); cost > budget, no bonus
    bad_insight = ToolInsight(
        transport_hours=7.5,
        transport_mode="train",
        weather_risk="high",
        weather_note="Stormy",
        estimated_total_cost=800,
    )
    assert score_destination(destination, request, bad_insight) == base_score - 2
