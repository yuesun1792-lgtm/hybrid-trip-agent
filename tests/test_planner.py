import json
import os
from unittest.mock import MagicMock, patch

import pytest

from backend.app.explainer import _fallback_explanation, generate_explanation
from backend.app.models import ToolInsight, TripRequest
from backend.app.planner import create_trip_plan, score_destination
from backend.app.tools import check_budget_fit, dispatch_tool, get_destination_details


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
# V2: mocked tool-use loop
# ---------------------------------------------------------------------------

def test_explain_with_tools_mocked_loop(monkeypatch):
    """Stub the Anthropic client to emit one tool_use then end_turn; verify dispatch + return."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    # Build mock content blocks
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "tu_1"
    tool_use_block.name = "check_budget_fit"
    tool_use_block.input = {"name": "Lisbon", "budget": 700}

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Lisbon is a great pick within budget!"

    first_response = MagicMock()
    first_response.stop_reason = "tool_use"
    first_response.content = [tool_use_block]

    second_response = MagicMock()
    second_response.stop_reason = "end_turn"
    second_response.content = [text_block]

    mock_create = MagicMock(side_effect=[first_response, second_response])
    mock_client = MagicMock()
    mock_client.messages.create = mock_create

    import backend.app.agent as agent_module
    with patch.object(agent_module, "anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value = mock_client
        plan = _make_plan()
        result = agent_module.explain_with_tools(plan)

    assert result == "Lisbon is a great pick within budget!"
    assert mock_create.call_count == 2


def test_generate_explanation_falls_back_on_agent_error(monkeypatch):
    """If the agent raises, generate_explanation returns the template fallback."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    import backend.app.agent as agent_module
    with patch.object(agent_module, "explain_with_tools", side_effect=RuntimeError("boom")):
        plan = _make_plan()
        result = generate_explanation(plan)

    assert plan.ranked_destinations[0].name in result


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
