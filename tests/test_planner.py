import json
import os
from unittest.mock import MagicMock, patch

import pytest

from backend.app.explainer import _fallback_explanation, generate_explanation
from backend.app.models import ToolInsight, TripRequest
from backend.app.planner import create_trip_plan, score_destination
from backend.app.currency_tool import convert_currency
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


def test_dispatch_tool_convert_currency(monkeypatch):
    """dispatch_tool routes convert_currency without making a real network call."""
    fake_response = {"amount": 100.0, "from": "EUR", "to": "USD",
                     "rate": 1.08, "converted": 108.0, "date": "2024-01-01"}

    def fake_convert(amount, from_currency, to_currency):
        return fake_response

    monkeypatch.setattr("backend.app.currency_tool.convert_currency", fake_convert)
    # Re-import so tools.py picks up the patched reference
    import importlib, backend.app.tools as tools_mod
    monkeypatch.setattr(tools_mod, "convert_currency", fake_convert)

    result = dispatch_tool("convert_currency", {"amount": 100, "from_currency": "EUR", "to_currency": "USD"})
    assert result["converted"] == 108.0
    assert result["rate"] == 1.08


# ---------------------------------------------------------------------------
# convert_currency: success and failure with stubbed urlopen
# ---------------------------------------------------------------------------

def _make_urlopen_mock(payload: dict):
    """Return a context-manager mock that yields a file-like object with json content."""
    import io
    fake_resp = MagicMock()
    fake_resp.__enter__ = lambda s: s
    fake_resp.__exit__ = MagicMock(return_value=False)
    fake_resp.read = MagicMock(return_value=json.dumps(payload).encode())
    # json.load calls resp.read() via the file protocol; patch at the json level instead
    fake_resp.__iter__ = MagicMock(return_value=iter([]))
    # urllib.request.urlopen is used as a context manager; make it return bytes via read
    cm = MagicMock()
    cm.__enter__ = lambda s: io.BytesIO(json.dumps(payload).encode())
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_convert_currency_success(monkeypatch):
    """Success path: urlopen returns valid ECB payload → correct converted amount."""
    payload = {"date": "2024-06-01", "rates": {"USD": 1.08}}

    with patch("backend.app.currency_tool.urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
        result = convert_currency(100.0, "EUR", "USD")

    assert result["converted"] == 108.0
    assert result["rate"] == 1.08
    assert result["from"] == "EUR"
    assert result["to"] == "USD"
    assert "error" not in result


def test_convert_currency_network_failure(monkeypatch):
    """Network failure → returns error dict, does not raise."""
    import urllib.error
    with patch("backend.app.currency_tool.urllib.request.urlopen",
               side_effect=urllib.error.URLError("timeout")):
        result = convert_currency(100.0, "EUR", "GBP")

    assert "error" in result
    assert "detail" in result


def test_convert_currency_same_currency():
    """Same currency → returns immediately without any network call."""
    result = convert_currency(200.0, "EUR", "EUR")
    assert result["converted"] == 200.0
    assert result["rate"] == 1.0
    assert "error" not in result


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


# ---------------------------------------------------------------------------
# home_currency: models + agent loop
# ---------------------------------------------------------------------------

def test_trip_request_home_currency_defaults_to_cny():
    req = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food"], transport_preference="train", physical_level="low",
    )
    assert req.home_currency == "CNY"


def test_trip_request_home_currency_custom():
    req = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food"], transport_preference="train", physical_level="low",
        home_currency="USD",
    )
    assert req.home_currency == "USD"


def test_destination_suggestion_accepts_home_currency_fields():
    from backend.app.models import DestinationSuggestion
    dest = DestinationSuggestion(
        name="Lisbon", country="Portugal", score=10,
        pros=["Food"], cons=["Hilly"], estimated_cost=650,
        estimated_cost_home=4680.5, home_currency="CNY",
    )
    assert dest.estimated_cost_home == 4680.5
    assert dest.home_currency == "CNY"


def test_destination_suggestion_home_currency_fields_optional():
    from backend.app.models import DestinationSuggestion
    dest = DestinationSuggestion(
        name="Prague", country="Czechia", score=8,
        pros=["Cheap"], cons=["Touristy"], estimated_cost=520,
    )
    assert dest.estimated_cost_home is None
    assert dest.home_currency is None


_VALID_SUBMIT_INPUT_WITH_HOME = {
    "ranked_destinations": [
        {"name": "Lisbon",    "country": "Portugal", "score": 10, "pros": ["Great food"],
         "cons": ["Hilly"],    "estimated_cost": 650,
         "estimated_cost_home": 4225.0, "home_currency": "CNY"},
        {"name": "Prague",    "country": "Czechia",  "score":  8, "pros": ["Cheap"],
         "cons": ["Touristy"], "estimated_cost": 520,
         "estimated_cost_home": 3380.0, "home_currency": "CNY"},
        {"name": "Ljubljana", "country": "Slovenia", "score":  6, "pros": ["Relaxed"],
         "cons": ["Small"],    "estimated_cost": 580,
         "estimated_cost_home": 3770.0, "home_currency": "CNY"},
    ],
    "itinerary": [{"day": 1, "destination": "Lisbon", "plan": "Explore."}],
    "risks": ["Costs are estimates."],
    "final_recommendation": "Choose Lisbon.",
    "explanation": "Lisbon is ideal within budget.",
}


def test_create_plan_calls_convert_currency_for_non_eur(monkeypatch):
    """Loop: convert_currency dispatched for CNY; estimated_cost_home arrives in final plan."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    convert_block = _make_tool_use_block(
        "convert_currency", "tu_1",
        {"amount": 650, "from_currency": "EUR", "to_currency": "CNY"},
    )
    submit_block = _make_tool_use_block("submit_trip_plan", "tu_2", _VALID_SUBMIT_INPUT_WITH_HOME)

    mock_create = MagicMock(side_effect=[
        _make_response("tool_use", [convert_block]),
        _make_response("tool_use", [submit_block]),
    ])
    mock_client = MagicMock()
    mock_client.messages.create = mock_create

    import backend.app.agent as agent_module
    import backend.app.currency_tool as ct_mod
    # Stub convert_currency so no real network call happens
    monkeypatch.setattr(ct_mod, "convert_currency",
                        lambda amount, from_currency, to_currency:
                        {"amount": amount, "from": from_currency, "to": to_currency,
                         "rate": 6.5, "converted": round(amount * 6.5, 2), "date": "2024-01-01"})
    import backend.app.tools as tools_mod
    monkeypatch.setattr(tools_mod, "convert_currency",
                        lambda amount, from_currency, to_currency:
                        {"amount": amount, "from": from_currency, "to": to_currency,
                         "rate": 6.5, "converted": round(amount * 6.5, 2), "date": "2024-01-01"})

    request = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food", "history"],
        transport_preference="train", physical_level="medium",
        home_currency="CNY",
    )
    with patch.object(agent_module, "anthropic") as mock_anthropic:
        mock_anthropic.Anthropic.return_value = mock_client
        plan = agent_module.create_plan(request)

    # convert_currency was dispatched (the loop processed the tool_use block)
    assert mock_create.call_count == 2
    # estimated_cost_home arrived from the submit_trip_plan input (from the tool result)
    assert plan.ranked_destinations[0].estimated_cost_home == 4225.0
    assert plan.ranked_destinations[0].home_currency == "CNY"


def test_no_key_fallback_returns_none_for_home_currency_fields(monkeypatch):
    """Without API key the deterministic planner runs; home-currency fields are absent."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    req = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food", "history"],
        transport_preference="train", physical_level="medium",
        home_currency="CNY",
    )
    plan = create_trip_plan(req)
    for dest in plan.ranked_destinations:
        assert dest.estimated_cost_home is None
        assert dest.home_currency is None


def test_build_agent_system_prompt_eur_skips_conversion():
    from backend.app.prompts import build_agent_system_prompt
    prompt = build_agent_system_prompt("EUR")
    assert "no currency conversion" in prompt
    assert "convert_currency" not in prompt.split("convert_currency")[1] if "convert_currency" in prompt else True


def test_build_agent_system_prompt_cny_includes_instruction():
    from backend.app.prompts import build_agent_system_prompt
    prompt = build_agent_system_prompt("CNY")
    assert "CNY" in prompt
    assert "convert_currency" in prompt
    assert "estimated_cost_home" in prompt


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
