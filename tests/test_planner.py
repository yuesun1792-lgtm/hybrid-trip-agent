import json
import os
from unittest.mock import MagicMock, patch

import pytest

from backend.app.explainer import _fallback_explanation, generate_explanation
from backend.app.models import CandidateDestination, EvidenceChunk, ToolInsight, TripRequest
from backend.app.planner import create_trip_plan, score_destination
from backend.app.currency_tool import convert_currency
from backend.app.planning_engine import deterministic_plan_from_state, state_from_request
from backend.app.rag_enhancements import ResearchDecision
from backend.app.candidate_discovery import (
    build_discovery_queries,
    discover_candidate_destinations,
)
from backend.app.candidate_ranker import rank_candidate_destinations, score_candidate
from backend.app.rag_plan_builder import build_trip_plan_from_ranked_state
from backend.app.costs import estimate_trip_cost, normalize_plan_costs
from backend.app.chat import extract_fields, handle_chat, missing_fields
from backend.app.models import ChatRequest
from backend.app.strategy import decide_strategy
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


def test_fallback_explanation_uses_request_budget_when_available():
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["food", "history", "nature"],
        transport_preference="train",
        physical_level="high",
    )
    plan = create_trip_plan(request)
    explanation = _fallback_explanation(plan, request)
    assert "€1000 budget" in explanation


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


def test_state_from_request_preserves_normalized_request():
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["food", "history", "nature"],
        transport_preference="train",
        physical_level="high",
    )
    state = state_from_request(request, conversation_id="conv-1")
    assert state.request == request
    assert state.conversation_id == "conv-1"
    assert state.stage == "ready_to_plan"
    assert state.candidate_destinations == []
    assert state.evidence == []


def test_deterministic_plan_from_state_keeps_existing_plan_shape():
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["food", "history", "nature"],
        transport_preference="train",
        physical_level="high",
    )
    state = state_from_request(request)
    plan = deterministic_plan_from_state(state, "test")
    assert len(plan.ranked_destinations) == 3
    assert len(plan.itinerary) == 4
    assert plan.planner_mode == "unified_itinerary_rule"
    assert plan.fallback_reason == "test"


def test_build_discovery_queries_include_request_constraints():
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["food", "history", "nature"],
        transport_preference="train",
        physical_level="high",
        taste_profile="I like mountain towns and dislike tourist traps.",
    )
    queries = build_discovery_queries(state_from_request(request))
    joined = " ".join(queries)
    assert "Munich" in joined
    assert "train" in joined
    assert "nature" in joined
    assert "high" in joined
    assert "mountain towns" in joined


def test_discover_candidate_destinations_groups_rag_evidence(monkeypatch):
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["nature", "history"],
        transport_preference="train",
        physical_level="high",
    )
    state = state_from_request(request)

    fake_results = [
        {
            "text": "Salzburg - Do\nMountain access and historic center.",
            "score": 0.9,
            "destination": "Salzburg",
            "section": "Do",
            "url": "https://en.wikivoyage.org/wiki/Salzburg",
        },
        {
            "text": "Innsbruck - See\nAlpine city with old town.",
            "score": 0.8,
            "destination": "Innsbruck",
            "section": "See",
            "url": "https://en.wikivoyage.org/wiki/Innsbruck",
        },
        {
            "text": "Salzburg - See\nBaroque old town.",
            "score": 0.7,
            "destination": "Salzburg",
            "section": "See",
            "url": "https://en.wikivoyage.org/wiki/Salzburg",
        },
    ]

    monkeypatch.setattr(
        "backend.app.candidate_discovery.search_travel_knowledge",
        lambda query, k=6: fake_results,
    )
    discovered = discover_candidate_destinations(state)
    assert discovered.stage == "candidates_discovered"
    assert [candidate.name for candidate in discovered.candidate_destinations] == [
        "Salzburg",
        "Innsbruck",
    ]
    assert discovered.candidate_destinations[0].source == "rag"
    assert len(discovered.candidate_destinations[0].evidence) == 2
    assert discovered.evidence


def test_score_candidate_adds_breakdown(monkeypatch):
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["nature", "history"],
        transport_preference="train",
        physical_level="high",
    )
    state = state_from_request(request)
    candidate = CandidateDestination(
        name="Salzburg",
        source="rag",
        score=0.9,
        evidence=[
            EvidenceChunk(
                text="Salzburg has nature, mountain access, and history in the old town.",
                destination="Salzburg",
                section="Do",
                score=0.9,
            )
        ],
    )
    monkeypatch.setattr(
        "backend.app.candidate_ranker.estimate_transport_time",
        lambda origin, destination: {"hours": 2.0, "mode": "train"},
    )
    scored = score_candidate(candidate, state, {"Salzburg": 150})
    assert scored.score > 0
    assert scored.score_breakdown["transport"] == 20
    assert scored.score_breakdown["budget"] == 25
    assert scored.score_breakdown["interests"] > 0


def test_estimate_trip_cost_scales_four_day_mock_cost():
    assert estimate_trip_cost("Salzburg", 2, {"Salzburg": 175}) == 350
    assert estimate_trip_cost("Salzburg", 4, {"Salzburg": 175}) == 700


def test_rank_candidate_destinations_orders_by_constraints(monkeypatch):
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["nature", "history"],
        transport_preference="train",
        physical_level="high",
    )
    state = state_from_request(request).model_copy(
        update={
            "candidate_destinations": [
                CandidateDestination(
                    name="Lisbon",
                    source="rag",
                    score=0.95,
                    evidence=[EvidenceChunk(text="Lisbon has history.", destination="Lisbon")],
                ),
                CandidateDestination(
                    name="Salzburg",
                    source="rag",
                    score=0.75,
                    evidence=[
                        EvidenceChunk(
                            text="Salzburg has nature, mountain access, and history.",
                            destination="Salzburg",
                        )
                    ],
                ),
            ]
        }
    )
    monkeypatch.setattr(
        "backend.app.candidate_ranker.estimate_transport_time",
        lambda origin, destination: {
            "Lisbon": {"hours": 3.0, "mode": "flight"},
            "Salzburg": {"hours": 2.0, "mode": "train"},
        }[destination],
    )
    ranked = rank_candidate_destinations(state)
    assert ranked.stage == "candidates_ranked"
    assert ranked.candidate_destinations[0].name == "Salzburg"
    assert ranked.candidate_destinations[0].score_breakdown["transport"] == 20
    assert ranked.candidate_destinations[1].score_breakdown["transport"] == -15


def test_rank_candidate_destinations_filters_over_budget_when_matches_exist(monkeypatch):
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=2,
        budget=400,
        interests=["nature", "history"],
        transport_preference="bus",
        physical_level="medium",
    )
    state = state_from_request(request).model_copy(
        update={
            "candidate_destinations": [
                CandidateDestination(
                    name="Salzburg",
                    source="rag",
                    score=0.7,
                    evidence=[EvidenceChunk(text="Salzburg has nature and history.")],
                ),
                CandidateDestination(
                    name="Paris",
                    source="rag",
                    score=0.95,
                    evidence=[EvidenceChunk(text="Paris has food, history, and architecture.")],
                ),
            ]
        }
    )
    monkeypatch.setattr(
        "backend.app.candidate_ranker.estimate_transport_time",
        lambda origin, destination: {"hours": 2.0, "mode": "bus"},
    )
    ranked = rank_candidate_destinations(state)
    names = [candidate.name for candidate in ranked.candidate_destinations]
    assert "Salzburg" in names
    assert "Paris" not in names
    assert ranked.filtered_candidates[0].name == "Paris"
    assert "exceeds €400 budget" in ranked.filtered_candidates[0].filtered_reason


def test_build_trip_plan_from_ranked_state_uses_candidates(monkeypatch):
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["nature", "history"],
        transport_preference="train",
        physical_level="high",
    )
    ranked_state = state_from_request(request).model_copy(
        update={
            "stage": "candidates_ranked",
            "candidate_destinations": [
                CandidateDestination(
                    name="Salzburg",
                    country="Austria",
                    source="rag",
                    score=66,
                    score_breakdown={"transport": 20, "budget": 15},
                    rationale="transport train from Munich; within budget",
                    evidence=[
                        EvidenceChunk(
                            text="Salzburg - Do\nMountain access and historic center.",
                            destination="Salzburg",
                            section="Do",
                            url="https://en.wikivoyage.org/wiki/Salzburg",
                        )
                    ],
                )
            ],
        }
    )
    monkeypatch.setattr(
        "backend.app.tools.gather_tool_insights",
        lambda origin, destination, date_range, trip_length: ToolInsight(
            transport_hours=2,
            transport_mode="train",
            weather_risk="low",
            weather_note="Warm",
            estimated_total_cost=700,
        ),
    )
    plan = build_trip_plan_from_ranked_state(ranked_state)
    assert plan.planner_mode == "unified_itinerary_rule"
    assert plan.ranked_destinations[0].name == "Salzburg"
    assert plan.ranked_destinations[0].estimated_cost == 700
    assert plan.ranked_destinations[0].score_breakdown["transport"] == 20
    assert plan.ranked_destinations[0].citations[0].url
    assert plan.tradeoff_summary
    assert len(plan.itinerary) == 4


def test_build_trip_plan_from_ranked_state_scales_cost_to_trip_length(monkeypatch):
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=2,
        budget=400,
        interests=["nature", "history"],
        transport_preference="bus",
        physical_level="medium",
    )
    ranked_state = state_from_request(request).model_copy(
        update={
            "stage": "candidates_ranked",
            "candidate_destinations": [
                CandidateDestination(
                    name="Salzburg",
                    country="Austria",
                    source="rag",
                    score=66,
                    score_breakdown={"transport": 20, "budget": 25},
                    rationale="transport bus from Munich; estimated €350",
                    evidence=[EvidenceChunk(text="Salzburg - Do\nHistoric center and nature.")],
                )
            ],
        }
    )
    monkeypatch.setattr(
        "backend.app.tools.gather_tool_insights",
        lambda origin, destination, date_range, trip_length: ToolInsight(
            transport_hours=2,
            transport_mode="bus",
            weather_risk="low",
            weather_note="Warm",
            estimated_total_cost=350,
        ),
    )
    plan = build_trip_plan_from_ranked_state(ranked_state)
    assert plan.ranked_destinations[0].estimated_cost == 350


def test_normalize_plan_costs_scales_agent_output_and_home_currency():
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=2,
        budget=400,
        interests=["nature"],
        transport_preference="bus",
        physical_level="medium",
    )
    plan = create_trip_plan(request).model_copy(
        update={
            "ranked_destinations": [
                create_trip_plan(request).ranked_destinations[0].model_copy(
                    update={
                        "name": "Salzburg",
                        "estimated_cost": 700,
                        "estimated_cost_home": 5434,
                    }
                )
            ]
        }
    )
    normalized = normalize_plan_costs(plan, request)
    assert normalized.ranked_destinations[0].estimated_cost == 350
    assert normalized.ranked_destinations[0].estimated_cost_home == 2717


def test_plan_trip_uses_rag_when_agent_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_ANTHROPIC_AGENT", "false")

    from backend.app.main import plan_trip
    import backend.app.main as main_module

    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["nature", "history"],
        transport_preference="train",
        physical_level="high",
    )
    expected = create_trip_plan(request).model_copy(
        update={"planner_mode": "rag_ranked", "fallback_reason": None}
    )
    monkeypatch.setattr(main_module, "plan_from_state", lambda state: expected)
    plan = plan_trip(request)
    assert plan.planner_mode == "rag_ranked"
    assert plan.fallback_reason is None


def test_train_preference_uses_actual_transport_signal():
    request = TripRequest(
        origin_city="Munich",
        date_range="July 10-15",
        trip_length=4,
        budget=1000,
        interests=["food", "history", "nature"],
        transport_preference="train",
        physical_level="high",
    )
    plan = create_trip_plan(request)
    assert plan.ranked_destinations[0].name != "Lisbon"
    assert plan.ranked_destinations[0].tool_signals.transport_mode == "train"


def test_chat_extract_fields_from_natural_language():
    fields = extract_fields(
        "I have 2 days from Munich in July, budget 400, food history nature, prefer bus, medium pace."
    )
    assert fields["origin_city"] == "Munich"
    assert fields["trip_length"] == 2
    assert fields["budget"] == 400
    assert fields["transport_preference"] == "bus"
    assert fields["physical_level"] == "medium"
    assert fields["date_range"] == "July"
    assert fields["interests"] == ["food", "history", "nature"]
    assert missing_fields(fields) == []


def test_chat_extract_fields_from_fixed_destination():
    fields = extract_fields("3 days in Paris for food and art")
    assert fields["destination"] == "Paris"
    assert fields["trip_length"] == 3
    assert fields["interests"] == ["food", "art"]


def test_chat_extract_fields_from_chinese_fixed_destination():
    fields = extract_fields("我想去巴黎玩三天")
    assert fields["destination"] == "Paris"
    assert fields["trip_length"] == 3


def test_strategy_routes_fixed_destination(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fields = extract_fields("3 days in Paris for food and art")
    decision = decide_strategy("3 days in Paris for food and art", fields)
    assert decision.intent == "plan_trip"
    assert decision.next_action == "run_planning"


def test_chat_asks_for_missing_fields():
    response = handle_chat(ChatRequest(message="I want food and history from Munich."))
    assert response.plan is None
    assert response.missing_fields
    assert response.extracted_fields["origin_city"] == "Munich"
    assert "?" in response.assistant_message


def test_chat_routes_fixed_destination_plan(monkeypatch):
    monkeypatch.setattr(
        "backend.app.knowledge.search_travel_knowledge",
        lambda query, k=6, destination=None: [
            {
                "text": "Paris - See\nMuseums, neighborhoods, and riverside walks.",
                "score": 0.9,
                "destination": "Paris",
                "section": "See",
                "url": "https://en.wikivoyage.org/wiki/Paris",
            }
        ],
    )
    response = handle_chat(ChatRequest(message="3 days in Paris for food and art"))
    assert response.plan is not None
    assert response.plan.planner_mode == "unified_itinerary_rule"
    assert response.plan.ranked_destinations[0].name == "Paris"
    assert len(response.plan.itinerary) == 3


def test_chat_generates_plan_when_fields_complete(monkeypatch):
    import backend.app.chat as chat_module
    request = ChatRequest(
        message="2 days from Munich in July, budget 400, food history nature, bus, medium"
    )
    monkeypatch.setattr("backend.app.knowledge.search_travel_knowledge", lambda *a, **k: [
        {"text": "Prague - See\nParks, museums and walks offer local activities.", "destination": "Prague", "section": "See", "url": "https://en.wikivoyage.org/wiki/Prague", "score": .9}])
    response = handle_chat(request)
    assert response.plan is not None
    assert response.planning_task["operation"] == "create"
    assert response.plan.planner_mode == "unified_itinerary_rule"
    assert response.missing_fields == []



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
    """Stub: search_travel_knowledge → submit_trip_plan. Assert valid TripPlan returned."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    search_block = _make_tool_use_block(
        "search_travel_knowledge", "tu_1", {"query": "budget history food Europe"}
    )
    submit_block = _make_tool_use_block("submit_trip_plan", "tu_2", _VALID_SUBMIT_INPUT)

    mock_create = MagicMock(side_effect=[
        _make_response("tool_use", [search_block]),
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


def test_create_plan_recovers_from_max_tokens(monkeypatch):
    """If the model hits max_tokens once, the loop nudges it to submit instead of fallback."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    monkeypatch.setenv("ANTHROPIC_AGENT_MAX_TOKENS", "4096")

    submit_block = _make_tool_use_block("submit_trip_plan", "tu_1", _VALID_SUBMIT_INPUT)

    mock_create = MagicMock(side_effect=[
        _make_response("max_tokens", []),
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

    assert plan.planner_mode == "llm_agent"
    assert plan.fallback_reason is None
    assert mock_create.call_count == 2


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
    assert plan.ranked_destinations[0].estimated_cost_home == 4238.0
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


def test_plan_trip_ignores_legacy_agent_flag(monkeypatch):
    monkeypatch.setenv("ENABLE_ANTHROPIC_AGENT", "false")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    from backend.app.main import plan_trip
    import backend.app.main as main_module
    req = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food"], transport_preference="train", physical_level="low",
    )
    expected = create_trip_plan(req).model_copy(update={"planner_mode": "rag_ranked"})
    monkeypatch.setattr(main_module, "plan_from_state", lambda state: expected)
    plan = plan_trip(req)
    assert plan.planner_mode == "rag_ranked"


def test_plan_trip_without_key_still_uses_shared_planner(monkeypatch):
    monkeypatch.setenv("ENABLE_ANTHROPIC_AGENT", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    from backend.app.main import plan_trip
    import backend.app.main as main_module
    req = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food"], transport_preference="train", physical_level="low",
    )
    expected = create_trip_plan(req).model_copy(update={"planner_mode": "rag_ranked"})
    monkeypatch.setattr(main_module, "plan_from_state", lambda state: expected)
    plan = plan_trip(req)
    assert plan.planner_mode == "rag_ranked"


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


# ---------------------------------------------------------------------------
# V4: knowledge.py — wikitext parsing + search (no network, no model)
# ---------------------------------------------------------------------------

_FIXTURE_WIKITEXT = """\
== Understand ==
Lisbon is the capital of Portugal, set on the Tagus estuary. It has a mild Atlantic climate
with over 300 days of sunshine per year, making it one of the sunniest capitals in Europe.

== See ==
The Belém Tower and Jerónimos Monastery are UNESCO World Heritage sites absolutely worth
visiting. The historic Alfama district with its medieval castle and narrow cobblestone streets
is also a highlight. Tram 28 offers a scenic ride through several historic neighbourhoods.

== Eat ==
Pastel de nata (custard tarts) and grilled sardines are must-try local specialties. Fresh
seafood is abundant and affordable. The Time Out Market in Cais do Sodré brings together the
best food stalls and restaurants from across the city under one roof. Local wine is excellent.

== Stay safe ==
Lisbon is generally safe for tourists. Pickpocketing is common on tram 28 and in Alfama.
Avoid poorly-lit streets near the waterfront late at night. The LNB neighbourhood can be rough.
Keep an eye on your belongings in crowded tourist areas and on public transport.

== Stay healthy ==
Tap water is safe to drink throughout the city. Pharmacies are widely available, well-stocked,
and staff often speak English. EU health card covers basic medical care for European visitors.
"""


def test_parse_sections_splits_by_heading():
    from scripts.build_index import parse_sections
    chunks = parse_sections(_FIXTURE_WIKITEXT, "Lisbon", "https://en.wikivoyage.org/wiki/Lisbon")
    sections = [c["metadata"]["section"] for c in chunks]
    assert "Stay safe" in sections
    assert "Eat" in sections
    assert "See" in sections


def test_parse_sections_chunk_text_contains_destination_prefix():
    from scripts.build_index import parse_sections
    chunks = parse_sections(_FIXTURE_WIKITEXT, "Lisbon", "https://en.wikivoyage.org/wiki/Lisbon")
    for c in chunks:
        assert c["text"].startswith("Lisbon —")


def test_parse_sections_metadata_fields():
    from scripts.build_index import parse_sections
    chunks = parse_sections(_FIXTURE_WIKITEXT, "Lisbon", "https://en.wikivoyage.org/wiki/Lisbon")
    for c in chunks:
        assert "destination" in c["metadata"]
        assert "section"     in c["metadata"]
        assert "url"         in c["metadata"]
        assert c["metadata"]["url"] == "https://en.wikivoyage.org/wiki/Lisbon"


def test_parse_sections_skips_tiny_sections():
    from scripts.build_index import parse_sections
    tiny = "== See ==\nOK.\n== Eat ==\nGreat food here with lots of variety and detail.\n" * 1
    chunks = parse_sections(tiny, "X", "http://x")
    # "OK." alone is < 20 words; should be skipped
    sections = [c["metadata"]["section"] for c in chunks]
    assert "See" not in sections


def test_clean_wikitext_strips_templates_and_links():
    from scripts.build_index import clean_wikitext
    raw = "{{listing|name=Foo}} [[Alfama|the Alfama]] area has '''great''' food."
    result = clean_wikitext(raw)
    assert "{{" not in result
    assert "[[" not in result
    assert "'''" not in result
    assert "the Alfama" in result


# ── Fixture index for search tests ──────────────────────────────────────────

import numpy as np

_FIXTURE_TEXTS = [
    "Lisbon — Stay safe\nPickpocketing common on tram 28. Avoid waterfront late at night.",
    "Lisbon — Eat\nPastel de nata and grilled sardines are must-try specialties.",
    "Venice — See\nThe Grand Canal and St Mark's Basilica are iconic. Best in spring.",
    "Venice — Stay safe\nVenice is very safe with almost no violent crime.",
    "Prague — Eat\nTraditional svíčková and goulash are Prague staples.",
]
_FIXTURE_DIM = 8
_rng = np.random.default_rng(42)
_FIXTURE_EMBEDDINGS = _rng.random((len(_FIXTURE_TEXTS), _FIXTURE_DIM), dtype=np.float32)
_FIXTURE_METADATA = [
    {"destination": "Lisbon", "section": "Stay safe", "url": "https://en.wikivoyage.org/wiki/Lisbon"},
    {"destination": "Lisbon", "section": "Eat",       "url": "https://en.wikivoyage.org/wiki/Lisbon"},
    {"destination": "Venice", "section": "See",       "url": "https://en.wikivoyage.org/wiki/Venice"},
    {"destination": "Venice", "section": "Stay safe", "url": "https://en.wikivoyage.org/wiki/Venice"},
    {"destination": "Prague", "section": "Eat",       "url": "https://en.wikivoyage.org/wiki/Prague"},
]


def _inject_fixture_index(monkeypatch):
    """Patch knowledge._index and _embed_query so no model or file is loaded."""
    import backend.app.knowledge as km
    monkeypatch.setattr(km, "_index", {
        "embeddings": _FIXTURE_EMBEDDINGS.copy(),
        "texts":      _FIXTURE_TEXTS[:],
        "metadata":   _FIXTURE_METADATA[:],
    })
    # Query vector pointing toward index[0] (Lisbon / Stay safe)
    query_vec = _FIXTURE_EMBEDDINGS[0].copy()
    monkeypatch.setattr(km, "_embed_query", lambda q: query_vec)


def test_search_travel_knowledge_returns_results(monkeypatch):
    _inject_fixture_index(monkeypatch)
    from backend.app.knowledge import search_travel_knowledge
    results = search_travel_knowledge("is Lisbon safe at night")
    assert len(results) > 0
    assert results[0]["destination"] == "Lisbon"
    assert results[0]["section"] == "Stay safe"


def test_search_travel_knowledge_result_schema(monkeypatch):
    _inject_fixture_index(monkeypatch)
    from backend.app.knowledge import search_travel_knowledge
    for r in search_travel_knowledge("Venice canal"):
        assert "text"        in r
        assert "score"       in r
        assert "destination" in r
        assert "section"     in r
        assert "url"         in r


def test_search_travel_knowledge_destination_filter(monkeypatch):
    _inject_fixture_index(monkeypatch)
    import backend.app.knowledge as km
    # Point query toward Venice chunk
    monkeypatch.setattr(km, "_embed_query",
                        lambda q: _FIXTURE_EMBEDDINGS[2].copy())
    from backend.app.knowledge import search_travel_knowledge
    results = search_travel_knowledge("Venice canal", destination="Venice")
    assert all(r["destination"] == "Venice" for r in results)


def test_search_travel_knowledge_no_index_returns_empty(monkeypatch):
    import backend.app.knowledge as km
    monkeypatch.setattr(km, "_index", None)
    monkeypatch.setattr(km, "_load_index", lambda: None)
    from backend.app.knowledge import search_travel_knowledge
    assert search_travel_knowledge("anything") == []


def test_dispatch_tool_search_travel_knowledge(monkeypatch):
    _inject_fixture_index(monkeypatch)
    result = dispatch_tool("search_travel_knowledge", {"query": "Lisbon safe"})
    assert "results" in result
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0


# ---------------------------------------------------------------------------
# V5: get_destination_budget
# ---------------------------------------------------------------------------

def test_get_destination_budget_known():
    from backend.app.tools import get_destination_budget
    result = get_destination_budget("Prague")
    assert result["destination"] == "Prague"
    assert result["daily_cost_eur"] == 130
    assert result["estimated_cost_eur"] == 130
    assert result["basis"] == "per_person_per_day"
    assert "error" not in result


def test_get_destination_budget_unknown():
    from backend.app.tools import get_destination_budget
    result = get_destination_budget("Atlantis")
    assert "error" in result
    assert result["destination"] == "Atlantis"


def test_get_destination_budget_covers_all_22():
    from backend.app.tools import get_destination_budget
    destinations = [
        "Lisbon", "Porto", "Algarve", "Barcelona", "Seville", "Paris", "Nice",
        "Chamonix", "Provence", "Rome", "Venice", "Florence", "Cinque Terre",
        "Athens", "Santorini", "Prague", "Vienna", "Ljubljana", "Salzburg",
        "Hallstatt", "Innsbruck", "Amsterdam",
    ]
    for dest in destinations:
        result = get_destination_budget(dest)
        assert "error" not in result, f"{dest} missing from destination_costs.json"
        assert isinstance(result["estimated_cost_eur"], int)


def test_dispatch_tool_get_destination_budget():
    result = dispatch_tool("get_destination_budget", {"destination": "Lisbon"})
    assert result["daily_cost_eur"] == 163
    assert result["estimated_cost_eur"] == 163
    assert result["destination"] == "Lisbon"


# ---------------------------------------------------------------------------
# V5: llm_rerank
# ---------------------------------------------------------------------------

def test_llm_rerank_reorders_by_score(monkeypatch):
    """LLM returns high score for index 1 → index 1 moves to position 0."""
    candidates = [
        {"text": "Lisbon cheap", "score": 0.9, "destination": "Lisbon", "section": "Eat", "url": ""},
        {"text": "Prague nightlife", "score": 0.8, "destination": "Prague", "section": "Do", "url": ""},
        {"text": "Venice canal", "score": 0.7, "destination": "Venice", "section": "See", "url": ""},
    ]

    fake_content = MagicMock()
    fake_content.text = "[3, 9, 1]"
    fake_response = MagicMock()
    fake_response.content = [fake_content]

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    import backend.app.reranker as reranker_mod
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    fake_response.text = fake_content.text
    with patch.object(reranker_mod, "generate_text", return_value=fake_response):
        result = reranker_mod.llm_rerank(
            [dict(c) for c in candidates],
            query="nightlife Europe",
            taste_profile="love nightlife, hate crowds",
            k=2,
        )

    assert result[0]["destination"] == "Prague"   # score 9 → first
    assert len(result) == 2


def test_llm_rerank_fallback_on_no_api_key(monkeypatch):
    """Without an API key llm_rerank returns cosine-ordered slice without calling LLM."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    candidates = [
        {"text": "A", "score": 0.9, "destination": "Lisbon",  "section": "Eat", "url": ""},
        {"text": "B", "score": 0.8, "destination": "Prague",  "section": "Do",  "url": ""},
    ]
    from backend.app.reranker import llm_rerank
    result = llm_rerank([dict(c) for c in candidates], "query", "profile", k=2)
    assert result[0]["destination"] == "Lisbon"
    assert len(result) == 2


def test_research_loop_stops_at_configured_round_limit(monkeypatch):
    import backend.app.unified_planner as engine
    from backend.app.planning_models import PlanningTask, TripConstraints
    from backend.app.llm import LLMResult
    calls = []
    monkeypatch.setattr(engine, "llm_rag_enhancements_enabled", lambda: True)
    monkeypatch.setattr(engine, "research_max_rounds", lambda: 3)
    def search(query, k=6, destination=None):
        calls.append((query, destination))
        return [{"text": "Prague food markets", "destination": "Prague", "section": "Eat", "score": .9}]
    decisions = []
    def propose(*args, **kwargs):
        decisions.append(len(decisions) + 1)
        return LLMResult(json.dumps({"queries": [f"query {len(decisions)}"]}), "test", "anthropic", None, None)
    monkeypatch.setattr("backend.app.knowledge.search_travel_knowledge", search)
    monkeypatch.setattr(engine, "generate_text", propose)
    candidates = engine.resolve_candidates(PlanningTask(constraints=TripConstraints(trip_length=3, interests=["food"])))
    assert decisions == [1, 2]
    assert len(calls) == 4
    assert candidates[0].name == "Prague"


def test_llm_research_failure_degrades_to_deterministic_rag(monkeypatch):
    import backend.app.unified_planner as engine
    from backend.app.planning_models import PlanningTask, TripConstraints
    monkeypatch.setattr(engine, "llm_rag_enhancements_enabled", lambda: True)
    monkeypatch.setattr(engine, "generate_text", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr("backend.app.knowledge.search_travel_knowledge", lambda *a, **k: [
        {"text": "Prague food markets", "destination": "Prague", "section": "Eat", "score": .9}])
    plan = engine.execute_task(PlanningTask(constraints=TripConstraints(trip_length=3, interests=["food"])))
    assert plan.planner_mode == "unified_itinerary_rule"
    assert plan.ranked_destinations[0].name == "Prague"


def test_no_key_path_uses_rag_backbone(monkeypatch):
    import backend.app.unified_planner as engine
    from backend.app.planning_models import PlanningTask, TripConstraints
    calls = []
    def search(query, k=6, destination=None):
        calls.append(destination)
        return [{"text": "Prague food markets", "destination": "Prague", "section": "Eat", "url": "https://en.wikivoyage.org/wiki/Prague", "score": .9}]
    monkeypatch.setattr("backend.app.knowledge.search_travel_knowledge", search)
    plan = engine.execute_task(PlanningTask(constraints=TripConstraints(trip_length=3, interests=["food"])))
    assert None in calls and "Prague" in calls
    assert plan.planner_mode == "unified_itinerary_rule"
    assert plan.evidence and all(day.activities for day in plan.itinerary)


def test_generated_copy_rejects_unknown_evidence_ids(monkeypatch):
    import backend.app.rag_enhancements as enhancements

    request = TripRequest(
        origin_city="Munich", date_range="June", trip_length=3, budget=700,
        interests=["food"], transport_preference="train", physical_level="medium",
    )
    state = state_from_request(request)
    candidates = [
        CandidateDestination(
            name="Prague",
            evidence=[
                EvidenceChunk(
                    text="Prague has food markets.",
                    destination="Prague",
                    url="https://en.wikivoyage.org/wiki/Prague",
                )
            ],
        )
    ]
    monkeypatch.setattr(
        enhancements,
        "_ask_json",
        lambda *args, **kwargs: {
            "candidates": [
                {
                    "name": "Prague",
                    "pro_points": [{"text": "Good food", "evidence_ids": ["Prague:99"]}],
                    "con_points": [],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="outside the retrieved candidate set"):
        enhancements.write_candidate_copy(candidates, state)


def test_dispatch_tool_rerank_travel_knowledge(monkeypatch):
    """dispatch_tool routes rerank_travel_knowledge; knowledge and reranker are monkeypatched."""
    _inject_fixture_index(monkeypatch)

    import backend.app.reranker as reranker_mod
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")

    fake_content = MagicMock()
    fake_content.text = "[5, 8, 3, 6, 4]"
    fake_response = MagicMock()
    fake_response.content = [fake_content]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    fake_response.text = fake_content.text
    with patch.object(reranker_mod, "generate_text", return_value=fake_response):
        result = dispatch_tool("rerank_travel_knowledge", {
            "query": "safe food destination",
            "taste_profile": "love local food, hate tourist traps",
        })

    assert "results" in result
    assert isinstance(result["results"], list)


# ---------------------------------------------------------------------------
# V5: taste_profile field
# ---------------------------------------------------------------------------

def test_trip_request_taste_profile_defaults_empty():
    req = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food"], transport_preference="train", physical_level="low",
    )
    assert req.taste_profile == ""


def test_trip_request_taste_profile_custom():
    req = TripRequest(
        origin_city="Munich", date_range="June 10-15", trip_length=4,
        budget=700, interests=["food"], transport_preference="train", physical_level="low",
        taste_profile="I love hiking and dislike tourist traps.",
    )
    assert req.taste_profile == "I love hiking and dislike tourist traps."


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
        transport_mode="train",
        weather_risk="low",
        weather_note="Sunny",
        estimated_total_cost=600,
    )
    assert score_destination(destination, request, good_insight) == base_score + 5

    # Both penalties: long travel (-1), high weather risk (-1); cost > budget, no bonus
    bad_insight = ToolInsight(
        transport_hours=7.5,
        transport_mode="train",
        weather_risk="high",
        weather_note="Stormy",
        estimated_total_cost=800,
    )
    assert score_destination(destination, request, bad_insight) == base_score
