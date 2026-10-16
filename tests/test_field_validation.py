import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.field_validation import normalize_trip_length
from backend.app.models import ChatRequest, StrategyDecision


@pytest.mark.parametrize("value", [3, "3", "3 days", "3 day", "3天", "三天", "三日游", " 3 DAYS "])
def test_trip_length_normalization(value):
    assert normalize_trip_length(value) == 3


@pytest.mark.parametrize("value", [0, -1, True, 3.5, [], {}, "three days", "3-5 days", "10月23日", "0 days"])
def test_invalid_trip_length_rejected(value):
    with pytest.raises(ValueError):
        normalize_trip_length(value)


def test_llm_known_fields_cannot_overwrite_day_count_with_unparsed_string():
    decision = StrategyDecision(intent="plan_fixed_destination", next_action="run_fixed_itinerary",
                                known_fields={"destination": "Paris", "trip_length": "3 days"})
    assert decision.known_fields["trip_length"] == 3
    with pytest.raises(ValidationError):
        StrategyDecision(intent="plan_fixed_destination", next_action="run_fixed_itinerary",
                         known_fields={"trip_length": "unknown"})


def test_chat_api_handles_llm_string_days(monkeypatch, tmp_path):
    import backend.app.chat as chat
    from backend.app.main import app
    import backend.app.strategy as strategy
    from backend.app.llm import LLMResult
    from backend.app.fixed_itinerary import FixedItineraryResult
    from backend.app.models import TripPlan

    monkeypatch.setenv("TRAJECTORY_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setattr(strategy, "generate_text", lambda *a, **k: LLMResult(
        text='{"intent":"plan_fixed_destination","next_action":"run_fixed_itinerary","known_fields":{"trip_length":"3 days"}}',
        model="test", provider="openai", usage=None, request_id=None))
    monkeypatch.setattr("backend.app.knowledge.search_travel_knowledge", lambda *a, **k: [])
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "3 days in Paris"})
    assert response.status_code == 200
    assert response.json()["extracted_fields"]["trip_length"] == 3


def test_invalid_llm_days_trigger_rule_fallback(monkeypatch):
    import backend.app.strategy as strategy
    from backend.app.llm import LLMResult
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setattr(strategy, "generate_text", lambda *a, **k: LLMResult(
        text='{"intent":"plan_fixed_destination","next_action":"run_fixed_itinerary","known_fields":{"trip_length":"unknown"}}',
        model="test", provider="openai", usage=None, request_id=None))
    decision = strategy.decide_strategy("Paris", {"destination": "Paris", "trip_length": 3})
    assert decision.router_mode == "rule"
    assert decision.next_action == "run_planning"
    assert decision.router_fallback_reason == "llm_router_error: ValidationError"


def test_client_fields_are_normalized_and_invalid_values_return_422():
    from backend.app.main import app
    assert ChatRequest(message="Paris", extracted_fields={"trip_length": "3 days"}).extracted_fields["trip_length"] == 3
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Paris", "extracted_fields": {"trip_length": "unknown"}})
    assert response.status_code == 422
