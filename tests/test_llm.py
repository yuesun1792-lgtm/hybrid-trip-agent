"""Exercise real SDK serialization/response parsing through an offline HTTP transport."""
import json
from types import SimpleNamespace

import httpx
import pytest

from backend.app import llm
from backend.app.chat import handle_chat
from backend.app.config import llm_api_key, llm_rag_enhancements_enabled
from backend.app.llm_usage import usage_from_response
from backend.app.models import ChatRequest, EvidenceChunk
from backend.app.rag_enhancements import _ask_json
from backend.app.reranker import llm_rerank
from backend.app.trajectory import Trajectory, _CURRENT


@pytest.fixture(autouse=True)
def configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAJECTORY_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-offline-anthropic")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    for key in ("OPENAI_INPUT_PRICE_PER_MTOK", "OPENAI_OUTPUT_PRICE_PER_MTOK", "OPENAI_CACHE_READ_PRICE_PER_MTOK"):
        monkeypatch.delenv(key, raising=False)


def mock_sdk(monkeypatch, replies, provider="openai"):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            return httpx.Response(reply[0], json=reply[1])
        if provider == "openai":
            body = {"id": "resp_test", "object": "response", "created_at": 1,
                    "model": "test-model", "status": "completed", "error": None,
                    "incomplete_details": None, "parallel_tool_calls": False,
                    "tools": [], "tool_choice": "auto",
                    "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                                "status": "completed", "content": [{"type": "output_text", "text": reply, "annotations": []}]}],
                    "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                              "input_tokens_details": {"cached_tokens": 40},
                              "output_tokens_details": {"reasoning_tokens": 0}}}
        else:
            body = {"id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
                    "content": [{"type": "text", "text": reply}], "stop_reason": "end_turn",
                    "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 20}}
        return httpx.Response(200, json=body, headers={"x-request-id": "req_test", "request-id": "req_test"})
    transport = httpx.MockTransport(handle)
    def client(provider, api_key):
        if provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=api_key, max_retries=0, http_client=httpx.Client(transport=transport))
        import anthropic
        return anthropic.Anthropic(api_key=api_key, max_retries=0, http_client=httpx.Client(transport=transport))
    monkeypatch.setattr(llm, "_make_client", client)
    return requests


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_sdk_text_and_usage(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    requests = mock_sdk(monkeypatch, ['{"ok":true}'], provider)
    result = llm.generate_text("Return JSON", stage="test", max_tokens=512)
    assert result.text == '{"ok":true}'
    assert result.provider == provider
    assert result.usage.input_tokens == 100
    assert result.request_id == "req_test"
    if provider == "openai":
        assert requests[0]["input"] == "Return JSON"
        assert requests[0]["max_output_tokens"] == 512
        assert requests[0]["store"] is False
        assert result.usage.cache_read_input_tokens == 40
        assert result.usage.estimated_cost_usd is None
    else:
        assert requests[0]["messages"][0]["content"] == "Return JSON"
        assert requests[0]["max_tokens"] == 512


def test_two_turn_chat_uses_openai_for_router_and_writer(monkeypatch):
    replies = [json.dumps({"intent": "plan_fixed_destination", "next_action": "ask_user",
                           "known_fields": {}, "missing_fields": ["trip_length"], "reason": "Need days"}),
               json.dumps({"intent": "plan_fixed_destination", "next_action": "run_fixed_itinerary",
                           "known_fields": {}, "missing_fields": [], "reason": "Ready"}),
               json.dumps({"days": [{"day": i, "activities": [{"title": "Museum", "description": "Explore museums and riverside walks.",
                   "category": "culture", "time_slot": "morning", "evidence_ids": [EvidenceChunk(text="Paris - See\nVisit museums and take a riverside walk.", destination="Paris", section="See", url="https://en.wikivoyage.org/wiki/Paris").evidence_id]}]} for i in range(1, 4)],
                   "pros": ["Culture"], "cons": [], "explanation": "Guide evidence"})]
    requests = mock_sdk(monkeypatch, replies)
    monkeypatch.setattr("backend.app.knowledge.search_travel_knowledge", lambda *a, **k: [{
        "text": "Paris - See\nVisit museums and take a riverside walk.", "destination": "Paris",
        "section": "See", "score": 0.9, "url": "https://en.wikivoyage.org/wiki/Paris"}])
    first = handle_chat(ChatRequest(message="Paris for food and art"))
    second = handle_chat(ChatRequest(message="3 days", conversation_id=first.conversation_id))
    assert first.plan is None
    assert second.plan.planner_mode == "unified_itinerary_llm"
    assert second.trajectory["status"] == "completed"
    assert second.trajectory["parent_run_id"] == first.run_id
    assert len(requests) == 3
    assert len(second.llm_usage) == 2
    assert all(u.provider == "openai" for u in second.llm_usage)
    assert {s["name"] for s in second.trajectory["steps"] if s["kind"] == "llm"} == {
        "task_router_api", "llm_itinerary_writer_api"}


@pytest.mark.parametrize("reply,exception", [
    (httpx.ConnectTimeout("connection timed out"), "APITimeoutError"),
    ((401, {"error": {"message": "invalid key", "type": "invalid_request_error", "code": "invalid_api_key"}}), "AuthenticationError"),
])
def test_openai_errors_recorded_and_router_falls_back(monkeypatch, reply, exception):
    mock_sdk(monkeypatch, [reply])
    response = handle_chat(ChatRequest(message="Paris"))
    assert response.trajectory["status"] == "completed_with_fallback"
    step = next(s for s in response.trajectory["steps"] if s["kind"] == "llm")
    assert step["error"]["type"] == exception
    assert step["llm"]["usage"] is None
    assert step["llm"]["provider"] == "openai"
    if exception == "APITimeoutError":
        assert step["error"]["cause"]["type"] == "ConnectTimeout"


def test_malformed_json_falls_back(monkeypatch):
    mock_sdk(monkeypatch, ["not JSON"])
    response = handle_chat(ChatRequest(message="Paris"))
    assert response.trace[1].status == "fallback"
    assert response.llm_usage[0].input_tokens == 100


def test_rag_and_chunk_reranker_use_unified_api(monkeypatch):
    mock_sdk(monkeypatch, ['{"action":"stop"}', '[2,9]'])
    assert _ask_json("Return JSON") == {"action": "stop"}
    chunks = [{"destination": name, "section": "See", "text": "Museum", "score": 0.5} for name in ["Paris", "Rome"]]
    ranked = llm_rerank(chunks, "culture", "art", 2, raise_on_error=True)
    assert ranked[0]["destination"] == "Rome"


def test_provider_key_isolation_and_feature_flag(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert llm_api_key() is None  # Anthropic key must not enable OpenAI.
    assert llm_rag_enhancements_enabled() is False
    monkeypatch.setenv("OPENAI_API_KEY", "your_key_here")
    assert llm_api_key() is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test")
    monkeypatch.setenv("ENABLE_LLM_RAG_ENHANCEMENTS", "true")
    assert llm_rag_enhancements_enabled() is True


def test_openai_pricing_excludes_cached_input_from_normal_rate(monkeypatch):
    monkeypatch.setenv("OPENAI_INPUT_PRICE_PER_MTOK", "2")
    monkeypatch.setenv("OPENAI_OUTPUT_PRICE_PER_MTOK", "8")
    monkeypatch.setenv("OPENAI_CACHE_READ_PRICE_PER_MTOK", "0.5")
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                                input_tokens_details=SimpleNamespace(cached_tokens=40)))
    usage = usage_from_response("test", "test-model", response, provider="openai")
    assert usage.estimated_cost_usd == 0.0003


def test_empty_response_rejected(monkeypatch):
    mock_sdk(monkeypatch, [""])
    with pytest.raises(llm.LLMResponseError):
        llm.generate_text("hello", stage="test")


def test_missing_model_is_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL")
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        llm.generate_text("hello", stage="test")


def test_incomplete_response_records_consumed_usage(monkeypatch):
    body = {"id": "resp_test", "object": "response", "created_at": 1,
            "model": "test-model", "status": "incomplete", "error": None,
            "incomplete_details": {"reason": "max_output_tokens"},
            "parallel_tool_calls": False, "tools": [], "tool_choice": "auto", "output": [],
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 20}}}
    mock_sdk(monkeypatch, [(200, body)])
    response = handle_chat(ChatRequest(message="Paris"))
    assert response.trace[1].status == "fallback"
    assert response.llm_usage[0].output_tokens == 20
    step = next(s for s in response.trajectory["steps"] if s["kind"] == "llm")
    assert step["error"]["type"] == "LLMResponseError"
    assert "incomplete" in step["error"]["message"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_real_client_initialization_without_network(monkeypatch, provider):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setenv("LLM_TRUST_ENV", "false")
    with llm._make_client(provider, "sk-offline-test") as client:
        assert client.timeout.read == 15
        assert client.max_retries == 1
        assert client._client._trust_env is False


def test_itinerary_request_enforces_exact_days_and_allowed_evidence(monkeypatch):
    from backend.app.plan_builder import build_itinerary
    from backend.app.models import DestinationAnalysis, EvidenceChunk
    from backend.app.planning_models import TripConstraints
    chunk = EvidenceChunk(text="Paris - See\nMuseums and riverside walks.", destination="Paris", section="See", url="https://en.wikivoyage.org/wiki/Paris")
    replies = [json.dumps({"days": [{"day": i, "activities": [{"title": "Museum", "description": "Visit museums.", "category": "culture", "time_slot": "morning", "evidence_ids": [chunk.evidence_id]}]} for i in range(1, 4)],
                          "pros": [], "cons": [], "explanation": "Evidence-backed"})]
    requests = mock_sdk(monkeypatch, replies)
    result = build_itinerary(DestinationAnalysis(destination="Paris", depth="detailed", evidence=[chunk]), TripConstraints(trip_length=3))
    assert result.llm_used and len(result.days) == 3
    format = requests[0]["text"]["format"]
    assert format["type"] == "json_schema" and format["strict"] is True
    schema = format["schema"]
    assert schema["properties"]["days"]["minItems"] == schema["properties"]["days"]["maxItems"] == 3
    assert schema["properties"]["days"]["items"]["properties"]["activities"]["items"]["properties"]["evidence_ids"]["items"]["enum"] == [chunk.evidence_id]
    assert schema["additionalProperties"] is False


def test_extra_summary_items_are_still_rejected_by_local_validation(monkeypatch):
    from backend.app.plan_builder import build_itinerary
    from backend.app.models import DestinationAnalysis, EvidenceChunk
    from backend.app.planning_models import TripConstraints
    replies = [json.dumps({"days": ["Day 1", "Day 2", "Day 3", "Summary", "Strengths", "Caveats"],
                          "pros": [], "cons": [], "explanation": "Test"})]
    mock_sdk(monkeypatch, replies)
    chunk = EvidenceChunk(text="Paris - See\nMuseums and riverside walks.", destination="Paris", section="See", url="https://en.wikivoyage.org/wiki/Paris")
    result = build_itinerary(DestinationAnalysis(destination="Paris", depth="detailed", evidence=[chunk]), TripConstraints(trip_length=3))
    assert not result.llm_used
    assert len(result.days) == 3 and all(day.activities for day in result.days)
    assert result.fallback_reason
