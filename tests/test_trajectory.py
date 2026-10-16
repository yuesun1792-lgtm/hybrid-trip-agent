import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from backend.app.chat import handle_chat
from backend.app.models import ChatRequest, StrategyDecision
from backend.app.trajectory import Trajectory, _CURRENT, clean, create_message, observed


def test_exception_group_preserves_inner_failure_and_redacts_key(monkeypatch):
    from backend.app.trajectory import error_details
    monkeypatch.setenv("GEOAPIFY_API_KEY", "geo-private-value")
    error = ExceptionGroup("MCP task group failed", [ValueError("Bad selection geo-private-value")])
    details = error_details(error)
    assert details["exceptions"][0]["type"] == "ValueError"
    assert details["exceptions"][0]["message"] == "Bad selection [REDACTED]"


@pytest.fixture(autouse=True)
def trajectory_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAJECTORY_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_chat_persists_linked_runs_and_field_changes(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.chat.decide_strategy", lambda *a, **k: StrategyDecision(
        intent="clarify", next_action="ask_user", missing_fields=["start_date"], reason="test", router_mode="llm"))
    first = handle_chat(ChatRequest(message="Paris"))
    second = handle_chat(ChatRequest(message="3 days", conversation_id=first.conversation_id))
    assert first.trajectory["result"]["plan_created"] is False
    assert second.trajectory["parent_run_id"] == first.run_id
    persisted = json.loads((tmp_path / f"{first.run_id}.traj.json").read_text())
    assert persisted == first.trajectory
    assert persisted["result"]["field_changes"]["destination"]["after"] == "Paris"
    assert persisted["steps"][0]["name"] == "field_extraction"
    events = [json.loads(line) for line in (tmp_path / f"{first.run_id}.events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"


def test_timeout_metadata_and_redaction(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    recorder = Trajectory("conv", None, "hello", {})
    token = _CURRENT.set(recorder)
    class APITimeoutError(Exception):
        pass
    def fail(**kwargs):
        raise APITimeoutError("timeout sk-test-secret")
    client = SimpleNamespace(messages=SimpleNamespace(create=fail), timeout=12, max_retries=2)
    try:
        with pytest.raises(APITimeoutError):
            create_message(client, stage="router", prompt_version="v1", model="test", max_tokens=700)
    finally:
        _CURRENT.reset(token)
    step = recorder.data["steps"][0]
    assert step["error"]["type"] == "APITimeoutError"
    assert "sk-test-secret" not in json.dumps(step)
    assert step["llm"]["usage"] is None
    assert step["llm"]["timeout_seconds"] == 12
    assert step["duration_ms"] >= 0
    assert clean({"api_key": "other-secret"})["api_key"] == "[REDACTED]"


def test_failed_run_is_saved_and_context_reset(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("router failed")
    monkeypatch.setattr("backend.app.chat.decide_strategy", fail)
    with pytest.raises(RuntimeError):
        handle_chat(ChatRequest(message="Paris"))
    data = json.loads(next(tmp_path.glob("*.traj.json")).read_text())
    assert data["status"] == "failed"
    assert data["error"]["type"] == "RuntimeError"
    assert _CURRENT.get() is None


def test_concurrent_runs_are_isolated(monkeypatch):
    monkeypatch.setattr("backend.app.chat.decide_strategy", lambda *a, **k: StrategyDecision(
        intent="clarify", next_action="ask_user", missing_fields=["destination"], reason="test", router_mode="llm"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda message: handle_chat(ChatRequest(message=message)), ["hello", "world"]))
    assert responses[0].run_id != responses[1].run_id
    for response, message in zip(responses, ["hello", "world"]):
        assert response.trajectory["input"]["message"] == message
        assert [step["name"] for step in response.trajectory["steps"]] == ["field_extraction", "task_preparation"]
        assert response.trajectory["status"] == "completed"


def test_persistence_failure_does_not_break_chat(tmp_path, monkeypatch):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    monkeypatch.setenv("TRAJECTORY_DIR", str(blocked))
    response = handle_chat(ChatRequest(message="Paris"))
    assert response.trajectory["persistence_error"]
    assert response.assistant_message


def test_fixed_itinerary_fallback_records_retrieval_and_errors(monkeypatch):
    monkeypatch.setattr("backend.app.knowledge.search_travel_knowledge", lambda *a, **k: [{
        "text": "Paris - See\nVisit museums and walk along the river.",
        "destination": "Paris", "section": "See", "score": 0.9,
        "url": "https://en.wikivoyage.org/wiki/Paris"}])
    response = handle_chat(ChatRequest(message="3 days in Paris"))
    assert response.trajectory["status"] == "completed_with_fallback"
    assert response.trajectory["result"]["plan_created"] is True
    steps = response.trajectory["steps"]
    analysis = next(s for s in steps if s["name"] == "destination_analysis")
    assert analysis["output"]["evidence"][0]["score"] == 0.9
    writer = next(s for s in steps if s["name"] == "structured_itinerary_builder")
    assert writer["status"] == "fallback"
    workflow = next(s for s in steps if s["name"] == "unified_planning_workflow")
    assert workflow["status"] == "fallback"
    assert writer["parent_step_id"] == workflow["step_id"]



def test_successful_sdk_call_records_usage_and_request_id():
    recorder = Trajectory("conv", None, "hello", {})
    token = _CURRENT.set(recorder)
    response = SimpleNamespace(_request_id="req_123", usage=SimpleNamespace(input_tokens=20, output_tokens=8))
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: response), timeout=12, max_retries=2)
    try:
        assert create_message(client, stage="router", prompt_version="v1", model="test") is response
    finally:
        _CURRENT.reset(token)
    step = recorder.data["steps"][0]
    assert step["status"] == "ok"
    assert step["llm"]["request_id"] == "req_123"
    assert step["llm"]["usage"]["input_tokens"] == 20
