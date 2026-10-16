import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.domain.models import ChatResponse, ExecutionTraceStep, TripPlan
from backend.app.domain.planning_models import Activity, Day
from scripts.eval_scenarios import Expectations, Suite, check_turn, main, run_suite


def plan():
    return TripPlan(ranked_destinations=[], risks=[], final_recommendation="Test",
                    itinerary=[Day(day=1, destination="Paris", activities=[
                        Activity(title="Dinner", category="food", time_slot="evening")])])


def test_reference_suite_repeats_in_isolated_conversations_and_writes_artifacts(tmp_path):
    suite = Suite.model_validate_json(Path("evals/scenarios/core.json").read_text())
    result = run_suite(suite, tmp_path, repeat=2)
    assert result["summary"]["failed"] == 0
    assert result["summary"]["cases"] == 28
    assert len({c["conversation_id"] for c in result["cases"]}) == 28
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["summary"] == result["summary"]
    assert "PASS" in (tmp_path / "report.md").read_text()
    first = saved["cases"][0]["turns"][0]
    assert Path(first["trajectory_file"]).is_file()
    assert first["canonical_after"]["version"] == 1
    assert first["fallback_used"]


@pytest.mark.parametrize("bad", [
    {"scenarios": [{"id": "test", "turns": [{"message": "Hello", "expect": {"outcome": "plan", "typo": True}}]}]},
    {"scenarios": [{"id": "test", "turns": [{"message": "Hello", "request": {"conversation_id": "shared"}, "expect": {"outcome": "plan"}}]}]},
    {"scenarios": [{"id": "test", "turns": [{"message": "Hello", "expect": {"outcome": "error"}}]}]},
])
def test_invalid_suites_are_rejected_before_running(bad):
    with pytest.raises(ValidationError):
        Suite.model_validate(bad)


def test_canonical_preservation_detects_unrequested_changes():
    before = plan()
    after = before.model_copy(deep=True)
    after.itinerary[0].activities[0].title = "Changed dinner"
    response = ChatResponse(conversation_id="test", assistant_message="Done", plan=after)
    checks = check_turn(Expectations(outcome="revision", changed_days=[], preserve_other_days=True,
                                    preserve_evening=True), response, before, after)
    failed = {c["name"] for c in checks if not c["passed"]}
    assert {"changed_days", "preserve_other_days", "preserve_evening", "revision_committed"} <= failed


def test_fallback_is_a_live_failure_unless_explicitly_expected():
    response = ChatResponse(conversation_id="test", assistant_message="Need dates", trace=[
        ExecutionTraceStep(stage="task_router", status="fallback", detail="Timeout")])
    strict = Expectations(outcome="clarification")
    assert not all(c["passed"] for c in check_turn(strict, response, None, None, mode="live"))
    assert all(c["passed"] for c in check_turn(strict, response, None, None, mode="offline"))
    strict.allow_fallback = True
    assert all(c["passed"] for c in check_turn(strict, response, None, None, mode="live"))


def test_failed_turn_skips_dependents_but_continues_next_scenario(tmp_path):
    suite = Suite.model_validate({"scenarios": [
        {"id": "fails", "turns": [
            {"message": "First", "expect": {"outcome": "plan"}},
            {"message": "Never run", "expect": {"outcome": "plan"}}]},
        {"id": "continues", "turns": [{"message": "Next", "expect": {"outcome": "clarification"}}]}
    ]})
    seen = []

    def handler(request):
        seen.append(request.message)
        return ChatResponse(conversation_id=request.conversation_id, assistant_message="Please clarify")

    report = run_suite(suite, tmp_path, handler=handler, snapshot=lambda _: None)
    assert seen == ["First", "Next"]
    assert report["summary"]["failed"] == 1
    assert report["summary"]["skipped_turns"] == 1


def test_error_requires_exact_type_and_no_committed_change():
    expected = Expectations(outcome="error", error_type="TimeoutError")
    assert all(c["passed"] for c in check_turn(expected, None, None, None, TimeoutError()))
    assert not all(c["passed"] for c in check_turn(expected, None, None, None, ValueError()))
    assert not all(c["passed"] for c in check_turn(expected, None, None, plan(), TimeoutError()))


def test_category_assertion_rejects_wrong_replacement():
    current = plan()
    response = ChatResponse(conversation_id="test", assistant_message="Done", plan=current)
    expected = Expectations(outcome="plan", activity_checks=[{"day": 1, "category": "shopping"}])
    checks = check_turn(expected, response, None, current)
    assert any(c["name"] == "activities_day_1_all_shopping" and not c["passed"] for c in checks)


def test_cli_returns_nonzero_for_unknown_case(monkeypatch):
    monkeypatch.setattr("sys.argv", ["eval_scenarios", "--case", "does_not_exist"])
    assert main() == 2


def test_cli_reports_assertion_failure_with_exit_one(monkeypatch, tmp_path):
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"scenarios": [{"id": "fails", "turns": [
        {"message": "巴黎三日游", "expect": {"outcome": "plan", "day_count": 99}}]}]}))
    monkeypatch.setattr("sys.argv", ["eval_scenarios", "--suite", str(suite), "--output-dir", str(tmp_path / "out")])
    assert main() == 1


def test_cli_live_requires_credentials(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr("sys.argv", ["eval_scenarios", "--mode", "live"])
    assert main() == 2
