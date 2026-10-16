import json
from datetime import date

import pytest

from backend.app.runtime.execution import RUN_CONTEXT, RunContext, BudgetExceeded
from backend.app.domain.models import CandidateDestination, EvidenceChunk, ItineraryDay, TripPlan
from backend.app.domain.planning_models import Activity, PlanningTask, TripConstraints, RevisionRequest, RevisionChange
from backend.app.planning.plan_store import InMemoryPlanStore, PlanVersionConflict
from backend.app.planning.plan_validation import validate_plan, PlanValidationError
from backend.app.planning.task_builder import task_from_fields, PlanningNeedsInput
from backend.app.planning.unified_planner import execute_task
from backend.app.llm.llm import LLMResult


@pytest.fixture
def evidence():
    return EvidenceChunk(text="Paris - See\nMuseums, galleries, shops and riverside walks offer cultural activities throughout the city.",
                         destination="Paris", section="See", url="https://en.wikivoyage.org/wiki/Paris", score=.9)


@pytest.fixture
def offline_tools(monkeypatch, evidence):
    calls = []
    def search(query, k=6, destination=None):
        calls.append((query, destination))
        names = [destination] if destination else ["Paris", "Rome", "Prague"]
        return [{"text": f"{name} - See\nMuseums, galleries, shops and riverside walks offer cultural activities throughout the city.",
                 "destination": name, "section": "See", "url": "https://en.wikivoyage.org/wiki/" + name, "score": .9} for name in names]
    monkeypatch.setattr("backend.app.rag.knowledge.search_travel_knowledge", search)
    return calls


def task(destination="Paris", output="itinerary"):
    return PlanningTask(constraints=TripConstraints(destination=destination, trip_length=3, interests=["culture"]),
                        candidate_destinations=[destination] if destination else [], requested_output=output)


def test_known_and_discovered_destinations_share_full_activity_builder(offline_tools):
    known = execute_task(task(), store=InMemoryPlanStore())
    unknown = execute_task(task(None), store=InMemoryPlanStore())
    for plan in (known, unknown):
        assert len(plan.itinerary) == 3
        assert all(len(day.activities) == 3 for day in plan.itinerary)
        assert plan.itinerary[0].activities[0].duration_minutes is None
        assert all(a.status == "needs_verification" for d in plan.itinerary for a in d.activities)
        assert any(a.depth == "detailed" for a in plan.candidate_analyses)
        assert plan.planner_mode == "unified_itinerary_rule"
    assert len([a for a in unknown.candidate_analyses if a.depth == "detailed"]) == 1
    assert len([a for a in unknown.candidate_analyses if a.depth == "overview"]) == 2


def test_options_do_not_generate_daily_itineraries(offline_tools, monkeypatch):
    def unexpected(*a, **k):
        raise AssertionError("Options must not invoke the full itinerary builder")
    monkeypatch.setattr("backend.app.planning.unified_planner.build_itinerary", unexpected)
    plan = execute_task(task(None, "options"), store=InMemoryPlanStore())
    assert plan.itinerary == []
    assert len(plan.ranked_destinations) == 3
    assert all(a.depth == "overview" for a in plan.candidate_analyses)
    assert len(offline_tools) == 2  # Overview reuses discovery evidence.


def test_freeform_named_activity_requirements_reach_itinerary_writer(offline_tools, monkeypatch):
    """Named stops/time slots are not represented by TripConstraints alone."""
    requested = task()
    requested.constraints.trip_length = 1
    requested.user_message = "Exactly two stops: Louvre Museum in the morning and Eiffel Tower in the afternoon."
    monkeypatch.setattr("backend.app.planning.plan_builder.llm_api_key", lambda: "offline-test")

    def write(prompt, **kwargs):
        assert requested.user_message in prompt
        from backend.app.domain.models import EvidenceChunk
        ref = EvidenceChunk(text="Paris - See\nMuseums, galleries, shops and riverside walks offer cultural activities throughout the city.",
                            destination="Paris", section="See", url="https://en.wikivoyage.org/wiki/Paris").evidence_id
        return LLMResult(json.dumps({"days": [{"day": 1, "activities": [
            {"title": name, "description": "Visit the named stop.", "category": "culture", "time_slot": slot, "evidence_ids": [ref]}
            for name, slot in [("Louvre Museum", "morning"), ("Eiffel Tower", "afternoon")]]}]}), "test", "openai", None, None)

    monkeypatch.setattr("backend.app.planning.plan_builder.generate_text", write)
    plan = execute_task(requested, store=InMemoryPlanStore())
    assert plan.planner_mode == "unified_itinerary_llm"
    assert [a.title for a in plan.itinerary[0].activities] == ["Louvre Museum", "Eiffel Tower"]


def test_revision_preserves_other_days_dinner_and_original_version(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    dinner = next(a for a in base.itinerary[1].activities if a.time_slot == "evening")
    revision = RevisionRequest(base_plan_id=base.plan_id, base_version=1,
        changes=[RevisionChange(day_numbers=[2], time_slot="afternoon", category="shopping", requirements="Replace with shopping")],
        preserve_activity_ids=[dinner.activity_id])
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base, supplied_revision=revision)
    updated = execute_task(prepared, base, store=store)
    assert updated.plan_id == base.plan_id and updated.version == 2 and updated.parent_version == 1
    assert updated.itinerary[0] == base.itinerary[0]
    assert updated.itinerary[2] == base.itinerary[2]
    assert next(a for a in updated.itinerary[1].activities if a.activity_id == dinner.activity_id) == dinner
    old_afternoon = next(a for a in base.itinerary[1].activities if a.time_slot == "afternoon")
    new_afternoon = next(a for a in updated.itinerary[1].activities if a.time_slot == "afternoon")
    assert old_afternoon.activity_id != new_afternoon.activity_id
    assert new_afternoon.category == "shopping"
    assert store.get(base.plan_id, 1) == base


def test_local_replacement_schema_cannot_duplicate_protected_activity(offline_tools, monkeypatch):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base,
        supplied_revision=RevisionRequest(changes=[RevisionChange(day_numbers=[2], time_slot="afternoon",
            requirements="Visit Musee d'Orsay; keep the existing morning unchanged.")]))
    monkeypatch.setattr("backend.app.planning.plan_builder.llm_api_key", lambda: "offline-test")

    def write(prompt, **kwargs):
        schema = kwargs["response_schema"]["properties"]["days"]["items"]["properties"]["activities"]
        assert schema["maxItems"] == 1
        ref = base.evidence[0].evidence_id
        return LLMResult(json.dumps({"days": [{"day": 2, "activities": [{"title": "Musee d'Orsay",
            "description": "Visit this museum.", "category": "culture", "time_slot": "afternoon",
            "evidence_ids": [ref]}]}]}), "test", "openai", None, None)

    monkeypatch.setattr("backend.app.planning.plan_builder.generate_text", write)
    result = execute_task(prepared, base, store=store)
    assert len(result.itinerary[1].activities) == len(base.itinerary[1].activities)
    before = {a.activity_id: a for a in base.itinerary[1].activities if a.time_slot != "afternoon"}
    after = {a.activity_id: a for a in result.itinerary[1].activities}
    assert all(after[identity] == activity for identity, activity in before.items())


def test_remove_and_relax_require_no_generation(offline_tools, monkeypatch):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    def unexpected(*a, **k):
        raise AssertionError("Removing activities does not need new generation")
    monkeypatch.setattr("backend.app.planning.unified_planner.build_itinerary", unexpected)
    change = RevisionRequest(changes=[RevisionChange(day_numbers=[2], action="relax")])
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base, supplied_revision=change)
    updated = execute_task(prepared, base, store=store)
    assert len(updated.itinerary[1].activities) == 1
    assert updated.itinerary[0] == base.itinerary[0]


def test_destination_change_replaces_evidence_and_costs(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    revision = RevisionRequest(constraints_updates={"destination": "罗马"})
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base, supplied_revision=revision)
    updated = execute_task(prepared, base, store=store)
    assert updated.constraints.destination == "Rome"
    assert all(d.destination == "Rome" for d in updated.itinerary)
    assert all(c.destination == "Rome" for c in updated.evidence)
    assert updated.ranked_destinations[0].country == "Italy"
    assert all(c.url.endswith("/Rome") for c in updated.ranked_destinations[0].citations)


def test_shortening_trip_cannot_delete_preserved_activity(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    preserved = base.itinerary[2].activities[0].activity_id
    revision = RevisionRequest(constraints_updates={"trip_length": 2}, preserve_activity_ids=[preserved])
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base, supplied_revision=revision)
    with pytest.raises(ValueError, match="preserving"):
        execute_task(prepared, base, store=store)
    assert store.latest(base.plan_id).version == 1


def test_stale_version_is_rejected_before_tools(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    store.commit(base, expected_version=1)
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base,
        supplied_revision=RevisionRequest(changes=[RevisionChange(day_numbers=[2], action="relax")]))
    before = len(offline_tools)
    with pytest.raises(PlanVersionConflict):
        execute_task(prepared, base, store=store)
    assert len(offline_tools) == before


def test_validation_rejects_wrong_evidence_city_and_duplicate_ids(evidence):
    activity = Activity(title="Museum", evidence_ids=[evidence.evidence_id], status="proposed")
    plan = TripPlan(constraints=TripConstraints(destination="Rome", trip_length=1), ranked_destinations=[],
        itinerary=[ItineraryDay(day=1, destination="Rome", activities=[activity, activity])], evidence=[evidence],
        risks=[], final_recommendation="Rome")
    with pytest.raises(PlanValidationError) as error:
        validate_plan(plan)
    assert {i.code for i in error.value.issues} >= {"evidence_destination", "duplicate_id"}


def test_unknown_cost_and_logistics_are_not_invented(offline_tools):
    plan = execute_task(task("Unknownville"), store=InMemoryPlanStore())
    assert plan.ranked_destinations[0].estimated_cost is None
    assert all(a.place_id is None and a.duration_minutes is None and a.estimated_cost is None for d in plan.itinerary for a in d.activities)


def test_date_change_updates_all_days(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    revision = RevisionRequest(constraints_updates={"start_date": "2026-10-23"})
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base, supplied_revision=revision)
    updated = execute_task(prepared, base, store=store)
    assert [d.date for d in updated.itinerary] == [date(2026, 10, 23), date(2026, 10, 24), date(2026, 10, 25)]


def test_budget_admission_and_registry_prevent_unbounded_calls():
    from backend.app.tool_calling.tool_executor import default_executor
    context = RunContext(max_tool_calls=0)
    token = RUN_CONTEXT.set(context)
    try:
        with pytest.raises(BudgetExceeded):
            default_executor().execute("search_travel_knowledge", {"query": "Paris"})
        with pytest.raises(ValueError, match="not registered"):
            default_executor().execute("unknown_tool", {})
    finally:
        RUN_CONTEXT.reset(token)


def test_full_generation_validates_per_activity_citations(monkeypatch, evidence):
    from backend.app.planning.plan_builder import build_itinerary
    from backend.app.domain.models import DestinationAnalysis
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    raw = {"days": [{"day": 1, "activities": [{"title": "Museum", "description": "Visit", "category": "art", "time_slot": "morning", "evidence_ids": ["missing"]}]}], "pros": [], "cons": [], "explanation": ""}
    monkeypatch.setattr("backend.app.planning.plan_builder.generate_text", lambda *a, **k: LLMResult(json.dumps(raw), "test", "anthropic", None, None))
    result = build_itinerary(DestinationAnalysis(destination="Paris", depth="detailed", evidence=[evidence]), TripConstraints(trip_length=1))
    assert not result.llm_used and result.fallback_reason
    assert all(a.status == "needs_verification" for a in result.days[0].activities)


def test_local_food_change_does_not_trigger_global_replan(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    fields = {**base.constraints.model_dump(), "interests": ["culture", "food"]}
    prepared = task_from_fields(fields, operation="revise", base_plan=base,
                                message="第二天下午改成美食，晚餐保留")
    assert not prepared.affected_scope.full_replan
    assert prepared.affected_scope.day_numbers == [2]
    assert prepared.constraints.interests == ["culture"]
    updated = execute_task(prepared, base, store=store)
    assert updated.itinerary[0] == base.itinerary[0]


def test_failed_validation_does_not_commit_draft(offline_tools, monkeypatch):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base,
        supplied_revision=RevisionRequest(changes=[RevisionChange(day_numbers=[2], action="relax")]))
    def reject(*a, **k):
        from backend.app.domain.planning_models import ValidationIssue
        raise PlanValidationError([ValidationIssue(code="test", severity="error", message="Rejected draft")])
    monkeypatch.setattr("backend.app.planning.unified_planner.validate_plan", reject)
    with pytest.raises(PlanValidationError):
        execute_task(prepared, base, store=store)
    assert store.latest(base.plan_id) == base


def test_chat_version_conflict_and_local_revision_api(offline_tools):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "3 days in Paris for culture"}).json()
        original = first["plan"]
        change = {"message": "Change day 2 afternoon to shopping, keep dinner", "conversation_id": first["conversation_id"],
                  "expected_plan_version": 1}
        response = client.post("/chat", json=change)
        assert response.status_code == 200
        revised = response.json()["plan"]
        assert revised["version"] == 2
        assert revised["itinerary"][0] == original["itinerary"][0]
        assert revised["itinerary"][2] == original["itinerary"][2]
        stale = client.post("/chat", json=change)
        assert stale.status_code == 409
        assert stale.json()["code"] == "plan_version_conflict"


def test_ambiguous_revision_waits_then_resumes_original_request(offline_tools):
    from backend.app.conversation.chat import handle_chat
    from backend.app.domain.models import ChatRequest
    first = handle_chat(ChatRequest(message="3 days in Paris for culture"))
    unclear = handle_chat(ChatRequest(message="换成购物", conversation_id=first.conversation_id))
    assert unclear.plan is None and "revision_scope" in unclear.missing_fields
    resumed = handle_chat(ChatRequest(message="第二天下午", conversation_id=first.conversation_id))
    assert resumed.plan.version == 2
    assert resumed.plan.itinerary[0] == first.plan.itinerary[0]
    assert next(a for a in resumed.plan.itinerary[1].activities if a.time_slot == "afternoon").category == "shopping"


def test_options_chat_can_be_promoted_to_full_plan(offline_tools):
    from backend.app.conversation.chat import handle_chat
    from backend.app.domain.models import ChatRequest
    options = handle_chat(ChatRequest(message="先比较几个城市，美食文化", requested_output="options"))
    assert options.plan.requested_output == "options" and options.plan.itinerary == []
    planned = handle_chat(ChatRequest(message="3 days in Paris", conversation_id=options.conversation_id))
    assert planned.plan.requested_output == "itinerary"
    assert len(planned.plan.itinerary) == 3 and all(d.activities for d in planned.plan.itinerary)


def test_whole_trip_budget_is_not_used_as_daily_activity_cost(offline_tools):
    plan = execute_task(PlanningTask(constraints=TripConstraints(destination="Paris", trip_length=3, budget=100),
                                    candidate_destinations=["Paris"]), store=InMemoryPlanStore())
    assert any(i.code == "estimated_budget_exceeded" for i in plan.validation_issues)
    assert any("exceeds" in message for message in plan.risks)
    assert all(a.estimated_cost is None for day in plan.itinerary for a in day.activities)


def test_multi_day_revision_keeps_distinct_categories(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    revision = RevisionRequest(changes=[
        RevisionChange(day_numbers=[1], time_slot="afternoon", category="shopping"),
        RevisionChange(day_numbers=[2], time_slot="afternoon", category="nature")])
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base, supplied_revision=revision)
    updated = execute_task(prepared, base, store=store)
    assert next(a.category for a in updated.itinerary[0].activities if a.time_slot == "afternoon") == "shopping"
    assert next(a.category for a in updated.itinerary[1].activities if a.time_slot == "afternoon") == "nature"
    assert updated.itinerary[2] == base.itinerary[2]


def test_invalid_duration_reply_can_be_corrected_in_next_turn(offline_tools):
    from backend.app.conversation.chat import handle_chat
    from backend.app.domain.models import ChatRequest
    invalid = handle_chat(ChatRequest(message="0 days in Paris"))
    assert invalid.plan is None and invalid.missing_fields == ["trip_length"]
    corrected = handle_chat(ChatRequest(message="3", conversation_id=invalid.conversation_id))
    assert corrected.plan and len(corrected.plan.itinerary) == 3


def test_preserved_city_conflict_is_clarified_before_generation(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    revision = RevisionRequest(constraints_updates={"destination": "Rome"},
                               preserve_activity_ids=[base.itinerary[0].activities[0].activity_id])
    with pytest.raises(ValueError, match="another destination"):
        task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base, supplied_revision=revision)
    assert store.latest(base.plan_id).version == 1


def test_output_token_budget_is_reserved_before_call():
    context = RunContext(max_output_tokens=1000)
    context.admit("llm", output_tokens=700)
    with pytest.raises(BudgetExceeded, match="output-token"):
        context.admit("llm", output_tokens=400)
    assert context.llm_calls == 1


def test_concurrent_chat_revisions_only_commit_one_expected_version(offline_tools):
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "3 days in Paris"}).json()
        payload = {"message": "slow down day 2", "conversation_id": first["conversation_id"], "expected_plan_version": 1,
                   "revision": {"changes": [{"action": "relax", "day_numbers": [2]}]}}
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: client.post("/chat", json=payload).status_code, range(2)))
    assert sorted(results) == [200, 409]


def test_adding_activity_preserves_existing_records(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    revision = RevisionRequest(changes=[RevisionChange(action="add", day_numbers=[2], time_slot="afternoon", category="shopping")])
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base, supplied_revision=revision)
    updated = execute_task(prepared, base, store=store)
    old = {a.activity_id: a for d in base.itinerary for a in d.activities}
    new = {a.activity_id: a for d in updated.itinerary for a in d.activities}
    assert all(old[identity] == new[identity] for identity in old)
    assert len(updated.itinerary[1].activities) == len(base.itinerary[1].activities) + 1


def test_structured_activity_frontend_renders_without_errors(offline_tools, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    script = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"
    monkeypatch.syspath_prepend(str(script.parent))
    plan = execute_task(task(), store=InMemoryPlanStore())
    app = AppTest.from_file(str(script))
    app.session_state["chat_latest_plan"] = plan.model_dump(mode="json")
    app.run()
    assert not app.exception
    assert any("Plan version: 1" in element.value for element in app.caption)
    assert any("Culture activity" in element.value for element in app.markdown)


def test_committed_plan_versions_are_readable_without_model_calls(offline_tools):
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as client:
        created = client.post("/chat", json={"message": "3 days in Paris"}).json()
        plan_id = created["plan"]["plan_id"]
        revised = client.post("/chat", json={"message": "relax day 2", "conversation_id": created["conversation_id"],
            "revision": {"changes": [{"action": "relax", "day_numbers": [2]}]}}).json()
        assert revised["plan"]["version"] == 2
        before = len(offline_tools)
        latest = client.get(f"/plans/{plan_id}").json()
        old = client.get(f"/plans/{plan_id}?version=1").json()
        assert latest["version"] == 2 and old["version"] == 1
        assert len(offline_tools) == before
        assert client.get(f"/plans/{plan_id}?version=99").status_code == 404


def test_presentation_mutation_cannot_change_session_plan(offline_tools):
    from backend.app.conversation.chat import handle_chat
    from backend.app.domain.models import ChatRequest
    from backend.app.conversation.conversation_state import get_conversation
    response = handle_chat(ChatRequest(message="3 days in Paris"))
    original_title = get_conversation(response.conversation_id).last_plan.itinerary[0].activities[0].title
    response.plan.itinerary[0].activities[0].title = "Mutated presentation"
    assert get_conversation(response.conversation_id).last_plan.itinerary[0].activities[0].title == original_title


def test_chinese_origin_is_not_a_confirmed_destination():
    from backend.app.conversation.chat import extract_fields
    from backend.app.conversation.strategy import extract_destination
    fields = extract_fields("从慕尼黑出发玩三天，预算900欧元，美食文化，火车，节奏适中")
    assert fields["origin_city"] == "Munich"
    assert "destination" not in fields
    assert fields["budget"] == 900 and fields["transport_preference"] == "train"
    assert extract_destination("从慕尼黑出发去巴黎") == "Paris"
    assert extract_destination("from Munich to Paris") == "Paris"


def test_debug_ranking_uses_same_analysis_scores(offline_tools):
    from backend.app.planning.planning_engine import state_from_request, discover_candidates_for_state, rank_candidates_for_state
    from backend.app.domain.models import TripRequest
    request = TripRequest(origin_city="Munich", date_range="July", trip_length=3, budget=1000,
                          interests=["culture"], transport_preference="train", physical_level="medium", requested_output="options")
    state = rank_candidates_for_state(discover_candidates_for_state(state_from_request(request)))
    from backend.app.planning.planning_engine import plan_from_state
    plan = plan_from_state(state_from_request(request))
    assert [candidate.name for candidate in state.candidate_destinations[:3]] == [candidate.name for candidate in plan.ranked_destinations]
    assert [candidate.score for candidate in state.candidate_destinations[:3]] == [candidate.score for candidate in plan.ranked_destinations]


def test_explicit_itinerary_request_overrides_router_options(offline_tools, monkeypatch):
    from backend.app.conversation.chat import handle_chat
    from backend.app.domain.models import ChatRequest, StrategyDecision
    monkeypatch.setattr("backend.app.conversation.chat.decide_strategy", lambda *a, **k: StrategyDecision(
        intent="recommend_destination", next_action="run_destination_discovery", requested_output="options", router_mode="llm"))
    response = handle_chat(ChatRequest(message="从慕尼黑出发玩三天，预算900欧元，喜欢美食文化，火车，节奏适中，请推荐目的地并安排具体行程"))
    assert response.plan.requested_output == "itinerary"
    assert len(response.plan.itinerary) == 3
    assert sum(a.depth == "detailed" for a in response.plan.candidate_analyses) == 1
    assert response.planning_task["candidate_destinations"] == []


def test_budget_recheck_removes_obsolete_warning(offline_tools):
    store = InMemoryPlanStore()
    prepared = task()
    prepared.constraints.budget = 100
    base = execute_task(prepared, store=store)
    assert any(i.code == "estimated_budget_exceeded" for i in base.validation_issues)
    revised_task = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base,
        supplied_revision=RevisionRequest(constraints_updates={"budget": 1000}))
    revised = execute_task(revised_task, base, store=store)
    assert not any(i.code == "estimated_budget_exceeded" for i in revised.validation_issues)
    assert not any("exceeds the requested budget" in risk for risk in revised.risks)
    assert revised.itinerary == base.itinerary


def test_relaxing_day_keeps_dinner_without_targeting_it(offline_tools):
    store = InMemoryPlanStore()
    base = execute_task(task(), store=store)
    prepared = task_from_fields(base.constraints.model_dump(), operation="revise", base_plan=base,
                                message="第二天太赶，晚餐保留")
    revised = execute_task(prepared, base, store=store)
    old_dinner = next(a for a in base.itinerary[1].activities if a.time_slot == "evening")
    assert len(revised.itinerary[1].activities) == 2
    assert old_dinner in revised.itinerary[1].activities


def test_evidence_identity_is_independent_of_city_alias():
    common = {"text": "Museum guide", "url": "https://en.wikivoyage.org/wiki/Paris", "section": "See"}
    assert EvidenceChunk(destination="巴黎", **common).evidence_id == EvidenceChunk(destination="Paris", **common).evidence_id


def test_long_trip_generation_is_batched_with_global_day_numbers(monkeypatch, evidence):
    from backend.app.planning.plan_builder import build_itinerary
    from backend.app.domain.models import DestinationAnalysis
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    calls = []
    def generate(prompt, **kwargs):
        numbers = kwargs["response_schema"]["properties"]["days"]["items"]["properties"]["day"]["enum"]
        calls.append(numbers)
        days = [{"day": number, "activities": [{"title": "Museum", "description": "Visit guide-backed museums.",
                  "category": "culture", "time_slot": "morning", "evidence_ids": [evidence.evidence_id]}]} for number in numbers]
        return LLMResult(json.dumps({"days": days, "pros": [], "cons": [], "explanation": "Guide evidence"}), "test", "anthropic", None, None)
    monkeypatch.setattr("backend.app.planning.plan_builder.generate_text", generate)
    built = build_itinerary(DestinationAnalysis(destination="Paris", depth="detailed", evidence=[evidence]), TripConstraints(trip_length=7))
    assert calls == [[1, 2, 3, 4], [5, 6, 7]]
    assert [day.day for day in built.days] == list(range(1, 8))
    assert built.llm_used and built.generation_mode == "llm"
    assert len({a.activity_id for day in built.days for a in day.activities}) == 7
