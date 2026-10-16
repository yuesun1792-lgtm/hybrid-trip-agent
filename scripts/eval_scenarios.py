"""Multi-turn behavioral evaluation: python -m scripts.eval_scenarios --help."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
from time import perf_counter
from typing import Literal
from unittest.mock import patch
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.domain.models import ChatRequest, LLMUsage
from backend.app.runtime.trajectory import clean, error_details


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ActivityExpectation(StrictModel):
    day: int = Field(gt=0)
    time_slot: Literal["morning", "afternoon", "evening", "flexible"] | None = None
    category: str | None = None
    min_count: int = Field(default=1, ge=0)
    max_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_counts(self):
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be >= min_count")
        return self


class Expectations(StrictModel):
    outcome: Literal["plan", "revision", "clarification", "error"]
    error_type: str | None = None
    destination: str | None = None
    day_count: int | None = Field(default=None, ge=0)
    language: str | None = None
    output: Literal["options", "itinerary"] | None = None
    min_candidates: int | None = Field(default=None, ge=1)
    detailed_count: int | None = Field(default=None, ge=0)
    changed_days: list[int] | None = None
    preserve_other_days: bool = False
    preserve_evening: bool = False
    fewer_activities_days: list[int] = Field(default_factory=list)
    version_increment: int | None = Field(default=None, ge=0)
    allow_fallback: bool = False
    activity_checks: list[ActivityExpectation] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)
    place_status: Literal["disabled", "completed", "partial", "failed"] | None = None
    min_resolved_activities: int | None = Field(default=None, ge=0)
    max_place_tool_calls: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_error(self):
        if (self.outcome == "error") != bool(self.error_type):
            raise ValueError("error outcome requires error_type; other outcomes cannot set it")
        if self.preserve_other_days and self.changed_days is None:
            raise ValueError("preserve_other_days requires changed_days")
        plan_checks = {"destination", "day_count", "output", "min_candidates", "detailed_count",
                       "changed_days", "preserve_other_days", "preserve_evening", "version_increment",
                       "activity_checks", "constraints", "fewer_activities_days", "place_status",
                       "min_resolved_activities", "max_place_tool_calls"}
        if self.outcome in {"error", "clarification"} and plan_checks & self.model_fields_set:
            raise ValueError("Plan assertions require plan/revision outcome")
        return self


class Turn(StrictModel):
    message: str = Field(min_length=1)
    request: dict = Field(default_factory=dict)
    expect: Expectations

    @model_validator(mode="after")
    def valid_request(self):
        if {"message", "conversation_id"} & self.request.keys():
            raise ValueError("message/conversation_id are managed by the runner")
        unknown = self.request.keys() - ChatRequest.model_fields.keys()
        if unknown:
            raise ValueError(f"Unknown request fields: {sorted(unknown)}")
        ChatRequest(message=self.message, **self.request)
        return self


class Scenario(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = ""
    turns: list[Turn] = Field(min_length=1)


class Suite(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({case.id for case in self.scenarios}) != len(self.scenarios):
            raise ValueError("Scenario IDs must be unique")
        return self


@contextmanager
def runtime(mode):
    """Offline uses the actual workflow with synthetic evidence and rule fallbacks."""
    with ExitStack() as stack:
        if mode == "offline":
            stack.enter_context(patch.dict(os.environ, {
                "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "HF_HUB_OFFLINE": "1",
                "GEOAPIFY_API_KEY": "", "ENABLE_PLACE_RESOLUTION": "false"}))

            def search(query, k=6, destination=None):
                return [{"destination": city, "section": "See", "score": .9,
                         "url": "https://en.wikivoyage.org/wiki/" + city,
                         "text": f"{city} - See\nMuseums, galleries, food markets, shops and riverside walks."}
                        for city in ([destination] if destination else ["Paris", "Rome", "Prague"])]

            stack.enter_context(patch("backend.app.rag.knowledge.search_travel_knowledge", search))
        yield


def check_turn(expect, response, before, after, error=None, mode="live"):
    checks = []

    def check(name, ok, detail=""):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    if expect.outcome == "error":
        check("expected_error", error is not None and type(error).__name__ == expect.error_type,
              type(error).__name__ if error else "No error")
        check("error_did_not_commit", before == after)
        return checks
    check("no_error", error is None, str(error) if error else "")
    if error or response is None:
        return checks
    plan = response.plan
    if expect.outcome == "clarification":
        check("clarification", plan is None and bool(response.assistant_message))
        check("clarification_did_not_commit", before == after)
    else:
        check("plan_returned", plan is not None)
        if plan is None:
            return checks
        if expect.outcome == "revision":
            check("same_plan", before is not None and before.plan_id == plan.plan_id)
            check("revision_committed", before is not None and after is not None and after.version > before.version)
        check("no_validation_errors", not any(i.severity == "error" for i in plan.validation_issues))
        ids = [a.activity_id for day in plan.itinerary for a in day.activities]
        check("unique_activity_ids", len(ids) == len(set(ids)))
        check("consecutive_days", [d.day for d in plan.itinerary] == list(range(1, len(plan.itinerary) + 1)))
        evidence = {e.evidence_id: e for e in plan.evidence}
        check("activity_evidence", all((a.evidence_ids or a.status == "needs_verification") and all(
            key in evidence and evidence[key].destination == day.destination for key in a.evidence_ids)
            for day in plan.itinerary for a in day.activities))
        if expect.destination is not None:
            check("destination", bool(plan.itinerary) and all(d.destination == expect.destination for d in plan.itinerary))
        if expect.day_count is not None:
            check("day_count", len(plan.itinerary) == expect.day_count)
        if expect.output is not None:
            check("output", plan.requested_output == expect.output)
            if expect.output == "options":
                check("options_have_no_schedule", not plan.itinerary and not plan.selected_destinations)
        if expect.min_candidates is not None:
            check("candidate_count", len(plan.ranked_destinations) >= expect.min_candidates)
        if expect.detailed_count is not None:
            check("detailed_count", sum(a.depth == "detailed" for a in plan.candidate_analyses) == expect.detailed_count)
        for target in expect.activity_checks:
            matches = [a for day in plan.itinerary if day.day == target.day for a in day.activities
                       if (target.time_slot is None or a.time_slot == target.time_slot) and
                          (target.category is None or a.category == target.category)]
            check(f"activities_day_{target.day}_{target.time_slot or 'all'}_{target.category or 'all'}",
                  len(matches) >= target.min_count and (target.max_count is None or len(matches) <= target.max_count),
                  f"actual_count={len(matches)}")
        actual_constraints = plan.constraints.model_dump(mode="json") if plan.constraints else {}
        for key, value in expect.constraints.items():
            check(f"constraint_{key}", key in actual_constraints and actual_constraints[key] == value)
        if expect.place_status is not None:
            check("place_status", plan.place_resolution.status == expect.place_status)
        if expect.min_resolved_activities is not None:
            check("resolved_activity_count", sum(bool(a.places) and a.place_resolution == "resolved"
                  for d in plan.itinerary for a in d.activities) >= expect.min_resolved_activities)
        if expect.max_place_tool_calls is not None:
            check("place_tool_call_limit", plan.place_resolution.tool_calls <= expect.max_place_tool_calls)
    if expect.language is not None:
        # Language code is routing metadata, not a prose-quality judgment.
        check("response_language", response.response_language == expect.language)
    if expect.version_increment is not None:
        check("version_increment", before is not None and after is not None and
              after.version == before.version + expect.version_increment)
    if expect.changed_days is not None or expect.preserve_other_days or expect.preserve_evening or expect.fewer_activities_days:
        check("canonical_snapshots_available", before is not None and after is not None)
        if before is not None and after is not None:
            old = {d.day: d for d in before.itinerary}
            new = {d.day: d for d in after.itinerary}
            changed = {number for number in old.keys() | new.keys() if old.get(number) != new.get(number)}
            if expect.changed_days is not None:
                check("changed_days", changed == set(expect.changed_days), f"actual={sorted(changed)}")
            if expect.preserve_other_days:
                check("preserve_other_days", expect.changed_days is not None and
                      all(new.get(n) == d for n, d in old.items() if n not in expect.changed_days))
            if expect.preserve_evening:
                evening = [a for d in before.itinerary for a in d.activities if a.time_slot == "evening"]
                activities = {a.activity_id: a for d in after.itinerary for a in d.activities}
                check("preserve_evening", bool(evening) and all(activities.get(a.activity_id) == a for a in evening))
            for number in expect.fewer_activities_days:
                check(f"fewer_activities_day_{number}", number in old and number in new and
                      len(new[number].activities) < len(old[number].activities))
    fallback = any(step.status == "fallback" for step in response.trace)
    check("fallback_policy", not fallback or expect.allow_fallback or mode == "offline")
    return checks


def run_suite(suite, output_dir, *, mode="offline", repeat=1, handler=None, snapshot=None):
    from backend.app.conversation.chat import handle_chat
    from backend.app.conversation.conversation_state import get_conversation
    from backend.app.config import llm_model, llm_provider

    if mode not in {"offline", "live"} or repeat < 1:
        raise ValueError("mode must be offline/live and repeat must be positive")
    handler = handler or handle_chat
    snapshot = snapshot or (lambda cid: get_conversation(cid).last_plan)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "1.0", "started_at": datetime.now(timezone.utc).isoformat(),
              "mode": mode, "provider": llm_provider() if mode == "live" else "none",
              "model": llm_model() if mode == "live" else "rule", "cases": [], "status": "running",
              "expected_cases": len(suite.scenarios) * repeat}
    with runtime(mode), patch.dict(os.environ, {"TRAJECTORY_DIR": str(output_dir / "trajectories")}):
        for iteration in range(1, repeat + 1):
            for case in suite.scenarios:
                cid = "eval-" + str(uuid4())
                result = {"id": case.id, "iteration": iteration, "conversation_id": cid, "turns": []}
                for index, turn in enumerate(case.turns, 1):
                    base = snapshot(cid)
                    before = base.model_copy(deep=True) if base else None
                    request = {"expected_plan_version": before.version} if before else {}
                    request.update(turn.request)
                    response, error = None, None
                    previous_run = get_conversation(cid).last_run_id
                    start = perf_counter()
                    try:
                        response = handler(ChatRequest(message=turn.message, conversation_id=cid, **request))
                    except Exception as exc:
                        error = exc
                    current = snapshot(cid)
                    after = current.model_copy(deep=True) if current else None
                    checks = check_turn(turn.expect, response, before, after, error, mode)
                    usage = response.llm_usage if response else []
                    run_id = response.run_id if response else get_conversation(cid).last_run_id
                    trajectory_path = output_dir / "trajectories" / (run_id + ".traj.json") if run_id and run_id != previous_run else None
                    if error and trajectory_path and trajectory_path.is_file():
                        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                        usage = [LLMUsage.model_validate(step["llm"]["usage"]) for step in trajectory["steps"]
                                 if step.get("llm", {}).get("usage") is not None]
                    record = {"turn": index, "input": turn.model_dump(mode="json"),
                              "duration_ms": round((perf_counter() - start) * 1000, 2),
                              "passed": all(c["passed"] for c in checks), "checks": checks,
                              "error": error_details(error) if error else None,
                              "response": response, "canonical_before": before, "canonical_after": after,
                              "trajectory_file": str(trajectory_path) if trajectory_path else None,
                              "fallback_used": any(s.status == "fallback" for s in response.trace) if response else None,
                              "input_tokens": sum(u.input_tokens for u in usage),
                              "output_tokens": sum(u.output_tokens for u in usage),
                              "known_estimated_cost_usd": sum(u.estimated_cost_usd or 0 for u in usage),
                              "estimated_cost_usd": sum(u.estimated_cost_usd for u in usage)
                                  if response and all(u.estimated_cost_usd is not None for u in usage) else None}
                    result["turns"].append(clean(record))
                    if not record["passed"]:
                        break  # Dependent turns are not run on an invalid baseline.
                result["passed"] = len(result["turns"]) == len(case.turns) and all(t["passed"] for t in result["turns"])
                result["skipped_turns"] = len(case.turns) - len(result["turns"])
                report["cases"].append(result)
                write_report(report, output_dir)  # Preserve progress after each case.
    report["status"] = "completed"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_report(report, output_dir)
    return report


def write_report(report, output_dir):
    turns = [t for case in report["cases"] for t in case["turns"]]
    passed = sum(c["passed"] for c in report["cases"])
    report["summary"] = {"cases": len(report["cases"]), "passed": passed,
        "failed": len(report["cases"]) - passed, "executed_turns": len(turns),
        "skipped_turns": sum(c["skipped_turns"] for c in report["cases"]),
        "fallback_turns": sum(t["fallback_used"] is True for t in turns),
        "input_tokens": sum(t["input_tokens"] for t in turns),
        "output_tokens": sum(t["output_tokens"] for t in turns),
        "known_estimated_cost_usd": sum(t["known_estimated_cost_usd"] for t in turns),
        "unknown_cost_turns": sum(t["estimated_cost_usd"] is None for t in turns),
        "median_turn_ms": statistics.median(t["duration_ms"] for t in turns) if turns else None}
    temporary = output_dir / "report.tmp"
    temporary.write_text(json.dumps(clean(report), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "report.json")
    lines = ["# Scenario evaluation", "", f"Mode: {report['mode']}; passed {passed}/{len(report['cases'])}.", "",
             "| Case | Repeat | Result | Failed checks |", "|---|---|---|---|"]
    for case in report["cases"]:
        failures = [f"turn {t['turn']}: {c['name']}" for t in case["turns"] for c in t["checks"] if not c["passed"]]
        lines.append(f"| {case['id']} | {case['iteration']} | {'PASS' if case['passed'] else 'FAIL'} | {', '.join(failures)} |")
    lines += ["", "See report.json for snapshots, usage and trajectory run IDs. Unknown costs are not counted as zero.",
              "Language checks cover metadata only; factual quality and translation need separate review."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("evals/scenarios/core.json"))
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--case", action="append", dest="cases", help="Scenario ID, repeatable")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.repeat < 1:
            raise ValueError("--repeat must be positive")
        suite = Suite.model_validate_json(args.suite.read_text(encoding="utf-8"))
        if args.cases:
            unknown = set(args.cases) - {case.id for case in suite.scenarios}
            if unknown:
                raise ValueError(f"Unknown cases: {sorted(unknown)}")
            suite.scenarios = [case for case in suite.scenarios if case.id in args.cases]
        if args.mode == "live":
            from dotenv import load_dotenv
            from backend.app.config import llm_api_key
            load_dotenv()
            if not llm_api_key():
                raise ValueError("Live mode requires the selected provider's API key")
        directory = args.output_dir or Path("data/evaluations") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        report = run_suite(suite, directory, mode=args.mode, repeat=args.repeat)
        print(json.dumps({"summary": report["summary"], "report": str(directory / "report.json")}, ensure_ascii=False, indent=2))
        return 1 if report["summary"]["failed"] else 0
    except (ValueError, OSError) as exc:
        print(json.dumps(error_details(exc), ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
