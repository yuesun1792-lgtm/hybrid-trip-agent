"""Deterministic business validation; factual entailment still needs verified data."""
from datetime import timedelta

from backend.app.models import TripPlan
from backend.app.planning_models import ValidationIssue
from backend.app.trajectory import observed


class PlanValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues if issue.severity == "error"))


@observed("plan_validation", "validation")
def validate_plan(plan: TripPlan, base_plan: TripPlan | None = None, preserve_ids: list[str] | None = None) -> TripPlan:
    issues = []
    def add(code, severity, message, target=None):
        issues.append(ValidationIssue(code=code, severity=severity, message=message, target_id=target))
    constraints = plan.constraints
    if plan.requested_output == "itinerary" and constraints and constraints.trip_length:
        if [day.day for day in plan.itinerary] != list(range(1, constraints.trip_length + 1)):
            add("day_count", "error", "Itinerary must contain exactly the requested consecutive days.")
    evidence = {chunk.evidence_id: chunk for chunk in plan.evidence}
    seen = set()
    for day in plan.itinerary:
        if day.day_id in seen:
            add("duplicate_id", "error", "Day IDs must be unique.", day.day_id)
        seen.add(day.day_id)
        if constraints and constraints.start_date and day.date != constraints.start_date + timedelta(days=day.day - 1):
            add("date_mismatch", "error", "Day date differs from the requested start date.", day.day_id)
        if constraints and constraints.destination and day.destination != constraints.destination:
            add("destination_mismatch", "error", "Day does not match the confirmed destination.", day.day_id)
        maximum = {"low": 3, "medium": 4, "high": 6}.get(constraints.physical_level if constraints else None, 4)
        if len(day.activities) > maximum:
            add("pacing_limit", "error", "Activity count exceeds the requested daily pacing limit.", day.day_id)
        if not day.activities:
            add("activities_missing", "error", "Each itinerary day must have structured activities.", day.day_id)
        for activity in day.activities:
            if activity.activity_id in seen:
                add("duplicate_id", "error", "Activity IDs must be unique.", activity.activity_id)
            seen.add(activity.activity_id)
            if activity.place_resolution == "resolved" and (not activity.places or activity.place_id != activity.places[0].place_id):
                add("place_resolution_invalid", "error", "Resolved activity needs matching provider place records.", activity.activity_id)
            if activity.place_resolution != "resolved" and activity.places:
                add("place_resolution_invalid", "error", "Unresolved activity must not contain selected places.", activity.activity_id)
            if any(not place.matches_city(day.destination) or place.place_id != f"{place.provider}:{place.provider_place_id}" for place in activity.places):
                add("place_destination", "error", "Selected place identity or city does not match the activity.", activity.activity_id)
            for ref in activity.evidence_ids:
                chunk = evidence.get(ref)
                if chunk is None:
                    add("evidence_missing", "error", "Activity references unknown evidence.", activity.activity_id)
                elif chunk.destination != day.destination:
                    add("evidence_destination", "error", "Activity evidence belongs to a different destination.", activity.activity_id)
            if not activity.evidence_ids or activity.status == "needs_verification":
                add("activity_unverified", "warning", "Activity requires verification before booking.", activity.activity_id)
        known_durations = [a.duration_minutes for a in day.activities]
        if known_durations and all(d is not None for d in known_durations) and sum(known_durations) > 16 * 60:
            add("day_overloaded", "error", "Known activity durations exceed 16 hours.", day.day_id)
    if base_plan:
        before = {a.activity_id: a for day in base_plan.itinerary for a in day.activities}
        after = {a.activity_id: a for day in plan.itinerary for a in day.activities}
        for identity in preserve_ids or []:
            if identity not in before or identity not in after or before[identity] != after[identity]:
                add("preservation_failed", "error", "An explicitly preserved activity was changed or removed.", identity)
    if constraints and constraints.budget is not None and plan.ranked_destinations:
        cost = plan.ranked_destinations[0].estimated_cost
        if cost is not None and cost > constraints.budget:
            add("estimated_budget_exceeded", "warning", "Curated estimated trip cost exceeds the requested budget; revise before booking.")
        if cost is None:
            add("budget_unknown", "warning", "Budget feasibility cannot be verified because cost is unknown.")
    add("logistics_unverified", "warning", "Opening hours and travel connections are not verified by live services.")
    if plan.place_resolution.status in {"partial", "failed"}:
        add("place_resolution_incomplete", "warning", plan.place_resolution.message)
    if any(issue.severity == "error" for issue in issues):
        raise PlanValidationError(issues)
    previous_warnings = {i.message for i in plan.validation_issues}
    base_risks = [message for message in plan.risks if message not in previous_warnings]
    risks = list(dict.fromkeys([*base_risks, *(i.message for i in issues if i.severity == "warning")]))
    return plan.model_copy(update={"validation_issues": issues, "risks": risks})
