from pydantic import ValidationError
import re
"""Adapt chat/form requests into one domain task; model routing cannot bypass prerequisites."""
from backend.app.domain.models import TripPlan, TripRequest
from backend.app.domain.planning_models import PlanningTask, TripConstraints, RevisionRequest
from backend.app.planning.plan_store import PlanVersionConflict
from backend.app.planning.revision import RevisionNeedsInput, parse_revision, compute_affected_scope
from backend.app.runtime.trajectory import observed


class PlanningNeedsInput(ValueError):
    def __init__(self, fields, message="More trip details are needed"):
        self.fields = fields
        super().__init__(message)


@observed("task_preparation", "preparation")
def task_from_fields(fields, *, operation="create", base_plan: TripPlan | None = None,
                     supplied_revision: RevisionRequest | None = None, expected_version=None,
                     message="", requested_output="itinerary", queries=None, rejected=None):
    try:
        constraints = TripConstraints.from_fields(fields)
    except (ValidationError, ValueError) as exc:
        missing = list(dict.fromkeys(str(error["loc"][0]) for error in exc.errors() if error["loc"])) if isinstance(exc, ValidationError) else ["trip_length"]
        raise PlanningNeedsInput(missing or ["trip_length"], "Some trip constraints are invalid; please correct them.") from exc
    task = PlanningTask(operation=operation, constraints=constraints, response_language=fields.get("response_language", "en"),
        requested_output=requested_output, user_message=message, queries=queries or [], rejected_destinations=rejected or [])
    if operation == "revise":
        if base_plan is None:
            raise PlanningNeedsInput(["existing_plan"], "A previous plan is needed before revision.")
        if expected_version is not None and expected_version != base_plan.version:
            raise PlanVersionConflict(f"Plan version changed: expected {expected_version}, current {base_plan.version}")
        old = base_plan.constraints or TripConstraints(destination=base_plan.itinerary[0].destination if base_plan.itinerary else None,
                                                        trip_length=len(base_plan.itinerary) or None)
        updates = {key: value for key, value in constraints.model_dump().items()
                   if value != getattr(old, key) and key in fields}
        if not re.search(r"兴趣|偏好|全程|整个|preferences|whole trip|entire trip", message, re.I):
            updates.pop("interests", None)  # A local food/shopping replacement is not a global preference change.
        revision = parse_revision(message, base_plan, supplied_revision, updates)
        if revision.base_plan_id is not None and revision.base_plan_id != base_plan.plan_id:
            raise PlanVersionConflict("Revision refers to a different plan.")
        if revision.base_version is not None and revision.base_version != base_plan.version:
            raise PlanVersionConflict("Revision refers to an outdated plan version.")
        revised = old.model_dump()
        revised.update(revision.constraints_updates)
        task.constraints = TripConstraints.model_validate(revised)
        task.revision = revision
        task.affected_scope = compute_affected_scope(base_plan, revision)
    if task.requested_output == "itinerary" and not task.constraints.trip_length:
        raise PlanningNeedsInput(["trip_length"])
    if not task.constraints.destination and not task.constraints.interests and not task.constraints.origin_city:
        raise PlanningNeedsInput(["interests"])
    task.candidate_destinations = [task.constraints.destination] if task.constraints.destination else list(fields.get("candidate_destinations") or [])
    return task


def task_from_request(request: TripRequest):
    fields = request.model_dump(mode="json")
    return task_from_fields(fields, requested_output=request.requested_output)
