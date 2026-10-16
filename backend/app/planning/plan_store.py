"""Versioned plan repository. Swap this adapter for a durable store when required."""
from threading import RLock
from typing import Protocol

from backend.app.domain.models import TripPlan
from backend.app.runtime.trajectory import observed


class PlanVersionConflict(ValueError):
    pass


class PlanStore(Protocol):
    def latest(self, plan_id: str) -> TripPlan | None: ...
    def get(self, plan_id: str, version: int) -> TripPlan | None: ...
    def commit(self, plan: TripPlan, expected_version: int | None = None) -> TripPlan: ...


class InMemoryPlanStore:
    def __init__(self):
        self._versions: dict[str, dict[int, TripPlan]] = {}
        self._lock = RLock()

    def latest(self, plan_id):
        with self._lock:
            history = self._versions.get(plan_id, {})
            return history[max(history)].model_copy(deep=True) if history else None

    def get(self, plan_id, version):
        with self._lock:
            plan = self._versions.get(plan_id, {}).get(version)
            return plan.model_copy(deep=True) if plan else None

    @observed("plan_commit", "commit")
    def commit(self, plan, expected_version=None):
        with self._lock:
            history = self._versions.get(plan.plan_id, {})
            current = max(history) if history else None
            if expected_version != current:
                raise PlanVersionConflict(f"Plan version changed: expected {expected_version}, current {current}")
            next_version = (current or 0) + 1
            committed = plan.model_copy(deep=True, update={"version": next_version, "parent_version": current})
            self._versions.setdefault(plan.plan_id, {})[next_version] = committed
            return committed.model_copy(deep=True)


PLAN_STORE = InMemoryPlanStore()
