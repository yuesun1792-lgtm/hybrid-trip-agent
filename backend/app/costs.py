import json

from backend.app.config import DESTINATION_COSTS_PATH

LEGACY_BASE_COST_DAYS = 4


def load_daily_costs() -> dict[str, int]:
    data = json.loads(DESTINATION_COSTS_PATH.read_text())
    costs = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(value.get("daily_cost_eur"), int):
            costs[key] = value["daily_cost_eur"]
        elif isinstance(value, int):
            # Backward compatibility for the old ~4-day total schema.
            costs[key] = max(1, round(value / LEGACY_BASE_COST_DAYS))
    return costs


def load_base_costs() -> dict[str, int]:
    """Backward-compatible alias returning per-day EUR costs."""
    return load_daily_costs()


def estimate_trip_cost(destination: str, trip_length_days: int, costs: dict[str, int] | None = None) -> int | None:
    """Estimate total trip cost from per-day EUR destination costs."""
    costs = costs or load_daily_costs()
    daily_cost = costs.get(destination)
    if daily_cost is None:
        return None
    return max(1, round(daily_cost * trip_length_days))


def normalize_plan_costs(plan, request):
    """Normalize known destination costs in a TripPlan to the requested trip length."""
    costs = load_daily_costs()
    updated_destinations = []
    for destination in plan.ranked_destinations:
        estimated_cost = estimate_trip_cost(destination.name, request.trip_length, costs)
        if estimated_cost is None:
            updated_destinations.append(destination)
            continue

        update = {"estimated_cost": estimated_cost}
        if destination.estimated_cost_home and destination.estimated_cost:
            inferred_rate = destination.estimated_cost_home / destination.estimated_cost
            update["estimated_cost_home"] = round(estimated_cost * inferred_rate, 2)
        updated_destinations.append(destination.model_copy(update=update))

    return plan.model_copy(update={"ranked_destinations": updated_destinations})
