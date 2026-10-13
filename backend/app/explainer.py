from backend.app.models import TripPlan, TripRequest
from backend.app.prompts import FALLBACK_EXPLANATION_TEMPLATE


def generate_explanation(plan: TripPlan, request: TripRequest | None = None) -> str:
    """Return a template-based explanation. Used on the no-key fallback path."""
    return _fallback_explanation(plan, request)


def _fallback_explanation(plan: TripPlan, request: TripRequest | None = None) -> str:
    top = plan.ranked_destinations[0]
    budget = request.budget if request is not None else top.estimated_cost
    return FALLBACK_EXPLANATION_TEMPLATE.format(
        top_dest=top.name,
        budget=budget,
    )
