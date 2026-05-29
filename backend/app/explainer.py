from backend.app.models import TripPlan
from backend.app.prompts import FALLBACK_EXPLANATION_TEMPLATE


def generate_explanation(plan: TripPlan) -> str:
    """Return a template-based explanation. Used on the no-key fallback path."""
    return _fallback_explanation(plan)


def _fallback_explanation(plan: TripPlan) -> str:
    top_dest = plan.ranked_destinations[0].name
    pros = plan.ranked_destinations[0].pros
    interests_str = ", ".join(pros[:2]) if pros else top_dest
    budget = plan.ranked_destinations[0].estimated_cost
    return FALLBACK_EXPLANATION_TEMPLATE.format(
        top_dest=top_dest,
        budget=budget,
        interests=interests_str,
    )
