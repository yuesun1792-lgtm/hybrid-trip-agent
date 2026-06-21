from backend.app.models import TripPlan
from backend.app.prompts import FALLBACK_EXPLANATION_TEMPLATE


def generate_explanation(plan: TripPlan) -> str:
    """Return a template-based explanation. Used on the no-key fallback path."""
    return _fallback_explanation(plan)


def _fallback_explanation(plan: TripPlan) -> str:
    top = plan.ranked_destinations[0]
    return FALLBACK_EXPLANATION_TEMPLATE.format(
        top_dest=top.name,
        budget=top.estimated_cost,
    )
