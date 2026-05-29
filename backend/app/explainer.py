import os

from backend.app.models import TripPlan
from backend.app.prompts import FALLBACK_EXPLANATION_TEMPLATE


def generate_explanation(plan: TripPlan) -> str:
    """Return a natural-language explanation for the trip plan.

    Delegates to the tool-use agent when ANTHROPIC_API_KEY is set.
    Falls back to a deterministic template on missing key, missing package, or any error.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_explanation(plan)

    try:
        import anthropic  # noqa: F401 — ensure package is available before delegating
        from backend.app.agent import explain_with_tools
        return explain_with_tools(plan)
    except Exception:
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
