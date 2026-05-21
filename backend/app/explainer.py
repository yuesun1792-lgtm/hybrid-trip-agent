import os

from backend.app.models import TripPlan
from backend.app.prompts import (
    EXPLANATION_SYSTEM_PROMPT,
    EXPLANATION_USER_PROMPT,
    FALLBACK_EXPLANATION_TEMPLATE,
)


def generate_explanation(plan: TripPlan) -> str:
    """Return a natural-language explanation for the trip plan.

    Calls Claude when ANTHROPIC_API_KEY is set; otherwise returns a
    deterministic template-based string so the app works without an API key.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_explanation(plan)

    # Lazy import so `anthropic` is not required when no key is configured.
    try:
        import anthropic
    except ImportError:
        return _fallback_explanation(plan)

    top_dest = plan.ranked_destinations[0].name
    interests = ", ".join(
        d.name for d in plan.ranked_destinations[:1]  # just the winner for brevity
    )
    # Use the traveller's interests from the pros list as a proxy
    interests_str = ", ".join(
        p for p in plan.ranked_destinations[0].pros[:2]
    ) or top_dest

    user_message = EXPLANATION_USER_PROMPT.format(
        top_dest=top_dest,
        budget=plan.ranked_destinations[0].estimated_cost,
        interests=interests_str,
        recommendation=plan.final_recommendation,
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        system=EXPLANATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


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
