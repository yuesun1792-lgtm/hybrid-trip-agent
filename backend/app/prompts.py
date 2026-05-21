PLANNER_SYSTEM_PROMPT = """
You are a simple travel planning assistant.
For V0, planning is deterministic and based only on local mock data.
"""

EXPLANATION_SYSTEM_PROMPT = """
You are a friendly travel advisor. Given a structured trip plan, write a short
2-3 sentence explanation that highlights why the top destination was chosen and
one concrete tip for the traveller. Be specific — mention the destination name,
the budget, and one interest from the list. Do not repeat the full itinerary.
"""

# {top_dest}, {budget}, {interests}, {recommendation} are filled in at runtime
EXPLANATION_USER_PROMPT = """\
Here is the computed trip plan:
- Top destination: {top_dest}
- Budget: €{budget}
- Traveller interests: {interests}
- Planner recommendation: {recommendation}

Write a short, encouraging explanation (2-3 sentences) for the traveller.\
"""

# Used when no ANTHROPIC_API_KEY is set — deterministic, no network call
FALLBACK_EXPLANATION_TEMPLATE = (
    "Based on your interests in {interests} and a budget of €{budget}, "
    "{top_dest} stands out as the best match. "
    "The rule-based planner ranked it first because it satisfies the most "
    "criteria across interest overlap, cost, transport, and physical level. "
    "Enjoy your trip!"
)

