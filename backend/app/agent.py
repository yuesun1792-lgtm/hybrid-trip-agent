"""Hand-rolled Claude tool-use loop for V2."""

import json
import os

import anthropic

from backend.app.models import TripPlan
from backend.app.prompts import AGENT_SYSTEM_PROMPT
from backend.app.tools import TOOL_SCHEMAS, dispatch_tool

_MAX_ITERATIONS = 5


def explain_with_tools(plan: TripPlan) -> str:
    """Call Claude with tools enabled and return a natural-language explanation.

    Loops until stop_reason == 'end_turn' or the iteration cap is reached.
    Raises on API errors so the caller (explainer.py) can fall back to the template.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    top = plan.ranked_destinations[0]
    user_message = (
        f"Here is the computed trip plan:\n"
        f"- Top destination: {top.name} ({top.country})\n"
        f"- Estimated cost: €{top.estimated_cost}\n"
        f"- Score: {top.score}\n"
        f"- All ranked destinations: {', '.join(d.name for d in plan.ranked_destinations)}\n"
        f"- Planner recommendation: {plan.final_recommendation}\n\n"
        f"Use the available tools to verify key details, then write a short explanation "
        f"(2-3 sentences) for the traveller."
    )

    messages = [{"role": "user", "content": user_message}]

    for _ in range(_MAX_ITERATIONS):
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=AGENT_SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        if response.stop_reason == "tool_use":
            # Append the assistant turn (may contain text + tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Build tool results for every tool_use block in this response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason — return whatever text we have
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        break

    return ""
