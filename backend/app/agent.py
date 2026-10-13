"""Hand-rolled Claude tool-use loop — V3: the LLM drives planning."""

import json
import os

import anthropic
from pydantic import ValidationError

from backend.app.config import ANTHROPIC_MODEL
from backend.app.models import TripPlan, TripRequest
from backend.app.prompts import build_agent_system_prompt
from backend.app.tools import TOOL_SCHEMAS, dispatch_tool
from backend.app.costs import normalize_plan_costs

_MAX_ITERATIONS = 16
_DEFAULT_MAX_TOKENS = 4096
_NUDGE = (
    "You have not submitted a plan yet. "
    "Please call submit_trip_plan now with the complete trip plan."
)
_TOKEN_LIMIT_NUDGE = (
    "Your previous response hit the output token limit. Do not explain or analyze. "
    "Call submit_trip_plan now with the best complete structured plan you can produce from "
    "the tool results already available."
)


def _agent_max_tokens() -> int:
    raw = os.environ.get("ANTHROPIC_AGENT_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS))
    try:
        return max(1024, int(raw))
    except ValueError:
        return _DEFAULT_MAX_TOKENS


def _build_user_message(request: TripRequest) -> str:
    lines = [
        "Plan a trip with these requirements:",
        f"- Origin city: {request.origin_city}",
        f"- Dates: {request.date_range}",
        f"- Duration: {request.trip_length} days",
        f"- Budget: €{request.budget}",
        f"- Interests: {', '.join(request.interests)}",
        f"- Transport preference: {request.transport_preference}",
        f"- Physical level: {request.physical_level}",
    ]
    if request.taste_profile:
        lines.append(f"- Taste profile: {request.taste_profile}")
    lines.append(
        "\nUse the available tools to research destinations and then call "
        "submit_trip_plan with a complete plan."
    )
    return "\n".join(lines)


def _deterministic_fallback(request: TripRequest, fallback_reason: str) -> TripPlan:
    from backend.app.explainer import _fallback_explanation
    from backend.app.planner import create_trip_plan
    plan = create_trip_plan(request)
    explanation = _fallback_explanation(plan, request)
    return plan.model_copy(
        update={
            "explanation": explanation,
            "planner_mode": "deterministic_fallback",
            "fallback_reason": fallback_reason,
        }
    )


def create_plan(request: TripRequest) -> TripPlan:
    """Drive planning via a Claude tool-use loop; ends when submit_trip_plan is called.

    Falls back to the deterministic planner if:
    - the model never calls submit_trip_plan (end_turn after nudge)
    - submit_trip_plan input fails Pydantic validation twice
    - the iteration cap is reached
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [{"role": "user", "content": _build_user_message(request)}]
    nudge_sent = False
    token_limit_nudge_sent = False
    validation_retries = 0

    for _ in range(_MAX_ITERATIONS):
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=_agent_max_tokens(),
            system=build_agent_system_prompt(request.home_currency),
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            if nudge_sent:
                return _deterministic_fallback(request, "agent_did_not_submit_plan")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": _NUDGE})
            nudge_sent = True
            continue

        if response.stop_reason == "max_tokens":
            if token_limit_nudge_sent:
                return _deterministic_fallback(request, "agent_stop_reason: max_tokens")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": _TOKEN_LIMIT_NUDGE})
            token_limit_nudge_sent = True
            continue

        if response.stop_reason != "tool_use":
            return _deterministic_fallback(request, f"agent_stop_reason: {response.stop_reason}")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "submit_trip_plan":
                try:
                    plan = TripPlan(**block.input).model_copy(
                        update={"planner_mode": "llm_agent", "fallback_reason": None}
                    )
                    return normalize_plan_costs(plan, request)
                except ValidationError as exc:
                    if validation_retries < 1:
                        validation_retries += 1
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Validation failed: {exc}. Fix the errors and call submit_trip_plan again.",
                            "is_error": True,
                        })
                    else:
                        return _deterministic_fallback(request, "agent_validation_failed")
            else:
                result = dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return _deterministic_fallback(request, "agent_iteration_limit")
