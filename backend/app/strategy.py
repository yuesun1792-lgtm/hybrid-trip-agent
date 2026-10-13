import json
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

from backend.app.config import ANTHROPIC_MODEL, WIKIVOYAGE_SOURCES_PATH
from backend.app.llm_usage import usage_from_response
from backend.app.models import StrategyDecision


load_dotenv()


RECOMMENDATION_REQUIRED_FIELDS = [
    "origin_city",
    "date_range",
    "trip_length",
    "budget",
    "interests",
    "transport_preference",
    "physical_level",
]

FIXED_ITINERARY_REQUIRED_FIELDS = ["destination", "trip_length"]

DESTINATION_ALIASES = {
    "巴黎": "Paris",
    "罗马": "Rome",
    "威尼斯": "Venice",
    "佛罗伦萨": "Florence",
    "布拉格": "Prague",
    "维也纳": "Vienna",
    "里斯本": "Lisbon",
    "波尔图": "Porto",
    "巴塞罗那": "Barcelona",
    "阿姆斯特丹": "Amsterdam",
    "雅典": "Athens",
    "圣托里尼": "Santorini",
    "萨尔茨堡": "Salzburg",
    "慕尼黑": "Munich",
}


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def missing_for_recommendation(fields: dict[str, Any]) -> list[str]:
    return [field for field in RECOMMENDATION_REQUIRED_FIELDS if _is_missing(fields.get(field))]


def missing_for_fixed_itinerary(fields: dict[str, Any]) -> list[str]:
    return [field for field in FIXED_ITINERARY_REQUIRED_FIELDS if _is_missing(fields.get(field))]


@lru_cache(maxsize=1)
def known_destinations() -> list[str]:
    try:
        sources = json.loads(WIKIVOYAGE_SOURCES_PATH.read_text())
    except Exception:
        return [
            "Lisbon",
            "Paris",
            "Rome",
            "Venice",
            "Prague",
            "Vienna",
            "Amsterdam",
        ]
    return sorted(
        {str(item.get("destination", "")).strip() for item in sources if item.get("destination")},
        key=len,
        reverse=True,
    )


def extract_destination(message: str) -> str | None:
    """Extract a known Wikivoyage destination from free text."""
    lower = message.lower()
    for alias, destination in DESTINATION_ALIASES.items():
        if alias in message:
            return destination

    for destination in known_destinations():
        escaped = re.escape(destination.lower())
        if not re.search(rf"\b{escaped}\b", lower):
            continue
        if re.search(rf"\bfrom\s+{escaped}\b", lower):
            # "from Paris" is more likely an origin unless another destination appears.
            continue
        return destination
    return None


def _json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM strategy response did not contain JSON")
    return json.loads(text[start : end + 1])


def _llm_decide_strategy(
    message: str,
    fields: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> StrategyDecision:
    import anthropic

    prompt = f"""You are the strategy router for a travel planning agent.
Return ONLY JSON matching this schema:
{{
  "intent": "recommend_destination" | "plan_fixed_destination" | "revise_plan" | "clarify",
  "next_action": "ask_user" | "run_destination_discovery" | "run_fixed_itinerary" | "revise_existing_plan",
  "known_fields": object,
  "missing_fields": array of strings,
  "queries": array of retrieval queries,
  "reason": string
}}

Routing rules:
- recommend_destination: user wants suggestions for where to go.
- plan_fixed_destination: user already names a destination and wants an itinerary there.
- revise_plan: user comments on or asks to modify the previous plan.
- clarify: user intent or required details are too ambiguous.
- If last_plan_exists is true and the user says a day is bad, too rushed, too expensive,
  asks to change a day, or says "last day", choose revise_plan.
- Do not answer travel facts from memory. Queries are only search strings for RAG.
- For fixed destination itinerary, require destination and trip_length.
- For destination recommendation, require origin_city, date_range, trip_length, budget,
  interests, transport_preference, and physical_level.

Current extracted fields:
{json.dumps(fields, ensure_ascii=True)}

Conversation context:
{json.dumps(context or {}, ensure_ascii=True)}

User message:
{message}"""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    decision = StrategyDecision.model_validate(_json_object(response.content[0].text.strip()))
    return decision.model_copy(
        update={
            "router_mode": "llm",
            "llm_usage": usage_from_response("task_router", ANTHROPIC_MODEL, response),
        }
    )


def _is_revision_request(message: str) -> bool:
    lower = message.lower()
    revision_terms = [
        "不好",
        "不太好",
        "不满意",
        "修改",
        "调整",
        "改一下",
        "换",
        "最后一天",
        "第一天",
        "第二天",
        "第三天",
        "第四天",
        "太赶",
        "太满",
        "太贵",
        "不要",
        "last day",
        "final day",
        "day 1",
        "day 2",
        "day 3",
        "day 4",
        "not good",
        "change",
        "revise",
        "replace",
        "too rushed",
        "too busy",
        "too expensive",
    ]
    return any(term in lower or term in message for term in revision_terms)


def _rule_decide_strategy(
    message: str,
    fields: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> StrategyDecision:
    context = context or {}
    if context.get("last_plan_exists") and _is_revision_request(message):
        return StrategyDecision(
            intent="revise_plan",
            next_action="revise_existing_plan",
            reason="User feedback refers to the existing plan, so route to plan revision.",
        )

    destination = fields.get("destination") or extract_destination(message)
    known_fields = {}
    if destination:
        known_fields["destination"] = destination

    fixed_fields = {**fields, **known_fields}
    if destination:
        missing = missing_for_fixed_itinerary(fixed_fields)
        interests = " ".join(fields.get("interests") or [])
        trip_length = fields.get("trip_length", "")
        return StrategyDecision(
            intent="plan_fixed_destination",
            next_action="ask_user" if missing else "run_fixed_itinerary",
            known_fields=known_fields,
            missing_fields=missing,
            queries=[
                f"{destination} see do eat get around {trip_length} days {interests}",
                f"{destination} museums food neighborhoods highlights {interests}",
            ],
            reason="A destination was explicitly named, so route to fixed-destination itinerary.",
        )

    missing = missing_for_recommendation(fields)
    if missing:
        return StrategyDecision(
            intent="clarify",
            next_action="ask_user",
            missing_fields=missing,
            reason="Destination recommendation needs more structured trip constraints.",
        )

    interests = " ".join(fields.get("interests") or [])
    return StrategyDecision(
        intent="recommend_destination",
        next_action="run_destination_discovery",
        queries=[
            (
                f"{fields.get('origin_city')} {fields.get('transport_preference')} "
                f"{interests} {fields.get('date_range')} {fields.get('trip_length')} days"
            )
        ],
        reason="No fixed destination was named and recommendation constraints are complete.",
    )


def decide_strategy(
    message: str,
    fields: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> StrategyDecision:
    """Choose the next planning workflow, using LLM routing when available."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key and api_key != "your_key_here":
        try:
            return _llm_decide_strategy(message, fields, context=context)
        except Exception as exc:
            decision = _rule_decide_strategy(message, fields, context=context)
            return decision.model_copy(
                update={"router_fallback_reason": f"llm_router_error: {type(exc).__name__}"}
            )
    decision = _rule_decide_strategy(message, fields, context=context)
    return decision.model_copy(update={"router_fallback_reason": "anthropic_api_key_missing"})
