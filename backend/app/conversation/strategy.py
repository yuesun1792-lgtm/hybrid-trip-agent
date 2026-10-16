from backend.app.runtime.trajectory import observed
from backend.app.llm.llm import generate_text
import json
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

from backend.app.config import WIKIVOYAGE_SOURCES_PATH, llm_api_key, llm_provider
from backend.app.domain.models import StrategyDecision


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

from backend.app.domain.destination_names import DESTINATION_ALIASES



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
    """Identify destinations without treating an explicitly named origin as a target."""
    lower = message.lower()
    origin_spans = [match.span(1) for match in re.finditer(r"从\s*([^，,。;\n]+?)\s*(?:出发|前往|到|去)", message)]
    def is_origin(position):
        return any(start <= position < end for start, end in origin_spans)
    for alias, destination in DESTINATION_ALIASES.items():
        if any(not is_origin(match.start()) for match in re.finditer(re.escape(alias), message)):
            return destination
    for destination in known_destinations():
        escaped = re.escape(destination.lower())
        for match in re.finditer(rf"\b{escaped}\b", lower):
            if is_origin(match.start()) or re.search(r"\bfrom\s+$", lower[:match.start()]):
                continue
            return destination
    return None


def _json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM strategy response did not contain JSON")
    return json.loads(text[start : end + 1])


@observed("task_router_llm", "llm_validation")
def _llm_decide_strategy(
    message: str,
    fields: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> StrategyDecision:
    prompt = f"""You are the strategy router for a travel planning agent.
Return ONLY JSON matching this schema:
{{
  "intent": "plan_trip" | "revise_plan" | "clarify",
  "next_action": "ask_user" | "run_planning" | "revise_existing_plan",
  "known_fields": object,
  "missing_fields": array of strings,
  "queries": array of retrieval queries,
  "requested_output": "options" | "itinerary",
  "reason": string
}}

Routing rules:
- plan_trip: create a new travel plan or compare destinations, whether a destination is already known or not.
- Use run_planning for both known and unknown destinations; the common scheduler resolves candidates.
- revise_plan: user comments on or asks to modify the previous plan.
- clarify: user intent or required details are too ambiguous.
- If last_plan_exists is true and the user says a day is bad, too rushed, too expensive,
  asks to change a day, or says "last day", choose revise_plan.
- Return known_fields.destination and origin_city as canonical ENGLISH names used by the index (巴黎 -> Paris).
- Do not turn an origin city into the destination when the user asks for recommendations.
- For dates without an explicit year, retain date_range and do not invent start_date.
- Return interests as English terms (美食 -> food, 文化 -> culture, 时尚 -> fashion).
- Generate ALL retrieval queries in English because the knowledge corpus and embedding model are English.
- Do not answer travel facts from memory. Queries are only search strings for RAG.
- For fixed destination itinerary, require destination and trip_length.
- For options/comparison, a destination or meaningful interests/origin are enough. Do not require optional budget or dates.
- For itinerary output require trip_length. Unknown destination starts candidate discovery before the same analysis/build pipeline.
- Choose requested_output=options only if the user asks to compare/list destinations WITHOUT daily plans or explicitly wants to choose before planning.
- If recommendations AND a specific/daily itinerary are requested, choose itinerary even without a destination: code discovers candidates, selects the best under constraints and builds its itinerary.
- If context.pending_revision is present, interpret the message as clarification of that revision.

Current extracted fields:
{json.dumps(fields, ensure_ascii=True)}

Conversation context:
{json.dumps(context or {}, ensure_ascii=True)}

User message:
{message}"""
    response = generate_text(prompt, stage="task_router", max_tokens=700, prompt_version="task_router_v1")
    decision = StrategyDecision.model_validate(_json_object(response.text))
    return decision.model_copy(
        update={
            "router_mode": "llm",
            "llm_usage": response.usage,
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
        "添加",
        "增加",
        "删除",
        "取消",
        "移除",
        "改成",
        "改为",
        "add ",
        "insert ",
        "remove ",
        "delete ",
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


def infer_output_scope(message: str, explicit=None, model_hint=None):
    if explicit:
        return explicit
    if re.search(r"只推荐|先比较|仅比较|只比较|只列出|不需要.*行程|不要.*行程|without.*itinerary|options only", message, re.I):
        return "options"
    if re.search(r"具体行程|每天|日程|安排.*行程|itinerary|daily plan", message, re.I):
        return "itinerary"
    if re.search(r"比较.*城市|列出.*城市|compare.*cities|list.*options", message, re.I):
        return "options"
    return model_hint or "itinerary"


def _rule_decide_strategy(message: str, fields: dict[str, Any], context: dict[str, Any] | None = None) -> StrategyDecision:
    context = context or {}
    if context.get("pending_revision") or (context.get("last_plan_exists") and _is_revision_request(message)):
        return StrategyDecision(intent="revise_plan", next_action="revise_existing_plan",
                                reason="User feedback refers to the existing plan.")
    destination = fields.get("destination") or extract_destination(message)
    output = infer_output_scope(message)
    missing = []
    if output == "itinerary" and _is_missing(fields.get("trip_length")):
        missing.append("trip_length")
    if not destination and not fields.get("interests") and not fields.get("origin_city"):
        missing.append("interests")
    return StrategyDecision(intent="clarify" if missing else "plan_trip",
        next_action="ask_user" if missing else "run_planning", requested_output=output,
        known_fields={"destination": destination} if destination else {}, missing_fields=missing,
        reason="Candidate resolution and destination analysis use the same planner; missing prerequisites are checked before execution.")


@observed("task_router", "router")
def decide_strategy(
    message: str,
    fields: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> StrategyDecision:
    """Choose the next planning workflow, using LLM routing when available."""
    api_key = llm_api_key()
    if api_key and api_key != "your_key_here":
        try:
            return _llm_decide_strategy(message, fields, context=context)
        except Exception as exc:
            decision = _rule_decide_strategy(message, fields, context=context)
            return decision.model_copy(
                update={"router_fallback_reason": f"llm_router_error: {type(exc).__name__}"}
            )
    decision = _rule_decide_strategy(message, fields, context=context)
    return decision.model_copy(update={"router_fallback_reason": f"{llm_provider()}_api_key_missing"})
