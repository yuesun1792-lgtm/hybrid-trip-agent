import re
import uuid
from typing import Any

from backend.app.config import llm_rag_enhancements_enabled
from backend.app.conversation_state import get_conversation
from backend.app.fixed_itinerary import build_fixed_destination_plan
from backend.app.models import ChatRequest, ChatResponse, ExecutionTraceStep, TripPlan, TripRequest
from backend.app.planning_engine import plan_from_state, state_from_request
from backend.app.strategy import (
    decide_strategy,
    extract_destination,
    missing_for_fixed_itinerary,
)

INTERESTS = ["food", "history", "art", "nature", "nightlife", "architecture"]
TRANSPORT_MODES = ["train", "flight", "bus"]
PHYSICAL_LEVELS = ["low", "medium", "high"]
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
REQUIRED_FIELDS = [
    "origin_city",
    "date_range",
    "trip_length",
    "budget",
    "interests",
    "transport_preference",
    "physical_level",
]
CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _parse_day_count(value: str) -> int:
    if value.isdigit():
        return int(value)
    return CHINESE_NUMBERS.get(value, 0)


def _has_day_reference(text: str) -> bool:
    lower = text.lower()
    return bool(
        "最后一天" in text
        or re.search(r"(最后[一二两三四五六七八九十]?|第[一二两三四五六七八九十\d]+)\s*天", text)
        or re.search(r"\b(last|final)\s+day\b", lower)
        or re.search(r"\bday\s+\d{1,2}\b", lower)
    )


def extract_fields(message: str, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Small rule-based parser for V6 chat bootstrapping."""
    fields = dict(existing or {})
    text = message.strip()
    lower = text.lower()

    origin = _first_match(
        r"\bfrom\s+([A-Za-z][A-Za-z\s-]{1,40}?)(?=,|\.|\s+in\s+|\s+for\s+|\s+with\s+|\s+budget|\s*$)",
        text,
    )
    if origin:
        fields["origin_city"] = origin.title()

    destination = extract_destination(text)
    if destination:
        fields["destination"] = destination

    length = _first_match(r"\b(\d{1,2})\s*(?:days?|day trip)\b", lower)
    if length:
        fields["trip_length"] = int(length)
    elif not _has_day_reference(text):
        zh_length = _first_match(r"([一二两三四五六七八九十]|\d{1,2})\s*天", text)
        if zh_length:
            parsed = _parse_day_count(zh_length)
            if parsed > 0:
                fields["trip_length"] = parsed

    budget = _first_match(r"(?:budget|under|within|around|about|€|eur)\s*€?\s*(\d{2,5})", lower)
    if budget:
        fields["budget"] = int(budget)

    for mode in TRANSPORT_MODES:
        if re.search(rf"\b{mode}\b", lower):
            fields["transport_preference"] = mode
            break

    for level in PHYSICAL_LEVELS:
        if re.search(rf"\b{level}\b", lower):
            fields["physical_level"] = level
            break

    interests = list(fields.get("interests") or [])
    for interest in INTERESTS:
        if re.search(rf"\b{interest}\b", lower) and interest not in interests:
            interests.append(interest)
    if interests:
        fields["interests"] = interests

    month_match = None
    for month in MONTHS:
        if month in lower:
            month_match = month.title()
            break
    date_span = _first_match(r"\b([A-Z][a-z]+\s+\d{1,2}\s*[-–]\s*\d{1,2})\b", text)
    if date_span:
        fields["date_range"] = date_span
    elif month_match:
        fields["date_range"] = month_match

    if "home_currency" not in fields:
        fields["home_currency"] = "CNY"

    return fields


def missing_fields(fields: dict[str, Any]) -> list[str]:
    missing = []
    for field in REQUIRED_FIELDS:
        value = fields.get(field)
        if value is None or value == "" or value == []:
            missing.append(field)
    return missing


def _question_for_missing(field: str) -> str:
    questions = {
        "origin_city": "Where are you starting from?",
        "date_range": "What month or dates are you thinking of?",
        "trip_length": "How many days do you have?",
        "budget": "What budget should I stay within in EUR?",
        "interests": "What are you interested in: food, history, art, nature, nightlife, or architecture?",
        "transport_preference": "Do you prefer train, flight, or bus?",
        "physical_level": "What physical level should I assume: low, medium, or high?",
        "destination": "Which destination should I build the itinerary for?",
    }
    return questions.get(field, f"Could you share {field}?")


def _trip_request_from_fields(fields: dict[str, Any]) -> TripRequest:
    return TripRequest(
        origin_city=fields["origin_city"],
        date_range=fields["date_range"],
        trip_length=int(fields["trip_length"]),
        budget=int(fields["budget"]),
        interests=list(fields["interests"]),
        transport_preference=fields["transport_preference"],
        physical_level=fields["physical_level"],
        home_currency=fields.get("home_currency", "CNY"),
        taste_profile=fields.get("taste_profile", ""),
    )


def _trace(stage: str, detail: str = "", status: str = "ok") -> ExecutionTraceStep:
    return ExecutionTraceStep(stage=stage, status=status, detail=detail)


def _target_day_from_message(message: str, plan: TripPlan) -> int:
    total_days = len(plan.itinerary)
    lower = message.lower()
    if "最后一天" in message or "last day" in lower or "final day" in lower:
        return total_days

    zh_day = _first_match(r"第\s*([一二两三四五六七八九十]|\d{1,2})\s*天", message)
    if zh_day:
        parsed = _parse_day_count(zh_day)
        if parsed > 0:
            return min(parsed, total_days)

    en_day = _first_match(r"\bday\s+(\d{1,2})\b", lower)
    if en_day:
        return min(int(en_day), total_days)

    return total_days


def _revise_existing_plan(plan: TripPlan, message: str, fields: dict[str, Any]) -> TripPlan:
    target_day = _target_day_from_message(message, plan)
    interests = ", ".join(fields.get("interests") or []) or "your priorities"
    destination = (
        plan.itinerary[target_day - 1].destination
        if 0 < target_day <= len(plan.itinerary)
        else fields.get("destination", "the destination")
    )
    lower = message.lower()

    if any(term in message for term in ["太赶", "太满"]) or any(term in lower for term in ["too rushed", "too busy"]):
        revision_note = "slow the pace down and leave more flexible time"
    elif "太贵" in message or "too expensive" in lower:
        revision_note = "reduce cost pressure by favoring low-cost walks, public spaces, and flexible meals"
    elif "不要" in message or "replace" in lower or "换" in message:
        revision_note = "replace the part you disliked while keeping the rest of the plan stable"
    else:
        revision_note = "make the day lighter and more coherent"

    revised_text = (
        f"Revised after your feedback: {revision_note}. "
        f"Keep Day {target_day} in {destination} focused on {interests}, with one main daytime block, "
        "one optional backup activity, and an easier evening instead of adding another dense route."
    )
    itinerary = [
        day.model_copy(update={"plan": revised_text}) if day.day == target_day else day
        for day in plan.itinerary
    ]
    planner_mode = plan.planner_mode
    if not planner_mode.endswith("_revised"):
        planner_mode = f"{planner_mode}_revised"
    tradeoff_summary = [
        *plan.tradeoff_summary,
        f"Conversation state detected a revision request and updated Day {target_day} without rebuilding the whole plan.",
    ]
    return plan.model_copy(
        update={
            "itinerary": itinerary,
            "planner_mode": planner_mode,
            "final_recommendation": f"Updated Day {target_day} of the {destination} itinerary.",
            "explanation": (
                f"I used the previous plan from this conversation and revised Day {target_day}, "
                "instead of treating your feedback as a new trip request."
            ),
            "tradeoff_summary": tradeoff_summary,
        }
    )


def handle_chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid.uuid4())
    conversation = get_conversation(conversation_id)
    llm_usage = []
    base_fields = {**conversation.extracted_fields, **request.extracted_fields}
    fields = extract_fields(request.message, base_fields)
    conversation.messages.append({"role": "user", "content": request.message})
    trace = [
        _trace(
            "field_extraction",
            f"Extracted fields: {', '.join(sorted(fields.keys())) or 'none'}",
        )
    ]
    decision = decide_strategy(request.message, fields, context=conversation.context())
    if decision.llm_usage:
        llm_usage.append(decision.llm_usage)
    fields.update({key: value for key, value in decision.known_fields.items() if value not in (None, "", [])})
    trace.append(
        _trace(
            "task_router",
            " ".join(
                [
                    f"{decision.router_mode} router selected intent={decision.intent},",
                    f"next_action={decision.next_action}.",
                    decision.reason,
                    f"Fallback reason: {decision.router_fallback_reason}."
                    if decision.router_fallback_reason
                    else "",
                ]
            ).strip(),
            "fallback" if decision.router_mode == "rule" else "ok",
        )
    )

    if decision.next_action == "revise_existing_plan":
        if conversation.last_plan is None:
            trace.append(_trace("revision", "No previous plan found for this conversation.", "fallback"))
            return ChatResponse(
                conversation_id=conversation_id,
                assistant_message="I can revise a plan after I have built one. What trip should we plan first?",
                extracted_fields=fields,
                missing_fields=[],
                trace=trace,
                llm_usage=llm_usage,
            )
        rejected_destination = extract_destination(request.message)
        if ("不要" in request.message or "not " in request.message.lower()) and rejected_destination:
            if rejected_destination not in conversation.rejected_destinations:
                conversation.rejected_destinations.append(rejected_destination)
            trace.append(_trace("state_update", f"Marked rejected destination: {rejected_destination}"))
        plan = _revise_existing_plan(conversation.last_plan, request.message, fields)
        conversation.last_plan = plan
        conversation.extracted_fields = fields
        conversation.current_stage = "plan_revised"
        assistant_message = f"I updated the existing plan: {plan.final_recommendation}"
        conversation.messages.append({"role": "assistant", "content": assistant_message})
        trace.append(
            _trace(
                "revision",
                "Reused conversation last_plan and updated only the referenced itinerary day.",
            )
        )
        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_message,
            extracted_fields=fields,
            missing_fields=[],
            trace=trace,
            llm_usage=llm_usage,
            plan=plan,
        )

    if decision.next_action == "run_fixed_itinerary":
        missing = missing_for_fixed_itinerary(fields)
        if missing:
            trace.append(_trace("clarification", f"Missing fixed-itinerary field: {missing[0]}"))
            conversation.extracted_fields = fields
            conversation.current_stage = "clarifying_fixed_itinerary"
            assistant_message = _question_for_missing(missing[0])
            conversation.messages.append({"role": "assistant", "content": assistant_message})
            return ChatResponse(
                conversation_id=conversation_id,
                assistant_message=assistant_message,
                extracted_fields=fields,
                missing_fields=missing,
                trace=trace,
                llm_usage=llm_usage,
            )
        trace.append(
            _trace(
                "fixed_itinerary_workflow",
                f"Building itinerary for {fields.get('destination')} using {len(decision.queries)} retrieval queries.",
            )
        )
        result = build_fixed_destination_plan(fields, queries=decision.queries)
        if result.llm_usage:
            llm_usage.append(result.llm_usage)
        plan = result.plan
        if result.llm_used:
            trace.append(
                _trace(
                    "llm_itinerary_writer",
                    f"LLM generated {len(plan.itinerary)} day plans from {result.evidence_count} retrieved chunks.",
                )
            )
        elif result.fallback_reason:
            trace.append(
                _trace(
                    "llm_itinerary_writer",
                    f"Skipped or failed; using rule itinerary builder. {result.fallback_reason}. Retrieved chunks={result.evidence_count}.",
                    "fallback",
                )
            )
        trace.append(
            _trace(
                "plan_builder",
                f"Returned planner_mode={plan.planner_mode}; evidence-backed citations={len(plan.ranked_destinations[0].citations) if plan.ranked_destinations else 0}.",
                "fallback" if plan.fallback_reason else "ok",
            )
        )
        conversation.last_plan = plan
        conversation.extracted_fields = fields
        conversation.current_stage = "fixed_itinerary_built"
        assistant_message = (
            f"I built a fixed-destination itinerary: {plan.final_recommendation} "
            "I skipped destination ranking because you already chose the city."
        )
        conversation.messages.append({"role": "assistant", "content": assistant_message})
        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_message,
            extracted_fields=fields,
            missing_fields=[],
            trace=trace,
            llm_usage=llm_usage,
            plan=plan,
        )

    if decision.next_action == "ask_user":
        missing = decision.missing_fields or missing_fields(fields)
        if not missing:
            missing = ["interests"]
        trace.append(_trace("clarification", f"Missing field: {missing[0]}"))
        conversation.extracted_fields = fields
        conversation.current_stage = "clarifying"
        assistant_message = _question_for_missing(missing[0])
        conversation.messages.append({"role": "assistant", "content": assistant_message})
        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_message,
            extracted_fields=fields,
            missing_fields=missing,
            trace=trace,
            llm_usage=llm_usage,
        )

    missing = missing_fields(fields)
    if missing:
        trace.append(_trace("clarification", f"Missing recommendation field: {missing[0]}"))
        conversation.extracted_fields = fields
        conversation.current_stage = "clarifying_recommendation"
        assistant_message = _question_for_missing(missing[0])
        conversation.messages.append({"role": "assistant", "content": assistant_message})
        return ChatResponse(
            conversation_id=conversation_id,
            assistant_message=assistant_message,
            extracted_fields=fields,
            missing_fields=missing,
            trace=trace,
            llm_usage=llm_usage,
        )

    trip_request = _trip_request_from_fields(fields)
    state = state_from_request(trip_request, conversation_id=conversation_id)
    trace.append(
        _trace(
            "destination_recommendation_workflow",
            (
                "Calling RAG-ranked planning engine: discover candidates -> rank -> build TripPlan. "
                f"LLM RAG enhancements enabled={llm_rag_enhancements_enabled()}."
            ),
        )
    )
    plan = plan_from_state(state)
    trace.append(
        _trace(
            "plan_builder",
            f"Returned planner_mode={plan.planner_mode}; fallback_reason={plan.fallback_reason or 'none'}.",
            "fallback" if plan.fallback_reason else "ok",
        )
    )
    if plan.planner_mode == "rag_ranked_llm":
        trace.append(
            _trace(
                "llm_rag_enhancements",
                "LLM participated in research control, candidate reranking, or candidate copy generation.",
            )
        )
    conversation.last_plan = plan
    conversation.extracted_fields = fields
    conversation.current_stage = "recommendation_built"
    assistant_message = (
        f"I found a plan: {plan.final_recommendation} "
        "I used your constraints to rank candidates and kept the tradeoffs visible."
    )
    conversation.messages.append({"role": "assistant", "content": assistant_message})
    return ChatResponse(
        conversation_id=conversation_id,
        assistant_message=assistant_message,
        extracted_fields=fields,
        missing_fields=[],
        trace=trace,
        llm_usage=llm_usage,
        state=state,
        plan=plan,
    )
