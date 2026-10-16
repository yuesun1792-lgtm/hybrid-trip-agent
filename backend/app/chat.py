from backend.app.task_builder import task_from_fields, PlanningNeedsInput
from backend.app.unified_planner import execute_task
from backend.app.revision import RevisionNeedsInput
from backend.app.plan_store import PlanVersionConflict
from backend.app.execution import RUN_CONTEXT, RunContext
from backend.app.language import choose_language, render_language
from backend.app.field_validation import normalize_trip_fields
from backend.app.trajectory import Trajectory, _CURRENT, observed, record_trace
import re
import uuid
from typing import Any

from backend.app.conversation_state import get_conversation
from backend.app.models import ChatRequest, ChatResponse, ExecutionTraceStep, TripRequest, LLMUsage, StrategyDecision
from backend.app.strategy import (
    decide_strategy,
    extract_destination,
    infer_output_scope,
)

INTERESTS = ["food", "history", "art", "culture", "fashion", "nature", "nightlife", "architecture"]
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


@observed("field_extraction", "parser")
def extract_fields(message: str, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Small rule-based parser for V6 chat bootstrapping."""
    fields = normalize_trip_fields(existing or {})
    text = message.strip()
    lower = text.lower()

    origin = _first_match(
        r"\bfrom\s+([A-Za-z][A-Za-z\s-]{1,40}?)(?=,|\.|\s+in\s+|\s+for\s+|\s+with\s+|\s+budget|\s*$)",
        text,
    )
    if origin:
        fields["origin_city"] = origin.title()
    zh_origin = _first_match(r"从\s*([^，,。;\n]+?)\s*(?:出发|前往|到|去)", text)
    if zh_origin:
        from backend.app.destination_names import canonical_destination
        fields["origin_city"] = canonical_destination(zh_origin)

    destination = extract_destination(text)
    if destination:
        fields["destination"] = destination

    length = _first_match(r"\b(\d{1,2})\s*(?:days?|day trip)\b", lower)
    if length:
        fields["trip_length"] = int(length)
    elif not _has_day_reference(text):
        duration_text = re.sub(r"\d{1,2}\s*月\s*\d{1,2}\s*日", "", text)
        zh_length = _first_match(r"([一二两三四五六七八九十]|\d{1,2})\s*(?:天|日(?:游)?)", duration_text)
        if zh_length:
            parsed = _parse_day_count(zh_length)
            if parsed > 0:
                fields["trip_length"] = parsed

    budget = _first_match(r"(?:budget|under|within|around|about|€|eur)\s*€?\s*(\d{2,5})", lower)
    zh_budget = _first_match(r"预算\s*(\d{2,5})\s*(?:欧元|EUR|€)", text)
    if budget or zh_budget:
        fields["budget"] = int(budget or zh_budget)

    for mode in TRANSPORT_MODES:
        if re.search(rf"\b{mode}\b", lower):
            fields["transport_preference"] = mode
            break

    for label, mode in {"火车": "train", "飞机": "flight", "巴士": "bus"}.items():
        if label in text:
            fields["transport_preference"] = mode
    for label, level in {"轻松": "low", "适中": "medium", "紧凑": "high"}.items():
        if label in text:
            fields["physical_level"] = level

    for level in PHYSICAL_LEVELS:
        if re.search(rf"\b{level}\b", lower):
            fields["physical_level"] = level
            break

    interests = list(fields.get("interests") or [])
    for interest in INTERESTS:
        if re.search(rf"\b{interest}\b", lower) and interest not in interests:
            interests.append(interest)
    for label, name in {"美食": "food", "文化": "culture", "时尚": "fashion", "艺术": "art", "历史": "history", "自然": "nature", "夜生活": "nightlife", "建筑": "architecture"}.items():
        if label in text and name not in interests:
            interests.append(name)
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
        "revision_scope": "Which day or activity should I change?",
        "existing_plan": "What trip should I plan before making revisions?",
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
    record_trace(stage, detail, status)
    return ExecutionTraceStep(stage=stage, status=status, detail=detail)


def _handle_chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid.uuid4())
    conversation = get_conversation(conversation_id)
    base_fields = {**request.extracted_fields, **conversation.extracted_fields, **conversation.pending_revision_fields}
    fields = extract_fields(request.message, base_fields)
    if conversation.pending_fields and re.fullmatch(r"\s*\d+\s*", request.message):
        pending = conversation.pending_fields[0]
        if pending in {"trip_length", "budget"}:
            fields[pending] = int(request.message.strip())
    fields["response_language"] = conversation.response_language
    conversation.messages.append({"role": "user", "content": request.message})
    trace = [_trace("field_extraction", f"Extracted fields: {', '.join(sorted(fields))}")]
    decision = (StrategyDecision(intent="revise_plan", next_action="revise_existing_plan", router_mode="rule",
                                 reason="Explicit typed revision bypasses model routing.") if request.revision else
                decide_strategy(request.message, fields, context=conversation.context()))
    fields.update({key: value for key, value in decision.known_fields.items() if value not in (None, "", [])})
    fields["response_language"] = conversation.response_language
    trace.append(_trace("task_router", f"{decision.router_mode} router selected intent={decision.intent}. {decision.reason}",
                        "fallback" if decision.router_fallback_reason else "ok"))
    operation = "revise" if request.revision or decision.intent == "revise_plan" or conversation.pending_revision_message else "create"
    message = request.message
    if conversation.pending_revision_message:
        clarification = f"day {request.message.strip()}" if conversation.pending_fields == ["revision_scope"] and request.message.strip().isdigit() else request.message
        message = f"{conversation.pending_revision_message}. Clarification: {clarification}"
    requested_output = infer_output_scope(request.message, request.requested_output, decision.requested_output)
    if operation == "revise" and conversation.last_plan:
        requested_output = conversation.last_plan.requested_output
    if operation == "revise" and any(term in request.message.lower() for term in ("不要", "not ", "avoid ")):
        rejected = extract_destination(request.message)
        if rejected and rejected not in conversation.rejected_destinations:
            conversation.rejected_destinations.append(rejected)
        if rejected and fields.get("destination") == rejected:
            fields["destination"] = None
    task = None
    try:
        task = task_from_fields(fields, operation=operation, base_plan=conversation.last_plan,
            supplied_revision=request.revision, expected_version=request.expected_plan_version,
            message=message, requested_output=requested_output, queries=decision.queries,
            rejected=conversation.rejected_destinations)
        plan = execute_task(task, base_plan=conversation.last_plan)
    except (PlanningNeedsInput, RevisionNeedsInput) as exc:
        missing = exc.fields if isinstance(exc, PlanningNeedsInput) else [exc.field]
        trace.append(_trace("clarification", str(exc)))
        conversation.pending_fields = missing
        if isinstance(exc, PlanningNeedsInput) and exc.__cause__:
            fields = {key: value for key, value in fields.items() if key not in missing}
        if operation == "revise":
            conversation.pending_revision_message = message
            conversation.pending_revision_fields = fields
            # Failed revisions must not overwrite committed constraints.
        else:
            conversation.extracted_fields = fields
        conversation.current_stage = "clarifying"
        assistant = _question_for_missing(missing[0]) if missing[0] not in {"revision_scope", "existing_plan"} else str(exc)
        conversation.messages.append({"role": "assistant", "content": assistant})
        return ChatResponse(conversation_id=conversation_id, assistant_message=assistant,
                            extracted_fields=conversation.extracted_fields, missing_fields=missing, trace=trace,
                            planning_task=task.model_dump(mode="json") if task else None)
    if plan.place_resolution.status != "disabled":
        trace.append(_trace("place_resolution", plan.place_resolution.message,
                            "fallback" if plan.place_resolution.status == "failed" else "ok"))
    trace.append(_trace("plan_builder", f"Returned planner_mode={plan.planner_mode}; version={plan.version}.",
                        "fallback" if plan.fallback_reason else "ok"))
    conversation.last_plan = plan  # Canonical committed plan, never the translated presentation.
    conversation.extracted_fields = {**task.constraints.model_dump(mode="json", exclude_none=True),
                                     "response_language": conversation.response_language}
    conversation.pending_revision_message = None
    conversation.pending_revision_fields = {}
    conversation.pending_fields = []
    conversation.current_stage = "plan_revised" if operation == "revise" else "plan_built"
    assistant = plan.final_recommendation
    conversation.messages.append({"role": "assistant", "content": assistant})
    return ChatResponse(conversation_id=conversation_id, assistant_message=assistant,
        extracted_fields=conversation.extracted_fields, missing_fields=[], trace=trace, plan=plan,
        planning_task=task.model_dump(mode="json"))


def handle_chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid.uuid4())
    conversation = get_conversation(conversation_id)
    # Serialize turns for one conversation; independent conversations remain concurrent.
    with conversation.lock:
        return _run_chat(request.model_copy(update={"conversation_id": conversation_id}), conversation)


def _run_chat(request, conversation):
    conversation_id = conversation.conversation_id
    conversation.response_language = choose_language(request.message, conversation.response_language, request.response_language)
    recorder = Trajectory(conversation_id, conversation.last_run_id, request.message,
                          {**conversation.extracted_fields, **request.extracted_fields})
    recorder.data["language"] = {"response_language": conversation.response_language,
                                 "retrieval_language": "en", "evidence_language": "en"}
    conversation.last_run_id = recorder.data["run_id"]
    token = _CURRENT.set(recorder)
    context = RunContext()
    context_token = RUN_CONTEXT.set(context)
    try:
        if request.expected_plan_version is not None and (not conversation.last_plan or request.expected_plan_version != conversation.last_plan.version):
            raise PlanVersionConflict("Client plan version is outdated; reload the current plan.")
        response = _handle_chat(request)
        response = response.model_copy(update={"response_language": conversation.response_language})
        response = render_language(response, conversation.response_language)
        if conversation.messages and conversation.messages[-1]["role"] == "assistant":
            conversation.messages[-1]["content"] = response.assistant_message
        recorded_usage = [LLMUsage.model_validate(step["llm"]["usage"])
                          for step in recorder.data["steps"] if step.get("llm", {}).get("usage") is not None]
        if recorded_usage:
            response = response.model_copy(update={"llm_usage": recorded_usage})
        recorder.data["budget"] = {"llm_calls": context.llm_calls, "tool_calls": context.tool_calls,
                                    "max_llm_calls": context.max_llm_calls, "max_tool_calls": context.max_tool_calls,
                                    "reserved_output_tokens": context.reserved_output_tokens,
                                    "max_output_tokens": context.max_output_tokens}
        recorder.finish(response=response)
        return response.model_copy(deep=True, update={"run_id": recorder.data["run_id"], "trajectory": recorder.data})
    except Exception as exc:
        recorder.finish(error=exc)
        raise
    finally:
        RUN_CONTEXT.reset(context_token)
        _CURRENT.reset(token)
