"""Build a full activity itinerary only for selected destinations or affected days."""
import json
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.config import llm_api_key
from backend.app.runtime.execution import BudgetExceeded
from backend.app.llm.llm import generate_text
from backend.app.domain.models import DestinationAnalysis, ItineraryDay
from backend.app.domain.planning_models import Activity, TripConstraints
from backend.app.planning.plan_renderer import render_day
from backend.app.runtime.trajectory import observed


class DraftActivity(BaseModel):
    title: str
    description: str
    category: str
    time_slot: str
    evidence_ids: list[str] = Field(default_factory=list)


class DraftDay(BaseModel):
    day: int
    activities: list[DraftActivity] = Field(min_length=1, max_length=6)


class ItineraryDraft(BaseModel):
    days: list[DraftDay]
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    explanation: str = ""


class BuildResult(BaseModel):
    days: list[ItineraryDay]
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    explanation: str = ""
    generation_mode: Literal["llm", "rule", "mixed"] = "rule"
    llm_used: bool = False
    fallback_reason: str | None = None


def draft_schema(day_numbers: list[int], evidence_ids: list[str], time_slots=None, max_activities=4) -> dict:
    activity = {"type": "object", "properties": {
        "title": {"type": "string"}, "description": {"type": "string"}, "category": {"type": "string"},
        "time_slot": {"type": "string", "enum": time_slots or ["morning", "afternoon", "evening", "flexible"]},
        "evidence_ids": {"type": "array", "items": {"type": "string", "enum": evidence_ids}, "minItems": 1}},
        "required": ["title", "description", "category", "time_slot", "evidence_ids"], "additionalProperties": False}
    day = {"type": "object", "properties": {"day": {"type": "integer", "enum": day_numbers},
        "activities": {"type": "array", "items": activity, "minItems": 1, "maxItems": max_activities}},
        "required": ["day", "activities"], "additionalProperties": False}
    properties = {"days": {"type": "array", "items": day, "minItems": len(day_numbers), "maxItems": len(day_numbers)},
        "pros": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "cons": {"type": "array", "items": {"type": "string"}, "maxItems": 3}, "explanation": {"type": "string"}}
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def fallback_days(analysis, constraints, day_numbers, category=None, requirements="", time_slots=None):
    sections = {chunk.section.lower() for chunk in analysis.evidence}
    categories = [("morning", "culture" if sections & {"see", "do", "understand"} else "general"),
                  ("afternoon", category or "exploration"), ("evening", "food")]
    if time_slots:
        categories = [(slot, category or "general") for slot in time_slots]
    output = []
    for number in day_numbers:
        activities = []
        for slot, kind in categories:
            relevant = [c.evidence_id for c in analysis.evidence if c.section.lower() in
                        ({"eat", "drink"} if kind == "food" else {"see", "do", "understand", "buy"})]
            activities.append(Activity(title=f"{kind.title()} activity to verify in {analysis.destination}",
                description=requirements or "Choose a specific activity after verifying guide evidence and logistics.",
                category=kind, time_slot=slot, evidence_ids=relevant[:2], status="needs_verification"))
        date = constraints.start_date + timedelta(days=number - 1) if constraints.start_date else None
        output.append(render_day(ItineraryDay(day=number, date=date, destination=analysis.destination, activities=activities)))
    return output


@observed("structured_itinerary_builder", "workflow")
def build_itinerary(analysis: DestinationAnalysis, constraints: TripConstraints, response_language: str = "en",
                    day_numbers: list[int] | None = None, requirements: str = "", category: str | None = None,
                    time_slots: list[str] | None = None, activity_limit: int | None = None) -> BuildResult:
    numbers = day_numbers or list(range(1, (constraints.trip_length or 1) + 1))
    if len(numbers) > 4 and llm_api_key() and analysis.evidence:
        batches = [build_itinerary(analysis, constraints, response_language, numbers[i:i + 4],
                                   requirements, category, time_slots, activity_limit) for i in range(0, len(numbers), 4)]
        all_model = all(batch.llm_used for batch in batches)
        some_model = any(batch.llm_used or batch.generation_mode == "mixed" for batch in batches)
        return BuildResult(days=[day for batch in batches for day in batch.days], llm_used=all_model,
            generation_mode="llm" if all_model else "mixed" if some_model else "rule",
            fallback_reason="; ".join(dict.fromkeys(b.fallback_reason for b in batches if b.fallback_reason)) or None,
            pros=list(dict.fromkeys(p for b in batches for p in b.pros))[:3],
            cons=list(dict.fromkeys(c for b in batches for c in b.cons))[:3],
            explanation=" ".join(b.explanation for b in batches))
    if not llm_api_key() or not analysis.evidence:
        reason = "llm_api_key_missing" if not llm_api_key() else "destination_evidence_missing"
        return BuildResult(days=fallback_days(analysis, constraints, numbers, category, requirements, time_slots), fallback_reason=reason)
    maximum = {"low": 3, "medium": 4, "high": 6}.get(constraints.physical_level, 4)
    if activity_limit is not None:
        maximum = min(maximum, max(1, activity_limit))
    evidence = [{"id": c.evidence_id, "destination": c.destination, "section": c.section,
                 "text": c.text[:2200], "url": c.url} for c in analysis.evidence]
    prompt = f"""Build a structured travel itinerary in language {response_language} using ONLY supplied evidence.
Return JSON with days, pros, cons and explanation. days must contain exactly day numbers {numbers} in that order.
Each day has day and activities. Each activity has title, description, category, time_slot and evidence_ids.
Use 1-{maximum} activities per day; favor a coherent morning/afternoon/evening plan over filling every slot.
Every activity must cite its supporting evidence IDs. A valid citation is not proof: avoid unsupported assertions.
Do not invent venues, durations, exact prices, opening hours, train stations or transport times.
Keep associations within evidence intact (a station or travel time for one place cannot be assigned to another).
Dates and numerical estimates will be added by code. Keep logistics uncertain if evidence is insufficient.
Destination: {analysis.destination}
Constraints: {constraints.model_dump_json()}
Scope requirements: {requirements}
Requested replacement category: {category or 'use relevant user interests'}
Allowed time slots for this scope: {time_slots or ['morning', 'afternoon', 'evening', 'flexible']}
If this is a time-slot-scoped replacement, generate only the requested new activities.
Requirements may mention existing or preserved activities: do not regenerate those activities
or move them into a different time slot. A preserved morning visit is not an afternoon replacement.
If a replacement category is specified, every returned activity must use that exact category.
Evidence: {json.dumps(evidence, ensure_ascii=False)}"""
    try:
        response = generate_text(prompt, stage="llm_itinerary_writer", max_tokens=min(3000, 600 + len(numbers) * 600),
            prompt_version="structured_itinerary_v1", response_schema=draft_schema(numbers, [c.evidence_id for c in analysis.evidence], time_slots, maximum))
        draft = ItineraryDraft.model_validate_json(response.text)
        if [day.day for day in draft.days] != numbers:
            raise ValueError("Draft did not preserve requested day numbers")
        allowed = {c.evidence_id for c in analysis.evidence}
        days = []
        for day in draft.days:
            if len(day.activities) > maximum:
                raise ValueError("Activity count exceeds the requested pacing limit")
            activities = []
            for item in day.activities:
                if not item.evidence_ids or not set(item.evidence_ids) <= allowed:
                    raise ValueError("Activity has invalid or missing evidence IDs")
                if time_slots and item.time_slot not in time_slots:
                    raise ValueError("Activity is outside the requested time scope")
                if category and item.category.casefold() != category.casefold():
                    raise ValueError("Activity does not match the requested replacement category")
                activity = Activity(**item.model_dump(), status="proposed")
                activities.append(activity)
            date = constraints.start_date + timedelta(days=day.day - 1) if constraints.start_date else None
            days.append(render_day(ItineraryDay(day=day.day, destination=analysis.destination, date=date, activities=activities)))
        return BuildResult(days=days, pros=draft.pros, cons=draft.cons, explanation=draft.explanation, llm_used=True, generation_mode="llm")
    except (ValueError, BudgetExceeded) as exc:
        return BuildResult(days=fallback_days(analysis, constraints, numbers, category, requirements, time_slots),
                           fallback_reason=f"llm_itinerary_writer_error: {type(exc).__name__}")
    except Exception as exc:
        # Provider/network failures degrade here; errors elsewhere in the engine remain visible.
        return BuildResult(days=fallback_days(analysis, constraints, numbers, category, requirements, time_slots),
                           fallback_reason=f"llm_itinerary_writer_error: {type(exc).__name__}")
