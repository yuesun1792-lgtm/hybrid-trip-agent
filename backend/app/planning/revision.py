"""Parse revisions, calculate dependencies and merge only affected activities."""
import json
import re
from datetime import timedelta

from backend.app.config import llm_api_key
from backend.app.llm.llm import generate_text
from backend.app.domain.models import TripPlan
from backend.app.domain.planning_models import AffectedScope, RevisionChange, RevisionRequest, TripConstraints
from backend.app.planning.plan_renderer import render_day
from backend.app.runtime.trajectory import observed


class RevisionNeedsInput(ValueError):
    def __init__(self, message, field="revision_scope"):
        self.field = field
        super().__init__(message)


_DIGITS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def rule_revision(message, base_plan, constraints_updates=None):
    lower = message.lower()
    numbers = []
    for match in re.finditer(r"(?:第\s*([一二两三四五六七八九十\d]+)\s*天|\bday\s+(\d+)\b)", message, re.I):
        raw = match.group(1) or match.group(2)
        numbers.append(int(raw) if raw.isdigit() else _DIGITS.get(raw, 0))
    if "最后一天" in message or "last day" in lower or "final day" in lower:
        numbers = [len(base_plan.itinerary)]
    all_days = any(word in lower for word in ("整个", "全程", "all days", "whole trip", "entire trip"))
    relax = any(word in lower for word in ("太赶", "太满", "轻松", "too rushed", "too busy", "slower"))
    if not numbers and (all_days or relax):
        numbers = [day.day for day in base_plan.itinerary]
    scope_text = re.sub(r"保留晚餐|晚餐保留|keep dinner|preserve dinner", "", lower)
    slot = next((slot for slot, terms in {"morning": ["上午", "早上", "morning"], "afternoon": ["下午", "afternoon"],
                                         "evening": ["晚上", "晚餐", "evening", "dinner"]}.items()
                 if any(term in scope_text for term in terms)), None)
    if ("下午" in message or "afternoon" in lower):
        slot = "afternoon"  # "afternoon ... preserve dinner" targets afternoon, not dinner.
    preserve = []
    if any(term in lower for term in ("保留晚餐", "晚餐保留", "keep dinner", "preserve dinner")):
        preserve = [a.activity_id for day in base_plan.itinerary if day.day in numbers for a in day.activities if a.time_slot == "evening"]
    if any(term in lower for term in ("不要保留", "不保留", "不用保留", "don't keep", "do not preserve")):
        preserve = []
    category = next((value for value, terms in {"shopping": ["购物", "shopping"], "food": ["美食", "food", "restaurant"],
                     "nature": ["自然", "nature"], "art": ["艺术", "art"], "culture": ["文化", "culture"]}.items()
                     if any(term in lower for term in terms)), None)
    changes = []
    if numbers:
        remove = any(term in lower for term in ("删除", "移除", "取消", "remove", "delete", "drop", "不要"))
        replace = any(term in lower for term in ("换", "改成", "replace", "instead"))
        add = any(term in lower for term in ("增加", "添加", "add ", "insert "))
        action = "add" if add else ("relax" if relax else ("remove" if remove and not replace else "replace"))
        changes.append(RevisionChange(action=action, day_numbers=list(dict.fromkeys(numbers)),
                                      time_slot=slot, requirements=message, category=category))
    return RevisionRequest(base_plan_id=base_plan.plan_id, base_version=base_plan.version,
                           changes=changes, preserve_activity_ids=preserve, constraints_updates=constraints_updates or {})


@observed("revision_interpretation", "interpretation")
def parse_revision(message: str, base_plan: TripPlan, supplied: RevisionRequest | None = None,
                   constraints_updates: dict | None = None) -> RevisionRequest:
    if supplied:
        return supplied
    rule = rule_revision(message, base_plan, constraints_updates)
    # Explicit day/slot and constraint corrections need no additional model call.
    if rule.changes or rule.constraints_updates:
        return rule
    if llm_api_key():
        ids = [a.activity_id for d in base_plan.itinerary for a in d.activities]
        prompt = f"""Interpret a travel-plan revision. Return only JSON with changes and preserve_activity_ids.
Each change has action (replace/remove/relax/add), activity_ids, day_numbers, time_slot (morning/afternoon/evening/flexible or null), requirements and category (string or null).
Use only existing IDs and day numbers. Do not choose a day arbitrarily when the request is ambiguous: return changes=[].
Do not add travel facts or update constraints. User: {message}
Existing days: {json.dumps([{'day': d.day, 'activities': [a.model_dump() for a in d.activities]} for d in base_plan.itinerary], ensure_ascii=False)}"""
        try:
            response = generate_text(prompt, stage="revision_interpreter", max_tokens=900, prompt_version="revision_v1")
            raw = json.loads(response.text)
            rule = RevisionRequest(base_plan_id=base_plan.plan_id, base_version=base_plan.version,
                                   changes=raw.get("changes", []), preserve_activity_ids=raw.get("preserve_activity_ids", []))
        except Exception:
            pass
    if not rule.changes and not rule.constraints_updates:
        raise RevisionNeedsInput("Which day or activity should I change?")
    return rule


@observed("revision_scope", "validation")
def compute_affected_scope(base: TripPlan, revision: RevisionRequest) -> AffectedScope:
    days = {day.day: day for day in base.itinerary}
    activities = {a.activity_id: (day.day, a) for day in base.itinerary for a in day.activities}
    if set(revision.preserve_activity_ids) - set(activities):
        raise RevisionNeedsInput("Preserved activity IDs are not in the current plan.")
    new_destination = revision.constraints_updates.get("destination")
    if new_destination:
        for identity in revision.preserve_activity_ids:
            number = activities[identity][0]
            if days[number].destination != new_destination:
                raise RevisionNeedsInput("Preserved activities belong to another destination. Remove preservation or keep the original city.")
    targets = set()
    affected_days = set()
    for change in revision.changes:
        if set(change.activity_ids) - set(activities) or set(change.day_numbers) - set(days):
            raise RevisionNeedsInput("The requested day or activity does not exist in this plan.")
        if change.action == "add":
            if not change.day_numbers:
                raise RevisionNeedsInput("Choose a day for the new activity.")
            affected_days.update(change.day_numbers)
            continue
        for identity in change.activity_ids:
            targets.add(identity)
            affected_days.add(activities[identity][0])
        for number in change.day_numbers:
            affected_days.add(number)
            targets.update(a.activity_id for a in days[number].activities if change.time_slot is None or a.time_slot == change.time_slot)
    full = bool(set(revision.constraints_updates) & {"destination", "trip_length", "start_date", "date_range", "origin_city", "transport_preference", "interests", "physical_level", "taste_profile"})
    if full:
        affected_days = set(days)
        targets = set(activities)
    targets -= set(revision.preserve_activity_ids)
    if not targets and revision.changes and not revision.constraints_updates and not any(c.action == "add" for c in revision.changes):
        raise RevisionNeedsInput("No activity matches this scope; choose an existing activity or time slot.")
    return AffectedScope(day_numbers=sorted(affected_days), activity_ids=sorted(targets),
                         preserve_activity_ids=revision.preserve_activity_ids, full_replan=full,
                         reasons=["Recheck adjacent connections and daily cost after activity changes."] +
                                 (["Constraint changes require reanalysis."] if revision.constraints_updates else []))


@observed("revision_merge", "merge")
def merge_revision(base: TripPlan, replacements: list, revision: RevisionRequest, scope: AffectedScope,
                   constraints: TripConstraints) -> list:
    drafted = {day.day: day for day in replacements}
    preserved = set(scope.preserve_activity_ids)
    targets = set(scope.activity_ids)
    output = []
    for original in base.itinerary:
        if scope.full_replan:
            if original.day > (constraints.trip_length or len(base.itinerary)):
                if any(a.activity_id in preserved for a in original.activities):
                    raise RevisionNeedsInput("Cannot shorten a trip while preserving activities on removed days.")
                continue
            replacement = drafted.get(original.day)
            if not replacement:
                raise RevisionNeedsInput("A replacement day is missing.")
            kept = [a for a in original.activities if a.activity_id in preserved]
            used_slots = {a.time_slot for a in kept}
            activities = [a for a in replacement.activities if a.time_slot not in used_slots] + kept
            activities.sort(key=lambda a: {"morning": 0, "afternoon": 1, "evening": 2, "flexible": 3}[a.time_slot])
            output.append(render_day(original.model_copy(update={"destination": replacement.destination, "date": replacement.date, "activities": activities})))
            continue
        if original.day not in scope.day_numbers:
            output.append(original.model_copy(deep=True))
            continue
        operations = [c for c in revision.changes if original.day in c.day_numbers or any(a.activity_id in c.activity_ids for a in original.activities)]
        relax_ids = set()
        for change in operations:
            affected = [a for a in original.activities if a.activity_id in targets and
                        (not change.activity_ids or a.activity_id in change.activity_ids) and
                        (change.time_slot is None or a.time_slot == change.time_slot)]
            if change.action == "relax" and len(affected) > 1:
                relax_ids.update(a.activity_id for a in affected[1:])
        incoming = drafted.get(original.day)
        remove_ids = set(relax_ids)
        replace_ids = set()
        slot_scopes = set()
        for change in operations:
            matched = [a for a in original.activities if a.activity_id in targets and
                       (not change.activity_ids or a.activity_id in change.activity_ids) and
                       (change.time_slot is None or a.time_slot == change.time_slot)]
            if change.action == "remove":
                remove_ids.update(a.activity_id for a in matched)
            elif change.action == "replace":
                replace_ids.update(a.activity_id for a in matched)
                if change.day_numbers:
                    slot_scopes.update(a.time_slot for a in matched)
        new_activities = [a.model_copy(deep=True) for a in original.activities
                          if a.activity_id not in remove_ids | replace_ids]
        if replace_ids:
            if incoming is None:
                raise RevisionNeedsInput("Replacement activities are missing.")
            needed_slots = {a.time_slot for a in original.activities if a.activity_id in replace_ids}
            for slot in needed_slots:
                choices = [a for a in incoming.activities if a.time_slot == slot]
                if not choices:
                    choices = [a for a in incoming.activities if a.time_slot == "flexible"]
                if not choices:
                    raise RevisionNeedsInput("Replacement did not provide an activity for the requested time slot.")
                count = sum(a.activity_id in replace_ids and a.time_slot == slot for a in original.activities)
                chosen = choices if slot in slot_scopes else choices[:count]
                for activity in chosen:
                    if not any(a.activity_id == activity.activity_id for a in new_activities):
                        new_activities.append(activity.model_copy(deep=True))
            new_activities.sort(key=lambda a: {"morning": 0, "afternoon": 1, "evening": 2, "flexible": 3}[a.time_slot])
        for change in operations:
            if change.action == "add":
                if incoming is None:
                    raise RevisionNeedsInput("New activities are missing.")
                slots = {change.time_slot} if change.time_slot else {"flexible"}
                for activity in incoming.activities:
                    if activity.time_slot in slots and not any(a.activity_id == activity.activity_id for a in new_activities):
                        new_activities.append(activity.model_copy(deep=True))
        new_activities.sort(key=lambda a: {"morning": 0, "afternoon": 1, "evening": 2, "flexible": 3}[a.time_slot])
        if not new_activities:
            raise RevisionNeedsInput("This change would remove every activity from a day. Provide a replacement.")
        output.append(render_day(original.model_copy(update={"activities": new_activities})))
    if scope.full_replan:
        output.extend(drafted[number] for number in sorted(drafted) if number > len(base.itinerary))
    return output
