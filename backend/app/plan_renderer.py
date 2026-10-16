"""Render compatibility prose from structured activities, without changing business state."""
from backend.app.models import ItineraryDay, TripPlan


def render_day(day: ItineraryDay, language="en") -> ItineraryDay:
    if not day.activities:
        return day
    labels = {"morning": "上午", "afternoon": "下午", "evening": "晚上", "flexible": "灵活安排"} if language.startswith("zh") else {}
    lines = [f"{a.display_time_slot or labels.get(a.time_slot, a.time_slot.title())}: {a.title}. {a.description}".strip() for a in day.activities]
    return day.model_copy(update={"plan": " ".join(lines)})


def render_plan(plan: TripPlan, language=None) -> TripPlan:
    language = language or plan.display_language
    return plan.model_copy(update={"itinerary": [render_day(day, language) for day in plan.itinerary], "display_language": language})
