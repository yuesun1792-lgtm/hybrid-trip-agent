import re
import json
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from backend.app.config import ANTHROPIC_MODEL
from backend.app.costs import estimate_trip_cost, load_daily_costs
from backend.app.knowledge import search_travel_knowledge
from backend.app.llm_usage import usage_from_response
from backend.app.models import Citation, DestinationSuggestion, EvidenceChunk, ItineraryDay, LLMUsage, TripPlan
from backend.app.tools import gather_tool_insights


load_dotenv()


COUNTRY_BY_DESTINATION = {
    "Lisbon": "Portugal",
    "Porto": "Portugal",
    "Algarve": "Portugal",
    "Barcelona": "Spain",
    "Seville": "Spain",
    "Paris": "France",
    "Nice": "France",
    "Chamonix": "France",
    "Provence": "France",
    "Rome": "Italy",
    "Venice": "Italy",
    "Florence": "Italy",
    "Cinque Terre": "Italy",
    "Athens": "Greece",
    "Santorini": "Greece",
    "Prague": "Czechia",
    "Vienna": "Austria",
    "Ljubljana": "Slovenia",
    "Salzburg": "Austria",
    "Hallstatt": "Austria",
    "Innsbruck": "Austria",
    "Amsterdam": "Netherlands",
}


class FixedItineraryCopy(BaseModel):
    days: list[str] = Field(min_length=1)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    explanation: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class FixedItineraryResult(BaseModel):
    plan: TripPlan
    evidence_count: int
    llm_used: bool = False
    fallback_reason: str | None = None
    llm_usage: LLMUsage | None = None


LOW_VALUE_SNIPPET_TERMS = [
    "* * *",
    "click on the name",
    "detailed information can be found",
    "fnac",
    "if you haven't planned",
    "individual district page",
    "jump lengthy queues",
    "thumb|",
    "require reservations",
    "requires reservations",
    "try to get a reservation",
    "tourist offices",
    "weeks, if not months",
    "opening hours",
    "last admission",
    "ticket office",
]


def _sentences(body: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", body).strip()
    if not cleaned:
        return []
    return [part.strip(" .") for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def _is_good_sentence(sentence: str) -> bool:
    if len(sentence) < 45:
        return False
    if not sentence[0].isalnum():
        return False
    if sentence[0].islower():
        return False
    lower = sentence.lower()
    return not any(term in lower for term in LOW_VALUE_SNIPPET_TERMS)


def _trim_sentence(sentence: str, limit: int = 180) -> str:
    if len(sentence) <= limit:
        return sentence
    trimmed = sentence[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{trimmed}..."


def _short_evidence(chunk_text: str) -> str:
    body = chunk_text.split("\n", 1)[-1].strip()
    sentences = _sentences(body)
    sentence = next((item for item in sentences if _is_good_sentence(item)), None)
    if sentence is None:
        fallback = next(
            (
                item
                for item in sentences
                if len(item) >= 45
                and item[0].isalnum()
                and not any(term in item.lower() for term in LOW_VALUE_SNIPPET_TERMS)
            ),
            None,
        )
        sentence = fallback
    return _trim_sentence(sentence) if sentence else "Supported by retrieved travel knowledge"


def _citations(chunks: list[EvidenceChunk], destination: str) -> list[Citation]:
    citations = []
    seen = set()
    preferred_sections = {"see", "do", "eat", "drink", "get around", "understand", "stay safe"}
    preferred = [chunk for chunk in chunks if _section_key(chunk.section) in preferred_sections]
    ordered_chunks = preferred or chunks
    for chunk in ordered_chunks:
        if not chunk.url:
            continue
        key = (chunk.url, chunk.section)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                title=f"{destination} - {chunk.section or 'Wikivoyage'}",
                url=chunk.url,
                section=chunk.section,
                snippet=_short_evidence(chunk.text),
            )
        )
    return citations[:5]


def _collect_evidence(destination: str, queries: list[str], k: int = 6) -> list[EvidenceChunk]:
    chunks = []
    seen = set()
    for query in queries:
        try:
            results = search_travel_knowledge(query, k=k, destination=destination)
        except Exception:
            results = []
        for result in results:
            key = (result.get("text", ""), result.get("section", ""), result.get("url", ""))
            if key in seen:
                continue
            seen.add(key)
            chunks.append(
                EvidenceChunk(
                    text=result.get("text", ""),
                    destination=result.get("destination", destination),
                    section=result.get("section", ""),
                    url=result.get("url", ""),
                    score=result.get("score"),
                )
            )
    chunks.sort(key=lambda item: item.score or 0.0, reverse=True)
    return chunks[:12]


def _evidence_payload(evidence: list[EvidenceChunk]) -> tuple[list[dict[str, str]], dict[str, EvidenceChunk]]:
    payload = []
    by_id = {}
    for index, chunk in enumerate(evidence[:10]):
        evidence_id = f"e{index}"
        by_id[evidence_id] = chunk
        payload.append(
            {
                "id": evidence_id,
                "section": chunk.section,
                "text": _short_evidence(chunk.text),
                "url": chunk.url,
            }
        )
    return payload, by_id


def _json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM itinerary response did not contain JSON")
    return json.loads(text[start : end + 1])


def _llm_fixed_itinerary_copy(
    destination: str,
    trip_length: int,
    interests: list[str],
    evidence: list[EvidenceChunk],
) -> tuple[FixedItineraryCopy, LLMUsage]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("ANTHROPIC_API_KEY is not configured")

    payload, by_id = _evidence_payload(evidence)
    if not payload:
        raise ValueError("No retrieved evidence available for LLM itinerary writing")

    prompt = f"""You write concise travel itineraries using ONLY supplied evidence.
Return ONLY JSON with:
{{
  "days": ["day 1 plan", "..."],
  "pros": ["1-3 short evidence-grounded strengths"],
  "cons": ["1-3 limitations or caveats"],
  "explanation": "one short sentence",
  "evidence_ids": ["e0", "e1"]
}}

Rules:
- Write exactly {trip_length} day plans.
- Each day must be practical and specific, with Morning/Afternoon/Evening.
- Use only facts supported by the evidence. Do not invent opening hours, routes, hotels, prices, or restaurants.
- Mention the user's interests when relevant: {", ".join(interests) or "local highlights"}.
- evidence_ids must be selected from the supplied IDs and must support the plan.

Destination: {destination}
Evidence:
{json.dumps(payload, ensure_ascii=True)}"""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    copy = FixedItineraryCopy.model_validate(_json_object(response.content[0].text.strip()))
    if len(copy.days) != trip_length:
        raise ValueError("LLM itinerary day count does not match trip length")
    allowed_ids = set(by_id)
    if not copy.evidence_ids or not set(copy.evidence_ids).issubset(allowed_ids):
        raise ValueError("LLM itinerary did not cite retrieved evidence IDs")
    return copy, usage_from_response("llm_itinerary_writer", ANTHROPIC_MODEL, response)


def _default_queries(fields: dict, destination: str) -> list[str]:
    interests = " ".join(fields.get("interests") or [])
    trip_length = fields.get("trip_length", "")
    date_range = fields.get("date_range", "")
    return [
        f"{destination} see do eat get around {trip_length} days {interests}",
        f"{destination} itinerary neighborhoods highlights {interests}",
        f"{destination} practical travel {date_range} safety crowds transport",
    ]


def _section_key(section: str) -> str:
    return section.strip().lower()


def _group_points_by_section(evidence: list[EvidenceChunk]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    seen = set()
    for chunk in evidence:
        point = _short_evidence(chunk.text)
        if not point or point == "Supported by retrieved travel knowledge":
            continue
        if point in seen:
            continue
        seen.add(point)
        grouped.setdefault(_section_key(chunk.section), []).append(point)
    return grouped


def _pick_points(grouped: dict[str, list[str]], sections: list[str], count: int) -> list[str]:
    points = []
    for section in sections:
        points.extend(grouped.get(section, []))
    unique = []
    seen = set()
    for point in points:
        if point in seen:
            continue
        seen.add(point)
        unique.append(point)
    return unique[:count]


def _summary_pros(grouped: dict[str, list[str]]) -> list[str]:
    pros = []
    if grouped.get("see") or grouped.get("do") or grouped.get("understand"):
        pros.append("Retrieved guide evidence covers sights, activities, or cultural context for the destination")
    if grouped.get("eat") or grouped.get("drink"):
        pros.append("Retrieved food or nightlife evidence can shape evening plans")
    if grouped.get("get around"):
        pros.append("Retrieved practical guidance can support movement between areas")
    return pros


def _build_day_plans(
    destination: str,
    trip_length: int,
    interests: list[str],
    evidence: list[EvidenceChunk],
) -> list[ItineraryDay]:
    grouped = _group_points_by_section(evidence)
    has_see = bool(grouped.get("see") or grouped.get("do") or grouped.get("understand"))
    has_food = bool(grouped.get("eat") or grouped.get("drink"))
    has_practical = bool(grouped.get("get around") or grouped.get("stay safe"))
    focus = ", ".join(interests) if interests else "local highlights"
    plans = []

    for day in range(1, trip_length + 1):
        if has_see or has_food:
            if day == 1:
                morning = (
                    "start with retrieved See/Do guidance to anchor the trip around "
                    "major sights, museums, or neighborhoods"
                )
                afternoon = "use retrieved activity evidence for a slower cultural walk or local highlight"
            elif day == trip_length:
                morning = "return to the strongest retrieved sight or neighborhood evidence"
                afternoon = "leave flexible time for practical guide notes, transport, and pacing"
            else:
                morning = "combine retrieved See and Do evidence into one focused half-day route"
                afternoon = (
                    "add a second neighborhood, museum, or activity from the guide evidence "
                    "without overloading the day"
                )
            evening = (
                "close with retrieved Eat/Drink guidance"
                if has_food
                else "keep dinner flexible because retrieved food evidence is limited"
            )
            practical = (
                " Check the retrieved practical sections before moving between areas."
                if has_practical
                else ""
            )
            plan_text = (
                f"Morning: {morning}. Afternoon: {afternoon}. "
                f"Evening: {evening}. Keep the day paced around {focus}.{practical}"
            )
        else:
            plan_text = (
                f"Explore {destination} with a focus on {focus}. "
                "The local knowledge index did not return strong destination evidence."
            )

        plans.append(ItineraryDay(day=day, destination=destination, plan=plan_text))
    return plans


def build_fixed_destination_plan(
    fields: dict,
    queries: list[str] | None = None,
) -> FixedItineraryResult:
    """Build an evidence-backed itinerary when the user already chose a destination."""
    destination = str(fields["destination"])
    trip_length = int(fields["trip_length"])
    interests = list(fields.get("interests") or [])
    origin_city = fields.get("origin_city", "")
    date_range = fields.get("date_range", "")

    retrieval_queries = queries or _default_queries(fields, destination)
    evidence = _collect_evidence(destination, retrieval_queries)
    grouped_points = _group_points_by_section(evidence)
    evidence_points = _pick_points(
        grouped_points,
        ["see", "do", "eat", "drink", "understand", "get around"],
        4,
    )
    citations = _citations(evidence, destination)

    day_plans = _build_day_plans(destination, trip_length, interests, evidence)
    llm_used = False
    fallback_reason = None
    llm_copy = None
    llm_usage = None
    try:
        llm_copy, llm_usage = _llm_fixed_itinerary_copy(destination, trip_length, interests, evidence)
        day_plans = [
            ItineraryDay(day=day, destination=destination, plan=plan_text)
            for day, plan_text in enumerate(llm_copy.days, start=1)
        ]
        llm_used = True
    except Exception as exc:
        fallback_reason = f"llm_itinerary_writer_error: {type(exc).__name__}"

    costs = load_daily_costs()
    estimated_cost = estimate_trip_cost(destination, trip_length, costs) or 0
    tool_signals = gather_tool_insights(origin_city, destination, date_range, trip_length)

    pros = (
        (llm_copy.pros if llm_copy and llm_copy.pros else [])
        or _summary_pros(grouped_points)
        or evidence_points[:2]
        or ["Destination was explicitly requested by the user"]
    )
    cons = ["This itinerary is evidence-backed but not connected to live bookings or opening hours"]
    if llm_copy and llm_copy.cons:
        cons = llm_copy.cons + cons
    if not evidence:
        cons.append("The local RAG index returned no strong evidence for this destination")

    suggestion = DestinationSuggestion(
        name=destination,
        country=COUNTRY_BY_DESTINATION.get(destination, "Unknown"),
        score=100 if evidence and llm_used else (85 if evidence else 60),
        score_breakdown={
            "user_specified_destination": 40.0,
            "retrieved_evidence": float(min(len(evidence), 10) * 4),
        },
        pros=pros[:3],
        cons=cons[:3],
        estimated_cost=estimated_cost,
        tool_signals=tool_signals,
        citations=citations,
    )

    tradeoffs = [
        "The task router detected a fixed destination, so destination discovery and ranking were skipped.",
        "RAG evidence is used to shape the itinerary, while costs, weather, and transport remain lightweight tool signals.",
    ]
    if not evidence:
        tradeoffs.append("Because retrieval was sparse, this result should be treated as a fallback itinerary.")

    planner_mode = "fixed_destination_itinerary_llm" if llm_used else "fixed_destination_itinerary_rule"
    plan = TripPlan(
        ranked_destinations=[suggestion],
        itinerary=day_plans,
        risks=[
            "The itinerary uses Wikivoyage-style local evidence, not live opening hours.",
            "Costs, transport, and weather are lightweight estimates unless connected to real APIs.",
            "Verify reservations and timed-entry attractions before booking.",
        ],
        final_recommendation=f"Use this {trip_length}-day itinerary for {destination}.",
        explanation=(
            llm_copy.explanation
            if llm_copy and llm_copy.explanation
            else (
                f"I treated {destination} as a fixed destination and built the plan around retrieved "
                "local guidance instead of comparing it against other cities."
            )
        ),
        planner_mode=planner_mode,
        fallback_reason=fallback_reason,
        tradeoff_summary=tradeoffs,
    )
    return FixedItineraryResult(
        plan=plan,
        evidence_count=len(evidence),
        llm_used=llm_used,
        fallback_reason=fallback_reason,
        llm_usage=llm_usage,
    )
