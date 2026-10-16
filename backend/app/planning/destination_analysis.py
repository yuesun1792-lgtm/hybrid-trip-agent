"""One destination-analysis capability, with bounded overview and detailed depths."""
from backend.app.config import research_queries_per_round
from backend.app.planning.costs import estimate_trip_cost, load_daily_costs
from backend.app.domain.destination_names import canonical_destination
from backend.app.domain.models import DestinationAnalysis, EvidenceChunk
from backend.app.domain.planning_models import TripConstraints
from backend.app.tool_calling.tool_executor import default_executor
from backend.app.runtime.trajectory import observed


def english_queries(destination, constraints, depth):
    interests = " ".join(term for term in constraints.interests if term.isascii()) or "local highlights"
    queries = [f"{destination} sights activities food culture {interests}"]
    if depth == "detailed":
        queries += [f"{destination} itinerary neighborhoods museums shopping {interests}",
                    f"{destination} practical transport safety opening reservations"]
    return queries


@observed("destination_analysis", "analysis")
def analyze_destination(destination: str, constraints: TripConstraints, depth: str = "overview",
                        seed_evidence: list[EvidenceChunk] | None = None,
                        queries: list[str] | None = None) -> DestinationAnalysis:
    destination = canonical_destination(destination)
    executor = default_executor()
    chunks = {c.evidence_id: c for c in seed_evidence or [] if canonical_destination(c.destination) == destination}
    # Discovery already provided overview evidence; don't regenerate full itineraries per candidate.
    if depth == "detailed" or not chunks:
        selected_queries = [q for q in (queries or []) if q.strip() and q.isascii()]
        selected_queries = selected_queries or english_queries(destination, constraints, depth)
        for query in list(dict.fromkeys(selected_queries))[:research_queries_per_round() if depth == "detailed" else 1]:
            result = executor.execute("search_travel_knowledge", {"query": query, "k": 8 if depth == "detailed" else 3, "destination": destination})
            for raw in result.data or []:
                if canonical_destination(raw.get("destination", destination)) != destination:
                    continue
                chunk = EvidenceChunk.model_validate({**raw, "destination": destination})
                chunks[chunk.evidence_id] = chunk
    evidence = sorted(chunks.values(), key=lambda c: c.score or 0, reverse=True)[:10 if depth == "detailed" else 3]
    cost = estimate_trip_cost(destination, constraints.trip_length, load_daily_costs()) if constraints.trip_length else None
    text = " ".join(c.text.lower() for c in evidence)
    hits = sum(term.lower() in text for term in constraints.interests)
    breakdown = {"retrieval": round(min(max((c.score or 0 for c in evidence), default=0), 1) * 20, 2),
                 "interests": float(min(hits * 8, 24))}
    taste_terms = [term.strip(".,;:!? ").lower() for term in constraints.taste_profile.split() if len(term) > 3]
    if taste_terms:
        breakdown["taste"] = float(min(sum(term in text for term in taste_terms) * 3, 12))
    if constraints.date_range:
        months = {"january": "winter", "february": "winter", "march": "spring", "april": "spring", "may": "spring", "june": "summer", "july": "summer", "august": "summer", "september": "autumn", "october": "autumn", "november": "autumn", "december": "winter"}
        requested = constraints.date_range.lower()
        season = next((season for month, season in months.items() if month in requested), None)
        breakdown["season"] = 5.0 if season and season in text else 0.0
    gaps = []
    if not evidence:
        gaps.append("No destination-specific evidence was retrieved; activities require verification.")
    if cost is None:
        gaps.append("Destination cost is unknown.")
    if constraints.budget is not None and cost is not None:
        breakdown["budget"] = 25.0 if cost <= constraints.budget else -60.0
    if constraints.origin_city and constraints.transport_preference:
        result = executor.execute("estimate_transport_time", {"origin": constraints.origin_city, "destination": destination})
        if result.data:
            breakdown["transport"] = 20.0 if result.data["mode"] == constraints.transport_preference else -15.0
            gaps.append("Transport fit uses mock estimates, not verified schedules.")
    return DestinationAnalysis(destination=destination, depth=depth, evidence=evidence,
        estimated_cost=cost, score=sum(breakdown.values()), score_breakdown=breakdown,
        activity_themes=sorted({c.section for c in evidence}), gaps=gaps)
