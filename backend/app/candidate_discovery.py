from collections import defaultdict

from backend.app.knowledge import search_travel_knowledge
from backend.app.models import CandidateDestination, EvidenceChunk, PlanningState


def build_discovery_queries(state: PlanningState) -> list[str]:
    """Build compact retrieval queries from the normalized planning state."""
    request = state.request
    interests = ", ".join(request.interests) if request.interests else "local highlights"
    queries = [
        (
            f"{request.origin_city} {request.transport_preference} accessible "
            f"{interests} {request.date_range} {request.trip_length} days"
        ),
        (
            f"Europe {interests} {request.physical_level} physical level "
            f"{request.date_range} itinerary"
        ),
    ]
    if request.taste_profile:
        queries.append(f"{request.taste_profile} {interests} Europe travel")
    return queries


def discover_candidate_destinations(
    state: PlanningState,
    queries: list[str] | None = None,
    per_query_k: int = 6,
    max_candidates: int = 8,
) -> PlanningState:
    """Populate PlanningState with RAG-discovered destinations and evidence chunks."""
    grouped: dict[str, list[EvidenceChunk]] = defaultdict(list)

    for candidate in state.candidate_destinations:
        grouped[candidate.name].extend(candidate.evidence)

    for query in queries or build_discovery_queries(state):
        for result in search_travel_knowledge(query, k=per_query_k):
            destination = result.get("destination", "").strip()
            if not destination:
                continue
            grouped[destination].append(
                EvidenceChunk(
                    text=result.get("text", ""),
                    destination=destination,
                    section=result.get("section", ""),
                    url=result.get("url", ""),
                    score=result.get("score"),
                )
            )

    candidates = []
    for destination, evidence in grouped.items():
        if destination in state.rejected_destinations:
            continue
        unique_evidence = []
        seen = set()
        for chunk in sorted(evidence, key=lambda item: item.score or 0.0, reverse=True):
            key = (chunk.text, chunk.section, chunk.url)
            if key in seen:
                continue
            seen.add(key)
            unique_evidence.append(chunk)
        best_score = max((chunk.score or 0.0 for chunk in unique_evidence), default=0.0)
        candidates.append(
            CandidateDestination(
                name=destination,
                source="rag",
                score=best_score,
                rationale="Discovered from travel knowledge matching the request.",
                evidence=unique_evidence[:3],
            )
        )

    candidates.sort(key=lambda candidate: candidate.score or 0.0, reverse=True)
    selected = candidates[:max_candidates]
    all_evidence = [chunk for candidate in selected for chunk in candidate.evidence]

    return state.model_copy(
        update={
            "stage": "candidates_discovered" if selected else "candidate_discovery_empty",
            "candidate_destinations": selected,
            "evidence": all_evidence,
        }
    )
