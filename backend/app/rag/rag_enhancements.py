from __future__ import annotations

from backend.app.runtime.trajectory import observed
from backend.app.llm.llm import generate_text

import json
from typing import Literal

from pydantic import BaseModel, Field

from backend.app.config import research_queries_per_round
from backend.app.domain.models import CandidateDestination, PlanningState
from backend.app.rag.reranker import llm_rerank


class ResearchDecision(BaseModel):
    action: Literal["search", "stop"]
    queries: list[str] = Field(default_factory=list)
    reason: str = ""
    evidence_gaps: list[str] = Field(default_factory=list)


class EvidenceBackedPoint(BaseModel):
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class CandidateCopy(BaseModel):
    name: str
    pro_points: list[EvidenceBackedPoint] = Field(default_factory=list)
    con_points: list[EvidenceBackedPoint] = Field(default_factory=list)

    @property
    def pros(self) -> list[str]:
        return [point.text for point in self.pro_points]

    @property
    def cons(self) -> list[str]:
        return [point.text for point in self.con_points]


def _json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def _ask_json(prompt: str, max_tokens: int = 700) -> dict:
    response = generate_text(prompt, stage="rag_enhancements", max_tokens=max_tokens,
                             prompt_version="rag_enhancements_v1")
    return _json_object(response.text)



@observed("research_controller", "workflow")
def decide_research(
    state: PlanningState,
    searched_queries: list[str],
    round_number: int,
) -> ResearchDecision:
    request = state.request
    candidate_summary = [
        {
            "name": candidate.name,
            "evidence_count": len(candidate.evidence),
            "best_similarity": candidate.score,
            "sections": sorted({chunk.section for chunk in candidate.evidence if chunk.section}),
        }
        for candidate in state.candidate_destinations
    ]
    payload = {
        "request": request.model_dump(),
        "round": round_number,
        "searched_queries": searched_queries,
        "candidates": candidate_summary,
    }
    prompt = f"""You are the research controller for a travel recommendation engine.
Decide whether the local Wikivoyage index needs another search. Search only for destination
facts relevant to the request. Do not propose an itinerary or facts from your own knowledge.
Return only JSON with: action (search or stop), queries (array), reason, evidence_gaps (array).
Generate at most {research_queries_per_round()} short, distinct queries. Stop when there are at
least 3 credible destinations with useful evidence, unless a material user constraint is uncovered.

State:
{json.dumps(payload, ensure_ascii=True)}"""
    decision = ResearchDecision.model_validate(_ask_json(prompt))
    if decision.action == "search":
        cleaned = []
        seen = {query.casefold() for query in searched_queries}
        for query in decision.queries:
            query = " ".join(query.split()).strip()
            if query and query.casefold() not in seen:
                cleaned.append(query)
                seen.add(query.casefold())
        decision.queries = cleaned[: research_queries_per_round()]
        if not decision.queries:
            decision.action = "stop"
    else:
        decision.queries = []
    return decision


@observed("candidate_reranker", "workflow")
def rerank_candidates(
    candidates: list[CandidateDestination],
    state: PlanningState,
) -> list[CandidateDestination]:
    if len(candidates) < 2:
        return candidates
    payload = [
        {
            "destination": candidate.name,
            "section": "combined evidence",
            "text": "\n".join(chunk.text[:300] for chunk in candidate.evidence[:3]),
            "score": candidate.score or 0.0,
        }
        for candidate in candidates
    ]
    query = " ".join(
        [
            state.request.origin_city,
            state.request.date_range,
            *state.request.interests,
            state.request.transport_preference,
            state.request.physical_level,
        ]
    )
    reranked = llm_rerank(
        payload,
        query=query,
        taste_profile=state.request.taste_profile,
        k=len(payload),
        raise_on_error=True,
    )
    ordered_names = [item["destination"] for item in reranked]
    by_name = {candidate.name: candidate for candidate in candidates}
    if set(ordered_names) != set(by_name) or len(ordered_names) != len(candidates):
        raise ValueError("LLM rerank did not preserve the candidate set")
    return [by_name[name] for name in ordered_names]


@observed("candidate_writer", "workflow")
def write_candidate_copy(
    candidates: list[CandidateDestination],
    state: PlanningState,
) -> dict[str, CandidateCopy]:
    evidence_ids: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        evidence_ids[candidate.name] = [
            {"id": f"{candidate.name}:{index}", "text": chunk.text[:500]}
            for index, chunk in enumerate(candidate.evidence[:3])
            if chunk.url
        ]
    prompt = f"""Write concise recommendation copy using only the supplied evidence.
Return only JSON: {{"candidates": [{{"name": exact name,
"pro_points": [{{"text": string, "evidence_ids": [ids]}}],
"con_points": [{{"text": string, "evidence_ids": [ids]}}]}}]}}.
Give 1-2 pros and 1-2 cons per destination. Every point must cite at least one supplied
evidence ID for that same destination. Do not invent logistics, prices, or safety claims.

Request: {json.dumps(state.request.model_dump(), ensure_ascii=True)}
Evidence: {json.dumps(evidence_ids, ensure_ascii=True)}"""
    result = _ask_json(prompt, max_tokens=1000)
    allowed = {candidate.name for candidate in candidates}
    copies = {}
    for raw in result.get("candidates", []):
        copy = CandidateCopy.model_validate(raw)
        if copy.name in allowed:
            allowed_ids = {item["id"] for item in evidence_ids[copy.name]}
            points = [*copy.pro_points, *copy.con_points]
            if not points or any(
                not set(point.evidence_ids).issubset(allowed_ids)
                for point in points
            ):
                raise ValueError("LLM copy cited evidence outside the retrieved candidate set")
            copies[copy.name] = copy
    if set(copies) != allowed:
        raise ValueError("LLM copy did not cover the ranked candidate set")
    return copies
