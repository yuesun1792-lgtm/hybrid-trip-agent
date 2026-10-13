# EuroTrip Agent - V5 Spec: State-Based Planning And Candidate Discovery

## Goal

Move the project from a single-shot form planner toward an interactive travel bot without
breaking the current Streamlit form. The form remains a useful debug and evaluation surface,
but the backend planning engine should operate on normalized planning state rather than UI
fields directly.

## Core Decision

`TripRequest` is now an internal normalized request schema, not the long-term product input
format. Form input and future chat input should both become `PlanningState`.

```text
Streamlit form -> TripRequest -> PlanningState -> planning engine -> TripPlan
Chat message   -> interpreted constraints -> PlanningState -> planning engine -> TripPlan or follow-up
```

## V5.0 Scope: State-Based Candidate Discovery

- Add `PlanningState`, `CandidateDestination`, and `EvidenceChunk` models.
- Add a thin `planning_engine.py` boundary.
- Route deterministic fallback through `PlanningState`.
- Keep `/plan` and the Streamlit form unchanged from the user's point of view.
- Use RAG to discover candidate destinations from `knowledge_index.npz`.
- Store discovered destinations and supporting evidence in `PlanningState`.
- Keep legacy `destination_notes.json` as fallback data only.
- Expose a debug endpoint for candidate discovery without changing the final `/plan` response.

## V5.1 Scope: Constraint-Aware Ranking

- Add constraint-aware ranking over candidates:
  - hard constraints: origin, trip length, budget, transport preference
  - soft preferences: interests, physical level, taste profile
  - context: date range / season
- Return score breakdowns and evidence-backed rationales.
- Expose a debug endpoint for ranked candidates before changing final `/plan` behavior.

## V5.2 Scope: Ranked Candidate TripPlan

- Build final `TripPlan` from RAG-discovered, constraint-ranked candidates.
- Keep legacy deterministic planning as a fallback only when RAG planning cannot produce candidates.
- Preserve debug endpoints for candidate discovery and ranking.

## V5.3 Scope: Evidence And Tradeoffs

- Add anti-repetition and tradeoff explanations:
  - why this destination
  - why not the obvious alternative
  - which user constraint drove the decision
- Preserve filtered candidates and the reason they were filtered.
- Return score breakdowns in final `TripPlan` destination cards.
- Surface tradeoff summaries and filtered candidates in the UI.

## Out Of Scope For V5.0

- Chat endpoint
- Persistent conversation storage
- Multi-agent orchestration
- MCP integration
- Real train/weather/maps APIs

## Invariants

- The Streamlit form remains available as a structured test harness.
- The planning engine should not depend on Streamlit-specific assumptions.
- Existing `/plan` clients should continue to work.
- Fallback mode must remain visible through `planner_mode` and `fallback_reason`.

## V6 Scope: Chat Planning Interface

- Add a stateless `/chat` endpoint.
- Parse natural-language messages into accumulated `extracted_fields`.
- Ask follow-up questions for missing fields.
- When required fields are complete, normalize into `TripRequest` and call the same planning engine.
- Keep Streamlit form unchanged; chat UI is a later layer.
