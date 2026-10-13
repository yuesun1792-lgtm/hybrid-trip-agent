# CLAUDE.md — EuroTrip Agent

## Project purpose

A learning-oriented project that builds a travel-planning agent one version at a time.
The goal is to understand a real agent pipeline end to end:
user input → an LLM that plans via tool calls → grounded structured output → frontend display.

Build the feature first, review and refine later. Each version is a self-contained,
verifiable chunk of work — not a stream of tiny edits.

## Current status: V5 in progress (V4 complete), V6 chat endpoint already started

**⚠️ Working-tree state (2026-07-27):** V5.0–V5.3 and a first cut of V6 are implemented but
**uncommitted** (`git status` shows large diffs + several untracked files), and
`pytest tests/` currently has **6 failing tests** — mostly stale assertions and mock fixtures
that haven't caught up with the RAG-first fallback behavior, not production regressions. See
`docs/architecture_review.md` for the full trace, a concept-by-concept status table, a spec-vs-code
diff, and a severity-ranked risk list. Read that file before starting new work on this branch —
don't re-derive it.

- **V0** — rule-based planner: ranked destinations + itinerary.
- **V1** — optional LLM explanation layer, with a deterministic fallback.
- **V2** — Claude tool-use loop: the model decides when to call mock tools.
- **V3** — the LLM drives planning itself: it selects destinations, gathers facts via tools,
  and emits a structured `TripPlan` by calling a terminal `submit_trip_plan` tool. The
  deterministic planner is demoted to an optional tool + the no-key fallback.
- **Post-V3 addition** — `currency_tool.py`: a REAL tool (`convert_currency`) hitting the live
  Frankfurter API. Proved that swapping a tool from mock data to a real API does not change the
  agent loop. Wired in via `TripRequest.home_currency` + `estimated_cost_home`.
- **V4 (complete)** — grounded knowledge via RAG: a real 22-destination Wikivoyage corpus
  (`data/wikivoyage_sources.json`) + an in-memory retrieval tool (`search_travel_knowledge`) the
  agent calls. See `SPEC_V4.md`.
- **V5 (in progress, uncommitted)** — state-based planning and RAG candidate discovery, per
  `SPEC_V5.md`:
  - `PlanningState` / `CandidateDestination` / `EvidenceChunk` — normalized planning state
    (`models.py`), independent of the request's origin (form, chat, or debug endpoint).
  - `planning_engine.py` — thin boundary that turns a `TripRequest` into a `PlanningState` and
    drives it through discovery → ranking → plan-building.
  - `candidate_discovery.py` — RAG-only destination discovery (fixed template queries against
    `search_travel_knowledge`, **not** agent-driven). Debug endpoint: `POST /plan/candidates`.
  - `candidate_ranker.py` — constraint-aware scoring (hard: budget/transport; soft:
    interests/taste; context: season) with a visible `score_breakdown`. Debug endpoint:
    `POST /plan/ranked-candidates`.
  - `rag_plan_builder.py` — builds the final `TripPlan` from ranked candidates, including
    `tradeoff_summary`, `filtered_candidates`, and per-destination `citations`.
  - `reranker.py` — LLM cross-encoder reranking (`rerank_travel_knowledge` tool) for the
    `taste_profile` field, used by the LLM-agent path only.
  - `main.py` now tries the RAG-ranked workflow **before** the deterministic planner whenever the
    LLM agent path isn't used (no key, or `ENABLE_ANTHROPIC_AGENT=false`) — this supersedes the
    old "no key → deterministic, no RAG" invariant below; the deterministic planner is now the
    fallback of the fallback.
  - **Known gaps** (see risk list in `docs/architecture_review.md`): `CandidateDestination.country`
    is never populated (always renders "Unknown"); `rejected_destinations` is read but never
    written anywhere (anti-repetition, V5.3, is not actually implemented); the LLM-agent path's
    `submit_trip_plan` schema has no `citations`/`score_breakdown` fields, so only the RAG-ranked
    workflow (not the agent path) returns them.
- **V6 (started early, ahead of spec)** — `chat.py` + `POST /chat` + a Streamlit chat tab already
  exist, even though `SPEC_V5.md` lists the chat endpoint under "Out of scope for V5.0" and
  describes it as a later layer. It's a stateless, regex-based slot-filler (`extract_fields`) that
  builds a `TripRequest` once all required fields are present, then calls the same RAG-ranked
  planning engine as `/plan`. No persistent conversation storage — the client re-sends
  `extracted_fields`/`conversation_id` every turn.

## How V4 fits

Retrieval is just another tool in the V3 loop. The agent calls `search_travel_knowledge` to
ground its recommendations in real travel facts (season, safety, food, hikes, hidden gems)
instead of relying on 4 toy JSON entries or its own memory.

- Corpus is built offline by `scripts/build_index.py` (fetch Wikivoyage → chunk → embed → save).
- The request path only loads and queries the prebuilt index — no fetching/embedding per request.
- In-memory cosine similarity for now, isolated in `knowledge.py` so Chroma is a later drop-in.
- Embeddings are local (`sentence-transformers`), no API key.
- `data/knowledge_index.npz` is gitignored — a fresh clone must run `scripts/build_index.py`
  once before RAG (agent tool or candidate discovery) returns anything; until then
  `search_travel_knowledge` silently returns `[]` and everything falls back to the deterministic
  planner.

## How V5 fits

V5 adds a second, non-agentic way to reach the same `TripPlan` schema: a fixed
discover → rank → build workflow over the same knowledge index, so RAG-grounded plans are
available on the no-key/no-agent path and from `/chat`, not only when the LLM agent runs.
This is retrieval-augmented generation but **not** agentic RAG — the queries are hardcoded
templates (`candidate_discovery.build_discovery_queries`), not model-chosen. Agentic RAG only
exists in `agent.py`'s tool-use loop. Keep this distinction in mind: the two paths currently
produce different levels of output richness (see "Known gaps" above), which is a real
inconsistency to resolve, not a design decision to preserve.

## Invariants that must hold

- `agent.py` loop unchanged — every new capability so far is just another tool.
- Numbers come from tools/data, never invented by the LLM.
- App still works with no API key (RAG-ranked workflow first, deterministic planner as its
  fallback — updated from the old V4 invariant, which assumed no-key meant no-RAG at all).
- No agent frameworks (no LangChain/LangGraph). Hand-rolled loop + Anthropic SDK.
- Response schema and frontend stay stable.

## Constraints (current)

- Mock/curated data plus a small number of real, free, no-key integrations:
  Frankfurter (currency) and Wikivoyage (corpus, fetched at ingest). Embeddings run locally.
- No database yet — `PlanningState` is a schema, not a persistence layer: it's rebuilt from the
  request on every call, and the chat endpoint's "memory" is just the client echoing
  `extracted_fields`/`conversation_id` back. A travel-fund / saved-trips / multi-turn-memory
  layer would need a real store and doesn't exist yet.
- No authentication, no persistence beyond the prebuilt index file.
- No CI configuration in the repo — `pytest` must be run manually to know if the branch is green.

## Architecture

```
eurotrip-agent/
  backend/app/
    config.py             # Path constants
    models.py              # Pydantic schemas: TripRequest, TripPlan, PlanningState,
                            #   CandidateDestination, EvidenceChunk, ChatRequest/ChatResponse
    planner.py             # Deterministic scorer — legacy tool + ultimate fallback (code unchanged)
    tools.py               # Tool implementations + Anthropic schemas (submit_trip_plan, etc.)
    currency_tool.py       # REAL tool: convert_currency via Frankfurter API
    knowledge.py           # V4: load index + search_travel_knowledge (in-memory cosine)
    reranker.py            # V5: LLM cross-encoder rerank (rerank_travel_knowledge tool)
    agent.py               # Claude tool-use loop that DRIVES planning; ends on submit_trip_plan
    planning_engine.py     # V5: thin boundary — TripRequest -> PlanningState -> plan
    candidate_discovery.py # V5: RAG-only (non-agentic) candidate discovery
    candidate_ranker.py    # V5: constraint-aware scoring + score_breakdown
    rag_plan_builder.py    # V5: ranked candidates -> TripPlan (citations, tradeoff_summary)
    costs.py               # V5: destination_costs.json loader + trip-length cost scaling
    chat.py                # V6 (started early): stateless slot-filling -> TripRequest -> plan
    explainer.py           # Template explanation used on the deterministic-fallback path
    prompts.py             # System prompt (LLM agent)
    main.py                # FastAPI: /plan, /plan/candidates, /plan/ranked-candidates, /chat
  frontend/
    streamlit_app.py # Form tab (POST /plan) + Chat tab (POST /chat), tradeoffs/citations UI
  scripts/
    build_index.py    # V4: one-time ingest — fetch Wikivoyage, chunk, embed, save index
    eval_retrieval.py  # V5: manual recall@k harness for the retriever (not wired into pytest/CI)
  data/
    destination_notes.json     # legacy 4-entry mock destinations (deterministic fallback only)
    wikivoyage_sources.json    # V4: curated list of 22 destinations/pages to fetch
    destination_costs.json     # V5: per-day EUR cost for all 22 destinations
    knowledge_index.npz        # V4: built corpus (chunks + embeddings + metadata) — gitignored
    user_preferences.json      # dead file — not read by any code, left over from an earlier iteration
  tests/
  requirements.txt
  .env.example
  docs/architecture_review.md  # full architecture audit — read before starting new work here
```

## Roadmap

- **V5 (current, uncommitted):** state-based planning, RAG candidate discovery/ranking,
  taste-profile personalization, tradeoff explanations. See `SPEC_V5.md` and the "Known gaps"
  above (country field, anti-repetition, agent-path citations) before treating V5 as done.
- **V6 (started early):** the `/chat` endpoint exists; still missing from the original V6 scope:
  persistent conversation storage (currently fully client-echoed) and any handling of a user
  correcting/updating an already-filled field mid-conversation.
- **Later (each its own layer):** map itinerary (geocoding + routing tool + map display,
  Wanderlog-style), travel fund (persistence), multi-trip planning across a semester budget,
  optional swap of the vector store to Chroma, a real plan-quality eval (not just retrieval
  recall), and CI.
