# CLAUDE.md — EuroTrip Agent

## Project purpose

A learning-oriented project that builds a travel-planning agent one version at a time.
The goal is to understand a real agent pipeline end to end:
user input → an LLM that plans via tool calls → grounded structured output → frontend display.

Build the feature first, review and refine later. Each version is a self-contained,
verifiable chunk of work — not a stream of tiny edits.

## Current status: V4 in progress (V3 complete)

- **V0** — rule-based planner: ranked destinations + itinerary.
- **V1** — optional LLM explanation layer, with a deterministic fallback.
- **V2** — Claude tool-use loop: the model decides when to call mock tools.
- **V3** — the LLM drives planning itself: it selects destinations, gathers facts via tools,
  and emits a structured `TripPlan` by calling a terminal `submit_trip_plan` tool. The
  deterministic planner is demoted to an optional tool + the no-key fallback.
- **Post-V3 addition** — `currency_tool.py`: a REAL tool (`convert_currency`) hitting the live
  Frankfurter API. Proved that swapping a tool from mock data to a real API does not change the
  agent loop. Wired in via `TripRequest.home_currency` + `estimated_cost_home`.
- **V4 (in progress)** — grounded knowledge via RAG: a real Wikivoyage corpus + an in-memory
  retrieval tool the agent can call. See `SPEC_V4.md`.

## How V4 fits

Retrieval is just another tool in the V3 loop. The agent calls `search_travel_knowledge` to
ground its recommendations in real travel facts (season, safety, food, hikes, hidden gems)
instead of relying on 4 toy JSON entries or its own memory.

- Corpus is built offline by `scripts/build_index.py` (fetch Wikivoyage → chunk → embed → save).
- The request path only loads and queries the prebuilt index — no fetching/embedding per request.
- In-memory cosine similarity for now, isolated in `knowledge.py` so Chroma is a later drop-in.
- Embeddings are local (`sentence-transformers`), no API key.

## Invariants that must hold

- `agent.py` loop unchanged — every new capability so far is just another tool.
- Numbers come from tools/data, never invented by the LLM.
- App still works with no API key via the deterministic fallback (no RAG on that path).
- No personalization yet — V4 retrieval ranks by relevance only. Taste/preferences are V5.
- No agent frameworks (no LangChain/LangGraph). Hand-rolled loop + Anthropic SDK.
- Response schema and frontend stay stable.

## Constraints (current)

- Mock/curated data plus a small number of real, free, no-key integrations:
  Frankfurter (currency) and Wikivoyage (corpus, fetched at ingest). Embeddings run locally.
- No database yet (a travel-fund / saved-trips state layer would change this — later).
- No authentication, no persistence beyond the prebuilt index file.

## Architecture

```
eurotrip-agent/
  backend/app/
    config.py        # Path constants
    models.py        # Pydantic schemas (TripRequest incl. home_currency, TripPlan)
    planner.py       # Deterministic scorer — a tool + no-key fallback (code unchanged)
    tools.py         # Tool implementations + Anthropic schemas (submit_trip_plan, etc.)
    currency_tool.py # REAL tool: convert_currency via Frankfurter API
    knowledge.py     # V4: load index + search_travel_knowledge (in-memory cosine)
    agent.py         # Claude tool-use loop that DRIVES planning; ends on submit_trip_plan
    explainer.py     # Template explanation used on the no-key fallback path
    prompts.py       # System prompt
    main.py          # FastAPI — key present → agent path, key absent → deterministic path
  frontend/
    streamlit_app.py # Form UI → POST → structured display (home_currency selector)
  scripts/
    build_index.py   # V4: one-time ingest — fetch Wikivoyage, chunk, embed, save index
  data/
    destination_notes.json     # legacy mock destinations
    wikivoyage_sources.json    # V4: curated list of pages to fetch
    knowledge_index.npz        # V4: built corpus (chunks + embeddings + metadata)
  tests/
  requirements.txt
  .env.example
```

## Roadmap

- **V4 (current):** grounded knowledge (Wikivoyage RAG + retrieval tool). See `SPEC_V4.md`.
- **V5:** personalization — a taste profile that ranks/filters retrieved results (demote
  hype-not-her-type, surface preference-matched hidden gems), and multi-trip planning across the
  semester (the "limited time, limited money, many places" optimization).
- **Later (each its own layer):** conversational/multi-turn refinement (backend state + chat UI),
  map itinerary (geocoding + routing tool + map display, Wanderlog-style), travel fund (persistence),
  optional swap of the vector store to Chroma, and evals.
