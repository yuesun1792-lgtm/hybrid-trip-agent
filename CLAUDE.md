# CLAUDE.md — EuroTrip Agent

## Project purpose

A learning-oriented project that builds a travel-planning agent one version at a time.
The goal is to understand a real agent pipeline end to end:
user input → an LLM that plans via tool calls → structured output → frontend display.

Build the feature first, review and refine later. Each version is a self-contained,
verifiable chunk of work — not a stream of tiny edits.

## Current status: V3 in progress (V2 complete)

- **V0** — rule-based planner returning ranked destinations + itinerary.
- **V1** — optional LLM explanation layer on top of the planner, with a deterministic fallback.
- **V2** — Claude tool-use loop: the model decides when to call mock tools, results feed back
  into its explanation. The deterministic planner was still the source of truth.
- **V3 (in progress)** — **the LLM now drives the planning itself.** Through a tool-use loop it
  selects destinations, gathers facts via tools, and emits a structured `TripPlan`. The
  deterministic planner is demoted from "source of truth" to an optional tool plus the no-key fallback.

### The big architectural change in V3

The old invariant "planner.py is the source of truth" is **removed.** Planning authority moves
to the LLM. The deterministic scorer survives in two reduced roles:

1. As a **tool** (`score_destinations`) the agent may choose to call for a baseline ranking.
2. As the **no-key fallback** — with no API key, the app runs the old deterministic planner so it
   still works offline.

### How structured output works now

The agent loop ends on a **terminal tool**, not on `end_turn`. `submit_trip_plan(...)` has an input
schema mirroring the `TripPlan` model; when the agent calls it, that input — validated by Pydantic —
becomes the response. This is how a free-form tool loop yields reliable structured JSON for the frontend.

### Invariants that must hold (revised)

- The **response schema is unchanged** — `/plan` still returns a `TripPlan`. The Streamlit frontend
  needs no changes.
- **All numbers come from tools/data, never the LLM.** Costs, budget deltas, and scores are computed
  in code; the model only orchestrates (selection, weighting, sequencing, narrative). The LLM must
  never invent a cost or do budget arithmetic.
- **Mock data only** — tools read `data/*.json`. No real external travel APIs.
- **Works with no API key** — falls back to the deterministic planner + template explanation, no network.
- **No agent frameworks** (no LangChain/LangGraph). The loop is hand-rolled Python + the Anthropic SDK.

## Constraints (current)

- Tools are mock, backed by `data/*.json`.
- No LangChain / LangGraph / agent frameworks.
- No database.
- No real external travel APIs.
- No authentication, no persistence, no async complexity.

## Architecture

```
eurotrip-agent/
  backend/app/
    config.py      # Path constants (PROJECT_ROOT, data file paths)
    models.py      # Pydantic schemas (TripRequest, TripPlan)
    planner.py     # Deterministic scorer — now a TOOL + no-key fallback (code unchanged)
    tools.py       # Tool implementations + Anthropic tool schemas (incl. submit_trip_plan)
    agent.py       # Claude tool-use loop that DRIVES planning; ends on submit_trip_plan
    explainer.py   # Template explanation used on the no-key fallback path
    prompts.py     # System prompt instructing the agent how to plan with tools
    main.py        # FastAPI app — key present → agent path, key absent → deterministic path
  frontend/
    streamlit_app.py  # Form UI → POST to backend → structured display (UNCHANGED)
  data/
    destination_notes.json  # mock European destinations
    user_preferences.json   # sample request payload (not loaded by code)
  tests/
    test_planner.py  # planner + tools + agent loop unit tests
  requirements.txt
  .env.example
```

### Request/response flow (V3)

```
User fills Streamlit form
  → POST /plan with TripRequest JSON
    → key present?
        YES → agent.create_plan(request)              ← LLM drives planning
              loop: model calls tools
                list_destinations / get_destination_details
                check_budget_fit / score_destinations (optional)
              → model calls submit_trip_plan(...)      ← terminal tool ends the loop
              → validate input as TripPlan
        NO  → planner.create_trip_plan(request)        ← deterministic fallback
              + explainer template explanation
    → returns TripPlan
  ← Streamlit displays ranked destinations, itinerary, risks, explanation
```

## The scoring rules (now behind a tool, not the pipeline)

These rules still exist — they back the `score_destinations` tool and the no-key fallback.
They are no longer the final word; the agent may call them, weigh them, or override them.

| Rule | Points |
|---|---|
| Each matching interest | +3 |
| Estimated cost ≤ budget | +2 |
| Estimated cost > budget | −2 |
| Transport preference matches | +1 |
| Physical level matches | +1 |

## How to run

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY for the agent path
                              # without it, the deterministic fallback runs

uvicorn backend.app.main:app --reload          # backend, terminal 1
streamlit run frontend/streamlit_app.py        # frontend, terminal 2
.venv/bin/python -m pytest tests/ -v           # tests
```

With no key the app still works — it runs the deterministic planner and a template explanation.

## Roadmap

- **V3 (current):** agentic planning — the LLM drives the plan via tools. See `SPEC_V3.md`.
- **V4 (next):** RAG. Expand `destination_notes.json` to ~25–40 richer entries, add a
  `search_destination_notes` retrieval tool the agent can call. Vector store still to be decided
  (simple in-memory cosine similarity vs. a library like Chroma).
