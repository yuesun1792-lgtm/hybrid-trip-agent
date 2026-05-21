# CLAUDE.md — EuroTrip Agent

## Project purpose

A learning-oriented portfolio project that simulates a travel planning agent.
The goal is to understand how a simple agent pipeline works end-to-end:
user input → backend logic → structured output → frontend display.

This project is intentionally minimal so it can be explained clearly in an interview.

## Current status: V1 complete

V1 adds an optional LLM explanation layer on top of the V0 rule-based planner.

### What was added in V1

- `backend/app/explainer.py` — `generate_explanation(plan)` reads `ANTHROPIC_API_KEY`
  from the environment. If the key is present it calls Claude (`claude-haiku-4-5`) for a
  2–3 sentence natural-language explanation. If the key is absent it returns a deterministic
  template-based fallback — no network call, no crash.
- `backend/app/prompts.py` — prompt templates for the LLM call and the fallback string.
- `backend/app/main.py` — calls `load_dotenv()` at startup then wires `generate_explanation`
  into the `/plan` endpoint; `explanation` is attached to the returned `TripPlan`.
- `frontend/streamlit_app.py` — displays `explanation` in a blue info box when present.
- `requirements.txt` — pinned deps including `python-dotenv` and `anthropic`.
- `.env.example` — template for the local secrets file.

### Invariants that must not change

- `planner.py` is the source of truth. The LLM reads the already-computed plan and
  adds a sentence; it does not influence scores or destination ranking.
- Scoring rules are unchanged (see table below).
- The app works without any API key — the fallback explanation is always returned.

## V1 constraints (do not add these yet)

- No real tool calls — Claude is called once for text only, no function/tool use
- No LangChain, LangGraph, or agent frameworks
- No database — data lives in `data/*.json`
- No real external travel APIs — destinations are hardcoded mock data
- No authentication, no persistence, no async complexity

## Architecture

```
eurotrip-agent/
  backend/app/
    config.py      # Path constants (PROJECT_ROOT, data file paths)
    models.py      # Pydantic request/response schemas
    planner.py     # Core logic: score destinations, build TripPlan
    explainer.py   # Optional LLM explanation; falls back to template if no key
    prompts.py     # System prompt and user prompt templates for LLM call
    main.py        # FastAPI app — loads .env, GET / and POST /plan
  frontend/
    streamlit_app.py  # Form UI → POST to backend → structured display
  data/
    destination_notes.json  # 4 mock European destinations
    user_preferences.json   # Sample request payload (not loaded by code)
  tests/
    test_planner.py  # Unit tests for planner and fallback explanation
  requirements.txt   # Pinned dependencies
  .env.example       # Copy to .env and add ANTHROPIC_API_KEY
```

### Request/response flow

```
User fills Streamlit form
  → POST /plan with TripRequest JSON
    → planner.create_trip_plan()          ← source of truth, unchanged
      → scores each destination
      → returns top 3 as TripPlan
    → explainer.generate_explanation()    ← reads ANTHROPIC_API_KEY
      → LLM call if key present
      → template fallback if key absent
    → returns TripPlan with explanation field
  ← Streamlit displays ranked destinations, itinerary, risks, explanation
```

## How the planner scores destinations

Each destination in `destination_notes.json` gets a numeric score:

| Rule | Points |
|---|---|
| Each matching interest | +3 |
| Estimated cost ≤ budget | +2 |
| Estimated cost > budget | −2 |
| Transport preference matches | +1 |
| Physical level matches | +1 |

Top 3 destinations by score become `ranked_destinations`.
The itinerary repeats the top destination for every requested day (V1 simplification).

## How to run

```bash
# 1. Copy the env template and add your key (optional — app works without it)
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# Backend (terminal 1)
uvicorn backend.app.main:app --reload

# Frontend (terminal 2)
streamlit run frontend/streamlit_app.py

# Tests
.venv/bin/python -m pytest tests/ -v
```

Running without `.env` is fine — the explanation falls back to a deterministic template.

## Next planned phase: V2 mock tool calling

V2 will teach the model to "call tools" using Claude's tool-use API, while keeping
the underlying data mock (no real external APIs). The goal is to understand how an
LLM decides when to invoke a tool and how the result feeds back into its response.

Planned scope (do not implement yet):
- Define 1–2 mock tools (e.g., `get_destination_details`, `check_budget_fit`)
- Wire Claude tool-use loop in `explainer.py` or a new `agent.py`
- The planner scoring logic stays unchanged — tools are for the explanation layer only
- No LangGraph, no database, no real API calls
