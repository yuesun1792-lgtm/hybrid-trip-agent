# CLAUDE.md — EuroTrip Agent

## Project purpose

A learning-oriented project that builds a travel-planning agent one version at a time.
The goal is to understand a real agent pipeline end to end:
user input → backend logic → LLM reasoning (now with tools) → structured output → frontend display.

Build the feature first, review and refine later. Each version is a self-contained,
verifiable chunk of work — not a stream of tiny edits.

## Current status: V2 in progress (V1 complete)

- **V0** — rule-based planner returning ranked destinations + itinerary.
- **V1** — optional LLM explanation layer on top of the planner, with a deterministic
  fallback when no API key is present.
- **V2 (in progress)** — upgrades the explanation layer into a real Claude **tool-use loop**
  over mock data. Claude decides when to call tools; tool results feed back into its explanation.

### What exists (V1, complete)

- `backend/app/explainer.py` — `generate_explanation(plan)` reads `ANTHROPIC_API_KEY`.
  Key present → calls Claude (`claude-haiku-4-5`) for a short explanation. Key absent →
  deterministic template fallback (no network, no crash).
- `backend/app/prompts.py` — prompt templates + fallback string.
- `backend/app/main.py` — `load_dotenv()` at startup, wires `generate_explanation` into `/plan`.
- `frontend/streamlit_app.py` — shows `explanation` in a blue info box.
- `requirements.txt`, `.env.example`.

### What V2 adds

- `backend/app/tools.py` — mock tool implementations (read `data/*.json`, do simple math)
  plus the Anthropic tool JSON schemas describing them to the model.
- `backend/app/agent.py` — hand-rolled Claude tool-use loop: call model → if `stop_reason`
  is `tool_use`, run the tool(s), append `tool_result`, call again → repeat until `end_turn`.
- `backend/app/explainer.py` — `generate_explanation(plan)` now delegates to
  `agent.explain_with_tools(plan)` when a key is present; otherwise returns the template fallback.
  The public interface and the no-key behavior are unchanged.
- `tests/test_planner.py` (+ new tests) — unit tests for the tool functions and the loop,
  runnable without hitting the real API.

### Invariants that must not change

- `planner.py` is the **source of truth**. Tools and the LLM only read the already-computed
  plan and the mock data; they never alter scores or destination ranking.
- Scoring rules are unchanged (see table below).
- The app works with **no API key** — the template fallback is always returned, with no network call.
- **Mock data only** — tools read `data/*.json`. No real external travel APIs.

## Constraints (current)

- Tool calling is allowed in V2, but only **mock** tools backed by `data/*.json`.
- No LangChain, LangGraph, or agent frameworks — the tool loop is hand-rolled.
- No database — data lives in `data/*.json`.
- No real external travel APIs.
- No authentication, no persistence, no async complexity.

## Architecture

```
eurotrip-agent/
  backend/app/
    config.py      # Path constants (PROJECT_ROOT, data file paths)
    models.py      # Pydantic request/response schemas
    planner.py     # Core logic: score destinations, build TripPlan  (UNCHANGED)
    tools.py       # V2: mock tool implementations + Anthropic tool schemas
    agent.py       # V2: Claude tool-use loop
    explainer.py   # Public entry: agent if key present, template fallback if not
    prompts.py     # System prompt + user prompt templates + fallback string
    main.py        # FastAPI app — loads .env, GET / and POST /plan
  frontend/
    streamlit_app.py  # Form UI → POST to backend → structured display
  data/
    destination_notes.json  # mock European destinations
    user_preferences.json   # sample request payload (not loaded by code)
  tests/
    test_planner.py  # planner + tool + loop unit tests
  requirements.txt
  .env.example
```

### Request/response flow (V2)

```
User fills Streamlit form
  → POST /plan with TripRequest JSON
    → planner.create_trip_plan()          ← source of truth, UNCHANGED
      → scores each destination, returns top 3 as TripPlan
    → explainer.generate_explanation(plan)
      → key present → agent.explain_with_tools(plan)
          → Claude may call get_destination_details / check_budget_fit
          → tool results fed back into the model
          → returns natural-language explanation
      → key absent → template fallback
    → returns TripPlan with explanation field
  ← Streamlit displays ranked destinations, itinerary, risks, explanation
```

## How the planner scores destinations (UNCHANGED)

| Rule | Points |
|---|---|
| Each matching interest | +3 |
| Estimated cost ≤ budget | +2 |
| Estimated cost > budget | −2 |
| Transport preference matches | +1 |
| Physical level matches | +1 |

Top 3 destinations by score become `ranked_destinations`.
The itinerary repeats the top destination for every requested day (carried over, V1 simplification).

## How to run

```bash
cp .env.example .env          # optional — app works without a key
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...

uvicorn backend.app.main:app --reload          # backend, terminal 1
streamlit run frontend/streamlit_app.py        # frontend, terminal 2
.venv/bin/python -m pytest tests/ -v           # tests
```

Running without `.env` is fine — the explanation falls back to a deterministic template.

## Roadmap

- **V2 (current):** mock tool calling — see `SPEC_V2.md`.
- **V3 (next):** expand `destination_notes.json` from a handful to ~25–40 richer entries,
  then add **RAG** over those notes to feed the explanation/recommendation. Retrieval store
  still to be decided (simple in-memory cosine similarity vs. a lightweight library like Chroma).
  RAG is gated on the larger dataset — there is nothing meaningful to retrieve over a handful of records.
