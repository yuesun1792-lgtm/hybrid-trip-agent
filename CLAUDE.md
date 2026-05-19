# CLAUDE.md — EuroTrip Agent

## Project purpose

A learning-oriented portfolio project that simulates a travel planning agent.
The goal is to understand how a simple agent pipeline works end-to-end:
user input → backend logic → structured output → frontend display.

This project is intentionally minimal so it can be explained clearly in an interview.

## V0 constraints (important — do not add these in V0)

- No database — all data lives in `data/*.json` files
- No real external APIs — destinations are hardcoded mock data
- No LangChain, LangGraph, or agent frameworks — logic is plain Python
- No LLM calls yet — `prompts.py` is a placeholder for a future V1 feature
- No authentication, no persistence, no async complexity

## Architecture

```
eurotrip-agent/
  backend/app/
    config.py      # Path constants (PROJECT_ROOT, data file paths)
    models.py      # Pydantic request/response schemas
    planner.py     # Core logic: score destinations, build TripPlan
    prompts.py     # Placeholder system prompt string (unused in V0)
    main.py        # FastAPI app — two endpoints: GET / and POST /plan
  frontend/
    streamlit_app.py  # Form UI → POST to backend → display raw JSON
  data/
    destination_notes.json  # 4 mock European destinations
    user_preferences.json   # Sample request payload (not loaded by code)
  tests/
    test_planner.py  # Unit test for create_trip_plan
```

### Request/response flow

```
User fills Streamlit form
  → POST /plan with TripRequest JSON
    → planner.create_trip_plan()
      → loads destination_notes.json
      → scores each destination against request
      → returns top 3 as TripPlan
  ← Streamlit displays TripPlan as raw JSON
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
The itinerary repeats the top destination for every requested day (V0 simplification).

## How to run

```bash
# Backend (terminal 1)
uvicorn backend.app.main:app --reload

# Frontend (terminal 2)
streamlit run frontend/streamlit_app.py

# Tests
pytest
```

Dependencies: `fastapi uvicorn streamlit requests pytest`

## What is deferred to V1+

- LLM explanation of the plan (the `prompts.py` stub is ready for this)
- Multi-city itinerary routing
- Real destination data from an external API
- Database persistence of plans
- LangGraph agent loop
