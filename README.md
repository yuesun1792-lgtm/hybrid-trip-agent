# EuroTrip Agent

Minimal V0 travel planning app for European trips.

## What V0 Does

- Collects basic trip preferences
- Uses local mock destination data
- Returns 3 ranked destinations and one simple itinerary
- Exposes a FastAPI backend
- Provides a Streamlit frontend

The current architecture keeps the Streamlit form as a structured debug surface while moving
the backend toward a state-based planning engine. Form input is normalized into `PlanningState`;
future chat input should use the same state boundary.

V0 does not use a database, real external APIs, LangChain, or LangGraph.

## Project Structure

```text
eurotrip-agent/
  backend/
    app/
      main.py
      models.py
      planner.py
      prompts.py
      config.py
  frontend/
    streamlit_app.py
  data/
    destination_notes.json
    user_preferences.json
  tests/
    test_planner.py
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install fastapi uvicorn streamlit requests pytest
```

## Run The Backend

```bash
uvicorn backend.app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Optional LLM agent mode is controlled through `.env`:

```bash
ENABLE_ANTHROPIC_AGENT=true
ANTHROPIC_AGENT_MAX_TOKENS=4096
ANTHROPIC_API_KEY=your_key_here
```

Set `ENABLE_ANTHROPIC_AGENT=false` to force the deterministic fallback.
The backend reads `.env` when the process starts, so restart `uvicorn` after changing it.

## Run The Frontend

In a second terminal, with the same virtual environment active:

```bash
streamlit run frontend/streamlit_app.py
```

## Candidate Discovery

V5.1 adds a debug endpoint for RAG candidate discovery:

```text
POST http://127.0.0.1:8000/plan/candidates
```

It accepts the same JSON body as `/plan`, then returns `PlanningState` with
`candidate_destinations` and supporting `evidence` populated from the local Wikivoyage index.

V5.1 also adds ranked candidate debugging:

```text
POST http://127.0.0.1:8000/plan/ranked-candidates
```

This runs discovery, then scores candidates with a visible `score_breakdown` for retrieval,
budget, transport, interests, taste, and season.

When the LLM agent is disabled or unavailable, `/plan` now tries the RAG-ranked planning path
first. It falls back to the legacy deterministic planner only if RAG planning cannot produce
ranked candidates.

RAG-ranked destination cards include `citations` with Wikivoyage source URLs. The Streamlit UI
renders these as source links under each destination.

Final plans also include `tradeoff_summary`, `filtered_candidates`, and destination-level
`score_breakdown` so ranking decisions are visible in the UI.

## Chat API

The chat endpoint is the primary natural-language entry point:

```text
POST http://127.0.0.1:8000/chat
```

Send a user message and, on later turns, pass back the returned `extracted_fields`.
The endpoint first runs a task router. It currently supports:

- destination recommendation, using the existing RAG-ranked planning engine
- fixed-destination itinerary planning, for messages like `3 days in Paris`
- clarification, when required fields are missing

When `ANTHROPIC_API_KEY` is present, the router can use an LLM to produce a structured
strategy decision. Without a key, it falls back to deterministic routing rules.

The Streamlit app uses this chat endpoint as the main interface and keeps
`conversation_id`, `extracted_fields`, chat history, and the latest plan in `st.session_state`.

The structured `/plan` endpoint is still available as a debug/regression surface for the
destination recommendation workflow.

## Run Tests

```bash
pytest
```
