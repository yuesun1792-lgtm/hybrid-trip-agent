# EuroTrip Agent

Minimal V0 travel planning app for European trips.

## What V0 Does

- Collects basic trip preferences
- Uses local mock destination data
- Returns 3 ranked destinations and one simple itinerary
- Exposes a FastAPI backend
- Provides a Streamlit frontend

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

## Run The Frontend

In a second terminal, with the same virtual environment active:

```bash
streamlit run frontend/streamlit_app.py
```

## Run Tests

```bash
pytest
```

