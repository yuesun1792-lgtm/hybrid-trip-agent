import os

from dotenv import load_dotenv
from fastapi import FastAPI

from backend.app.explainer import generate_explanation
from backend.app.models import TripPlan, TripRequest
from backend.app.planner import create_trip_plan

# Load .env when present; no-op if the file does not exist.
load_dotenv()

app = FastAPI(title="EuroTrip Agent API")


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "eurotrip-agent"}


@app.post("/plan", response_model=TripPlan)
def plan_trip(request: TripRequest) -> TripPlan:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401 — ensure package present before importing agent
            from backend.app.agent import create_plan
            return create_plan(request)
        except Exception:
            pass  # fall through to deterministic path

    plan = create_trip_plan(request)
    explanation = generate_explanation(plan)
    return plan.model_copy(update={"explanation": explanation})
