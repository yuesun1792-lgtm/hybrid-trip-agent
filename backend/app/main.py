from dotenv import load_dotenv
from fastapi import FastAPI

from backend.app.chat import handle_chat
from backend.app.models import ChatRequest, ChatResponse, PlanningState, TripPlan, TripRequest
from backend.app.planning_engine import (
    discover_candidates_for_state,
    plan_from_state,
    rank_candidates_for_state,
    state_from_request,
)

# Load .env when present; no-op if the file does not exist.
load_dotenv()

app = FastAPI(title="EuroTrip Agent API")


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "eurotrip-agent"}


@app.post("/plan", response_model=TripPlan)
def plan_trip(request: TripRequest) -> TripPlan:
    return plan_from_state(state_from_request(request))


@app.post("/plan/candidates", response_model=PlanningState)
def discover_candidates(request: TripRequest) -> PlanningState:
    state = state_from_request(request)
    return discover_candidates_for_state(state)


@app.post("/plan/ranked-candidates", response_model=PlanningState)
def rank_candidates(request: TripRequest) -> PlanningState:
    state = state_from_request(request)
    discovered = discover_candidates_for_state(state)
    return rank_candidates_for_state(discovered)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return handle_chat(request)
