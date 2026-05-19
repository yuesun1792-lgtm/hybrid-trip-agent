from fastapi import FastAPI

from backend.app.models import TripPlan, TripRequest
from backend.app.planner import create_trip_plan


app = FastAPI(title="EuroTrip Agent API")


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "eurotrip-agent"}


@app.post("/plan", response_model=TripPlan)
def plan_trip(request: TripRequest) -> TripPlan:
    return create_trip_plan(request)

