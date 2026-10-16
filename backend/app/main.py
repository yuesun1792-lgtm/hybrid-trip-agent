from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from backend.app.planning.plan_store import PlanVersionConflict, PLAN_STORE
from backend.app.planning.plan_validation import PlanValidationError
from backend.app.planning.task_builder import PlanningNeedsInput
from backend.app.planning.revision import RevisionNeedsInput
from backend.app.runtime.execution import BudgetExceeded, RunContext, RUN_CONTEXT

from backend.app.conversation.chat import handle_chat
from backend.app.domain.models import ChatRequest, ChatResponse, PlanningState, TripPlan, TripRequest
from backend.app.planning.planning_engine import (
    discover_candidates_for_state,
    plan_from_state,
    rank_candidates_for_state,
    state_from_request,
)

# Load .env when present; no-op if the file does not exist.
load_dotenv()

app = FastAPI(title="EuroTrip Agent API")


@app.exception_handler(PlanVersionConflict)
async def version_conflict(request: Request, exc: PlanVersionConflict):
    return JSONResponse(status_code=409, content={"detail": str(exc), "code": "plan_version_conflict"})


@app.exception_handler(PlanValidationError)
async def invalid_plan(request: Request, exc: PlanValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc), "issues": [i.model_dump() for i in exc.issues]})


@app.exception_handler(PlanningNeedsInput)
@app.exception_handler(RevisionNeedsInput)
async def missing_input(request: Request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(BudgetExceeded)
async def exhausted_budget(request: Request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc), "code": "run_budget_exceeded"})


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "eurotrip-agent"}


@app.post("/plan", response_model=TripPlan)
def plan_trip(request: TripRequest) -> TripPlan:
    token = RUN_CONTEXT.set(RunContext())
    try:
        plan = plan_from_state(state_from_request(request))
        from backend.app.conversation.language import render_language
        return render_language(ChatResponse(conversation_id="form", assistant_message=plan.final_recommendation,
                                            plan=plan, response_language=request.response_language), request.response_language).plan
    finally:
        RUN_CONTEXT.reset(token)


@app.post("/plan/candidates", response_model=PlanningState)
def discover_candidates(request: TripRequest) -> PlanningState:
    state = state_from_request(request)
    return discover_candidates_for_state(state)


@app.post("/plan/ranked-candidates", response_model=PlanningState)
def rank_candidates(request: TripRequest) -> PlanningState:
    state = state_from_request(request)
    discovered = discover_candidates_for_state(state)
    return rank_candidates_for_state(discovered)


@app.get("/plans/{plan_id}", response_model=TripPlan)
def read_plan(plan_id: str, version: int | None = Query(default=None, gt=0)) -> TripPlan:
    """Read canonical committed state without an LLM call or changing conversation state."""
    plan = PLAN_STORE.get(plan_id, version) if version is not None else PLAN_STORE.latest(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Plan version not found")
    return plan


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return handle_chat(request)
