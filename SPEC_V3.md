# EuroTrip Agent — V3 Spec: Agentic Planning

## Goal

Move planning authority from the deterministic scorer to the LLM. Through a tool-use loop,
the model selects destinations, gathers facts via tools, assembles the itinerary, and emits a
structured `TripPlan`. The old scorer is demoted to an optional tool plus the no-key fallback.

This is one phase. Implement all of it, then verify against the acceptance criteria below.

## Why

V2 let the model call tools to enrich an explanation, but `planner.py` still decided everything.
V3 hands the decision-making to the model: which destinations to consider, how to weigh budget vs.
interests, how to sequence the itinerary. This is the step that turns "an app that calls an LLM"
into "an agent." The learning target is how an LLM orchestrates multiple tool calls toward a goal
and how you get reliable structured output back out of that loop.

## Core design decisions (baked in)

1. **Structured output via a terminal tool.** Define `submit_trip_plan(...)` whose input schema
   mirrors the `TripPlan` model. The loop ends when the agent calls it; the tool input, validated by
   Pydantic, becomes the `/plan` response. The loop does NOT end on `end_turn` — if the model stops
   without submitting, see "Loop safety" below.

2. **Keep `planner.py`; demote it.** The deterministic `create_trip_plan` is no longer the authority.
   It now serves two reduced roles: (a) the `score_destinations` tool, and (b) the no-key fallback.
   `planner.py`'s code is unchanged — only how it is called changes.

3. **Math stays in code.** Costs, budget deltas, and scores come only from tools/data. The LLM must
   never invent a number or do budget arithmetic. The model orchestrates; the tools compute.

## Scope

### Tools (in `tools.py`)

- `list_destinations() -> list[dict]` — names + a one-line summary of each, so the agent knows what's available.
- `get_destination_details(name: str) -> dict` — full record (carried from V2).
- `check_budget_fit(name: str, budget: float) -> dict` — budget math (carried from V2).
- `score_destinations(request: dict) -> list[dict]` — wraps the existing deterministic scorer;
  returns ranked destinations with scores. Optional for the agent to call.
- `submit_trip_plan(...)` — **terminal tool.** Input schema mirrors `TripPlan`
  (ranked_destinations, pros/cons, estimated_cost, itinerary, risks, recommendation, explanation).
  Calling it ends the loop.
- `TOOL_SCHEMAS` — the Anthropic `tools` list for all of the above.
- `dispatch_tool(name, tool_input) -> dict` — maps tool name to implementation.

### Agent loop (in `agent.py`)

- `create_plan(request) -> TripPlan` — the tool-use loop:
  1. Build the initial message from the system prompt + the user's `TripRequest`.
  2. Call Claude with `tools=TOOL_SCHEMAS`.
  3. While the model issues `tool_use` blocks: run each via `dispatch_tool`, append `tool_result`,
     call again. If a `tool_use` is `submit_trip_plan`, validate its input as a `TripPlan` and return it.
  4. Hard iteration cap (e.g. 8).
- The system prompt (in `prompts.py`) must tell the model: use the tools to gather facts, never invent
  costs, and finish by calling `submit_trip_plan` exactly once with the final plan.

### Loop safety

- Hard iteration cap; on exceeding it, fall back to the deterministic planner.
- If the model returns `end_turn` without calling `submit_trip_plan`, send one nudge message asking it
  to submit; if it still doesn't, fall back to the deterministic planner.
- If `submit_trip_plan` input fails Pydantic validation, retry once with the validation error fed back;
  on a second failure, fall back.

### Changed wiring

- `main.py` — `/plan` branches on the API key: key present → `agent.create_plan(request)`;
  key absent → `planner.create_trip_plan(request)` + `explainer` template explanation.
- `explainer.py` — keeps the template explanation used on the no-key fallback path.
- `models.py` — confirm `TripPlan` fields line up with the `submit_trip_plan` schema; adjust only if needed.
- `planner.py` — **code unchanged.**

### Tests (`tests/`)

- Unit-test `list_destinations`, `get_destination_details`, `check_budget_fit`, `score_destinations`,
  and `dispatch_tool` directly (no API).
- Test the no-key fallback: returns a valid `TripPlan` with a template explanation, no network call.
- Test the loop with a stubbed Anthropic client that emits a couple of `tool_use` blocks then a
  `submit_trip_plan` call; assert the tools were dispatched and a valid `TripPlan` is returned.
- Test loop safety: stub a client that never submits → assert it falls back without crashing.

## Invariants (must hold)

- Response schema unchanged — `/plan` returns a `TripPlan`; the Streamlit frontend is untouched.
- All costs/deltas/scores come from tools or data — never invented by the LLM.
- Mock data only — tools read `data/*.json`.
- App works with no API key via the deterministic fallback, with no network call.
- No agent framework. Hand-rolled loop using the Anthropic SDK.
- Same model as before (`claude-haiku-4-5`); a stronger model is optional.

## Out of scope (do NOT do in V3)

- RAG, embeddings, vector stores, retrieval tools (that's V4).
- Expanding the destination dataset (V4).
- Database, authentication, persistence.
- Real travel APIs.
- Changing the frontend or the `TripPlan` response shape beyond field alignment.

## Acceptance criteria (definition of done)

The phase is done when **all** of these are true:

1. `.venv/bin/python -m pytest tests/ -v` passes.
2. **With** a valid `ANTHROPIC_API_KEY`, `POST /plan` returns a valid `TripPlan` assembled by the
   agent; run logs show the model called at least two tools, ending with `submit_trip_plan`.
3. Every monetary value and score in the returned plan traces to tool/data output — none are invented.
4. **Without** a key, `POST /plan` returns a valid `TripPlan` via the deterministic planner +
   template explanation, with no network call.
5. The loop has a hard cap and a defined fallback if the agent never submits or submits invalid data.
6. The `TripPlan` response schema is unchanged and the Streamlit frontend renders it without edits.

## Suggested commit boundary

One commit (or a small series) ending with all acceptance criteria met and tests green.
