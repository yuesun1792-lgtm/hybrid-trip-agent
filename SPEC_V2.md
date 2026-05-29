# EuroTrip Agent — V2 Spec: Mock Tool Calling

## Goal

Turn the V1 explanation layer into a real Claude **tool-use loop**. The model should
decide *when* to call a tool, receive the result, and fold it into a short explanation
of the trip plan. The tools are mock — they read `data/*.json` and do simple math — but
the tool-use mechanics are real.

This is one phase. Implement all of it, then verify against the acceptance criteria below.

## Why

V1 already explains the plan, but it just gets the whole plan handed to it as text. V2
teaches the model to *fetch what it needs*: pull a destination's details, check whether a
cost fits the budget. The learning target is the tool-use loop itself — how the model
signals a tool call, how results are returned, and how the conversation continues to a final answer.

## Scope

### New files

- `backend/app/tools.py`
  - Tool implementations (plain Python functions over mock data):
    - `get_destination_details(name: str) -> dict` — return the full record for a
      destination from `destination_notes.json` (description, estimated cost, activities,
      transport, physical level, etc.). If the name is unknown, return a clear error dict
      (e.g. `{"error": "unknown destination", "name": name}`) rather than raising.
    - `check_budget_fit(name: str, budget: float) -> dict` — return whether the
      destination's estimated cost is within budget, plus the numeric difference,
      e.g. `{"name": ..., "estimated_cost": ..., "budget": ..., "fits": true, "delta": -120}`.
  - `TOOL_SCHEMAS` — the Anthropic `tools` list (JSON schema) describing both tools to the model.
  - `dispatch_tool(name: str, tool_input: dict) -> dict` — map a tool name to its function.

- `backend/app/agent.py`
  - `explain_with_tools(plan) -> str` — the hand-rolled tool-use loop:
    1. Build the initial messages from the system prompt + the already-computed plan.
    2. Call Claude with `tools=TOOL_SCHEMAS`.
    3. While `response.stop_reason == "tool_use"`: for each `tool_use` block, run
       `dispatch_tool`, append a `tool_result` block, and call the model again.
    4. When `stop_reason == "end_turn"`, return the final text as the explanation.
    5. Cap the loop (e.g. max 5 iterations) so it can never spin forever.

### Changed files

- `backend/app/explainer.py`
  - `generate_explanation(plan)` keeps the same signature and role. If `ANTHROPIC_API_KEY`
    is present, delegate to `agent.explain_with_tools(plan)`. If absent — **or if the agent
    call raises** — return the existing template fallback. Never crash the `/plan` endpoint.

- `backend/app/prompts.py`
  - Add/adjust the system prompt so the model knows the tools exist and may use them to
    verify details before explaining. Keep the fallback string.

- `tests/test_planner.py` (or a new `tests/test_agent.py`)
  - Unit-test `get_destination_details`, `check_budget_fit`, and `dispatch_tool` directly
    (no API needed).
  - Test the fallback path: with no key, `generate_explanation` returns the template and
    makes no network call.
  - Optionally test the loop with a stubbed/mocked Anthropic client that returns a
    `tool_use` then an `end_turn`, asserting the tool was dispatched and the final text returned.

## Invariants (must hold)

- `planner.py` is unchanged. Tools only **read** the plan and mock data; they never affect
  scoring or ranking.
- App works with no API key: template fallback, no network call.
- Mock data only — tools read `data/*.json`. No real external APIs.
- No agent framework (no LangChain/LangGraph). The loop is plain Python + the Anthropic SDK.
- Same model as V1 (`claude-haiku-4-5`). A stronger model is optional but not required.

## Out of scope (do NOT do in V2)

- RAG, embeddings, vector stores (that's V3).
- Expanding the destination dataset (also V3).
- A database, authentication, or persistence.
- Real travel APIs.
- Changing the planner's scoring or the response schema, beyond what already carries `explanation`.

## Acceptance criteria (definition of done)

The phase is done when **all** of these are true:

1. `.venv/bin/python -m pytest tests/ -v` passes.
2. **With** a valid `ANTHROPIC_API_KEY`, `POST /plan` returns a `TripPlan` whose
   `explanation` reflects at least one tool result (e.g. a budget-fit or detail fetched via a tool),
   and the run logs show at least one tool call.
3. **Without** an API key, `POST /plan` still returns a valid `TripPlan` with a template
   explanation, with no network call and no error.
4. The tool functions in `tools.py` are pure and unit-tested independently of the model.
5. The loop has a hard iteration cap and cannot run forever.
6. No file outside the list above is modified; `planner.py` is byte-for-byte unchanged.

## Suggested commit boundary

One commit (or a small series) that ends with all acceptance criteria met and tests green.
