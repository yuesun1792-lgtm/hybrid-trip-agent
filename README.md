# EuroTrip Agent

A travel-planning app with a controlled workflow, English Wikivoyage retrieval,
structured daily activities, OpenAI/Anthropic support and multilingual chat.

## Current architecture

Specified and unspecified destinations use the same planner. Unspecified destinations
first go through candidate discovery; all candidates receive a common overview analysis.
Only selected destinations receive detailed research and a daily activity itinerary.
Comparison-only requests skip itinerary generation. Revisions reuse the same capabilities,
regenerate only affected scopes and commit a new version after validation.

See [current architecture](docs/current_architecture.md) for contracts, module boundaries,
revision behavior, budgets, compatibility and limitations.

Optional real place resolution uses a bounded model-directed MCP node with Geoapify:
see [setup and verification](docs/place_resolution.md). Start with
`python -m scripts.test_places --discover` (no keys/network required).

Read [MCP walkthrough](docs/mcp_walkthrough.md) for a conversation-to-code example.
[Resume project notes](docs/resume_project.md) describe the implemented scope and limits.

Multi-turn behavioral evaluation: `python -m scripts.eval_scenarios` (offline) or
`python -m scripts.eval_scenarios --mode live --case paris_local_revision` (real API).
See [scenario evaluation](docs/scenario_evaluation.md) for the 14 reference cases,
assertions, reports, repeat runs and automation exit codes.

```text
backend/app/
  main.py          # FastAPI entry point
  config.py        # Shared configuration and project paths
  conversation/    # Chat, state, intent routing and language
  domain/          # Shared schemas, normalization and destination names
  planning/        # Current planning workflow, revision, validation and storage
  rag/             # Retrieval, reranking and discovery research
  tool_calling/    # Tool implementations, schemas and execution
  places/          # Place resolution, Geoapify and MCP client/server
  llm/             # Model adapters, token usage and prompts
  runtime/         # Execution budgets and trajectories
  legacy/          # Earlier planners and compatibility adapters
frontend/          # Streamlit UI and HTTP client
scripts/           # Index building, evaluation and smoke-test entry points
tests/             # Automated regression tests
```

The legacy `legacy/agent.py` is Anthropic-only; its behavior is unchanged; HTTP endpoints use the unified
scheduler. Former fixed/ranked builders are compatibility adapters.

See [directory organization](docs/directory_structure.md) for classification rules and the complete module map.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn backend.app.main:app --reload
```

In another terminal:

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py
```

API: `http://127.0.0.1:8000`, docs: `/docs`. Restart the backend after `.env` changes.
The Streamlit page disables developer cache shortcuts and bypasses system proxies for
its local backend requests. On macOS use Command+C for native browser copying.

## LLM configuration

OpenAI:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_actual_api_key
OPENAI_MODEL=gpt-5.4-mini
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=0
LLM_TRUST_ENV=true
```

Anthropic is the default when `LLM_PROVIDER` is omitted:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_actual_api_key
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
```

Only the selected provider's key enables calls. Without a key the app still returns
comparisons and structured placeholders marked `needs_verification`. OpenAI requires an
explicit model available to your account and uses Responses API with `store=false`.

`ENABLE_LLM_RAG_ENHANCEMENTS=false` disables dynamic discovery query research, not routing
or itinerary writing. `LLM_TRUST_ENV=false` bypasses outbound proxies where a direct
connection is appropriate. The deprecated `ENABLE_ANTHROPIC_AGENT` flag is not used by API.

Optional OpenAI model-specific prices in USD per million tokens:

```dotenv
OPENAI_INPUT_PRICE_PER_MTOK=0.75
OPENAI_OUTPUT_PRICE_PER_MTOK=4.50
OPENAI_CACHE_READ_PRICE_PER_MTOK=0.075
```

Verify prices for the chosen model. Missing prices yield null cost estimates. Anthropic
uses `LLM_*_PRICE_PER_MTOK`. OpenAI cached input is counted within total input and billed once.

Run bounds (defaults):

```dotenv
RUN_TIMEOUT_SECONDS=180
RUN_MAX_LLM_CALLS=12
RUN_MAX_TOOL_CALLS=40
RUN_MAX_OUTPUT_TOKENS=32000
RAG_RESEARCH_MAX_ROUNDS=3
RAG_RESEARCH_QUERIES_PER_ROUND=3
RAG_RESEARCH_MAX_CANDIDATES=8
```

## API and examples

`POST /chat` is the primary natural-language surface:

```json
{"message":"计划三日巴黎游 美食 文化 时尚"}
```

Other examples:

- `从慕尼黑出发玩三天，预算900欧元，美食文化，火车，节奏适中，请推荐目的地并安排具体行程`
- `先比较几个城市，美食文化` with `requested_output="options"`
- `第二天下午换成购物，晚餐保留` with the previous `conversation_id`
- `添加第二天下午的购物活动`

`response_language="zh"` or another language code explicitly controls presentation.
Language detection is heuristic. Numbers/dates preserve the conversation's language.
Canonical city names and interests remain English; source excerpts and trace remain
original. Non-English prose rendering uses an additional recorded LLM call. Without a
configured LLM, full plan translation is unavailable and Chinese users receive a notice.

For precise revisions:

```json
{
  "conversation_id": "existing-conversation-id",
  "message": "Replace the selected activity",
  "expected_plan_version": 1,
  "revision": {
    "base_plan_id": "existing-plan-id",
    "base_version": 1,
    "changes": [
      {
        "action": "replace",
        "activity_ids": ["existing-activity-id"],
        "category": "shopping",
        "requirements": "Keep the afternoon light"
      }
    ],
    "preserve_activity_ids": ["existing-dinner-id"]
  }
}
```

`constraints_updates` supports fields such as destination, trip_length, start_date,
budget and interests. Dates use ISO `YYYY-MM-DD`; budget and curated costs are in EUR.
Unknown item prices, durations and locations remain null. Numerical estimates come from
curated data/tools rather than model guesses.

`POST /plan` keeps the structured `TripRequest` interface and accepts optional destination,
candidate_destinations, requested_output and response_language. Debug APIs `/plan/candidates`
and `/plan/ranked-candidates` use the same resolution/overview analysis rules.

Read canonical versions with `GET /plans/{plan_id}` or `?version=1`. Stale version conflicts
return 409; invalid drafts return 422; exhausted execution budgets return 503. Plan/session
storage is currently single-process memory, so restart loses history. Persisted trajectories
are diagnostics and do not provide execution recovery.

## Trajectory and verification

Every chat response includes `run_id`, `planning_task` and `trajectory`. Download the latest
JSON in Streamlit. The backend writes `<run_id>.traj.json` and incremental
`<run_id>.events.jsonl` under `data/trajectories/`; override with `TRAJECTORY_DIR`.
Credentials are redacted, user messages/evidence are included and files are Git-ignored.
SDK call timing includes internal retries; individual SDK attempts are not observed.

```bash
# Offline tests; no real keys or model downloads used by fixtures
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p no:capture -q tests
# Real configured-provider smoke tests (billable API calls)
python -m scripts.test_llm
python -m scripts.test_llm --chat
python -m scripts.test_llm --chat --message '计划三日巴黎游 美食 文化 时尚' \
  --revision-message '第二天下午换成购物，晚餐保留'
```

The revision smoke checks versions and exact preservation of unaffected days/explicitly
preserved activities. Missing credentials/model exit 2; fallback or failed checks exit 1.

Build the English corpus index separately when needed:

```bash
python scripts/build_index.py
```

## Practical limits

The system validates structure, identity, city/evidence consistency, preservation and
estimated constraints. It does not prove factual entailment or verify live venue opening
hours, reservations and travel connections. Activity statuses and validation warnings
expose that uncertainty. External writes, MCP, LangGraph and multi-agent orchestration
are extension points, not enabled features.
