# Scenario evaluation

The evaluator exercises the real in-process chat workflow over isolated, multi-turn
conversations. It does not start FastAPI or Streamlit. HTTP/UI tests remain separate.

## Run

From the repository root:

```bash
# No credentials, API calls or embedding downloads; synthetic evidence + rule builder.
.venv/bin/python -m scripts.eval_scenarios

# Selected scenarios, repeated with independent conversations.
.venv/bin/python -m scripts.eval_scenarios --case paris_local_revision --case discover_itinerary --repeat 3

# Real configured provider and actual local RAG index; consumes API tokens.
.venv/bin/python -m scripts.eval_scenarios --mode live --case paris_local_revision

# Full real suite; repeat to measure variability rather than exact response text.
.venv/bin/python -m scripts.eval_scenarios --mode live --repeat 3

# Custom input and report directory.
.venv/bin/python -m scripts.eval_scenarios --suite evals/scenarios/core.json --output-dir data/evaluations/manual
```

Live mode loads `.env` and requires the selected provider's key and model. Build the RAG
index as described in README before evaluating retrieval quality. Missing evidence or
provider failures can produce explicit fallbacks and fail strict live assertions.
Offline mode never loads `.env` and disables both provider keys even if already exported.
It tests control flow, data integrity and fallback behavior, not LLM quality or English
retrieval relevance. A passing offline run does not establish that live generation works.

Exit codes: **0** all cases passed, **1** behavioral assertion failed, **2** invalid
configuration/suite. These codes support CI or scheduled execution. Run the offline
suite alongside pytest on each change; run live scenarios after prompt/model/retrieval
changes and before release. Existing per-turn execution budgets still apply. There is
no whole-suite monetary cap; `--case` and `--repeat` determine how many turns can run.

## Reference scenarios

`evals/scenarios/core.json` includes 14 cases: known destination + local replacement,
discovery comparisons, discovery + itinerary, options promoted to a selected itinerary,
missing duration + reply, relaxed day preserving dinner, destination change, budget-only
change, seven-day trip, invalid duration + correction, stale version, add, remove, and
extended duration. Some revisions intentionally use typed requests to independently
test merge/commit behavior; natural-language cases test interpretation as well.

Each repetition starts a new conversation. Expected plan versions are sent automatically
after the first committed plan, unless a turn overrides them (e.g. stale-version case).
A failed turn stops dependent turns in its case; other cases continue. Scenarios run
sequentially. Concurrency and injected transport/model failures are covered by pytest,
not simulated by this runner.

## Format

```json
{
  "schema_version": "1.0",
  "scenarios": [{
    "id": "paris_revision",
    "turns": [
      {
        "message": "计划巴黎三日游，美食文化",
        "expect": {"outcome": "plan", "destination": "Paris", "day_count": 3}
      },
      {
        "message": "第二天下午换成购物，晚餐保留",
        "expect": {
          "outcome": "revision",
          "changed_days": [2],
          "preserve_other_days": true,
          "preserve_evening": true,
          "version_increment": 1,
          "activity_checks": [{"day": 2, "time_slot": "afternoon", "category": "shopping"}]
        }
      }
    ]
  }]
}
```

Unknown fields and duplicate case IDs are rejected. `request` may contain ChatRequest
fields except `message`/`conversation_id`, which the runner manages. Expected invalid
business inputs should be written as user messages; structurally invalid ChatRequests
are rejected during suite loading. Expected exceptions use `outcome: "error"` and exact
`error_type`, and must leave the committed plan unchanged.

Available assertions:

| Field | Meaning |
|---|---|
| `outcome` | plan, revision, clarification, error |
| `destination`, `day_count` | Every itinerary day uses canonical city; exact schedule length |
| `output` | options or itinerary; options must have no schedule or selected city |
| `min_candidates`, `detailed_count` | Ranked candidate minimum; number of detailed analyses |
| `language` | Response language metadata (not prose language quality) |
| `constraints` | Exact subset of canonical constraint fields |
| `changed_days` | Exact days whose canonical records changed; `[]` means none |
| `preserve_other_days` | All prior days outside changed_days are identical |
| `preserve_evening` | Every prior evening activity retains its ID and full content; at least one must exist |
| `fewer_activities_days` | Each listed day has fewer activities than before |
| `activity_checks` | Count activities matching day, optional slot/category, min_count (default 1), max_count |
| `version_increment` | Exact increment from previous committed snapshot |
| `allow_fallback` | Explicitly permit live fallback; default false; offline always permits it |

Every returned plan also receives structural checks for unique activity IDs, consecutive
days, validation errors and evidence references/cities. Evidence-free placeholders are
accepted only when explicitly marked `needs_verification`. Checks compare canonical
snapshots, so translated presentation cannot hide unwanted mutations. They do not prove
factual entailment, logistics feasibility or natural-language quality.

## Artifacts and interpretation

Default output is a unique directory under `data/evaluations/` (gitignored):

- `report.json`: checks, inputs, responses, canonical before/after snapshots, model,
  token usage, timing, fallback flags and trajectory paths for each turn.
- `report.md`: compact case/repetition pass/fail table and failed check names.
- `trajectories/`: ordinary run JSON and event JSONL, including failed calls.

JSON is atomically replaced after each case and marked `running` until all cases finish.
An interrupted evaluation retains its partial report. Do not share one output directory
between simultaneous evaluators. Reports contain user text and plans; keys are redacted
using the standard trajectory cleaner.

Known estimated cost is a subtotal, not a bill: unknown prices and failed calls without
usage remain explicit (`unknown_cost_turns`), and recorded usage from failed trajectories
is retained where available. A stale-version rejection may also have unknown cost despite
making no model calls; inspect its trajectory. Assertions and model-generated content are
not identical things: a passing structural suite still needs manual or separate quality
evaluation for relevance, source support and translation. No LLM-as-judge is added here.
