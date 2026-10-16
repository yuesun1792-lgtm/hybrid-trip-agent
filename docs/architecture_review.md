# EuroTrip Agent — Architecture Review

**Scope:** read-only review of the working tree as of 2026-07-27 (branch `main`, on top of
commit `5498ac8`, plus a large uncommitted diff — see §5). No code was changed to produce this
document.

**Headline finding:** the working tree contains a materially more advanced system than either
`CLAUDE.md` or the committed git history describe. V5.0–V5.3 (state-based RAG discovery,
constraint ranking, tradeoff explanations) and V6 (chat endpoint) are already implemented and
wired into `/plan`, `/chat`, and the Streamlit UI — but none of this is committed, `CLAUDE.md`
still says "V4 in progress", and the test suite currently has **6 failing tests** (85 passing).

---

## 1. Data flow: form input → `TripPlan` output

There are three independent entry points that all converge on the same `TripPlan` shape. Each
is traced below in call order.

### 1a. Form path (`/plan`) — the primary, spec'd path

```
Streamlit form                     frontend/streamlit_app.py:88-137
  → POST /plan (TripRequest JSON)
FastAPI route                      backend/app/main.py:43 plan_trip()
  ├─ agent disabled or no key  →   backend/app/main.py:46,49 → _rag_or_deterministic_plan()
  │                                backend/app/main.py:27-35
  │     state_from_request()       backend/app/planning/planning_engine.py:9-18
  │     rag_plan_from_state()      backend/app/planning/planning_engine.py:47-51
  │       discover_candidate_destinations()  backend/app/rag/candidate_discovery.py:26-82
  │         build_discovery_queries()        backend/app/rag/candidate_discovery.py:7-23
  │         search_travel_knowledge()        backend/app/rag/knowledge.py:55-103  (loads data/knowledge_index.npz)
  │       rank_candidate_destinations()      backend/app/legacy/candidate_ranker.py:87-121
  │         score_candidate()                backend/app/legacy/candidate_ranker.py:19-84
  │           estimate_trip_cost()           backend/app/planning/costs.py:27-33 (data/destination_costs.json)
  │           estimate_transport_time()      backend/app/tool_calling/tools.py:34-43 (mock lookup table)
  │       build_trip_plan_from_ranked_state() backend/app/legacy/rag_plan_builder.py:133-174
  │         gather_tool_insights()            backend/app/tool_calling/tools.py:417-433
  │     on ANY exception in the RAG path →   deterministic_plan_from_state()
  │                                           backend/app/planning/planning_engine.py:21-34
  │       create_trip_plan()                 backend/app/legacy/planner.py:58-109 (data/destination_notes.json, 4 entries)
  │       generate_explanation()             backend/app/legacy/explainer.py:5-16 (template, no LLM)
  │
  └─ agent enabled + key present →  backend/app/legacy/agent.py:53 create_plan()
        _build_user_message()               backend/app/legacy/agent.py:35-52
        loop: client.messages.create(...)   backend/app/legacy/agent.py:84-91  (model "claude-haiku-4-5")
          tool_use → dispatch_tool()         backend/app/tool_calling/tools.py:394-410
            search_travel_knowledge | rerank_travel_knowledge | get_destination_budget | convert_currency
          tool_use == submit_trip_plan → TripPlan(**block.input)   backend/app/legacy/agent.py:119-135
            normalize_plan_costs()           backend/app/planning/costs.py:36-52
          end_turn / max_tokens / validation failure / iteration cap →
            _deterministic_fallback()        backend/app/legacy/agent.py:55-66 (same planner.py path as above)
TripPlan JSON returned                       backend/app/main.py:43 → FastAPI response_model=TripPlan
render_plan()                                 frontend/streamlit_app.py:13-83
```

Two structurally different planners can produce the response body: the **LLM agent loop**
(`legacy/agent.py`, tool-driven, no `score_breakdown`/`citations` in its output schema) and the
**RAG-ranked workflow** (`rag/candidate_discovery.py` → `legacy/candidate_ranker.py` → `legacy/rag_plan_builder.py`,
fixed pipeline, richer output). Which one runs is decided per-request by `main.py:27-56`, and the
response shape is the union of both (`TripPlan` carries fields only one path populates — see §3).

### 1b. Chat path (`/chat`) — V6, not V5

```
Streamlit chat tab                  frontend/streamlit_app.py:140-203
  → POST /chat (ChatRequest: message, conversation_id, extracted_fields)
FastAPI route                       backend/app/main.py:72-74 chat()
  handle_chat()                     backend/app/conversation/chat.py:122-148
    extract_fields()                backend/app/conversation/chat.py:31-83  (regex-based slot filling)
    missing_fields()                backend/app/conversation/chat.py:86-92
    if missing → ask one question, return (no plan yet)
    else → _trip_request_from_fields() → TripRequest
           state_from_request() → rag_plan_from_state()   (same RAG workflow as 1a, never the LLM agent)
ChatResponse JSON (conversation_id, extracted_fields, plan)
Streamlit stores conversation_id/extracted_fields in st.session_state and re-sends them next turn
```

There is no server-side session store. "State" survives only because the client (Streamlit's
`st.session_state`, or any API caller) echoes `extracted_fields`/`conversation_id` back on the
next call — the server is fully stateless per request.

### 1c. Debug paths

`POST /plan/candidates` (`main.py:59-62`) and `POST /plan/ranked-candidates` (`main.py:65-69`)
expose the discovery/ranking stages of 1a directly, returning `PlanningState` instead of
`TripPlan`, for inspection without going through the full pipeline.

### 1d. Offline ingest (not on any request path)

```
scripts/build_index.py  (run manually, never imported by the backend)
  fetch_wikitext()      scripts/build_index.py:47-75   (MediaWiki API, 22 destinations)
  parse_sections()      scripts/build_index.py:94-126  (chunk by Wikivoyage section)
  embed_chunks()        scripts/build_index.py:139-144  (sentence-transformers, local)
  → data/knowledge_index.npz   (gitignored — must be rebuilt on a fresh clone)
```

---

## 2. Concept mapping — agent-engineering concepts vs. this repo

| Concept | Status | Where / evidence |
|---|---|---|
| **Workflow** (fixed, code-defined control flow) | ✅ Implemented | `planning_engine.rag_plan_from_state` ([backend/app/planning/planning_engine.py:47-51](backend/app/planning/planning_engine.py#L47-L51)) is a hardcoded 3-step pipeline: discover → rank → build. No branching is decided by an LLM. Same for `deterministic_plan_from_state`. |
| **Agent** (LLM decides next action/tool in a loop) | ✅ Implemented, one path only | `agent.py:69-147` — the model chooses which tools to call, how many rounds, and when to stop (via `submit_trip_plan`). This is the *only* agentic path; it only runs when `ANTHROPIC_API_KEY` is set and `ENABLE_ANTHROPIC_AGENT` is true. |
| **RAG pipeline** (retrieve → augment → generate) | ✅ Implemented | `knowledge.py:55-103` (retrieve, in-memory cosine over `data/knowledge_index.npz`) feeds both the agent's `search_travel_knowledge` tool call and the non-agent `rag/candidate_discovery.py` workflow. |
| **Agentic RAG** (the *model* decides what/when to retrieve, not a fixed pipeline) | ⚠️ Partial — only in the LLM-agent path | In `legacy/agent.py`, retrieval is model-initiated (the prompt instructs it to call `search_travel_knowledge`/`rerank_travel_knowledge` itself — [backend/app/llm/prompts.py:16-24](backend/app/llm/prompts.py#L16-L24)). In the RAG-ranked workflow (used by `/chat`, the debug endpoints, and the no-key/no-agent fallback), retrieval queries are hardcoded template strings ([backend/app/rag/candidate_discovery.py:7-23](backend/app/rag/candidate_discovery.py#L7-L23)) — this is **RAG, not agentic RAG**, despite living in the same codebase and producing the same response schema. Worth being precise about this distinction when studying the code. |
| **Reranking (cross-encoder pattern)** | ✅ Implemented, LLM-agent path only | `reranker.py:22-72` — a dedicated Haiku call scores chunks against `taste_profile`; degrades to cosine order on failure or missing key. Not used by the non-agent RAG workflow (`rag/candidate_discovery.py` uses raw cosine `search_travel_knowledge` only, never `rerank_travel_knowledge`). |
| **Tool calling / tool-use loop** | ✅ Implemented | `agent.py:84-145`, tool schemas in `tools.py:222-360`, dispatch in `tools.py:394-410`. Textbook Anthropic tool-use loop with a terminal tool (`submit_trip_plan`), iteration cap (`_MAX_ITERATIONS = 16`, [agent.py:14](backend/app/legacy/agent.py#L14)), and a nudge-then-fallback safety net. |
| **State management** | ⚠️ Partial — schema exists, no persistence | `PlanningState` ([backend/app/domain/models.py:72-80](backend/app/domain/models.py#L72-L80)) models conversation stage, candidates, filtered candidates, evidence, and `rejected_destinations`. But it is rebuilt from scratch on every request (`state_from_request`, [planning_engine.py:9-18](backend/app/planning/planning_engine.py#L9-L18)) — there is no database, cache, or in-memory session store keyed by `conversation_id`. "State" is really "a struct the client must echo back." `rejected_destinations` is read in two places ([candidate_discovery.py:51](backend/app/rag/candidate_discovery.py#L51), [candidate_ranker.py:93](backend/app/legacy/candidate_ranker.py#L93)) but **never written anywhere** — the anti-repetition mechanism the field implies doesn't exist yet. |
| **Structured output / schema-constrained generation** | ✅ Implemented | Terminal-tool pattern: `submit_trip_plan`'s `input_schema` ([tools.py:242-298](backend/app/tool_calling/tools.py#L242-L298)) mirrors `TripPlan`; the tool call's `input` is validated by Pydantic ([agent.py:120-135](backend/app/legacy/agent.py#L120-L135)) with one retry on validation failure. |
| **Grounding / hallucination control** | ✅ Implemented, enforced by prompt + schema, not by code | `prompts.py:26-29` mandates every factual claim trace to retrieved text; `get_destination_budget` schema explicitly forbids inventing costs ([tools.py:224-229](backend/app/tool_calling/tools.py#L224-L229)). This is prompt-level discipline, not a hard guarantee — nothing validates post-hoc that `pros`/`cons` text actually appears in retrieved chunks. |
| **Evals** | ⚠️ Partial — retrieval only, not wired into CI/tests | `scripts/eval_retrieval.py` is a real, three-tier (easy/medium/hard) recall@k harness for the retriever — good practice. But it's a standalone script, not a pytest test, not run in any CI (there is no CI config in the repo at all), and there is **no** eval for plan *quality* (no LLM-as-judge, no golden trip plans, no check that citations/pros/cons are actually grounded). |
| **Multi-agent orchestration** | ❌ Not present (correctly, per spec — out of scope through V6) | Single agent loop, single reranker call. No handoffs, no sub-agents. |
| **MCP integration** | ❌ Not present (correctly, out of scope) | All tools are hand-rolled Python functions + Anthropic tool schemas. |

---

## 3. Deviations from `SPEC_V5.md`

`SPEC_V5.md` is the newest spec file (untracked) and the most accurate statement of intent. Comparing it line-by-line to the working tree:

### Spec promised, code delivers (on track)

- `PlanningState`, `CandidateDestination`, `EvidenceChunk` models — done ([models.py:47-80](backend/app/domain/models.py#L47-L80)).
- Thin `planning/planning_engine.py` boundary — done, and appropriately thin (51 lines, pure delegation).
- Deterministic fallback routed through `PlanningState` — done ([planning_engine.py:21-34](backend/app/planning/planning_engine.py#L21-L34)).
- RAG discovery from `knowledge_index.npz`, `destination_notes.json` demoted to fallback-only data — done. `destination_notes.json` is now read only by the deterministic path and the orphaned V3 tools (see below), never by the RAG workflow.
- Debug endpoints for discovery and ranking — done (`/plan/candidates`, `/plan/ranked-candidates`).
- Constraint-aware ranking with hard/soft/context breakdown — done ([candidate_ranker.py:19-84](backend/app/legacy/candidate_ranker.py#L19-L84)).
- Tradeoff summaries, filtered candidates, score breakdowns surfaced in `TripPlan` and the UI — done ([rag_plan_builder.py:104-130](backend/app/legacy/rag_plan_builder.py#L104-L130), [streamlit_app.py:32-41](frontend/streamlit_app.py#L32-L41)).

### Spec promised, code does not fully deliver

- **Anti-repetition (V5.3)** — spec explicitly lists this as in-scope. `rejected_destinations` exists in the schema and is *read* in two ranking/discovery spots, but nothing in the codebase ever *writes* it. There is no mechanism that actually avoids re-recommending a destination the user already rejected in an earlier turn. This is a stub, not an implementation.
- **`CandidateDestination.country`** is never populated. `candidate_discovery.py:62-70` constructs candidates with only `name`, `source`, `score`, `rationale`, `evidence` — no `country`. `rag_plan_builder._country_from_url` ([rag_plan_builder.py:17-19](backend/app/legacy/rag_plan_builder.py#L17-L19)) falls back to the literal string `"Unknown"`. Every RAG-ranked plan (the fallback path, `/chat`, and both debug endpoints) will show `"Unknown"` as the country for every destination. This isn't in the spec's "out of scope" list — it's a plain gap.
- **Citations/attribution only reach the RAG-ranked workflow, not the LLM agent.** SPEC_V4 promised source attribution for every retrieved chunk; `rag_plan_builder._citations_from_candidate` ([rag_plan_builder.py:28-46](backend/app/legacy/rag_plan_builder.py#L28-L46)) delivers this — but `submit_trip_plan`'s tool schema ([tools.py:250-276](backend/app/tool_calling/tools.py#L250-L276)) has no `citations` or `score_breakdown` property, so the LLM-agent path (the one used whenever an API key is present, i.e. the "main" experience) can never return them. The richer, more transparent output is only reachable via the *fallback* code path.

### Code delivers what the spec doesn't mention (spec is behind the code)

- **The `/chat` endpoint and Streamlit chat tab are fully implemented** ([chat.py](backend/app/conversation/chat.py), [main.py:72-74](backend/app/main.py#L72-L74), [streamlit_app.py:140-203](frontend/streamlit_app.py#L140-L203)) — but `SPEC_V5.md` lists "Chat endpoint" under **"Out Of Scope For V5.0"** and describes it as V6, "a later layer." The code has already jumped a full version ahead of the spec that's supposed to be governing current work.
- **Four V3-era tools are now dead code from the LLM's perspective.** `list_destinations`, `get_destination_details`, `check_budget_fit`, and `score_destinations` still exist in `tool_calling/tools.py` (lines 161-219) and are still unit-tested directly, but none of them appear in `TOOL_SCHEMAS` ([tools.py:222-360](backend/app/tool_calling/tools.py#L222-L360)) and `dispatch_tool` ([tools.py:394-410](backend/app/tool_calling/tools.py#L394-L410)) has no branch for their names — the agent cannot call them even if it wanted to. This is an undocumented, silent deprecation; neither `SPEC_V3.md` nor `CLAUDE.md` nor any commit message marks them as retired.
- **`/plan` without a key or with the agent disabled no longer means "deterministic, no RAG."** `CLAUDE.md`'s own invariant says *"App still works with no API key via the deterministic fallback (no RAG on that path)"* (a V4-era invariant). The current code tries the RAG-ranked workflow **first** even with no key ([main.py:27-35](backend/app/main.py#L27-L35)) — which is a reasonable, arguably better, evolution (SPEC_V5 explicitly wants this), but it directly contradicts a standing invariant that nothing has removed, and it also runs a local embedding model (sentence-transformers) on every no-key request, which changes the no-key path's dependency/latency profile.

---

## 4. Risk list (most to least severe)

1. **The test suite is currently red: 6 of 91 tests fail (`pytest tests/ -q`).** This is not a hypothetical risk, it's the present state of the branch. Concretely:
   - `test_plan_trip_marks_fallback_when_agent_disabled` and `test_plan_trip_marks_fallback_when_key_missing` fail because they assert `planner_mode == "deterministic_fallback"`, but the code now legitimately returns `"rag_ranked"` first (see §3, last bullet). These are **stale tests describing old, superseded behavior** — they need to be rewritten to match the intended new contract, or the new contract needs to be reconsidered; right now nothing distinguishes "expected change" from "regression" for whoever runs `pytest` next.
   - `test_score_candidate_adds_breakdown` and `test_tool_signals_influence_score_deterministically` fail from **fixture bugs**, not production bugs: the first passes `{"Salzburg": 700}` as a *daily* cost (per `planning/costs.py`'s schema) when the test's own comment implies it meant a €700 *total* for a 4-day trip — `estimate_trip_cost` correctly multiplies 700×4=2800 and blows the budget, producing a large negative score the assertion doesn't expect. The second sets `transport_mode="flight"` in its fixture while `transport_preference="train"` in the request, so the "match" bonus the test's comment describes never fires.
   - `test_discover_candidate_destinations_groups_rag_evidence` fails because its mock (`lambda query, k=6: fake_results`) returns the *same* 3 chunks regardless of query text; `build_discovery_queries` now issues 2 queries, so the mock's duplicate chunks collide during dedup and Salzburg ends up with 2 unique evidence chunks instead of the expected 3 — an artifact of the fixture not varying by query, likely left behind by the "discovery batching" change in the most recent commit.
   - Net effect: **anyone who runs the test suite right now cannot tell, from the test output alone, whether the codebase is healthy.** All 6 failures happen to be test-side, but that's exactly the situation where a real regression would also go unnoticed.

2. **~950 lines across 8 new/changed backend files are uncommitted.** `git status` shows the entire V5.0–V5.3 candidate-discovery/ranking/plan-building pipeline plus the V6 chat endpoint as either modified-uncommitted or untracked (`rag/candidate_discovery.py`, `legacy/candidate_ranker.py`, `conversation/chat.py`, `planning/costs.py`, `planning/planning_engine.py`, `legacy/rag_plan_builder.py`, `scripts/eval_retrieval.py`, `SPEC_V5.md`, plus large diffs in `legacy/agent.py`, `main.py`, `domain/models.py`, `tool_calling/tools.py`, `streamlit_app.py`, `test_planner.py`). If this machine's working tree were lost, all of V5/V6 would be lost with it — there is no commit boundary to recover to. Combined with risk #1, this is uncommitted *and* failing.

3. **`CandidateDestination.country` is never populated**, so every RAG-ranked `TripPlan` (the fallback path, `/chat`, both debug endpoints — i.e. most traffic once V5 ships) shows `"Unknown"` as the country for every destination. Low code complexity to fix (thread country through from `wikivoyage_sources.json` or infer from Wikivoyage metadata), but currently a visible, user-facing defect in the "new" primary path.

4. **Model-string inconsistency between the two Anthropic call sites.** `agent.py:86` calls `model="claude-haiku-4-5"` while `reranker.py:56` calls `model="claude-haiku-4-5-20251001"` — the same nominal model referenced two different ways in the same codebase. If the undated alias ever stops resolving (Anthropic's aliases are not guaranteed to be permanent), the main agent loop breaks while the reranker keeps working, or vice versa if the pinned dated snapshot is deprecated first. At minimum this should be one constant, not two independent literals.

5. **`data/knowledge_index.npz` is gitignored**, so a fresh clone has no retrieval corpus until someone manually runs `scripts/build_index.py` (which fetches live Wikivoyage pages and downloads a ~80MB embedding model). Until that happens, `search_travel_knowledge` silently returns `[]` ([knowledge.py:66-67](backend/app/rag/knowledge.py#L66-L67)), `discover_candidate_destinations` returns zero candidates, and `rag_plan_from_state` raises `ValueError("No ranked candidates available")` — which *is* caught and falls back to the deterministic planner ([main.py:29-35](backend/app/main.py#L29-L35)), so nothing crashes, but a new contributor gets silently degraded (Unknown-country-free, but generic 4-destination) output with no signal that the corpus was never built. Worth an explicit startup check or log line.

6. **The LLM-agent path (the one that runs whenever a real API key is configured) cannot return `citations` or `score_breakdown`**, because `submit_trip_plan`'s input schema doesn't ask for them (§3). This means the most "production-like" path — real LLM, real tool calls — is *less* transparent about its reasoning than the fallback path, which is backwards from what you'd want if citations/tradeoffs are meant to build user trust.

7. **No persistence for `PlanningState` means `rejected_destinations` (and any multi-turn "don't suggest that again" behavior) is architecturally impossible right now**, not just unimplemented — there's no `conversation_id`-keyed store to hang it off, and the client only echoes `extracted_fields`, not full state. Anti-repetition (V5.3) can't be finished without first deciding where state lives (server-side cache/db vs. fully client-echoed).

8. **Four legacy tool functions (`list_destinations`, `get_destination_details`, `check_budget_fit`, `score_destinations`) are dead from the agent's perspective but still tested and still read `destination_notes.json`.** Low severity — they're harmless — but they add maintenance surface and reader confusion (a newcomer reading `TOOL_SCHEMAS` has no way to know these four functions in the same file used to be tools and no longer are, since nothing marks the change).

9. **No CI configuration exists anywhere in the repo** (no `.github/workflows`, no other CI). The failing tests in risk #1 would not be caught automatically on push; they were only found by running `pytest` manually as part of this review.

10. **`eval_retrieval.py` has no pass/fail threshold and isn't invoked by `pytest` or any automation** — it's a good manual diagnostic but doesn't function as a regression gate. If a future change to chunking or the embedding model quietly degrades recall, nothing will flag it.

11. **Minor:** `README.md`'s "Project Structure" section still lists `data/user_preferences.json` and a V0-era file tree; `data/user_preferences.json` itself is present in the repo but is not read by any code (`grep` finds zero references) — it appears to be dead data left over from an earlier iteration.

---

## 5. Notes on method

This review was produced by reading every backend, frontend, script, and test file in the
repository (not sampling), reading `SPEC_V2` through `SPEC_V5`, diffing the working tree against
`HEAD`, and running `pytest tests/ -q` once to establish current pass/fail ground truth. No files
were modified. Line numbers refer to the working tree at review time; if the branch is committed
or further edited, line numbers will drift — treat them as pointers to re-locate, not permanent
anchors.
