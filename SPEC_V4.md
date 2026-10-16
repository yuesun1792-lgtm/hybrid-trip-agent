# EuroTrip Agent — V4 Spec: Grounded Travel Knowledge (RAG)

## Goal

Replace the 4-entry toy data as the agent's source of *facts* with a real, retrievable
corpus built from Wikivoyage, and add a retrieval tool the agent can call. By the end, the
agent answers grounded in real data — season, safety, food, hikes, hidden gems — instead of
the toy JSON. This is the "knowledge / data" layer; personalization is V5.

This is one phase. Implement all of it, then verify against the acceptance criteria.

## Decisions (baked in — do not re-litigate)

- **Vector store:** in-memory cosine similarity, hand-rolled with numpy. Isolated to one
  module (`knowledge.py`) so it can later be swapped for Chroma without touching anything else.
- **Embeddings:** local `sentence-transformers` (model `all-MiniLM-L6-v2`). No API key, runs
  offline. (First run downloads ~80MB once.)
- **Corpus source:** a *curated, focused* set of Wikivoyage pages — the user's wishlist
  (Spain, Portugal, France incl. Mont Blanc & South of France, Italy incl. Rome & Venice,
  Greece) plus a handful of Munich-reachable gems. NOT all of Europe.
- **Chunking:** by Wikivoyage section (Understand, Get in, Get around, See, Do, Eat, Drink,
  Sleep, Stay safe), with a size cap. Keep destination name + section + source URL as metadata.
- **Attribution:** Wikivoyage is CC BY-SA. Store the source URL per chunk and make it available
  for attribution.

## Scope

### Ingest (one-time, offline — not on the request path)

- `data/wikivoyage_sources.json` — the curated list of destinations/pages to fetch (user-editable).
- `scripts/build_index.py` — fetch each page, parse into sections, chunk, embed with
  sentence-transformers, and save the index to `data/knowledge_index.npz` (or pickle):
  a list of `{chunk_text, embedding, metadata: {destination, section, url}}`.
  Run by the user once (and again when the source list changes). The request path never fetches or re-embeds.

### Retrieval (on the request path)

- `backend/app/rag/knowledge.py` — loads the saved index at startup and exposes:
  `search_travel_knowledge(query: str, k: int = 5, destination: str | None = None) -> list[dict]`
  → embed the query, cosine-rank against stored vectors, optional filter by destination,
  return top-k chunks with their metadata. **This is the only file that changes if you swap to Chroma.**

- `backend/app/tool_calling/tools.py` — add `SEARCH_TRAVEL_KNOWLEDGE_SCHEMA` to `TOOL_SCHEMAS`; wire
  `"search_travel_knowledge"` into `dispatch_tool`.

- `backend/app/llm/prompts.py` — system prompt tells the agent it can search travel knowledge to
  ground its recommendations (season, safety, food, hikes, etc.), and to prefer retrieved facts
  over its own memory.

- `backend/app/legacy/agent.py` — **loop unchanged.** The retrieval tool slots into the existing V3 loop.

- `requirements.txt` — add `sentence-transformers`, `numpy`, `requests` (or `httpx`) if not present.

### Tests (`tests/`)

- Test section-parsing and chunking on a small saved HTML fixture (no network).
- Test `search_travel_knowledge` against a tiny fixture index (a few hand-made chunks +
  precomputed/stubbed vectors) and assert relevant chunks come back for a known query.
- **Never download the model or hit the network in tests** — stub the embedding function or use
  a tiny fixture index.
- Test the no-key fallback path still returns a valid `TripPlan` without RAG.

## Invariants (must hold)

- `agent.py` loop unchanged — retrieval is just another tool.
- **No personalization yet** (that's V5): retrieval ranks by relevance only, taste-blind.
- App still works with no API key via the deterministic fallback (which does not use RAG).
- Numbers still come from tools/data, never invented by the LLM.
- The only new "real external" calls are fetching Wikivoyage during ingest (one-time) and local
  embedding. The request path stays offline-capable for retrieval (loads a local index).
- Response schema and frontend unchanged (beyond the agent now grounding its text).

## Out of scope (do NOT do in V4)

- Personalization / taste ranking / preference-based hidden-gem surfacing (V5).
- Multi-trip planning over the semester (V5+).
- Maps, routing, travel fund, conversation/chat (later).
- Swapping the vector store for Chroma (in-memory only for now).

## Acceptance criteria (definition of done)

1. `pytest tests/` passes (network and model stubbed in tests).
2. `python scripts/build_index.py` builds `data/knowledge_index.*` from the curated source list.
3. `search_travel_knowledge` returns relevant chunks for queries like "is Lisbon safe at night"
   or "best months to visit Venice" — from the right destination and section.
4. With an API key, a `POST /plan` run shows the agent calling `search_travel_knowledge`, and the
   plan/explanation reflects retrieved Wikivoyage facts (not invented).
5. Each retrieved chunk carries its source metadata (destination, section, URL) for attribution.
6. `agent.py`'s loop logic is unchanged.

## Suggested commit boundary

One commit (or a small series) ending with all acceptance criteria met and tests green.
The built `knowledge_index.*` may be committed or gitignored — your call; note which in the commit.
