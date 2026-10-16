"""
reranker.py — LLM-based chunk reranking for personalized retrieval (V5).

llm_rerank() is a dedicated LLM call that scores candidate chunks against
the user's taste_profile. This is the cross-encoder pattern:

  cosine similarity (bi-encoder) — fast, context-unaware
  LLM reranker (cross-encoder)  — one extra call, sees query + chunk + taste together

Called from tools.py after an initial cosine retrieval step.
Gracefully degrades to cosine order if the API key is absent or the call fails.
"""

from __future__ import annotations

import json

from backend.app.config import llm_api_key
from backend.app.llm.llm import generate_text


def llm_rerank(
    candidates: list[dict],
    query: str,
    taste_profile: str,
    k: int,
    raise_on_error: bool = False,
) -> list[dict]:
    """Score candidates by relevance to (query, taste_profile) via one LLM call.

    Each chunk gets a 0–10 score; chunks are re-sorted descending and the top k
    are returned. On any failure, returns the original cosine-ranked slice.
    """
    api_key = llm_api_key()
    if not api_key or not candidates:
        return candidates[:k]

    chunk_lines = "\n".join(
        f"[{i}] {c['destination']} / {c['section']}: "
        f"{c['text'][:200].replace(chr(10), ' ')}"
        for i, c in enumerate(candidates)
    )

    prompt = (
        f"User query: {query}\n"
        f"User taste profile: {taste_profile}\n\n"
        f"Rate each chunk 0–10 for how useful it is for this specific user.\n"
        f"Consider: does the content match their taste? Is it relevant to the query?\n"
        f"Reply with ONLY a JSON array of numbers, one per chunk, in order.\n"
        f"Example for 3 chunks: [7, 2, 9]\n\n"
        f"Chunks:\n{chunk_lines}"
    )

    try:
        response = generate_text(prompt, stage="chunk_reranker", max_tokens=256,
                                 prompt_version="chunk_reranker_v1")
        scores_raw = response.text
        scores = json.loads(scores_raw)
        for i, chunk in enumerate(candidates):
            chunk["rerank_score"] = float(scores[i]) / 10.0 if i < len(scores) else chunk["score"]
    except Exception:
        if raise_on_error:
            raise
        for chunk in candidates:
            chunk.setdefault("rerank_score", chunk["score"])

    return sorted(
        candidates,
        key=lambda c: c.get("rerank_score", c["score"]),
        reverse=True,
    )[:k]
