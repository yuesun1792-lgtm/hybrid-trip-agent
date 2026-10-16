"""
knowledge.py — in-memory vector index for travel knowledge (V4).

Loads a prebuilt index from data/knowledge_index.npz and exposes
search_travel_knowledge() as a tool callable from the agent loop.

Everything that might need to be swapped for Chroma later lives here.
"""

from __future__ import annotations

import re

import numpy as np

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded on first call)
# ---------------------------------------------------------------------------

_index: dict | None = None        # {"embeddings": np.ndarray, "texts": list, "metadata": list}
_st_model = None                   # SentenceTransformer instance


def _load_index() -> dict | None:
    global _index
    if _index is not None:
        return _index
    from backend.app.config import KNOWLEDGE_INDEX_PATH
    if not KNOWLEDGE_INDEX_PATH.exists():
        return None
    data = np.load(str(KNOWLEDGE_INDEX_PATH), allow_pickle=True)
    _index = {
        "embeddings": data["embeddings"].astype(np.float32),
        "texts":      data["texts"].tolist(),
        "metadata":   data["metadata"].tolist(),
    }
    return _index


def _get_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def _embed_query(query: str) -> np.ndarray:
    """Embed a query string; monkeypatched in tests to avoid loading the model."""
    return _get_model().encode([query], show_progress_bar=False)[0].astype(np.float32)


_STOPWORDS = {
    "a", "an", "and", "are", "around", "at", "best", "day", "days", "do", "for",
    "from", "get", "how", "in", "is", "itinerary", "of", "on", "or", "the",
    "to", "travel", "trip", "what", "with",
}


def _query_terms(query: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", query.lower())
        if term not in _STOPWORDS
    }


def _lexical_search(
    query: str,
    texts: list[str],
    metadata: list[dict],
    k: int,
    destination: str | None,
) -> list[dict]:
    """Small no-model fallback over the stored corpus text."""
    terms = _query_terms(query)
    if destination:
        terms.add(destination.lower())
    scored = []
    dest_lower = destination.strip().lower() if destination else ""
    for index, text in enumerate(texts):
        meta = metadata[index]
        if dest_lower and meta.get("destination", "").lower() != dest_lower:
            continue
        haystack = f"{meta.get('destination', '')} {meta.get('section', '')} {text}".lower()
        overlap = sum(1 for term in terms if term in haystack)
        if overlap <= 0:
            continue
        scored.append((overlap / max(len(terms), 1), index))

    scored.sort(reverse=True)
    results = []
    for score, index in scored[:k]:
        results.append({
            "text": texts[index],
            "score": float(score),
            "destination": metadata[index].get("destination", ""),
            "section": metadata[index].get("section", ""),
            "url": metadata[index].get("url", ""),
        })
    return results


# ---------------------------------------------------------------------------
# Public search function
# ---------------------------------------------------------------------------

def search_travel_knowledge(
    query: str,
    k: int = 5,
    destination: str | None = None,
) -> list[dict]:
    """Cosine-rank corpus chunks against query; optionally filter by destination.

    Returns up to k dicts: {text, score, destination, section, url}.
    Returns [] if the index hasn't been built yet (graceful degradation).
    """
    if destination:
        from backend.app.domain.destination_names import canonical_destination
        destination = canonical_destination(destination)
    index = _load_index()
    if not index:
        return []

    embeddings: np.ndarray = index["embeddings"]   # (N, D)
    texts: list[str]       = index["texts"]
    metadata: list[dict]   = index["metadata"]

    try:
        query_vec = _embed_query(query)
    except Exception:
        return _lexical_search(query, texts, metadata, min(k, 10), destination)

    # Cosine similarity: dot product of L2-normalised vectors
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    q_norm    = np.linalg.norm(query_vec) + 1e-10
    sims = (embeddings / emb_norms) @ (query_vec / q_norm)  # (N,)

    # A fixed destination must never silently broaden to other cities.
    if destination:
        dest_lower = destination.strip().lower()
        mask = np.array(
            [m.get("destination", "").lower() == dest_lower for m in metadata]
        )
        if not mask.any():
            return []
        sims = np.where(mask, sims, -2.0)

    k = min(k, 10)
    top_idx = np.argsort(sims)[::-1][:k]

    results = []
    for i in top_idx:
        if sims[i] < -1.0:   # skip items masked out
            continue
        results.append({
            "text":        texts[i],
            "score":       float(sims[i]),
            "destination": metadata[i].get("destination", ""),
            "section":     metadata[i].get("section", ""),
            "url":         metadata[i].get("url", ""),
        })
    return results
