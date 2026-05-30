"""
knowledge.py — in-memory vector index for travel knowledge (V4).

Loads a prebuilt index from data/knowledge_index.npz and exposes
search_travel_knowledge() as a tool callable from the agent loop.

Everything that might need to be swapped for Chroma later lives here.
"""

from __future__ import annotations

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
    index = _load_index()
    if not index:
        return []

    embeddings: np.ndarray = index["embeddings"]   # (N, D)
    texts: list[str]       = index["texts"]
    metadata: list[dict]   = index["metadata"]

    query_vec = _embed_query(query)

    # Cosine similarity: dot product of L2-normalised vectors
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    q_norm    = np.linalg.norm(query_vec) + 1e-10
    sims = (embeddings / emb_norms) @ (query_vec / q_norm)  # (N,)

    # Optional destination filter — fall back to global search if nothing matches
    if destination:
        dest_lower = destination.strip().lower()
        mask = np.array(
            [m.get("destination", "").lower() == dest_lower for m in metadata]
        )
        if mask.sum() > 0:
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
