"""
scripts/build_index.py — one-time offline ingest for V4 RAG.

  python scripts/build_index.py [--dry-run] [--sources data/wikivoyage_sources.json]

Fetches each Wikivoyage page via the MediaWiki API (wikitext), splits it
into per-section chunks, embeds them with sentence-transformers, and saves
the index to data/knowledge_index.npz.

Run again whenever wikivoyage_sources.json changes.
Corpus: CC BY-SA — source URLs are stored in metadata for attribution.
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.config import KNOWLEDGE_INDEX_PATH, WIKIVOYAGE_SOURCES_PATH

MEDIAWIKI_API = "https://en.wikivoyage.org/w/api.php"
HEADERS = {"User-Agent": "eurotrip-agent/1.0 (educational project)"}

# Sections we care about (case-insensitive match against wikitext headings)
KEEP_SECTIONS = {
    "understand", "get in", "get around", "see", "do",
    "eat", "drink", "sleep", "stay safe", "stay healthy",
    "respect", "connect", "go next",
}

MAX_CHUNK_WORDS = 400   # split oversized sections into smaller pieces


# ── Wikitext helpers ─────────────────────────────────────────────────────────

def fetch_wikitext(title: str, retries: int = 3) -> str | None:
    params = urllib.parse.urlencode({
        "action":    "query",
        "prop":      "revisions",
        "titles":    title,
        "rvslots":   "main",
        "rvprop":    "content",
        "formatversion": "2",
        "format":    "json",
    })
    url = f"{MEDIAWIKI_API}?{params}"
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.load(resp)
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                return None
            rev = pages[0].get("revisions", [{}])[0]
            return rev.get("slots", {}).get("main", {}).get("content")
        except Exception as exc:
            wait = 5 * (attempt + 1)
            if attempt < retries - 1:
                print(f"  [retry {attempt+1}] {exc} — waiting {wait}s …", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [warn] fetch failed after {retries} attempts: {exc}", file=sys.stderr)
    return None


def clean_wikitext(text: str) -> str:
    text = re.sub(r"<ref[^>]*/?>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/?>", " ", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)          # {{templates}}
    text = re.sub(r"\{\{[^{}]*\{\{[^{}]*\}\}[^{}]*\}\}", " ", text)  # nested
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)    # [[link|text]]
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)        # [url text]
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"'{2,5}", "", text)                   # bold/italic
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)                 # remaining HTML
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def parse_sections(wikitext: str, destination: str, url: str) -> list[dict]:
    """Split wikitext into section chunks with metadata."""
    # Split on level-2 headings: == Section ==
    parts = re.split(r"\n==([^=\n]+)==", wikitext)
    chunks = []

    for i in range(1, len(parts), 2):
        section_raw = parts[i].strip()
        body_raw    = parts[i + 1] if i + 1 < len(parts) else ""

        section_key = section_raw.lower()
        if KEEP_SECTIONS and not any(k in section_key for k in KEEP_SECTIONS):
            continue

        body = clean_wikitext(body_raw)
        if len(body.split()) < 20:          # skip near-empty sections
            continue

        # Strip level-3 sub-headings but keep their content
        body = re.sub(r"===+[^=\n]+===+", "", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()

        words = body.split()
        if len(words) <= MAX_CHUNK_WORDS:
            chunks.append(_make_chunk(destination, section_raw, url, body))
        else:
            # Split oversized sections into MAX_CHUNK_WORDS-word pieces
            for start in range(0, len(words), MAX_CHUNK_WORDS):
                piece = " ".join(words[start: start + MAX_CHUNK_WORDS])
                if len(piece.split()) >= 20:
                    chunks.append(_make_chunk(destination, section_raw, url, piece))

    return chunks


def _make_chunk(destination: str, section: str, url: str, body: str) -> dict:
    text = f"{destination} — {section}\n{body}"
    return {
        "text":     text,
        "metadata": {"destination": destination, "section": section, "url": url},
    }


# ── Embedding ────────────────────────────────────────────────────────────────

def embed_chunks(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"  Embedding {len(texts)} chunks …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True).astype(np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse only, skip embedding and saving")
    parser.add_argument("--sources", default=str(WIKIVOYAGE_SOURCES_PATH))
    parser.add_argument("--only", nargs="+", metavar="DEST",
                        help="Only fetch/embed these destinations (append to existing index)")
    args = parser.parse_args()

    sources_path = Path(args.sources)
    if not sources_path.exists():
        sys.exit(f"Sources file not found: {sources_path}")

    sources = json.loads(sources_path.read_text())
    if args.only:
        only_set = {d.lower() for d in args.only}
        sources = [s for s in sources if s["destination"].lower() in only_set]
    print(f"Sources: {len(sources)} destinations")

    # Load existing index so we can append
    existing_texts:     list[str]  = []
    existing_metadata:  list[dict] = []
    existing_embeddings: np.ndarray | None = None
    if args.only and KNOWLEDGE_INDEX_PATH.exists():
        print("Loading existing index for append …")
        data = np.load(str(KNOWLEDGE_INDEX_PATH), allow_pickle=True)
        existing_embeddings = data["embeddings"].astype(np.float32)
        existing_texts      = data["texts"].tolist()
        existing_metadata   = data["metadata"].tolist()
        print(f"  {len(existing_texts)} existing chunks")

    all_chunks: list[dict] = []

    for entry in sources:
        dest = entry["destination"]
        url  = entry["url"]
        title = url.split("/wiki/")[-1]
        print(f"\n[{dest}] fetching …", end=" ", flush=True)

        wikitext = fetch_wikitext(title)
        if not wikitext:
            print("SKIPPED (fetch failed)")
            continue

        chunks = parse_sections(wikitext, dest, url)
        print(f"{len(chunks)} chunks")
        all_chunks.extend(chunks)
        time.sleep(1.0)   # be polite to the API

    print(f"\nTotal chunks: {len(all_chunks)}")

    if args.dry_run:
        print("Dry run — not saving.")
        return

    if not all_chunks:
        sys.exit("No chunks produced — nothing to save.")

    new_texts    = [c["text"]     for c in all_chunks]
    new_metadata = [c["metadata"] for c in all_chunks]
    new_embeddings = embed_chunks(new_texts)

    # Merge with existing if appending
    if existing_embeddings is not None and len(existing_texts) > 0:
        final_embeddings = np.concatenate([existing_embeddings, new_embeddings], axis=0)
        final_texts      = existing_texts + new_texts
        final_metadata   = existing_metadata + new_metadata
    else:
        final_embeddings = new_embeddings
        final_texts      = new_texts
        final_metadata   = new_metadata

    KNOWLEDGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(KNOWLEDGE_INDEX_PATH),
        embeddings=final_embeddings,
        texts=np.array(final_texts,    dtype=object),
        metadata=np.array(final_metadata, dtype=object),
    )
    print(f"\nSaved → {KNOWLEDGE_INDEX_PATH}  ({final_embeddings.shape[0]} chunks, dim {final_embeddings.shape[1]})")


if __name__ == "__main__":
    main()
