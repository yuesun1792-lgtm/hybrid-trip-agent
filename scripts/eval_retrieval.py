"""
scripts/eval_retrieval.py — measure retrieval quality for the knowledge index.

For each test case we ask:
  1. Did the right destination appear in top-k results?
  2. At what rank did it first appear?
  3. What did the top result actually say?

Three difficulty tiers reveal different failure modes:
  easy   — destination named in the query (tests basic recall)
  medium — destination implied but not named (tests semantic matching)
  hard   — pure semantic, no destination cue (tests embedding quality)

Run:  python scripts/eval_retrieval.py [--k 5]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.knowledge import search_travel_knowledge

# (query, expected_destination, difficulty, note)
EVAL_CASES = [
    # ── Easy: destination named ──────────────────────────────────────────────
    ("is Lisbon safe at night",                      "Lisbon",     "easy",   "safety"),
    ("what to eat in Prague",                        "Prague",     "easy",   "food"),
    ("best time to visit Venice",                    "Venice",     "easy",   "season"),
    ("hiking trails near Chamonix",                  "Chamonix",   "easy",   "activity"),
    ("Amsterdam cycling museums culture",            "Amsterdam",  "easy",   "culture"),
    ("Florence Uffizi art Renaissance",              "Florence",   "easy",   "art"),

    # ── Medium: destination implied ──────────────────────────────────────────
    ("alpine glacier hiking July high altitude",     "Chamonix",   "medium", "activity implied"),
    ("cheap medieval old town eastern Europe",       "Prague",     "medium", "budget+arch implied"),
    ("Greek island caldera romantic views",          "Santorini",  "medium", "scenery implied"),
    ("baroque architecture Mozart classical music",  "Salzburg",   "medium", "culture implied"),
    ("canal bridges cycling flat city",              "Amsterdam",  "medium", "landmark implied"),

    # ── Hard: pure semantic, no destination cue ──────────────────────────────
    ("pickpocketing warning crowded trams",          "Lisbon",     "hard",   "specific safety detail"),
    ("custard tarts pastry local bakery",            "Lisbon",     "hard",   "specific food"),
    ("salt mine world heritage lake village",        "Hallstatt",  "hard",   "specific landmark"),
    ("colosseum ancient ruins forum roman",          "Rome",       "hard",   "landmark no name"),
    ("meze seafood taverna aegean coast",            "Athens",     "hard",   "food culture no name"),
]


def rank_of(results: list[dict], expected_dest: str) -> int | None:
    for i, r in enumerate(results):
        if r["destination"].lower() == expected_dest.lower():
            return i + 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Top-k to retrieve (default 5)")
    parser.add_argument("--verbose", action="store_true", help="Print all top-k results")
    args = parser.parse_args()

    k = args.k
    hits: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    totals: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    hits_at: dict[int, int] = {1: 0, 3: 0, k: 0}

    col_q   = 46
    col_exp = 12
    col_r   = 5
    col_top = 38

    header = f"{'Query':<{col_q}} {'Expected':<{col_exp}} {'Rank':<{col_r}} {'Top-1 result (dest / section / score)'}"
    print(header)
    print("─" * len(header))

    for query, expected, difficulty, note in EVAL_CASES:
        results = search_travel_knowledge(query, k=k)
        rank = rank_of(results, expected)

        top1 = results[0] if results else None
        top1_str = (
            f"{top1['destination']} / {top1['section']} ({top1['score']:.2f})"
            if top1 else "—"
        )[:col_top]

        rank_str = str(rank) if rank else "—"
        marker = "✓" if rank == 1 else ("~" if rank and rank <= 3 else ("·" if rank else "✗"))

        print(f"{marker} {query:<{col_q-2}} {expected:<{col_exp}} {rank_str:<{col_r}} {top1_str}  [{difficulty}]")

        totals[difficulty] += 1
        if rank:
            hits[difficulty] += 1
            if rank <= 1: hits_at[1] += 1
            if rank <= 3: hits_at[3] += 1
            if rank <= k: hits_at[k] += 1

        if args.verbose and results:
            for i, r in enumerate(results[:k]):
                marker_r = ">>>" if r["destination"].lower() == expected.lower() else "   "
                print(f"   {marker_r} [{i+1}] {r['destination']} / {r['section']} ({r['score']:.2f})")
                print(f"         {r['text'][:120].replace(chr(10), ' ')}")
            print()

    total = len(EVAL_CASES)
    print()
    print("─" * 55)
    print(f"Recall@1 : {hits_at[1]:>2}/{total} = {hits_at[1]/total:>4.0%}")
    print(f"Recall@3 : {hits_at[3]:>2}/{total} = {hits_at[3]/total:>4.0%}")
    print(f"Recall@{k} : {hits_at[k]:>2}/{total} = {hits_at[k]/total:>4.0%}")
    print()
    print("By difficulty:")
    for tier in ("easy", "medium", "hard"):
        t = totals[tier]
        h = hits[tier]
        bar = "█" * h + "░" * (t - h)
        print(f"  {tier:<8} {bar}  {h}/{t}")
    print()
    print("Legend: ✓ rank-1  ~ top-3  · top-k  ✗ not found")


if __name__ == "__main__":
    main()
