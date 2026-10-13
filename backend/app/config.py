import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DESTINATION_NOTES_PATH = DATA_DIR / "destination_notes.json"
WIKIVOYAGE_SOURCES_PATH = DATA_DIR / "wikivoyage_sources.json"
KNOWLEDGE_INDEX_PATH = DATA_DIR / "knowledge_index.npz"
DESTINATION_COSTS_PATH = DATA_DIR / "destination_costs.json"


ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def llm_rag_enhancements_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and env_flag(
        "ENABLE_LLM_RAG_ENHANCEMENTS",
        default=True,
    )


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def research_max_rounds() -> int:
    return _bounded_int("RAG_RESEARCH_MAX_ROUNDS", 3, 1, 3)


def research_queries_per_round() -> int:
    return _bounded_int("RAG_RESEARCH_QUERIES_PER_ROUND", 3, 1, 4)


def research_max_candidates() -> int:
    return _bounded_int("RAG_RESEARCH_MAX_CANDIDATES", 8, 3, 12)


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def llm_input_price_per_mtok() -> float:
    return _bounded_float("LLM_INPUT_PRICE_PER_MTOK", 1.0, 0.0, 1000.0)


def llm_output_price_per_mtok() -> float:
    return _bounded_float("LLM_OUTPUT_PRICE_PER_MTOK", 5.0, 0.0, 1000.0)


def llm_cache_write_price_per_mtok() -> float:
    return _bounded_float("LLM_CACHE_WRITE_PRICE_PER_MTOK", 1.25, 0.0, 1000.0)


def llm_cache_read_price_per_mtok() -> float:
    return _bounded_float("LLM_CACHE_READ_PRICE_PER_MTOK", 0.10, 0.0, 1000.0)
