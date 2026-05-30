from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DESTINATION_NOTES_PATH = DATA_DIR / "destination_notes.json"
WIKIVOYAGE_SOURCES_PATH = DATA_DIR / "wikivoyage_sources.json"
KNOWLEDGE_INDEX_PATH = DATA_DIR / "knowledge_index.npz"

