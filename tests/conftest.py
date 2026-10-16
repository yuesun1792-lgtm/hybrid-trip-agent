"""Unit tests never use developer credentials or download embedding models."""
import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("ENABLE_PLACE_RESOLUTION", "false")
    monkeypatch.setenv("GEOAPIFY_API_KEY", "")
    monkeypatch.setenv("TRAJECTORY_DIR", str(tmp_path / "trajectories"))
    from backend.app.rag import knowledge
    def no_embedding_download():
        raise RuntimeError("Use injected embeddings or lexical fallback in unit tests")
    monkeypatch.setattr(knowledge, "_get_model", no_embedding_download)
