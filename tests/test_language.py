import json
import re

import numpy as np
import pytest

from backend.app.chat import extract_fields, handle_chat
from backend.app.field_validation import normalize_trip_fields
from backend.app.language import choose_language
from backend.app.llm import LLMResult
from backend.app.models import ChatRequest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAJECTORY_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("OPENAI_MODEL", "test")


def result(text):
    return LLMResult(text, "test", "openai", None, None)


@pytest.mark.parametrize("message,language", [("计划三日巴黎游", "zh"), ("3 days in Paris", "en"),
    ("3 jours à Paris", "fr"), ("3 Tage in Paris", "de"), ("3 días en París", "es"), ("京都へ旅行", "ja")])
def test_language_detection(message, language):
    assert choose_language(message) == language


def test_numeric_reply_preserves_language_and_override_is_respected():
    assert choose_language("3", "zh") == "zh"
    assert choose_language("2026-10-23", "zh") == "zh"
    assert choose_language("计划巴黎游，请用英文回复", "zh") == "en"
    assert choose_language("Paris", "en", explicit="fr") == "fr"


def test_chinese_fields_have_canonical_english_values():
    fields = extract_fields("计划三日巴黎游 美食 文化 时尚")
    assert fields["destination"] == "Paris"
    assert fields["trip_length"] == 3
    assert fields["interests"] == ["food", "culture", "fashion"]
    assert normalize_trip_fields({"destination": "巴黎", "interests": ["美食", "文化"]}) == {
        "destination": "Paris", "interests": ["food", "culture"]}
    assert "trip_length" not in extract_fields("10月23日")


def test_chinese_clarification_without_api_and_numeric_followup(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    first = handle_chat(ChatRequest(message="我想去巴黎"))
    second = handle_chat(ChatRequest(message="3", conversation_id=first.conversation_id))
    assert first.assistant_message == "你计划旅行几天？"
    assert second.response_language == "zh"
    assert second.extracted_fields["trip_length"] == 3
    assert second.plan is not None


def test_chinese_pipeline_keeps_rag_english_and_filters_other_cities(monkeypatch):
    import backend.app.strategy as strategy
    import backend.app.language as language
    captured = []
    monkeypatch.setattr(strategy, "generate_text", lambda prompt, **k: result(json.dumps({
        "intent": "plan_fixed_destination", "next_action": "run_fixed_itinerary",
        "known_fields": {"destination": "巴黎", "trip_length": "3 days", "interests": ["美食", "文化", "时尚"]},
        "queries": ["Paris food culture fashion", "巴黎美食"]})))
    def search(query, k=6, destination=None):
        captured.append((query, destination))
        return [{"text": "Paris - See\nMuseums and riverside walks offer visitors cultural activities throughout Paris.", "destination": "Paris", "section": "See", "url": "https://en.wikivoyage.org/wiki/Paris", "score": .9},
                {"text": "Rome - Eat\nLocal pasta.", "destination": "Rome", "section": "Eat", "url": "https://en.wikivoyage.org/wiki/Rome", "score": .5}]
    monkeypatch.setattr("backend.app.knowledge.search_travel_knowledge", search)
    def write(prompt, **kwargs):
        assert "language en" in prompt
        assert "Museums" in prompt
        assert "Local pasta" not in prompt
        from backend.app.models import EvidenceChunk
        ref = EvidenceChunk(text="Paris - See\nMuseums and riverside walks offer visitors cultural activities throughout Paris.", destination="Paris", section="See", url="https://en.wikivoyage.org/wiki/Paris").evidence_id
        return result(json.dumps({"days": [{"day": i, "activities": [{"title": "Museum", "description": "Visit a museum.", "category": "culture", "time_slot": "morning", "evidence_ids": [ref]}]} for i in range(1, 4)],
            "pros": ["Cultural activities"], "cons": [], "explanation": "Three days using guide evidence."}))
    monkeypatch.setattr("backend.app.plan_builder.generate_text", write)
    def translate(prompt, **kwargs):
        texts = json.loads(prompt.split("Texts: ", 1)[1])
        return result(json.dumps({"texts": [t if re.search(r"[\u3400-\u9fff]", t) else "中文说明：" + t for t in texts]}))
    monkeypatch.setattr(language, "generate_text", translate)
    response = handle_chat(ChatRequest(message="计划三日巴黎游 美食 文化 时尚"))
    assert response.response_language == "zh"
    assert response.extracted_fields["destination"] == "Paris"
    assert captured[-1] == ("Paris food culture fashion", "Paris")
    assert all(destination == "Paris" and query.isascii() for query, destination in captured)
    suggestion = response.plan.ranked_destinations[0]
    assert suggestion.country == "France"
    assert suggestion.estimated_cost > 0
    assert all(c.url.endswith("/Paris") for c in suggestion.citations)
    assert response.plan.itinerary[0].destination == "Paris"
    assert "中文说明" in response.plan.itinerary[0].plan
    from backend.app.conversation_state import get_conversation
    assert get_conversation(response.conversation_id).last_plan.itinerary[0].activities[0].title == "Museum"
    assert suggestion.citations[0].snippet.startswith("Museums")
    assert response.trajectory["language"]["retrieval_language"] == "en"


def test_unknown_destination_does_not_broaden_vector_search(monkeypatch):
    import backend.app.knowledge as knowledge
    monkeypatch.setattr(knowledge, "_load_index", lambda: {"embeddings": np.array([[1., 0.]]),
        "texts": ["Paris museums"], "metadata": [{"destination": "Paris", "section": "See"}]})
    monkeypatch.setattr(knowledge, "_embed_query", lambda query: np.array([1., 0.]))
    assert knowledge.search_travel_knowledge("Museums", destination="Unknown city") == []
    assert knowledge.search_travel_knowledge("Museums", destination="巴黎")[0]["destination"] == "Paris"


def test_translation_failure_keeps_original_facts_and_marks_fallback(monkeypatch):
    from backend.app.language import render_language
    from backend.app.models import ChatResponse
    def fail(*a, **k):
        raise ValueError("translation failed")
    monkeypatch.setattr("backend.app.language.generate_text", fail)
    response = render_language(ChatResponse(conversation_id="test", assistant_message="Plan ready"), "zh")
    assert response.trace[-1].status == "fallback"
    assert "语言转换" in response.assistant_message
