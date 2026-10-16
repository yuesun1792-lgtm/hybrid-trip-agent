"""Keep user-facing language separate from canonical English planning fields."""
import re

from backend.app.config import llm_api_key
from backend.app.llm import generate_text
from backend.app.models import ExecutionTraceStep


_LANGUAGE_NAMES = {"zh": "Chinese", "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "ja": "Japanese", "ko": "Korean", "ru": "Russian", "ar": "Arabic"}


def choose_language(message: str, previous: str = "en", explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if re.search(r"(?:用英文|英语回复|reply in English|answer in English)", message, re.I):
        return "en"
    if re.search(r"(?:用中文|中文回复|reply in Chinese|answer in Chinese)", message, re.I):
        return "zh"
    for code, pattern in [("ja", r"[\u3040-\u30ff]"), ("ko", r"[\uac00-\ud7af]"), ("zh", r"[\u3400-\u9fff]"), ("ru", r"[\u0400-\u04ff]"), ("ar", r"[\u0600-\u06ff]")]:
        if re.search(pattern, message):
            return code
    for code, pattern in [("fr", r"\b(bonjour|jours|voyage|je veux)\b"), ("de", r"\b(hallo|tage|reise|ich möchte)\b"), ("es", r"\b(hola|días|viaje|quiero)\b"), ("it", r"\b(ciao|giorni|viaggio|vorrei)\b"), ("en", r"\b(plan|days|please|want|from|change|revise)\b")]:
        if re.search(pattern, message, re.I):
            return code
    return previous  # Dates, numbers and city names should not change conversation language.


def language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code.split("-")[0], code)


def chinese_question(field: str) -> str:
    return {"origin_city": "你从哪个城市出发？", "date_range": "你计划在哪个月或哪些日期出行？",
            "start_date": "你计划哪天出发？", "trip_length": "你计划旅行几天？", "budget": "你的预算是多少欧元？",
            "interests": "你更喜欢美食、文化、艺术、自然还是其他活动？", "transport_preference": "你偏好火车、飞机还是巴士？",
            "physical_level": "你希望行程轻松、适中还是紧凑？", "destination": "你想去哪个目的地？", "revision_scope": "你希望修改哪一天或哪个活动？", "existing_plan": "请先告诉我需要规划的旅行，再修改行程。"}.get(field, "请补充相关出行信息。")


def render_language(response, target: str):
    """Translate prose only. Preserve field values, numbers, source excerpts, URLs and trace."""
    if target.startswith("en"):
        return response
    base = target.split("-")[0]
    if response.plan is None and response.missing_fields and base == "zh":
        return response.model_copy(update={"assistant_message": chinese_question(response.missing_fields[0])})
    if not llm_api_key():
        if base == "zh":
            message = "已生成行程；当前未启用语言转换，以下部分内容保留英文。" if response.plan else "请告诉我你的旅行安排。"
            return response.model_copy(update={"assistant_message": message})
        return response
    data = response.model_dump(mode="json")
    paths = [("assistant_message",)]
    if response.plan:
        paths += [("plan", key) for key in ("explanation", "final_recommendation")]
        if response.plan.place_resolution.status != "disabled":
            paths.append(("plan", "place_resolution", "message"))
        for key in ("risks", "tradeoff_summary"):
            paths += [("plan", key, i) for i in range(len(data["plan"][key]))]
        for i, day in enumerate(data["plan"]["itinerary"]):
            if not day.get("activities"):
                paths.append(("plan", "itinerary", i, "plan"))
            for j, activity in enumerate(day.get("activities", [])):
                activity["display_time_slot"] = activity["time_slot"].title()
                paths += [("plan", "itinerary", i, "activities", j, field) for field in ("title", "description", "display_time_slot")]
                if activity.get("place_resolution_reason"):
                    paths.append(("plan", "itinerary", i, "activities", j, "place_resolution_reason"))
        paths += [("plan", "validation_issues", i, "message") for i in range(len(data["plan"].get("validation_issues", [])))]
        for i, destination in enumerate(data["plan"]["ranked_destinations"]):
            for key in ("pros", "cons"):
                paths += [("plan", "ranked_destinations", i, key, j) for j in range(len(destination[key]))]
            if destination.get("tool_signals"):
                paths.append(("plan", "ranked_destinations", i, "tool_signals", "weather_note"))
    def get(path):
        value = data
        for part in path:
            value = value[part]
        return value
    paths = [p for p in paths if isinstance(get(p), str) and get(p)]
    texts = [get(p) for p in paths]
    import json
    schema = {"type": "object", "properties": {"texts": {"type": "array", "items": {"type": "string"}, "minItems": len(texts), "maxItems": len(texts)}}, "required": ["texts"], "additionalProperties": False}
    prompt = f"""Translate each text into {language_name(target)} ({target}). Return only JSON {{"texts": [...]}}.
Keep the same order and number of texts. Preserve meaning, facts, uncertainty and numbers.
Do not add activities or travel facts. Keep any text already in the target language unchanged.
Texts: {json.dumps(texts, ensure_ascii=False)}"""
    try:
        result = generate_text(prompt, stage="response_translation", max_tokens=3000,
                               prompt_version="response_translation_v1", response_schema=schema)
        translated = json.loads(result.text)["texts"]
        if len(translated) != len(paths) or any(not isinstance(t, str) or not t.strip() for t in translated):
            raise ValueError("Translation did not preserve text count")
        for path, text in zip(paths, translated):
            value = data
            for part in path[:-1]:
                value = value[part]
            value[path[-1]] = text
        if data.get("plan"):
            data["plan"]["display_language"] = target
        translated_response = type(response).model_validate(data)
        if translated_response.plan:
            from backend.app.plan_renderer import render_plan
            translated_response = translated_response.model_copy(update={"plan": render_plan(translated_response.plan, target)})
        return translated_response
    except Exception as exc:
        trace = [*response.trace, ExecutionTraceStep(stage="response_translation", status="fallback", detail=f"Could not render language={target}: {type(exc).__name__}; retained original content.")]
        message = "行程已生成，但语言转换暂时失败，部分内容保留原文。" if base == "zh" else response.assistant_message
        return response.model_copy(update={"trace": trace, "assistant_message": message})
