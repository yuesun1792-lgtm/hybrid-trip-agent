"""Normalize structured trip fields before they reach planning workflows."""
import re
from typing import Any
from backend.app.domain.destination_names import canonical_destination


_CHINESE_DAYS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                 "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def normalize_trip_length(value: Any) -> int:
    if type(value) is int:
        result = value
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+|[一二两三四五六七八九十])\s*(?:days?|天|日(?:游)?)?\s*", value, re.IGNORECASE)
        if not match:
            raise ValueError("trip_length must be a positive whole number of days")
        number = match.group(1)
        result = int(number) if number.isdigit() else _CHINESE_DAYS[number]
    else:
        raise ValueError("trip_length must be a positive whole number of days")
    if result <= 0:
        raise ValueError("trip_length must be a positive whole number of days")
    return result


def normalize_trip_fields(fields: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(fields)
    if normalized.get("trip_length") not in (None, ""):
        normalized["trip_length"] = normalize_trip_length(normalized["trip_length"])
    for key in ("destination", "origin_city"):
        if isinstance(normalized.get(key), str):
            normalized[key] = canonical_destination(normalized[key])
    aliases = {"火车": "train", "飞机": "flight", "巴士": "bus", "汽车": "car", "轻松": "low", "适中": "medium", "紧凑": "high"}
    for key in ("transport_preference", "physical_level"):
        if isinstance(normalized.get(key), str):
            normalized[key] = aliases.get(normalized[key], normalized[key].lower())
    if "candidate_destinations" in normalized:
        names = normalized["candidate_destinations"]
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValueError("candidate_destinations must be an array of strings")
        normalized["candidate_destinations"] = list(dict.fromkeys(canonical_destination(name) for name in names))
    if isinstance(normalized.get("interests"), list):
        if any(not isinstance(item, str) for item in normalized["interests"]):
            raise ValueError("interests must contain strings")
        aliases = {"美食": "food", "文化": "culture", "时尚": "fashion", "艺术": "art", "历史": "history", "自然": "nature", "夜生活": "nightlife", "建筑": "architecture"}
        normalized["interests"] = list(dict.fromkeys(aliases.get(item, item) for item in normalized["interests"]))
    return normalized
