"""Stable destination names shared by field parsing, retrieval and cost lookup."""
import json
from functools import lru_cache
from backend.app.config import WIKIVOYAGE_SOURCES_PATH

DESTINATION_ALIASES = {
    "巴黎": "Paris",
    "罗马": "Rome",
    "威尼斯": "Venice",
    "佛罗伦萨": "Florence",
    "布拉格": "Prague",
    "维也纳": "Vienna",
    "里斯本": "Lisbon",
    "波尔图": "Porto",
    "巴塞罗那": "Barcelona",
    "阿姆斯特丹": "Amsterdam",
    "雅典": "Athens",
    "圣托里尼": "Santorini",
    "萨尔茨堡": "Salzburg",
    "慕尼黑": "Munich",
}

@lru_cache(maxsize=1)
def canonical_names():
    names = set(DESTINATION_ALIASES.values())
    try:
        names.update(item["destination"] for item in json.loads(WIKIVOYAGE_SOURCES_PATH.read_text()) if item.get("destination"))
    except (OSError, ValueError, TypeError):
        pass
    return {name.casefold(): name for name in names}


def canonical_destination(value: str) -> str:
    value = value.strip()
    return DESTINATION_ALIASES.get(value, canonical_names().get(value.casefold(), value))


COUNTRY_BY_DESTINATION = {
    "Lisbon": "Portugal",
    "Porto": "Portugal",
    "Algarve": "Portugal",
    "Barcelona": "Spain",
    "Seville": "Spain",
    "Paris": "France",
    "Nice": "France",
    "Chamonix": "France",
    "Provence": "France",
    "Rome": "Italy",
    "Venice": "Italy",
    "Florence": "Italy",
    "Cinque Terre": "Italy",
    "Athens": "Greece",
    "Santorini": "Greece",
    "Prague": "Czechia",
    "Vienna": "Austria",
    "Ljubljana": "Slovenia",
    "Salzburg": "Austria",
    "Hallstatt": "Austria",
    "Innsbruck": "Austria",
    "Amsterdam": "Netherlands",
}
