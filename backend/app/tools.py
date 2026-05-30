"""
Mock tool functions for V2 enrichment.

Each function is a plain Python lookup — no network calls, fully deterministic.
gather_tool_insights() calls all three and returns a ToolInsight for one destination.
"""

from backend.app.models import ToolInsight
from .currency_tool import convert_currency, CONVERT_CURRENCY_SCHEMA

# ---------------------------------------------------------------------------
# Tool 1: transport time
# ---------------------------------------------------------------------------

# (origin_lowercase, destination) → (hours, mode)
_TRANSPORT: dict[tuple[str, str], tuple[float, str]] = {
    ("munich", "Lisbon"):    (3.0, "flight"),
    ("munich", "Prague"):    (4.5, "train"),
    ("munich", "Ljubljana"): (4.0, "train"),
    ("munich", "Amsterdam"): (5.5, "train"),
}

# Fallback by destination only — used when origin is not in the table
_TRANSPORT_FALLBACK: dict[str, tuple[float, str]] = {
    "Lisbon":    (3.5, "flight"),
    "Prague":    (3.0, "train"),
    "Ljubljana": (3.5, "train"),
    "Amsterdam": (3.5, "train"),
}

_TRANSPORT_DEFAULT: tuple[float, str] = (4.0, "train")


def estimate_transport_time(origin: str, destination: str) -> dict:
    """Return estimated one-way travel time from origin to destination."""
    key = (origin.strip().lower(), destination)
    if key in _TRANSPORT:
        hours, mode = _TRANSPORT[key]
    elif destination in _TRANSPORT_FALLBACK:
        hours, mode = _TRANSPORT_FALLBACK[destination]
    else:
        hours, mode = _TRANSPORT_DEFAULT
    return {"hours": hours, "mode": mode}


# ---------------------------------------------------------------------------
# Tool 2: weather risk
# ---------------------------------------------------------------------------

# (destination, month_3char_lowercase) → (risk, note)
# risk is one of: "low", "medium", "high"
_WEATHER: dict[tuple[str, str], tuple[str, str]] = {
    ("Lisbon", "jan"): ("medium", "Mild but rainy, fewer crowds"),
    ("Lisbon", "feb"): ("medium", "Mild and rainy"),
    ("Lisbon", "mar"): ("low",    "Pleasant spring weather"),
    ("Lisbon", "apr"): ("low",    "Warm and sunny"),
    ("Lisbon", "may"): ("low",    "Warm and sunny"),
    ("Lisbon", "jun"): ("low",    "Warm and sunny, ideal for sightseeing"),
    ("Lisbon", "jul"): ("low",    "Hot and dry, very popular season"),
    ("Lisbon", "aug"): ("low",    "Hot and dry, still very popular"),
    ("Lisbon", "sep"): ("low",    "Still warm, less crowded"),
    ("Lisbon", "oct"): ("medium", "Warm but some rain possible"),
    ("Lisbon", "nov"): ("medium", "Mild but rainy"),
    ("Lisbon", "dec"): ("medium", "Mild but rainy, fewer crowds"),

    ("Prague", "jan"): ("high",   "Cold and icy, limited outdoor activities"),
    ("Prague", "feb"): ("high",   "Cold, some snow possible"),
    ("Prague", "mar"): ("medium", "Cool and variable"),
    ("Prague", "apr"): ("low",    "Mild spring weather"),
    ("Prague", "may"): ("low",    "Warm and pleasant"),
    ("Prague", "jun"): ("low",    "Warm and mostly dry"),
    ("Prague", "jul"): ("low",    "Warm with occasional thunderstorms"),
    ("Prague", "aug"): ("low",    "Warm with occasional thunderstorms"),
    ("Prague", "sep"): ("low",    "Mild and dry"),
    ("Prague", "oct"): ("medium", "Cool and foggy"),
    ("Prague", "nov"): ("medium", "Cold and grey"),
    ("Prague", "dec"): ("high",   "Cold and icy"),

    ("Ljubljana", "jan"): ("high",   "Cold with possible snow"),
    ("Ljubljana", "feb"): ("high",   "Cold, snowy"),
    ("Ljubljana", "mar"): ("medium", "Cool and variable"),
    ("Ljubljana", "apr"): ("medium", "Mild but showery"),
    ("Ljubljana", "may"): ("low",    "Warm with some rain"),
    ("Ljubljana", "jun"): ("medium", "Warm but occasional afternoon showers"),
    ("Ljubljana", "jul"): ("low",    "Warm and sunny"),
    ("Ljubljana", "aug"): ("low",    "Warm and sunny"),
    ("Ljubljana", "sep"): ("low",    "Pleasant and mild"),
    ("Ljubljana", "oct"): ("medium", "Cool with some rain"),
    ("Ljubljana", "nov"): ("medium", "Cool and rainy"),
    ("Ljubljana", "dec"): ("high",   "Cold with possible snow"),

    ("Amsterdam", "jan"): ("high",   "Cold, grey, and rainy"),
    ("Amsterdam", "feb"): ("high",   "Cold and windy"),
    ("Amsterdam", "mar"): ("medium", "Cool and variable"),
    ("Amsterdam", "apr"): ("medium", "Mild but unpredictable"),
    ("Amsterdam", "may"): ("low",    "Mild and pleasant"),
    ("Amsterdam", "jun"): ("medium", "Mild but unpredictable, bring a layer"),
    ("Amsterdam", "jul"): ("low",    "Warmest month with some rain"),
    ("Amsterdam", "aug"): ("low",    "Warm with occasional showers"),
    ("Amsterdam", "sep"): ("medium", "Cooling down, more rain"),
    ("Amsterdam", "oct"): ("medium", "Cool and rainy"),
    ("Amsterdam", "nov"): ("high",   "Cold and rainy"),
    ("Amsterdam", "dec"): ("high",   "Cold, grey, and rainy"),
}

_WEATHER_DEFAULT: tuple[str, str] = (
    "medium",
    "Weather data not available for this destination and period",
)


def _month_key(date_range: str) -> str:
    """Extract a 3-char lowercase month key from strings like 'June 10-15' or 'June'."""
    return date_range.strip().lower()[:3]


def estimate_weather_risk(destination: str, date_range: str) -> dict:
    """Return weather risk level and a short note for the destination and travel period."""
    key = (destination, _month_key(date_range))
    risk, note = _WEATHER.get(key, _WEATHER_DEFAULT)
    return {"risk": risk, "note": note}


# ---------------------------------------------------------------------------
# Tool 3: total cost estimate
# ---------------------------------------------------------------------------

# Daily per-person costs in euros, broken down by category
_DAILY_COSTS: dict[str, dict[str, int]] = {
    "Lisbon":    {"accommodation": 120, "food": 40, "local_transport": 8},
    "Prague":    {"accommodation": 80,  "food": 25, "local_transport": 5},
    "Ljubljana": {"accommodation": 90,  "food": 30, "local_transport": 5},
    "Amsterdam": {"accommodation": 180, "food": 50, "local_transport": 10},
}

_DAILY_COST_DEFAULT: dict[str, int] = {
    "accommodation": 100,
    "food": 35,
    "local_transport": 8,
}


def estimate_total_cost(destination: str, trip_length_days: int) -> dict:
    """Return a cost estimate scaled by trip length, with a per-category breakdown."""
    daily = _DAILY_COSTS.get(destination, _DAILY_COST_DEFAULT)
    breakdown = {category: rate * trip_length_days for category, rate in daily.items()}
    return {"total": sum(breakdown.values()), "breakdown": breakdown}


# ---------------------------------------------------------------------------
# V2 tools: destination details + budget fit (exposed to the LLM via TOOL_SCHEMAS)
# ---------------------------------------------------------------------------

def _load_destination_notes() -> list[dict]:
    import json
    from backend.app.config import DESTINATION_NOTES_PATH
    with DESTINATION_NOTES_PATH.open() as f:
        return json.load(f)


def get_destination_details(name: str) -> dict:
    """Return the full destination record from destination_notes.json, or an error dict."""
    for dest in _load_destination_notes():
        if dest["name"].lower() == name.strip().lower():
            return dest
    return {"error": "unknown destination", "name": name}


def check_budget_fit(name: str, budget: float) -> dict:
    """Return whether the destination's estimated cost fits the budget."""
    details = get_destination_details(name)
    if "error" in details:
        return details
    estimated_cost = details["estimated_cost"]
    return {
        "name": details["name"],
        "estimated_cost": estimated_cost,
        "budget": budget,
        "fits": estimated_cost <= budget,
        "delta": budget - estimated_cost,
    }


def list_destinations() -> list[dict]:
    """Return name, country, and a one-line summary for every available destination."""
    return [
        {
            "name": d["name"],
            "country": d["country"],
            "summary": f"{', '.join(d['interests'][:3])}; ~€{d['estimated_cost']}",
        }
        for d in _load_destination_notes()
    ]


def score_destinations(request: dict) -> list[dict]:
    """Score all destinations using the deterministic scorer; return them ranked best-first."""
    from backend.app.models import TripRequest
    from backend.app.planner import load_destination_notes as _load_notes, score_destination

    try:
        trip_request = TripRequest(**request)
    except Exception as exc:
        return [{"error": f"invalid request: {exc}"}]

    destinations = _load_notes()
    results = [
        {
            "name": d["name"],
            "country": d["country"],
            "score": score_destination(d, trip_request),
            "estimated_cost": d["estimated_cost"],
            "pros": d["pros"],
            "cons": d["cons"],
            "interests": d["interests"],
        }
        for d in destinations
    ]
    return sorted(results, key=lambda x: x["score"], reverse=True)


TOOL_SCHEMAS = [
    {
        "name": "list_destinations",
        "description": "List all available European destinations with a brief summary of interests and cost.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_destination_details",
        "description": (
            "Get full details for a European destination: estimated cost, interests, "
            "transport options, physical level, pros, and cons."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Destination name, e.g. 'Lisbon'."}
            },
            "required": ["name"],
        },
    },
    {
        "name": "check_budget_fit",
        "description": (
            "Check whether a destination's estimated cost fits the traveller's budget. "
            "Returns fits (bool) and delta (positive = under budget)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Destination name."},
                "budget": {"type": "number", "description": "Traveller's total budget in euros."},
            },
            "required": ["name", "budget"],
        },
    },
    {
        "name": "score_destinations",
        "description": (
            "Score all destinations using the rule-based scorer and return them ranked best-first. "
            "Pass the traveller's full request so scores reflect their interests and constraints."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin_city":          {"type": "string"},
                "date_range":           {"type": "string"},
                "trip_length":          {"type": "integer"},
                "budget":               {"type": "number"},
                "interests":            {"type": "array", "items": {"type": "string"}},
                "transport_preference": {"type": "string"},
                "physical_level":       {"type": "string"},
            },
            "required": [
                "origin_city", "date_range", "trip_length", "budget",
                "interests", "transport_preference", "physical_level",
            ],
        },
    },
    {
        "name": "submit_trip_plan",
        "description": (
            "Submit the final trip plan. Call this exactly once when you have gathered "
            "enough information. This ends the planning loop and returns the plan to the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ranked_destinations": {
                    "type": "array",
                    "description": "Exactly 3 destinations, best first.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":                {"type": "string"},
                            "country":             {"type": "string"},
                            "score":               {"type": "integer"},
                            "pros":                {"type": "array", "items": {"type": "string"}},
                            "cons":                {"type": "array", "items": {"type": "string"}},
                            "estimated_cost":      {"type": "integer"},
                            "estimated_cost_home": {
                                "type": "number",
                                "description": "Estimated cost converted to home currency via convert_currency tool.",
                            },
                            "home_currency": {
                                "type": "string",
                                "description": "ISO code of the home currency (e.g. CNY, USD).",
                            },
                        },
                        "required": ["name", "country", "score", "pros", "cons", "estimated_cost"],
                    },
                },
                "itinerary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day":         {"type": "integer"},
                            "destination": {"type": "string"},
                            "plan":        {"type": "string"},
                        },
                        "required": ["day", "destination", "plan"],
                    },
                },
                "risks":                {"type": "array", "items": {"type": "string"}},
                "final_recommendation": {"type": "string"},
                "explanation":          {"type": "string", "description": "2-3 sentence explanation for the traveller."},
            },
            "required": [
                "ranked_destinations", "itinerary", "risks",
                "final_recommendation", "explanation",
            ],
        },
    },
    CONVERT_CURRENCY_SCHEMA,
    {
        "name": "search_travel_knowledge",
        "description": (
            "Search the Wikivoyage travel knowledge base for grounded facts about a destination. "
            "Returns relevant text chunks with source attribution (destination, section, URL). "
            "Use this to ground claims about safety, best seasons, food, activities, and local "
            "tips in real data rather than your own memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look up, e.g. 'Lisbon safety at night' or 'Venice best time to visit'.",
                },
                "destination": {
                    "type": "string",
                    "description": "Optional: restrict results to this destination.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
]


def dispatch_tool(name: str, tool_input: dict) -> dict:
    """Route a tool call by name to its implementation.

    submit_trip_plan is a terminal tool — the agent loop handles it directly
    and never passes it here.
    """
    print(f"[tool] {name} {tool_input}", flush=True)
    if name == "list_destinations":
        return {"destinations": list_destinations()}
    if name == "get_destination_details":
        return get_destination_details(**tool_input)
    if name == "check_budget_fit":
        return check_budget_fit(**tool_input)
    if name == "score_destinations":
        return {"ranked": score_destinations(tool_input)}
    if name == "convert_currency":
        return convert_currency(**tool_input)
    if name == "search_travel_knowledge":
        from backend.app.knowledge import search_travel_knowledge
        return {"results": search_travel_knowledge(**tool_input)}
    return {"error": "unknown tool", "name": name}


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def gather_tool_insights(
    origin: str,
    destination: str,
    date_range: str,
    trip_length_days: int,
) -> ToolInsight:
    """Call all three mock tools and return a single ToolInsight for one destination."""
    transport = estimate_transport_time(origin, destination)
    weather = estimate_weather_risk(destination, date_range)
    cost = estimate_total_cost(destination, trip_length_days)
    return ToolInsight(
        transport_hours=transport["hours"],
        transport_mode=transport["mode"],
        weather_risk=weather["risk"],
        weather_note=weather["note"],
        estimated_total_cost=cost["total"],
    )
