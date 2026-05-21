"""
Mock tool functions for V2 enrichment.

Each function is a plain Python lookup — no network calls, fully deterministic.
gather_tool_insights() calls all three and returns a ToolInsight for one destination.
"""

from backend.app.models import ToolInsight

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
