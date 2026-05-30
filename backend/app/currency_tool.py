"""
currency_tool.py — a REAL tool for the EuroTrip agent.

Same shape as your mock tools (a plain function + an Anthropic tool schema),
but the body calls a live API instead of reading a local JSON file.

The whole point: agent.py does NOT change. Only this function's body differs
from a mock tool. From Claude's side it's identical — name, schema, return dict.

Data source: Frankfurter (European Central Bank rates). Free, no API key.
Frankfurter only returns rates, not conversions, so we do the multiply ourselves.

Feel it:  python currency_tool.py
"""

import json
import urllib.request
import urllib.error

FRANKFURTER_URL = "https://api.frankfurter.app/latest"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert `amount` from one currency to another using live ECB rates.

    Returns a dict on success, or {"error": ...} on failure — the same
    'never raise, return an error dict' convention your mock tools use,
    so the agent loop handles it identically.
    """
    frm = from_currency.upper()
    to = to_currency.upper()

    # Same currency: no network call needed.
    if frm == to:
        return {"amount": amount, "from": frm, "to": to,
                "rate": 1.0, "converted": round(amount, 2), "date": "n/a"}

    url = f"{FRANKFURTER_URL}?from={frm}&to={to}"
    req = urllib.request.Request(url, headers={"User-Agent": "eurotrip-agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return {"error": "currency service unavailable", "detail": str(e)}

    rate = data.get("rates", {}).get(to)
    if rate is None:
        return {"error": "unknown currency pair", "from": frm, "to": to}

    return {
        "amount": amount,
        "from": frm,
        "to": to,
        "rate": rate,
        "converted": round(amount * rate, 2),
        "date": data.get("date"),
    }


# The schema Claude sees — identical in style to your mock tools' schemas.
CONVERT_CURRENCY_SCHEMA = {
    "name": "convert_currency",
    "description": (
        "Convert a money amount from one currency to another using live "
        "European Central Bank exchange rates. Use this to express a "
        "destination's estimated cost in the traveller's home currency. "
        "Currencies are ISO codes, e.g. EUR, USD, GBP, JPY."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "The amount to convert"},
            "from_currency": {"type": "string", "description": "ISO code to convert from, e.g. EUR"},
            "to_currency": {"type": "string", "description": "ISO code to convert to, e.g. USD"},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
}


if __name__ == "__main__":
    # Convert a sample trip cost (1200 EUR) into a few currencies.
    for target in ("USD", "GBP", "JPY"):
        print(convert_currency(1200, "EUR", target))
