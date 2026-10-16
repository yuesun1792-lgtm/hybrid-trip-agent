"""
Tests for backend/app/tool_calling/tools.py — no API key required.
All tool functions are pure Python lookups.
"""

from backend.app.domain.models import ToolInsight
from backend.app.tool_calling.tools import (
    estimate_total_cost,
    estimate_transport_time,
    estimate_weather_risk,
    gather_tool_insights,
)

KNOWN_DESTINATIONS = ["Lisbon", "Prague", "Ljubljana", "Amsterdam"]


# ---------------------------------------------------------------------------
# estimate_transport_time
# ---------------------------------------------------------------------------

class TestEstimateTransportTime:
    def test_known_munich_to_lisbon(self):
        result = estimate_transport_time("Munich", "Lisbon")
        assert result["hours"] == 3.0
        assert result["mode"] == "flight"

    def test_known_munich_to_prague(self):
        result = estimate_transport_time("Munich", "Prague")
        assert result["hours"] == 4.5
        assert result["mode"] == "train"

    def test_unknown_origin_falls_back_to_destination_default(self):
        result = estimate_transport_time("SomeUnknownCity", "Amsterdam")
        assert result["hours"] == 3.5
        assert result["mode"] == "train"

    def test_unknown_origin_and_destination_returns_generic_default(self):
        result = estimate_transport_time("Nowhere", "Nowhere")
        assert isinstance(result["hours"], float)
        assert result["mode"] in ("flight", "train", "bus")

    def test_origin_matching_is_case_insensitive(self):
        lower = estimate_transport_time("munich", "Lisbon")
        upper = estimate_transport_time("MUNICH", "Lisbon")
        assert lower == upper


# ---------------------------------------------------------------------------
# estimate_weather_risk
# ---------------------------------------------------------------------------

class TestEstimateWeatherRisk:
    def test_lisbon_june_is_low_risk(self):
        result = estimate_weather_risk("Lisbon", "June 10-15")
        assert result["risk"] == "low"
        assert result["note"]

    def test_prague_december_is_high_risk(self):
        result = estimate_weather_risk("Prague", "December")
        assert result["risk"] == "high"

    def test_amsterdam_june_is_medium_risk(self):
        result = estimate_weather_risk("Amsterdam", "June")
        assert result["risk"] == "medium"

    def test_unknown_month_returns_medium_fallback(self):
        result = estimate_weather_risk("Lisbon", "NotAMonth")
        assert result["risk"] == "medium"
        assert result["note"]

    def test_unknown_destination_returns_fallback(self):
        result = estimate_weather_risk("UnknownCity", "June")
        assert result["risk"] in ("low", "medium", "high")
        assert result["note"]

    def test_risk_values_are_valid(self):
        for destination in KNOWN_DESTINATIONS:
            result = estimate_weather_risk(destination, "June")
            assert result["risk"] in ("low", "medium", "high"), (
                f"Unexpected risk level for {destination}"
            )


# ---------------------------------------------------------------------------
# estimate_total_cost
# ---------------------------------------------------------------------------

class TestEstimateTotalCost:
    def test_known_destination_returns_nonzero_total(self):
        result = estimate_total_cost("Prague", 4)
        assert result["total"] > 0

    def test_breakdown_has_expected_categories(self):
        result = estimate_total_cost("Prague", 4)
        assert "accommodation" in result["breakdown"]
        assert "food" in result["breakdown"]
        assert "local_transport" in result["breakdown"]

    def test_total_equals_sum_of_breakdown(self):
        result = estimate_total_cost("Lisbon", 5)
        assert result["total"] == sum(result["breakdown"].values())

    def test_cost_scales_linearly_with_trip_length(self):
        cost_4 = estimate_total_cost("Amsterdam", 4)["total"]
        cost_8 = estimate_total_cost("Amsterdam", 8)["total"]
        assert cost_8 == cost_4 * 2

    def test_unknown_destination_uses_fallback_and_does_not_crash(self):
        result = estimate_total_cost("UnknownCity", 3)
        assert result["total"] > 0


# ---------------------------------------------------------------------------
# gather_tool_insights
# ---------------------------------------------------------------------------

class TestGatherToolInsights:
    def test_returns_tool_insight_instance(self):
        insight = gather_tool_insights("Munich", "Lisbon", "June 10-15", 4)
        assert isinstance(insight, ToolInsight)

    def test_all_fields_are_populated(self):
        insight = gather_tool_insights("Munich", "Prague", "June 10-15", 4)
        assert insight.transport_hours > 0
        assert insight.transport_mode in ("flight", "train", "bus")
        assert insight.weather_risk in ("low", "medium", "high")
        assert insight.weather_note
        assert insight.estimated_total_cost > 0

    def test_unknown_destination_does_not_crash(self):
        insight = gather_tool_insights("Munich", "UnknownCity", "June 10-15", 4)
        assert isinstance(insight, ToolInsight)

    def test_cost_reflects_trip_length(self):
        short = gather_tool_insights("Munich", "Prague", "June", 3)
        long = gather_tool_insights("Munich", "Prague", "June", 6)
        assert long.estimated_total_cost == short.estimated_total_cost * 2
