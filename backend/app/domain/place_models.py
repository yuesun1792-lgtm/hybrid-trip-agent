"""Provider-grounded place records; coordinates never come from model output."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Place(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    place_id: str = Field(min_length=1)
    provider: Literal["geoapify"] = "geoapify"
    provider_place_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    city: str = Field(min_length=1)
    country: str = ""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    address: str = ""
    categories: list[str] = Field(default_factory=list)
    source_url: str
    attribution: str = "Powered by Geoapify | © OpenStreetMap contributors"
    retrieved_at: datetime

    def matches_city(self, city):
        from backend.app.domain.destination_names import COUNTRY_BY_DESTINATION
        return self.city == city and (not COUNTRY_BY_DESTINATION.get(city) or
                                      self.country == COUNTRY_BY_DESTINATION[city])


class SearchPlacesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=250)
    limit: int = Field(default=3, ge=1, le=5)


class PlaceDetailsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    place_id: str = Field(min_length=1, max_length=600)


class PlaceToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "not_found", "error"]
    places: list[Place] = Field(default_factory=list)
    error_type: str | None = None
    message: str = ""
    retryable: bool = False
    diagnostic_code: str | None = None
    network: dict[str, bool] = Field(default_factory=dict)
    requested_place_id: str | None = None

    @property
    def fallback_reason(self):
        return f"{self.error_type}: {self.message}" if self.status == "error" else None


class PlaceResolutionSummary(BaseModel):
    status: Literal["disabled", "completed", "partial", "failed"] = "disabled"
    resolved_activities: int = 0
    attempted_activities: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    rounds: int = 0
    message: str = "Place resolution is disabled."
