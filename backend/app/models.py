from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    origin_city: str
    date_range: str
    trip_length: int = Field(gt=0)
    budget: int = Field(gt=0)
    interests: list[str]
    transport_preference: str
    physical_level: str


class DestinationSuggestion(BaseModel):
    name: str
    country: str
    score: int
    pros: list[str]
    cons: list[str]
    estimated_cost: int


class ItineraryDay(BaseModel):
    day: int
    destination: str
    plan: str


class TripPlan(BaseModel):
    ranked_destinations: list[DestinationSuggestion]
    itinerary: list[ItineraryDay]
    risks: list[str]
    final_recommendation: str
    # Optional LLM-generated explanation; None when no API key is configured
    explanation: str | None = None

