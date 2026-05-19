# EuroTrip Agent V0 Spec

## Goal
Build a minimal travel planning app for European trips.

## V0 Scope
The app takes a user travel request and returns 3 destination suggestions plus one recommended itinerary.

## Input fields
- origin city
- date range
- trip length
- budget
- interests
- transport preference
- physical level

## Output
Return structured JSON:
- ranked destinations
- pros and cons
- estimated cost
- itinerary
- risks
- final recommendation

## Tech stack
- Backend: Python FastAPI
- Frontend: Streamlit
- No database in V0
- No real external APIs in V0
- Use mock destination data first
- Keep code simple and readable

## Learning goal
I want to understand the architecture, not just generate code. Please keep the implementation minimal and explain each file.
