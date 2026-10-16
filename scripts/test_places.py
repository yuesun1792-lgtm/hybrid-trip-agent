"""Probe MCP discovery, real place tools, or the model-directed place node."""
import argparse
import asyncio
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="Paris")
    parser.add_argument("--query", default="Louvre Museum")
    parser.add_argument("--discover", action="store_true", help="Only start the real MCP subprocess and discover tools; no keys/network needed")
    parser.add_argument("--resolve", action="store_true", help="Use model-directed search/details and return a resolved activity")
    parser.add_argument("--diagnose-network", action="store_true", help="Probe HTTPS without sending an API key or place query")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    import os
    from backend.app.trajectory import clean, error_details, Trajectory, _CURRENT
    from backend.app.places_mcp_client import places_session
    from backend.app.execution import RUN_CONTEXT, RunContext
    from backend.app.models import ChatResponse, TripPlan, ItineraryDay, ExecutionTraceStep
    from backend.app.planning_models import Activity
    if args.diagnose_network:
        from backend.app.place_provider import GeoapifyPlaces
        result = GeoapifyPlaces().check_connection()
        print(json.dumps(clean(result), ensure_ascii=False, indent=2))
        return 0 if result["reachable"] else 1

    async def probe():
        async with places_session() as client:
            tools = await client.discover()
            if args.discover:
                return {"tools": tools}
            search = await client.execute("search_places", {"city": args.city, "query": args.query})
            # Probe all returned candidates; this does not select one for a plan.
            details = [await client.execute("get_place_details", {"place_id": p.place_id}) for p in search.places]
            return {"search": search, "details": details}

    if not args.discover and not os.environ.get("GEOAPIFY_API_KEY", "").strip():
        print("Configure GEOAPIFY_API_KEY in .env first.")
        return 2
    if args.resolve and not args.discover:
        from backend.app.config import llm_api_key, llm_model
        try:
            if not llm_api_key():
                raise ValueError("Configure the selected LLM provider key first")
            llm_model()
        except ValueError as exc:
            print(json.dumps(error_details(exc), ensure_ascii=False))
            return 2
    recorder = Trajectory("places-smoke", None, args.query, {})
    token = _CURRENT.set(recorder)
    budget_token = RUN_CONTEXT.set(RunContext())
    try:
        if args.resolve and not args.discover:
            from backend.app.place_resolution import resolve_plan_places
            from unittest.mock import patch
            seed = TripPlan(ranked_destinations=[], itinerary=[ItineraryDay(day=1, destination=args.city,
                activities=[Activity(title=args.query, description="Visit this named place.", status="proposed")])],
                risks=[], final_recommendation="Place resolution smoke test")
            with patch.dict(os.environ, {"ENABLE_PLACE_RESOLUTION": "true"}):
                plan = resolve_plan_places(seed)
            result = {"plan": plan}
            passed = plan.place_resolution.status == "completed" and plan.place_resolution.resolved_activities == 1
        else:
            result = asyncio.run(probe())
            passed = args.discover or (result["search"].status == "ok" and bool(result["details"]) and
                                      all(item.status == "ok" for item in result["details"]))
        recorder.finish(response=ChatResponse(conversation_id="places-smoke", assistant_message="Passed" if passed else "Failed",
            plan=result.get("plan"), trace=[ExecutionTraceStep(stage="place_resolution", status="ok" if passed else "fallback")]))
        print(json.dumps(clean({**result, "passed": passed, "trajectory_file": str(recorder.path)}), ensure_ascii=False, indent=2))
        return 0 if passed else 1
    except Exception as exc:
        recorder.finish(error=exc)
        print(json.dumps(error_details(exc), ensure_ascii=False, indent=2))
        return 1
    finally:
        RUN_CONTEXT.reset(budget_token)
        _CURRENT.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())
