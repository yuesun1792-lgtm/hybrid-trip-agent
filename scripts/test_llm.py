"""Real provider smoke test: python -m scripts.test_llm [--chat]."""
import argparse
import json

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat", action="store_true", help="Exercise real chat routing, retrieval and itinerary writing")
    parser.add_argument("--message", help="Custom chat message; test fields are inferred from this message")
    parser.add_argument("--output", choices=["options", "itinerary"], help="Explicit output scope")
    parser.add_argument("--language", help="Explicit response language code, for example zh or fr")
    parser.add_argument("--revision-message", help="After creating a plan, revise it in the same process and verify unaffected days")
    args = parser.parse_args()
    load_dotenv()
    from backend.app.config import llm_api_key, llm_model, llm_provider
    from backend.app.trajectory import Trajectory, _CURRENT, clean, error_details
    from backend.app.models import ChatRequest, ChatResponse
    from backend.app.llm import generate_text

    try:
        if not llm_api_key():
            raise ValueError(f"Configure {llm_provider().upper()}_API_KEY in .env")
        model = llm_model()
    except ValueError as exc:
        print(json.dumps(error_details(exc), ensure_ascii=False, indent=2))
        return 2

    if args.chat:
        from backend.app.chat import handle_chat
        try:
            response = handle_chat(ChatRequest(
                message=args.message or "Plan 3 days in Paris for food and art, from Munich in July, budget 1000, train, medium.",
                response_language=args.language,
                requested_output=args.output,
                extracted_fields={"destination": "Paris", "trip_length": 3, "origin_city": "Munich",
                                  "date_range": "July", "budget": 1000, "interests": ["food", "art"],
                                  "transport_preference": "train", "physical_level": "medium"} if not args.message else {},
            ))
            print(json.dumps(clean({"run_id": response.run_id, "provider": llm_provider(), "model": model,
                                  "status": response.trajectory["status"], "trace": response.trace,
                                  "response_language": response.response_language,
                                  "assistant_message": response.assistant_message,
                                  "extracted_fields": response.extracted_fields,
                                  "plan_id": response.plan.plan_id if response.plan else None,
                                  "version": response.plan.version if response.plan else None,
                                  "activity_count": sum(len(d.activities) for d in response.plan.itinerary) if response.plan else 0,
                                  "llm_usage": response.llm_usage}), ensure_ascii=False, indent=2))
            if args.revision_message and response.plan:
                from backend.app.conversation_state import get_conversation
                before = get_conversation(response.conversation_id).last_plan.model_copy(deep=True)
                revised = handle_chat(ChatRequest(message=args.revision_message, conversation_id=response.conversation_id,
                                                 expected_plan_version=before.version))
                if not revised.plan:
                    print(json.dumps({"revision_needs_input": revised.assistant_message}, ensure_ascii=False))
                    return 1
                after = get_conversation(response.conversation_id).last_plan
                scope = revised.planning_task["affected_scope"]
                unchanged = [day for day in before.itinerary if day.day not in scope["day_numbers"]]
                assert all(day == next(d for d in after.itinerary if d.day == day.day) for day in unchanged)
                old_activities = {a.activity_id: a for d in before.itinerary for a in d.activities}
                new_activities = {a.activity_id: a for d in after.itinerary for a in d.activities}
                assert all(old_activities[identity] == new_activities[identity] for identity in scope["preserve_activity_ids"])
                assert after.version == before.version + 1 and after.plan_id == before.plan_id
                print(json.dumps(clean({"revision_run_id": revised.run_id, "status": revised.trajectory["status"],
                                       "version": after.version, "affected_scope": scope, "preservation_checks": "passed",
                                       "assistant_message": revised.assistant_message, "llm_usage": revised.llm_usage}), ensure_ascii=False, indent=2))
                return 0 if revised.trajectory["status"] == "completed" else 1
            return 0 if response.plan and response.trajectory["status"] == "completed" else 1
        except Exception as exc:
            print(json.dumps(error_details(exc), ensure_ascii=False, indent=2))
            return 1

    recorder = Trajectory("llm-smoke-test", None, "Reply with OK.", {})
    token = _CURRENT.set(recorder)
    try:
        result = generate_text("Reply with OK.", stage="smoke_test", max_tokens=1024, prompt_version="smoke_v1")
        recorder.finish(response=ChatResponse(conversation_id="llm-smoke-test", assistant_message=result.text,
                                             llm_usage=[result.usage] if result.usage else []))
        print(json.dumps(clean({"result": result, "run_id": recorder.data["run_id"],
                              "trajectory_file": str(recorder.path)}), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        recorder.finish(error=exc)
        print(json.dumps(clean({"error": error_details(exc), "trajectory_file": str(recorder.path)}), ensure_ascii=False, indent=2))
        return 1
    finally:
        _CURRENT.reset(token)


if __name__ == "__main__":
    raise SystemExit(main())
