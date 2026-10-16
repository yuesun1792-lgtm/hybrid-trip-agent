import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import socket
import ssl

import httpx
import pytest

from backend.app.llm import llm
from backend.app.runtime.execution import RUN_CONTEXT, RunContext
from backend.app.domain.models import TripPlan, ItineraryDay
from backend.app.domain.planning_models import Activity
from backend.app.domain.place_models import Place, PlaceToolResult
from backend.app.places.place_provider import GeoapifyPlaces
from backend.app.places.place_resolution import resolve_plan_places
from backend.app.places.places_mcp_client import PlacesMCPClient, places_session
from backend.app.places.places_mcp_server import create_server


def props(city="Paris", identity="abc"):
    return {"place_id": identity, "city": city, "country": "France" if city == "Paris" else "Italy", "name": "Louvre Museum", "lat": 48.86, "lon": 2.33,
            "result_type": "amenity", "formatted": "Louvre, Paris", "categories": ["tourism.museum"]}


def place(city="Paris"):
    return Place(place_id="geoapify:abc", provider_place_id="abc", name="Louvre Museum", city=city, country="France",
        latitude=48.86, longitude=2.33, source_url="https://www.geoapify.com/", retrieved_at=datetime.now(timezone.utc))


def seed():
    return TripPlan(ranked_destinations=[], itinerary=[ItineraryDay(day=1, destination="Paris", activities=[
        Activity(title="Louvre Museum", description="Visit the Louvre", status="proposed")])],
        risks=[], final_recommendation="Test")


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("ENABLE_PLACE_RESOLUTION", "true")
    monkeypatch.setenv("GEOAPIFY_API_KEY", "private-geo-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-key")


def test_provider_search_details_city_filter_cache_and_provenance(enabled):
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path.endswith("search"):
            return httpx.Response(200, json={"results": [props(), props("Rome", "wrong")]})
        return httpx.Response(200, json={"features": [{"properties": props()}]})

    provider = GeoapifyPlaces(httpx.Client(transport=httpx.MockTransport(handle)))
    search = provider.search_places("Paris", "Louvre")
    assert search.status == "ok" and len(search.places) == 1
    assert provider.search_places("Paris", "Louvre") == search
    detail = provider.get_place_details(search.places[0].place_id)
    assert detail.places[0].latitude == 48.86
    assert len(calls) == 2
    assert calls[1].url.params["features"] == "details"
    assert "private-geo-key" not in detail.model_dump_json()


@pytest.mark.parametrize("code,retryable", [(401, False), (429, True), (503, True)])
def test_provider_failures_do_not_expose_key_or_fabricate_coordinates(enabled, code, retryable):
    provider = GeoapifyPlaces(httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(code))))
    result = provider.search_places("Paris", "Louvre")
    assert result.status == "error" and result.retryable == retryable and not result.places
    assert "private-geo-key" not in result.model_dump_json()


def test_provider_rejects_city_level_results_and_bad_coordinates(enabled):
    bad = props()
    bad["lat"] = 999
    provider = GeoapifyPlaces(httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"results": [bad]}))))
    assert provider.search_places("Paris", "Louvre").status == "error"
    city = props()
    city["result_type"] = "city"
    provider = GeoapifyPlaces(httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"results": [city]}))))
    assert provider.search_places("Paris", "Paris").status == "not_found"


def test_real_sdk_mcp_protocol_with_http_provider_mock(enabled):
    from mcp.shared.memory import create_connected_server_and_client_session
    provider = GeoapifyPlaces(httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"results": [props()]} if req.url.path.endswith("search")
                                  else {"features": [{"properties": props()}]}))))

    async def run():
        async with create_connected_server_and_client_session(create_server(provider)) as session:
            client = PlacesMCPClient(session)
            assert {t["name"] for t in await client.discover()} == {"search_places", "get_place_details"}
            result = await client.execute("search_places", {"city": "Paris", "query": "Louvre"})
            assert result.status == "ok"
            details = await client.execute("get_place_details", {"place_id": result.places[0].place_id})
            assert details.places[0].city == "Paris"
            assert (await client.execute("delete_trip", {})).error_type == "UnknownTool"
            assert (await client.execute("search_places", {"city": "Paris", "query": "Louvre", "url": "fake"})).error_type == "InvalidArguments"

    asyncio.run(run())


def test_stdio_subprocess_discovery_without_keys():
    async def run():
        async with places_session() as client:
            assert len(await client.discover()) == 2
            result = await client.execute("search_places", {"city": "Paris", "query": "Louvre"})
            assert result.status == "error" and not result.places
    asyncio.run(run())


class FakeMCP:
    def __init__(self):
        self.calls = []

    async def discover(self):
        return []

    async def execute(self, name, args):
        self.calls.append((name, args))
        return PlaceToolResult(status="ok", places=[place()])


def install_turns(monkeypatch, target, *, selected="geoapify:abc", repeated=False):
    calls = [llm.ToolCall("s1", "search_places", {"city": "Paris", "query": "Louvre"})]
    if repeated:
        calls.append(llm.ToolCall("s2", "search_places", {"city": "Paris", "query": "Louvre"}))
    turns = iter([
        llm.ModelTurn("", calls, [], "anthropic"),
        llm.ModelTurn("", [llm.ToolCall("d1", "get_place_details", {"place_id": "geoapify:abc"})], [], "anthropic"),
        llm.ModelTurn(json.dumps({"activities": [{"activity_id": target.activity_id, "status": "resolved",
            "place_ids": [selected], "reason": "Named venue and city match."}]}), [], [], "anthropic")])
    monkeypatch.setattr("backend.app.places.place_resolution.generate_tool_turn", lambda *a, **k: next(turns))


def factory(client):
    @asynccontextmanager
    async def session():
        yield client
    return session


def test_node_uses_only_tool_coordinates_and_deduplicates_requests(enabled, monkeypatch):
    original = seed()
    install_turns(monkeypatch, original.itinerary[0].activities[0], repeated=True)
    client = FakeMCP()
    result = resolve_plan_places(original, session_factory=factory(client))
    activity = result.itinerary[0].activities[0]
    assert result.place_resolution.status == "completed"
    assert activity.places[0].latitude == 48.86 and activity.place_id == "geoapify:abc"
    assert len(client.calls) == 2 and result.place_resolution.tool_calls == 3
    assert original.itinerary[0].activities[0].places == []


def test_forged_selection_never_commits_coordinates(enabled, monkeypatch):
    original = seed()
    install_turns(monkeypatch, original.itinerary[0].activities[0], selected="geoapify:invented")
    result = resolve_plan_places(original, session_factory=factory(FakeMCP()))
    assert result.place_resolution.status == "failed"
    assert not result.itinerary[0].activities[0].places


def test_short_final_selection_handle_attaches_only_verified_provider_record(enabled, monkeypatch):
    original = seed()
    install_turns(monkeypatch, original.itinerary[0].activities[0], selected="place_1")
    result = resolve_plan_places(original, session_factory=factory(FakeMCP()))
    activity = result.itinerary[0].activities[0]
    assert result.place_resolution.status == "completed"
    assert activity.place_id == "geoapify:abc" and activity.places[0].latitude == 48.86


def test_revision_preserves_all_old_activity_records_without_tool_calls(enabled):
    before = seed()
    before.itinerary[0].activities[0].places = [place()]
    before.itinerary[0].activities[0].place_id = "geoapify:abc"
    before.itinerary[0].activities[0].place_resolution = "resolved"
    result = resolve_plan_places(before.model_copy(deep=True), before, session_factory=lambda: pytest.fail("No tools for retained activities"))
    assert result.itinerary == before.itinerary
    assert result.place_resolution.attempted_activities == 0


def test_unknown_detail_id_is_rejected_before_mcp(enabled, monkeypatch):
    original = seed()
    turns = iter([llm.ModelTurn("", [llm.ToolCall("d", "get_place_details", {"place_id": "geoapify:fake"})], [], "anthropic"),
        llm.ModelTurn(json.dumps({"activities": [{"activity_id": original.itinerary[0].activities[0].activity_id,
            "status": "not_found", "place_ids": [], "reason": "No match"}]}), [], [], "anthropic")])
    monkeypatch.setattr("backend.app.places.place_resolution.generate_tool_turn", lambda *a, **k: next(turns))
    client = FakeMCP()
    result = resolve_plan_places(original, session_factory=factory(client))
    assert not client.calls and not result.itinerary[0].activities[0].places
    assert result.place_resolution.status == "failed" and result.place_resolution.tool_errors == 1


def test_repeated_model_calls_stop_at_node_limit(enabled, monkeypatch):
    monkeypatch.setenv("PLACE_MAX_TOOL_CALLS", "1")
    original = seed()
    install_turns(monkeypatch, original.itinerary[0].activities[0])
    result = resolve_plan_places(original, session_factory=factory(FakeMCP()))
    assert result.place_resolution.status == "failed" and result.place_resolution.tool_calls == 1
    assert not result.itinerary[0].activities[0].places


def test_ambiguous_result_stays_unresolved_without_queries(enabled, monkeypatch):
    original = seed()
    target = original.itinerary[0].activities[0]
    response = {"activities": [{"activity_id": target.activity_id, "status": "ambiguous", "place_ids": [], "reason": "Multiple plausible venues."}]}
    monkeypatch.setattr("backend.app.places.place_resolution.generate_tool_turn", lambda *a, **k:
        llm.ModelTurn(json.dumps(response), [], [], "anthropic"))
    client = FakeMCP()
    result = resolve_plan_places(original, session_factory=factory(client))
    assert result.place_resolution.status == "completed" and not client.calls
    assert result.itinerary[0].activities[0].place_resolution == "ambiguous"
    assert not result.itinerary[0].activities[0].places


def test_wrong_country_same_city_is_rejected(enabled):
    wrong = props()
    wrong["country"] = "United States"
    provider = GeoapifyPlaces(httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"results": [wrong]}))))
    assert provider.search_places("Paris", "Louvre").status == "not_found"


@pytest.mark.parametrize("cause,code", [
    (socket.gaierror(-2, "hostname failed"), "dns_resolution_failed"),
    (ssl.SSLCertVerificationError(1, "certificate_verify_failed"), "tls_verification_failed"),
    (ConnectionRefusedError(61, "refused"), "connection_refused")])
def test_network_error_classification_never_leaks_urls_or_keys(enabled, cause, code):
    error = httpx.ConnectError("https://api.geoapify.com?apiKey=private-geo-key")
    error.__cause__ = cause
    result = GeoapifyPlaces()._failure(error)
    assert result.diagnostic_code == code
    assert "private-geo-key" not in result.model_dump_json()
    assert "https://api.geoapify.com?" not in result.model_dump_json()


def test_explicit_proxy_and_ca_are_independent_of_trust_env(monkeypatch):
    from backend.app.places.place_provider import places_http_client
    monkeypatch.setenv("PLACES_TRUST_ENV", "false")
    monkeypatch.setenv("PLACES_PROXY_URL", "http://localhost:7890")
    monkeypatch.setenv("PLACES_CA_BUNDLE", "/trusted.pem")
    ca = object()
    monkeypatch.setattr("backend.app.places.place_provider.ssl.create_default_context", lambda **kw: ca)
    monkeypatch.setattr("backend.app.places.place_provider.httpx.Client", lambda **kw: kw)
    options = places_http_client()
    assert options["trust_env"] is False and options["proxy"] == "http://localhost:7890"
    assert options["verify"] is ca


def test_mcp_child_receives_system_proxy_and_no_model_keys(monkeypatch):
    from backend.app.places.places_mcp_client import places_server_environment
    for key in ["HTTPS_PROXY", "https_proxy", "PLACES_PROXY_URL", "NO_PROXY", "no_proxy"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PLACES_TRUST_ENV", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-pass")
    monkeypatch.setenv("ALL_PROXY", "http://localhost:7890")
    monkeypatch.setenv("https_proxy", "")
    monkeypatch.setattr("backend.app.places.places_mcp_client.getproxies", lambda: {"https": "http://localhost:7890", "no": "localhost"})
    env = places_server_environment()
    assert env["HTTPS_PROXY"] == "http://localhost:7890"
    assert env["ALL_PROXY"] == "http://localhost:7890" and env["NO_PROXY"] == "localhost"
    assert "OPENAI_API_KEY" not in env
    monkeypatch.setenv("PLACES_TRUST_ENV", "false")
    assert "HTTPS_PROXY" not in places_server_environment()


def test_provider_preserves_lookup_relationship_for_canonicalized_id(enabled):
    provider = GeoapifyPlaces(httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"features": [{"properties": props(identity="canonical")}]}))))
    result = provider.get_place_details("geoapify:search-id")
    assert result.status == "ok"
    assert result.requested_place_id == "geoapify:search-id"
    assert result.places[0].place_id == "geoapify:canonical"


def test_node_accepts_canonical_details_id_only_via_explicit_lookup(enabled, monkeypatch):
    original = seed()
    install_turns(monkeypatch, original.itinerary[0].activities[0])

    class CanonicalMCP(FakeMCP):
        async def execute(self, name, args):
            if name == "get_place_details":
                canonical = place().model_copy(update={"place_id": "geoapify:canonical", "provider_place_id": "canonical"})
                return PlaceToolResult(status="ok", places=[canonical], requested_place_id=args["place_id"])
            return await super().execute(name, args)

    result = resolve_plan_places(original, session_factory=factory(CanonicalMCP()))
    activity = result.itinerary[0].activities[0]
    assert result.place_resolution.status == "completed"
    assert activity.place_id == activity.places[0].place_id == "geoapify:canonical"


def test_only_new_activity_is_resolved_during_revision(enabled, monkeypatch):
    before = seed()
    old_activity = before.itinerary[0].activities[0]
    old_activity.places = [place()]
    old_activity.place_id = "geoapify:abc"
    old_activity.place_resolution = "resolved"
    updated = before.model_copy(deep=True)
    added = Activity(title="Louvre Museum", status="proposed")
    updated.itinerary[0].activities.append(added)
    install_turns(monkeypatch, added)
    result = resolve_plan_places(updated, before, session_factory=factory(FakeMCP()))
    assert result.place_resolution.attempted_activities == 1
    assert result.itinerary[0].activities[0] == old_activity
    assert result.itinerary[0].activities[1].places


def test_complete_node_with_real_mcp_and_mocked_provider_http(enabled, monkeypatch, tmp_path):
    from mcp.shared.memory import create_connected_server_and_client_session
    from backend.app.runtime.trajectory import Trajectory, _CURRENT
    from backend.app.domain.models import ChatResponse
    original = seed()
    install_turns(monkeypatch, original.itinerary[0].activities[0])
    provider = GeoapifyPlaces(httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"results": [props()]} if req.url.path.endswith("search")
                                  else {"features": [{"properties": props()}]}))))

    @asynccontextmanager
    async def session_factory():
        async with create_connected_server_and_client_session(create_server(provider)) as session:
            yield PlacesMCPClient(session)

    context = RunContext()
    recorder = Trajectory("place-test", None, "Resolve Louvre", {})
    token = _CURRENT.set(recorder)
    budget_token = RUN_CONTEXT.set(context)
    try:
        result = resolve_plan_places(original, session_factory=session_factory)
        recorder.finish(response=ChatResponse(conversation_id="place-test", assistant_message="Done", plan=result))
        assert result.place_resolution.status == "completed"
        assert context.tool_calls == 2
        steps = recorder.data["steps"]
        assert len([s for s in steps if s["name"] == "places_mcp_call"]) == 2
        assert all(s["status"] == "ok" for s in steps)
        assert "private-geo-key" not in recorder.path.read_text()
    finally:
        _CURRENT.reset(token)
        RUN_CONTEXT.reset(budget_token)


def test_plan_validation_rejects_place_from_other_city():
    from backend.app.planning.plan_validation import validate_plan, PlanValidationError
    current = seed()
    activity = current.itinerary[0].activities[0]
    activity.places = [place("Rome")]
    activity.place_id = "geoapify:abc"
    activity.place_resolution = "resolved"
    with pytest.raises(PlanValidationError, match="place identity or city"):
        validate_plan(current)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_native_tool_turn_preserves_calls_and_returns_results(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    monkeypatch.setenv("OPENAI_MODEL", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if provider == "openai":
            output = [{"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "search_places",
                       "arguments": '{"city":"Paris","query":"Louvre"}'}] if len(requests) == 1 else [
                {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "Done", "annotations": []}]}]
            if len(requests) == 1:
                output.insert(0, {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"})
            body = {"id": "resp_1", "object": "response", "created_at": 1, "model": "test", "status": "completed",
                    "output": output, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
        else:
            content = [{"type": "tool_use", "id": "c1", "name": "search_places", "input": {"city": "Paris", "query": "Louvre"}}] if len(requests) == 1 else [{"type": "text", "text": "Done"}]
            body = {"id": "msg_1", "type": "message", "role": "assistant", "model": "test", "content": content,
                    "stop_reason": "tool_use" if len(requests) == 1 else "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}
        return httpx.Response(200, json=body)

    def make_client(kind, key):
        http = httpx.Client(transport=httpx.MockTransport(handle))
        if kind == "openai":
            from openai import OpenAI
            return OpenAI(api_key=key, http_client=http)
        from anthropic import Anthropic
        return Anthropic(api_key=key, http_client=http)

    monkeypatch.setattr(llm, "_make_client", make_client)
    history = [{"role": "user", "content": "Find Louvre"}]
    tools = [{"name": "search_places", "description": "Find place", "input_schema": {"type": "object", "properties": {}}}]
    first = llm.generate_tool_turn(history, tools)
    assert first.tool_calls[0].arguments["city"] == "Paris"
    history.extend(first.continuation)
    history.extend(llm.tool_result_messages(provider, [("c1", {"status": "ok"})]))
    final = llm.generate_tool_turn(history, tools)
    assert final.text == "Done" and not final.tool_calls
    if provider == "openai":
        assert requests[1]["input"][-1]["call_id"] == "c1"
        assert requests[1]["input"][-2]["type"] == "function_call"
        assert requests[1]["input"][1]["encrypted_content"] == "opaque"
        assert requests[0]["include"] == ["reasoning.encrypted_content"]
    else:
        assert requests[1]["messages"][-1]["content"][0]["tool_use_id"] == "c1"
