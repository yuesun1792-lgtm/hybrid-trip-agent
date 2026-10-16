# Place resolution through MCP

This is the first bounded tool-using node in the unified planning workflow. The node
uses native OpenAI Responses/Anthropic tool calls, a local stdio MCP client/server and
real Geoapify HTTP endpoints. Existing RAG/transport tools remain on the local registry;
this change does not migrate every tool, add TREK writes or replace the scheduler.

## Configure

Install updated `requirements.txt`. Register a Geoapify project and obtain its own API
key; model-provider keys do not work for geographic queries. Add to `.env`:

```dotenv
ENABLE_PLACE_RESOLUTION=true
GEOAPIFY_API_KEY=your_geoapify_key
PLACE_MAX_ACTIVITIES=6
PLACE_MAX_ROUNDS=6
PLACE_MAX_TOOL_CALLS=12
# Optional network settings, separate from LLM_TRUST_ENV:
PLACES_TRUST_ENV=true
# PLACES_PROXY_URL=http://127.0.0.1:7890
# PLACES_CA_BUNDLE=/absolute/path/to/trusted-ca-bundle.pem
```

Restart the backend. The app launches its own local MCP subprocess for each node run;
there is no separate HTTP server/port to start. Only geographic-provider/proxy environment
variables are forwarded; model keys are not passed to that subprocess. If the explicit
enable flag is absent, supplying a Geoapify key enables the node automatically. Missing
required keys produces a disabled summary explaining why, with no fabricated data.

The host forwards shell proxy variables (including lowercase and ALL_PROXY) and resolves
macOS system HTTPS proxy settings for its child. Set `PLACES_PROXY_URL` explicitly when
needed; `PLACES_TRUST_ENV=false` disables automatic proxy/environment configuration but
does not disable an explicit proxy. Custom CA bundles retain TLS verification.
Do not change API keys to troubleshoot a connection failure without an HTTP response.
To isolate connectivity without transmitting credentials:

```bash
python -m scripts.test_places --diagnose-network
```

Any HTTP response (often 401 without credentials) proves HTTPS connectivity, not key
validity. Error results classify DNS, TLS verification, proxy, refused connection,
timeout and HTTP authentication problems without exposing URLs or proxy credentials.

The adapter uses:

- `GET https://api.geoapify.com/v1/geocode/search` for named venues/addresses.
- `GET https://api.geoapify.com/v2/place-details` with `features=details` for identity and coordinates.

See the official [geocoding](https://apidocs.geoapify.com/docs/geocoding/forward-geocoding/)
and [place details](https://apidocs.geoapify.com/docs/place-details/) documentation. Calls
are subject to the account's quota/pricing. No public Nominatim endpoint is used.
This is named-place search, not a generic category/nearby POI discovery engine.

## Workflow

```text
Selected destination -> generate structured activities
  -> copy plan, select only NEW activities (bounded count)
  -> start MCP session, discover the two allowed tools
  -> native model tool turn
  -> validate request, check scope/budget, execute MCP call
  -> return result with matching call ID to model
  -> repeat within node bounds, then validate structured selections
  -> attach provider records, validate full plan, commit version
```

The model gets `search_places(city, query, limit=3)` and
`get_place_details(place_id)`. Tool descriptions and JSON schemas come from MCP discovery;
only these two names are approved. Independent calls can be requested together but are
executed sequentially. Native provider continuation items are retained; OpenAI stateless
turns request encrypted reasoning continuation. The existing text-generation API remains.

Search must target a city present in this node. Details calls must use IDs previously
returned by search in the same node. The final JSON must cover each target activity once;
only IDs returned by both search and details can be selected. The application attaches
coordinates from validated tool records, never from final model JSON. Selecting a new
venue does not rewrite an activity's title, category, description or source evidence.
Geoapify search IDs can be canonicalized by the details endpoint. Details results carry
`requested_place_id`; the node accepts a selected search ID only through that explicit
lookup relationship, and attaches the provider's returned canonical ID and coordinates.
Final selections may use host-issued short handles (`place_1`), mapped only to verified
detail records, so the model need not reproduce long provider IDs in its final JSON.

Search results are filtered by canonical city, and known destination countries are
checked to prevent e.g. Paris in the wrong country. Unknown cities lack that country
cross-check. City/street-level geocodes are rejected as venue matches. Conservative city
checks can reject nearby suburbs or landmarks with incomplete addresses; they remain
unresolved rather than being silently assigned.

## Data and failure behavior

Activities now contain `places[]`, a compatibility `place_id` (first selected place),
`place_resolution` and `place_resolution_reason`. A place contains provider ID, name,
city/country, coordinates, address, categories, provider link, retrieval timestamp and
attribution. It is a provider identity/location record, not proof of current opening
hours, booking availability or suitability. Activity `status` is not upgraded to confirmed.

Resolution statuses: `unattempted`, `resolved`, `ambiguous`, `not_found`, `not_applicable`,
`error`. Vague activities/free time need not map to a point. Ambiguous/not-found results
have no selected coordinates. Multiple places per activity are permitted.

`TripPlan.place_resolution` summarizes this node run: enabled/completion status, attempted
and resolved activity counts, rounds/calls, and a message. The activity cap can leave
later activities `unattempted`; this is reported as partial, never silently complete.
Failed optional enrichment returns a valid core plan with errors and a validation warning;
it does not pretend that place resolution succeeded. Shared run exhaustion can still stop
the whole request. Local revisions only resolve new activity IDs and preserve every old
record, including old unresolved records. Retry/backfill of existing activities is not
implemented in this first version.

The node memoizes identical tool calls; the provider caches responses for one hour in
the MCP process (bounded 256 entries). Retrieval timestamps refer to the provider fetch,
not cache access. Caches do not persist across subprocesses. Calls are bounded by local
node limits and shared RunContext limits. Node deadline is at most 90 seconds within the
remaining run deadline; underlying HTTP/SDK timeouts bound cleanup of worker threads.

Trajectories record native turns/usage, MCP arguments/results/errors and node outcome.
Geoapify credentials are redacted and provider HTTP errors do not expose request URLs.
Streamlit displays returned names, coordinates and attribution. Translations affect only
presentation reasons, preserving canonical location records and IDs.

## Verify

```bash
# Actual SDK + subprocess handshake/discovery, no external calls or keys needed.
python -m scripts.test_places --discover

# Actual MCP -> Geoapify search + details; probes candidates without selecting a plan.
python -m scripts.test_places --city Paris --query 'Louvre Museum'

# Model chooses search/details and returns a place-backed activity (billable LLM).
python -m scripts.test_places --resolve --city Paris --query 'Louvre Museum'

# Full planning/multi-turn evaluation. Both keys and enable flag required.
python -m scripts.eval_scenarios --mode live --suite evals/scenarios/places.json

# Offline tests include real MCP protocol with mocked HTTP and native SDK requests.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p no:capture -q tests
```

The live place suite intentionally fails when resolution is disabled/unavailable; it
must not pass by using offline placeholders. The ordinary offline scene suite explicitly
disables this feature and never calls real geographic services. CLI exit codes remain
0 success, 1 failed behavior, 2 missing/invalid configuration. No live quality claim is
made by mocked HTTP tests.
