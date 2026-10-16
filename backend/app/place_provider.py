"""Real Geoapify geocoding/details adapter, also used by the MCP server."""
from datetime import datetime, timezone
from hashlib import sha256
import os
import socket
import ssl
from threading import Lock
from time import monotonic

import httpx

from backend.app.destination_names import canonical_destination, COUNTRY_BY_DESTINATION
from backend.app.config import env_flag
from backend.app.place_models import Place, PlaceToolResult, SearchPlacesArguments, PlaceDetailsArguments


def places_network_settings():
    return {"trust_env": env_flag("PLACES_TRUST_ENV", True),
            "explicit_proxy": bool(os.environ.get("PLACES_PROXY_URL", "").strip()),
            "custom_ca": bool(os.environ.get("PLACES_CA_BUNDLE", "").strip())}


def places_http_client():
    options = {"follow_redirects": False, "trust_env": env_flag("PLACES_TRUST_ENV", True)}
    proxy = os.environ.get("PLACES_PROXY_URL", "").strip()
    if proxy:
        options["proxy"] = proxy
    ca_bundle = os.environ.get("PLACES_CA_BUNDLE", "").strip()
    if ca_bundle:
        options["verify"] = ssl.create_default_context(cafile=ca_bundle)
    return httpx.Client(**options)


class GeoapifyPlaces:
    def __init__(self, client=None):
        self.client = client
        self.cache = {}
        self.lock = Lock()

    def _request(self, endpoint, params):
        key = os.environ.get("GEOAPIFY_API_KEY", "").strip()
        if not key:
            raise ValueError("GEOAPIFY_API_KEY is not configured")
        cache_key = (sha256(key.encode()).hexdigest(), endpoint, tuple(sorted(params.items())))
        with self.lock:
            cached = self.cache.get(cache_key)
            if cached and monotonic() - cached[0] < 3600:
                return cached[1]
        options = {"params": {**params, "apiKey": key}, "timeout": 8.0}
        if self.client:
            response = self.client.get("https://api.geoapify.com" + endpoint, **options)
        else:
            with places_http_client() as client:
                response = client.get("https://api.geoapify.com" + endpoint, **options)
        response.raise_for_status()
        data = response.json()
        data["_retrieved_at"] = datetime.now(timezone.utc).isoformat()
        with self.lock:
            if len(self.cache) >= 256:
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = (monotonic(), data)
        return data

    def _place(self, props, retrieved_at):
        identity = str(props.get("place_id") or "")
        city = canonical_destination(props.get("city") or props.get("town") or props.get("village") or "")
        if not identity or not city or props.get("lat") is None or props.get("lon") is None:
            return None
        country = props.get("country", "")
        if COUNTRY_BY_DESTINATION.get(city) and country != COUNTRY_BY_DESTINATION[city]:
            return None  # Same city name in a different country is not a valid match.
        # A city/street result is not a resolved venue. Category is descriptive only.
        if props.get("result_type") in {"country", "state", "city", "county", "postcode", "street", "suburb", "district"}:
            return None
        return Place(place_id="geoapify:" + identity, provider_place_id=identity,
            name=props.get("name") or props.get("address_line1") or props.get("formatted") or "",
            city=city, country=country, latitude=props["lat"], longitude=props["lon"], address=props.get("formatted", ""),
            categories=props.get("categories") or [], source_url="https://www.geoapify.com/",
            retrieved_at=retrieved_at)

    def _failure(self, exc):
        # Never expose HTTP exception URLs: provider credentials are query parameters.
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        chain = []
        cause = exc
        while cause is not None and len(chain) < 8 and all(cause is not item for item in chain):
            chain.append(cause)
            cause = cause.__cause__ or cause.__context__
        text = " ".join(str(item).lower() for item in chain)
        code, hint = "provider_error", "Inspect provider configuration or response format."
        if any(isinstance(item, socket.gaierror) for item in chain):
            code, hint = "dns_resolution_failed", "Could not resolve the provider or proxy hostname. Check DNS and proxy settings."
        elif any(isinstance(item, ssl.SSLCertVerificationError) for item in chain) or "certificate_verify_failed" in text:
            code, hint = "tls_verification_failed", "TLS certificate verification failed. Configure PLACES_CA_BUNDLE for your trusted CA; certificate verification stays enabled."
        elif isinstance(exc, httpx.ProxyError):
            code, hint = "proxy_connection_failed", "Proxy connection failed. Check PLACES_PROXY_URL and whether the proxy is running."
        elif any(isinstance(item, ConnectionRefusedError) for item in chain):
            code, hint = "connection_refused", "Connection was refused. Check provider/proxy connectivity."
        elif isinstance(exc, httpx.ConnectTimeout):
            code, hint = "connection_timeout", "Connection timed out. Check whether this network requires an explicit PLACES_PROXY_URL."
        elif isinstance(exc, httpx.ConnectError):
            code, hint = "connection_failed", "Connection failed before an HTTP response. Check DNS, proxy and TLS configuration."
        elif status in {401, 403}:
            code, hint = "authentication_or_permission_failed", "Provider rejected access. Check the API key and its IP/origin restrictions."
        elif status == 429:
            code, hint = "rate_limited", "Provider rate limit or quota was reached."
        elif status:
            code, hint = "http_error", "Provider returned an HTTP error."
        elif str(exc) == "GEOAPIFY_API_KEY is not configured":
            code, hint = "missing_api_key", "GEOAPIFY_API_KEY is not configured."
        return PlaceToolResult(status="error", error_type=type(exc).__name__,
            diagnostic_code=code, network=places_network_settings(),
            message=f"Place provider request failed{f' (HTTP {status})' if status else ''}. {hint}",
            retryable=isinstance(exc, httpx.TransportError) or status == 429 or bool(status and status >= 500))

    def check_connection(self):
        """No API key or query is sent. Any HTTP response establishes connectivity only."""
        try:
            with places_http_client() as client:
                response = client.get("https://api.geoapify.com/v1/geocode/search", timeout=8)
            return {"reachable": True, "http_status": response.status_code, "network": places_network_settings(),
                    "message": "HTTP response received without credentials; this does not validate an API key."}
        except (httpx.HTTPError, ValueError, OSError) as exc:
            return {"reachable": False, "error": self._failure(exc).model_dump(mode="json")}

    def search_places(self, city, query, limit=3):
        args = SearchPlacesArguments(city=city, query=query, limit=limit)
        try:
            raw = self._request("/v1/geocode/search", {"text": f"{args.query}, {args.city}",
                "limit": args.limit, "format": "json", "lang": "en"})
            places = [place for props in raw["results"] if (place := self._place(props, raw["_retrieved_at"])) is not None
                      and place.city == canonical_destination(args.city)]
            return PlaceToolResult(status="ok" if places else "not_found", places=places)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError) as exc:
            return self._failure(exc)

    def get_place_details(self, place_id):
        args = PlaceDetailsArguments(place_id=place_id)
        if not args.place_id.startswith("geoapify:"):
            return PlaceToolResult(status="error", error_type="InvalidPlaceID", message="Expected geoapify: place ID.")
        identity = args.place_id.removeprefix("geoapify:")
        try:
            raw = self._request("/v2/place-details", {"id": identity, "features": "details", "lang": "en"})
            places = [place for feature in raw["features"]
                      if feature["properties"].get("feature_type", "details") == "details"
                      and (place := self._place(feature["properties"], raw["_retrieved_at"])) is not None]
            # Geoapify may canonicalize a geocoding ID (including its encoded name)
            # in the details response. Keep the queried ID as explicit provenance.
            return PlaceToolResult(status="ok" if places else "not_found", places=places[:1], requested_place_id=args.place_id)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError) as exc:
            return self._failure(exc)
