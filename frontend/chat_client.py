"""HTTP client for the local chat backend."""
import requests


def request_chat(url: str, payload: dict) -> dict:
    # The backend runs on loopback. System proxies (including macOS settings)
    # can intercept localhost even when no HTTP_PROXY environment variable exists.
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(url, json=payload, timeout=180)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            if isinstance(detail, str):
                raise requests.HTTPError(detail, response=response) from exc
            raise
        return response.json()
