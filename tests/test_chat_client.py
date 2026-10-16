from unittest.mock import MagicMock

import pytest
import requests

from frontend.chat_client import request_chat


def test_chat_bypasses_system_proxy_and_preserves_payload(monkeypatch):
    session = MagicMock()
    session.__enter__.return_value = session
    response = session.post.return_value
    response.json.return_value = {"conversation_id": "conv", "assistant_message": "hello"}
    monkeypatch.setattr(requests, "Session", lambda: session)
    payload = {"message": "计划巴黎三日游 美食 文化"}
    result = request_chat("http://127.0.0.1:8000/chat", payload)
    assert session.trust_env is False
    session.post.assert_called_once_with("http://127.0.0.1:8000/chat", json=payload, timeout=180)
    response.raise_for_status.assert_called_once()
    assert result["conversation_id"] == "conv"
    session.__exit__.assert_called_once()


def test_chat_http_errors_propagate_to_ui(monkeypatch):
    session = MagicMock()
    session.__enter__.return_value = session
    session.post.return_value.raise_for_status.side_effect = requests.HTTPError("502")
    monkeypatch.setattr(requests, "Session", lambda: session)
    with pytest.raises(requests.HTTPError):
        request_chat("http://127.0.0.1:8000/chat", {"message": "Paris"})


def test_streamlit_starts_with_script_directory_on_path(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    script = Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py"
    # streamlit run inserts the script directory, rather than its parent, on sys.path.
    monkeypatch.syspath_prepend(str(script.parent))
    app = AppTest.from_file(str(script)).run()
    assert not app.exception, [error.message for error in app.exception]
