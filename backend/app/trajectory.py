from dataclasses import asdict, is_dataclass
"""Per-chat trajectories. Context-local spans also work for concurrent API requests."""
import inspect
import json
import logging
import os
import re
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import perf_counter

from backend.app.config import PROJECT_ROOT

_CURRENT = ContextVar("trajectory", default=None)
_PARENT = ContextVar("trajectory_parent", default=None)
_LOG = logging.getLogger(__name__)


def clean(value):
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if any(s in str(k).lower() for s in ("api_key", "authorization", "secret", "password", "access_token")) else clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, str):
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEOAPIFY_API_KEY"):
            secret = os.environ.get(key)
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def error_details(exc):
    result = {"type": type(exc).__name__, "message": str(exc),
                  "cause": {"type": type(exc.__cause__).__name__, "message": str(exc.__cause__)} if exc.__cause__ else None,
                  "request_id": getattr(exc, "request_id", None),
                  "status_code": getattr(exc, "status_code", None)}
    if isinstance(exc, BaseExceptionGroup):
        result["exceptions"] = [error_details(child) for child in exc.exceptions[:5]]
    return clean(result)


class Trajectory:
    def __init__(self, conversation_id, parent_run_id, message, fields):
        self.started = perf_counter()
        self.data = {"schema_version": "1.0", "conversation_id": conversation_id,
                     "run_id": str(uuid.uuid4()), "parent_run_id": parent_run_id,
                     "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                     "input": clean({"message": message, "fields_before": fields}), "steps": []}
        directory = Path(os.environ.get("TRAJECTORY_DIR", str(PROJECT_ROOT / "data" / "trajectories")))
        self.path = directory / (self.data["run_id"] + ".traj.json")
        self.events_path = directory / (self.data["run_id"] + ".events.jsonl")
        self.event("run_started", {k: v for k, v in self.data.items() if k != "steps"})

    def event(self, event, payload):
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(clean({"event": event, "at": datetime.now(timezone.utc).isoformat(), **payload}), ensure_ascii=False, allow_nan=False) + "\n")
        except (OSError, ValueError):
            _LOG.exception("Could not persist trajectory event")

    def finish(self, response=None, error=None):
        self.data["duration_ms"] = round((perf_counter() - self.started) * 1000, 3)
        self.data["finished_at"] = datetime.now(timezone.utc).isoformat()
        if error:
            self.data.update(status="failed", error=error_details(error))
        else:
            fallback = any(s.status == "fallback" for s in response.trace)
            self.data["status"] = "completed_with_fallback" if fallback else "completed"
            self.data["result"] = clean({"assistant_message": response.assistant_message,
                "fields_after": response.extracted_fields, "missing_fields": response.missing_fields,
                "field_changes": {k: {"before": self.data["input"]["fields_before"].get(k), "after": v}
                                  for k, v in response.extracted_fields.items()
                                  if self.data["input"]["fields_before"].get(k) != v},
                "fallback_used": fallback, "plan_created": response.plan is not None,
                "plan": response.plan, "llm_usage": response.llm_usage, "trace": response.trace})
        self.event("run_finished", {"status": self.data["status"], "duration_ms": self.data["duration_ms"]})
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(clean(self.data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(self.path)
        except (OSError, ValueError):
            _LOG.exception("Could not persist trajectory JSON")
            self.data["persistence_error"] = "Could not persist trajectory JSON"


def record_trace(stage, detail, status):
    recorder = _CURRENT.get()
    if recorder:
        recorder.event("trace", {"stage": stage, "detail": detail, "status": status})


def observed(name, kind):
    """Record function boundaries, validated outputs and errors without changing behavior."""
    def decorate(function):
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapper(*args, **kwargs):
                recorder = _CURRENT.get()
                if recorder is None:
                    return await function(*args, **kwargs)
                bound = inspect.signature(function).bind(*args, **kwargs)
                step = {"step_id": str(uuid.uuid4()), "parent_step_id": _PARENT.get(),
                        "name": name, "kind": kind, "status": "running", "input": clean(bound.arguments)}
                recorder.data["steps"].append(step)
                recorder.event("step_started", {"step": step})
                token = _PARENT.set(step["step_id"])
                started = perf_counter()
                try:
                    result = await function(*args, **kwargs)
                    step.update(status="ok", output=clean(result))
                    reason = getattr(result, "fallback_reason", None)
                    if reason:
                        step.update(status="fallback", fallback={"reason": reason})
                    return result
                except BaseException as exc:
                    step.update(status="error", error=error_details(exc))
                    raise
                finally:
                    step["duration_ms"] = round((perf_counter() - started) * 1000, 3)
                    recorder.event("step_finished", {"step": step})
                    _PARENT.reset(token)
            return async_wrapper
        @wraps(function)
        def wrapper(*args, **kwargs):
            recorder = _CURRENT.get()
            if recorder is None:
                return function(*args, **kwargs)
            bound = inspect.signature(function).bind(*args, **kwargs)
            step = {"step_id": str(uuid.uuid4()), "parent_step_id": _PARENT.get(),
                    "name": name, "kind": kind, "status": "running", "input": clean(bound.arguments)}
            recorder.data["steps"].append(step)
            recorder.event("step_started", {"step": step})
            token = _PARENT.set(step["step_id"])
            started = perf_counter()
            try:
                result = function(*args, **kwargs)
                step.update(status="ok", output=clean(result))
                if kind == "retrieval" and isinstance(result, list):
                    step["chunk_count"] = len(result)
                reason = getattr(result, "router_fallback_reason", None) or getattr(result, "fallback_reason", None)
                if reason:
                    step.update(status="fallback", fallback={"reason": reason})
                return result
            except Exception as exc:
                step.update(status="error", error=error_details(exc))
                raise
            finally:
                step["duration_ms"] = round((perf_counter() - started) * 1000, 3)
                recorder.event("step_finished", {"step": step})
                _PARENT.reset(token)
        return wrapper
    return decorate


def create_message(client, *, stage, prompt_version, **kwargs):
    """Record SDK-call timing; SDK internal retries are not separate observed attempts."""
    recorder = _CURRENT.get()
    if recorder is None:
        return client.messages.create(**kwargs)

    @observed(stage + "_api", "llm")
    def call(metadata):
        step = next(s for s in recorder.data["steps"] if s["step_id"] == _PARENT.get())
        step["llm"] = metadata.copy()
        response = client.messages.create(**kwargs)
        from backend.app.llm_usage import usage_from_response
        step = next(s for s in recorder.data["steps"] if s["step_id"] == _PARENT.get())
        step["llm"] = {**metadata, "request_id": getattr(response, "_request_id", None),
                       "usage": clean(usage_from_response(stage, kwargs["model"], response))}
        return response

    timeout = getattr(client, "timeout", None)
    metadata = {"provider": "anthropic", "model": kwargs["model"], "prompt_version": prompt_version,
                "max_tokens": kwargs.get("max_tokens"), "max_retries": getattr(client, "max_retries", None),
                "timeout_seconds": {k: getattr(timeout, k, None) for k in ("connect", "read", "write", "pool")} if hasattr(timeout, "read") else timeout,
                "usage": None, "attempt_visibility": "sdk_call_including_internal_retries"}
    return call(metadata)
