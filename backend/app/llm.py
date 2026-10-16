"""Provider-neutral text generation for routing, itinerary writing and RAG."""
from dataclasses import dataclass
import json

from backend.app.config import (
    env_flag, llm_api_key, llm_max_retries, llm_model, llm_provider, llm_timeout_seconds,
)
from backend.app.llm_usage import usage_from_response
from backend.app.execution import RUN_CONTEXT
from backend.app.models import LLMUsage
from backend.app.trajectory import _CURRENT, _PARENT, clean, observed


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    usage: LLMUsage | None
    request_id: str | None


class LLMResponseError(ValueError):
    """No complete text was returned by the provider."""


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall]
    continuation: list[dict]
    provider: str
    usage: LLMUsage | None = None


def _make_client(provider: str, api_key: str):
    import httpx

    if provider == "openai":
        from openai import OpenAI as SDKClient
    else:
        from anthropic import Anthropic as SDKClient
    context = RUN_CONTEXT.get()
    seconds = min(llm_timeout_seconds(), context.remaining()) if context else llm_timeout_seconds()
    timeout = httpx.Timeout(seconds, connect=min(10.0, seconds))
    options = {"api_key": api_key, "timeout": timeout, "max_retries": llm_max_retries(),
               "http_client": httpx.Client(timeout=timeout, trust_env=env_flag("LLM_TRUST_ENV", True))}
    return SDKClient(**options)


def generate_text(prompt: str, *, stage: str, max_tokens: int = 700,
                  prompt_version: str = "v1", response_schema: dict | None = None) -> LLMResult:
    provider = llm_provider()
    metadata = {"provider": provider, "model": None, "prompt_version": prompt_version,
                "max_tokens": max_tokens, "max_retries": llm_max_retries(),
                "timeout_seconds": {"connect": min(10.0, llm_timeout_seconds()),
                                    "read": llm_timeout_seconds(), "write": llm_timeout_seconds(),
                                    "pool": llm_timeout_seconds()},
                "usage": None, "request_id": None,
                "response_format": "json_schema" if provider == "openai" and response_schema else "text",
                "attempt_visibility": "sdk_call_including_internal_retries"}

    @observed(stage + "_api", "llm")
    def call(metadata):
        recorder = _CURRENT.get()
        step = next((s for s in recorder.data["steps"] if s["step_id"] == _PARENT.get()), None) if recorder else None
        if step is not None:
            step["llm"] = metadata
        model = llm_model()
        metadata["model"] = model
        api_key = llm_api_key()
        if not api_key:
            raise ValueError(f"{provider.upper()}_API_KEY is not configured")
        context = RUN_CONTEXT.get()
        if context:
            context.admit("llm", output_tokens=max_tokens)
            remaining = min(llm_timeout_seconds(), context.remaining())
            metadata["timeout_seconds"] = {"connect": min(10.0, remaining), "read": remaining,
                                           "write": remaining, "pool": remaining}
        with _make_client(provider, api_key) as client:
            if provider == "openai":
                options = {}
                if response_schema is not None:
                    options["text"] = {"format": {"type": "json_schema", "name": "structured_response",
                                                    "strict": True, "schema": response_schema}}
                response = client.responses.create(model=model, input=prompt, max_output_tokens=max_tokens,
                                                   store=False, **options)
                text = response.output_text
                metadata["request_id"] = getattr(response, "_request_id", None)

            else:
                response = client.messages.create(model=model, max_tokens=max_tokens,
                                                   messages=[{"role": "user", "content": prompt}])
                text = "".join(block.text for block in response.content if getattr(block, "type", "text") == "text")
                metadata["request_id"] = getattr(response, "_request_id", None)

            usage = usage_from_response(stage, model, response, provider=provider) if getattr(response, "usage", None) is not None else None
            metadata["usage"] = clean(usage)
            if provider == "openai" and response.status != "completed":
                raise LLMResponseError(f"OpenAI response status={response.status}; details={getattr(response, 'incomplete_details', None)}")
            if provider == "anthropic" and getattr(response, "stop_reason", None) in {"max_tokens", "refusal"}:
                raise LLMResponseError(f"Anthropic response stop_reason={response.stop_reason}")
            if not text or not text.strip():
                raise LLMResponseError("LLM returned no text (empty response or refusal)")
            return LLMResult(text=text.strip(), model=model, provider=provider,
                             usage=usage, request_id=metadata["request_id"])

    return call(metadata)


def tool_result_messages(provider: str, results: list[tuple[str, dict]]) -> list[dict]:
    if provider == "openai":
        return [{"type": "function_call_output", "call_id": identity, "output": json.dumps(data, ensure_ascii=False)}
                for identity, data in results]
    return [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": identity,
        "content": json.dumps(data, ensure_ascii=False), "is_error": data.get("status") == "error"}
        for identity, data in results]}]


def generate_tool_turn(history: list[dict], tools: list[dict], *, stage="place_resolver", max_tokens=1800) -> ModelTurn:
    """Native function calling; preserve all provider continuation items, including reasoning."""
    provider = llm_provider()
    metadata = {"provider": provider, "model": llm_model(), "prompt_version": "places_tools_v1",
                "usage": None, "max_tokens": max_tokens, "request_id": None}

    @observed(stage + "_api", "llm")
    def call(metadata):
        recorder = _CURRENT.get()
        if recorder:
            step = next(s for s in recorder.data["steps"] if s["step_id"] == _PARENT.get())
            step["llm"] = metadata
        key = llm_api_key()
        if not key:
            raise ValueError("Selected provider API key is not configured")
        context = RUN_CONTEXT.get()
        if context:
            context.admit("llm", output_tokens=max_tokens)
        with _make_client(provider, key) as client:
            if provider == "openai":
                response = client.responses.create(model=metadata["model"], input=history,
                    include=["reasoning.encrypted_content"],
                    tools=[{"type": "function", "name": t["name"], "description": t["description"],
                            "parameters": t["input_schema"], "strict": False} for t in tools],
                    tool_choice="auto", parallel_tool_calls=True, max_output_tokens=max_tokens, store=False)
                metadata["usage"] = clean(usage_from_response(stage, metadata["model"], response, provider=provider)) if response.usage else None
                metadata["request_id"] = getattr(response, "_request_id", None)
                if response.status != "completed":
                    raise LLMResponseError(f"Tool response status={response.status}")
                continuation = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
                calls = [ToolCall(item.call_id, item.name, json.loads(item.arguments))
                         for item in response.output if item.type == "function_call"]
                text = response.output_text or ""
            else:
                response = client.messages.create(model=metadata["model"], messages=history,
                    tools=tools, max_tokens=max_tokens)
                metadata["usage"] = clean(usage_from_response(stage, metadata["model"], response, provider=provider)) if response.usage else None
                metadata["request_id"] = getattr(response, "_request_id", None)
                if response.stop_reason in {"max_tokens", "refusal", "pause_turn"}:
                    raise LLMResponseError(f"Tool response stop_reason={response.stop_reason}")
                continuation = [{"role": "assistant", "content": [b.model_dump(mode="json", exclude_none=True) for b in response.content]}]
                calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
                text = "".join(b.text for b in response.content if b.type == "text")
            if any(not isinstance(c.arguments, dict) for c in calls):
                raise LLMResponseError("Tool arguments must be JSON objects")
            if not calls and not text.strip():
                raise LLMResponseError("Empty tool turn")
            return ModelTurn(text=text, tool_calls=calls, continuation=continuation, provider=provider,
                             usage=LLMUsage.model_validate(metadata["usage"]) if metadata["usage"] else None)

    return call(metadata)
