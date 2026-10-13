from typing import Any

from backend.app.config import (
    llm_cache_read_price_per_mtok,
    llm_cache_write_price_per_mtok,
    llm_input_price_per_mtok,
    llm_output_price_per_mtok,
)
from backend.app.models import LLMUsage


def _usage_value(usage: Any, name: str) -> int:
    value = getattr(usage, name, 0) if usage is not None else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
) -> float:
    cost = (
        input_tokens / 1_000_000 * llm_input_price_per_mtok()
        + output_tokens / 1_000_000 * llm_output_price_per_mtok()
        + cache_creation_input_tokens / 1_000_000 * llm_cache_write_price_per_mtok()
        + cache_read_input_tokens / 1_000_000 * llm_cache_read_price_per_mtok()
    )
    return round(cost, 8)


def usage_from_response(stage: str, model: str, response: Any) -> LLMUsage:
    usage = getattr(response, "usage", None)
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    cache_creation_input_tokens = _usage_value(usage, "cache_creation_input_tokens")
    cache_read_input_tokens = _usage_value(usage, "cache_read_input_tokens")
    return LLMUsage(
        stage=stage,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        estimated_cost_usd=_cost_usd(
            input_tokens,
            output_tokens,
            cache_creation_input_tokens,
            cache_read_input_tokens,
        ),
    )
