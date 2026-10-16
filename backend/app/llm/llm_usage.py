import os
import math
from typing import Any

from backend.app.config import (
    llm_cache_read_price_per_mtok,
    llm_cache_write_price_per_mtok,
    llm_input_price_per_mtok,
    llm_output_price_per_mtok,
)
from backend.app.domain.models import LLMUsage


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


def usage_from_response(stage: str, model: str, response: Any, provider: str = "anthropic") -> LLMUsage:
    usage = getattr(response, "usage", None)
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    cache_creation_input_tokens = _usage_value(usage, "cache_creation_input_tokens")
    cache_read_input_tokens = _usage_value(usage, "cache_read_input_tokens")
    cost = _cost_usd(input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens)
    if provider == "openai":
        cache_creation_input_tokens = 0
        cache_read_input_tokens = _usage_value(getattr(usage, "input_tokens_details", None), "cached_tokens")
        # OpenAI input_tokens includes cached tokens; do not bill them twice.
        cost = None
        try:
            input_price = float(os.environ["OPENAI_INPUT_PRICE_PER_MTOK"])
            output_price = float(os.environ["OPENAI_OUTPUT_PRICE_PER_MTOK"])
            cache_price = float(os.environ["OPENAI_CACHE_READ_PRICE_PER_MTOK"]) if cache_read_input_tokens else 0.0
            if all(math.isfinite(v) and v >= 0 for v in (input_price, output_price, cache_price)):
                cost = round(((input_tokens - cache_read_input_tokens) * input_price
                              + cache_read_input_tokens * cache_price + output_tokens * output_price) / 1_000_000, 8)
        except (KeyError, ValueError):
            pass
    return LLMUsage(
        provider=provider,
        stage=stage,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        estimated_cost_usd=cost,
    )
