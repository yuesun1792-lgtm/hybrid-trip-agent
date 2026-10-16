"""Bounded execution context, independent of the workflow scheduler."""
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic
from threading import Lock

from backend.app.config import _bounded_int, _bounded_float


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RunContext:
    deadline: float = field(default_factory=lambda: monotonic() + _bounded_float("RUN_TIMEOUT_SECONDS", 180, 10, 600))
    max_llm_calls: int = field(default_factory=lambda: _bounded_int("RUN_MAX_LLM_CALLS", 12, 1, 50))
    max_tool_calls: int = field(default_factory=lambda: _bounded_int("RUN_MAX_TOOL_CALLS", 40, 1, 200))
    max_output_tokens: int = field(default_factory=lambda: _bounded_int("RUN_MAX_OUTPUT_TOKENS", 32000, 1000, 100000))
    reserved_output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    lock: Lock = field(default_factory=Lock, repr=False)

    def remaining(self):
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise BudgetExceeded("Run deadline exceeded")
        return remaining

    def admit(self, kind, output_tokens=0):
        with self.lock:
            self.remaining()
            attr, limit = ("llm_calls", self.max_llm_calls) if kind == "llm" else ("tool_calls", self.max_tool_calls)
            if getattr(self, attr) >= limit:
                raise BudgetExceeded(f"Run {kind} call budget exceeded")
            if kind == "llm":
                if self.reserved_output_tokens + output_tokens > self.max_output_tokens:
                    raise BudgetExceeded("Run output-token reservation budget exceeded")
                self.reserved_output_tokens += output_tokens
            setattr(self, attr, getattr(self, attr) + 1)


RUN_CONTEXT = ContextVar("planning_run_context", default=None)
