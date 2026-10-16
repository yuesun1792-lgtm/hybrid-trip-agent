"""Typed read-only tools; remote/MCP adapters can implement the same boundary."""
from typing import Any, Callable

from pydantic import BaseModel, Field

from backend.app.execution import RUN_CONTEXT
from backend.app.trajectory import observed


class SearchArguments(BaseModel):
    query: str
    k: int = Field(default=6, ge=1, le=10)
    destination: str | None = None


class TransportArguments(BaseModel):
    origin: str
    destination: str


class ToolResult(BaseModel):
    tool_name: str
    data: Any = None
    error_type: str | None = None
    error_message: str | None = None
    retryable: bool = False


    @property
    def fallback_reason(self):
        return f"{self.error_type}: {self.error_message}" if self.error_type else None


class ToolExecutor:
    def __init__(self):
        self.registry: dict[str, tuple[type[BaseModel], Callable]] = {}

    def register(self, name, input_model, function):
        self.registry[name] = (input_model, function)

    @observed("tool_execution", "tool")
    def execute(self, name: str, arguments: dict) -> ToolResult:
        if name not in self.registry:
            raise ValueError(f"Tool is not registered: {name}")
        schema, function = self.registry[name]
        validated = schema.model_validate(arguments)
        context = RUN_CONTEXT.get()
        if context:
            context.admit("tool")
        try:
            return ToolResult(tool_name=name, data=function(**validated.model_dump()))
        except (TimeoutError, ConnectionError, OSError) as exc:
            return ToolResult(tool_name=name, error_type=type(exc).__name__, error_message=str(exc), retryable=True)


def default_executor():
    from backend.app import knowledge, tools
    executor = ToolExecutor()
    executor.register("search_travel_knowledge", SearchArguments, knowledge.search_travel_knowledge)
    executor.register("estimate_transport_time", TransportArguments, tools.estimate_transport_time)
    return executor
