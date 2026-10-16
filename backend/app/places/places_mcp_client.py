"""A bounded stdio MCP session. No arbitrary commands or model-supplied endpoints."""
from contextlib import asynccontextmanager
import os
import sys
from datetime import timedelta
from urllib.request import getproxies

from backend.app.config import PROJECT_ROOT
from backend.app.config import env_flag
from backend.app.runtime.execution import RUN_CONTEXT
from backend.app.domain.place_models import SearchPlacesArguments, PlaceDetailsArguments, PlaceToolResult
from backend.app.runtime.trajectory import observed

SCHEMAS = {"search_places": SearchPlacesArguments, "get_place_details": PlaceDetailsArguments}


def places_server_environment():
    names = ("PATH", "GEOAPIFY_API_KEY", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
             "https_proxy", "http_proxy", "all_proxy", "no_proxy", "PLACES_TRUST_ENV", "PLACES_PROXY_URL", "PLACES_CA_BUNDLE")
    environment = {key: os.environ[key] for key in names if key in os.environ}
    # On macOS proxy settings may exist only in System Settings, not shell variables.
    # Resolve them in the host so a restricted child sees the same HTTPS proxy.
    if env_flag("PLACES_TRUST_ENV", True) and not environment.get("PLACES_PROXY_URL"):
        proxies = getproxies()
        if not environment.get("HTTPS_PROXY") and not environment.get("https_proxy") and proxies.get("https"):
            environment["HTTPS_PROXY"] = proxies["https"]
        if not environment.get("NO_PROXY") and not environment.get("no_proxy") and proxies.get("no"):
            environment["NO_PROXY"] = proxies["no"]
    return environment


class PlacesMCPClient:
    def __init__(self, session):
        self.session = session
        self.tools = []

    async def discover(self):
        response = await self.session.list_tools()
        # Only explicitly approved names, irrespective of server annotations.
        self.tools = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                      for t in response.tools if t.name in SCHEMAS]
        if {t["name"] for t in self.tools} != set(SCHEMAS):
            raise ValueError("Places MCP server must expose search_places and get_place_details")
        return self.tools

    @observed("places_mcp_call", "tool")
    async def execute(self, name, arguments):
        if name not in SCHEMAS:
            return PlaceToolResult(status="error", error_type="UnknownTool", message="Tool is not allowed in this node.")
        context = RUN_CONTEXT.get()
        if context:
            context.admit("tool")
        try:
            args = SCHEMAS[name].model_validate(arguments)
        except ValueError:
            return PlaceToolResult(status="error", error_type="InvalidArguments", message="Arguments do not match tool schema.")
        timeout = min(12.0, context.remaining()) if context else 12.0
        try:
            result = await self.session.call_tool(name, args.model_dump(), read_timeout_seconds=timedelta(seconds=timeout))
            if result.isError:
                return PlaceToolResult(status="error", error_type="MCPToolError", message="MCP tool execution failed.")
            data = result.structuredContent
            if data is None:
                import json
                data = json.loads(next(item.text for item in result.content if item.type == "text"))
            return PlaceToolResult.model_validate(data)
        except (TimeoutError, ValueError, StopIteration) as exc:
            return PlaceToolResult(status="error", error_type=type(exc).__name__, message="Invalid or timed-out MCP result.", retryable=True)


@asynccontextmanager
async def places_session():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    # Never forward model-provider keys to the tool server.
    environment = places_server_environment()
    parameters = StdioServerParameters(command=sys.executable,
        args=["-m", "backend.app.places.places_mcp_server"], cwd=str(PROJECT_ROOT), env=environment)
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=12)) as session:
            await session.initialize()
            yield PlacesMCPClient(session)
