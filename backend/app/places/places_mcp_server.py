"""Local read-only MCP server: python -m backend.app.places.places_mcp_server (stdio)."""
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from backend.app.domain.place_models import PlaceToolResult
from backend.app.places.place_provider import GeoapifyPlaces


def create_server(provider=None):
    provider = provider or GeoapifyPlaces()
    server = FastMCP("eurotrip-places", log_level="WARNING")
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)

    @server.tool(annotations=annotations)
    def search_places(city: str, query: str, limit: int = 3) -> PlaceToolResult:
        """Find a named venue or address within a city using real Geoapify data.

        Use specific names from the activity/evidence, not generic interests or a whole
        city. Results may be ambiguous or empty. Search results are candidates, not an
        automatic selection. Use their exact place_id with get_place_details next.
        """
        return provider.search_places(city, query, limit)

    @server.tool(annotations=annotations)
    def get_place_details(place_id: str) -> PlaceToolResult:
        """Fetch real provider details for an exact geoapify: ID returned by search_places.

        Never invent an ID or supply a URL. Details confirm identity/coordinates only,
        not current opening hours, booking availability or route feasibility.
        """
        return provider.get_place_details(place_id)

    return server


if __name__ == "__main__":
    create_server().run(transport="stdio")
