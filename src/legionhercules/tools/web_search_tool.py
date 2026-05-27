"""Web search tool using DuckDuckGo for LEGIONHERCULES."""

from __future__ import annotations

from typing import Any

from legionhercules.tools.base import Tool, ToolResult
from legionhercules.utils.logging import get_logger

logger = get_logger(__name__)


class WebSearchTool(Tool):
    """Tool for searching the web using DuckDuckGo."""
    
    def __init__(self, max_results: int = 5):
        super().__init__(
            name="web_search",
            description="Search the web using DuckDuckGo. Returns search results with titles, URLs, and snippets."
        )
        self.max_results = max_results
    
    async def execute(
        self,
        query: str,
        max_results: int | None = None,
        region: str = "wt-wt",
        safesearch: str = "moderate"
    ) -> ToolResult:
        """Search the web using DuckDuckGo.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            region: Region code (e.g., 'us-en', 'wt-wt' for worldwide)
            safesearch: Safe search level ('on', 'moderate', 'off')
        """
        try:
            # Import here to handle optional dependency gracefully
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return ToolResult(
                    success=False,
                    error="DuckDuckGo search not available. Install with: pip install duckduckgo-search"
                )
            
            num_results = max_results or self.max_results
            
            logger.debug(f"Searching DuckDuckGo: {query[:50]}...")
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch,
                    max_results=num_results
                )
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                    })
            
            logger.debug(f"Found {len(results)} results")
            
            # Format output
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {result['title']}\n"
                    f"   URL: {result['url']}\n"
                    f"   {result['snippet'][:200]}..."
                )
            
            output = "\n\n".join(formatted_results) if formatted_results else "No results found"
            
            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "query": query,
                    "results_count": len(results),
                    "region": region,
                }
            )
            
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            return ToolResult(success=False, error=str(e))
    
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                },
                "region": {
                    "type": "string",
                    "description": "Region code (e.g., 'us-en', 'wt-wt' for worldwide)",
                    "default": "wt-wt"
                },
                "safesearch": {
                    "type": "string",
                    "description": "Safe search level ('on', 'moderate', 'off')",
                    "default": "moderate"
                }
            },
            "required": ["query"]
        }
