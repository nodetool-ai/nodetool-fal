"""
OpenAPI schema fetcher for FAL endpoints.

This module handles fetching and caching OpenAPI schemas from FAL.ai endpoints.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Optional
import httpx


class SchemaFetcher:
    """Fetches and caches OpenAPI schemas for FAL endpoints."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the schema fetcher.
        
        Args:
            cache_dir: Directory for caching schemas. Defaults to .codegen_cache/
        """
        self.cache_dir = cache_dir or Path(".codegen_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, endpoint_id: str) -> str:
        """Generate cache key for endpoint."""
        return hashlib.sha256(endpoint_id.encode()).hexdigest()[:16]

    def _cache_path(self, endpoint_id: str) -> Path:
        """Get cache file path for endpoint."""
        return self.cache_dir / f"{self._cache_key(endpoint_id)}.json"

    async def fetch_schema(
        self, endpoint_id: str, use_cache: bool = True
    ) -> dict[str, Any]:
        """
        Fetch OpenAPI schema for a FAL endpoint.
        
        Args:
            endpoint_id: FAL endpoint ID (e.g., 'fal-ai/flux/dev')
            use_cache: Whether to use cached schema if available
            
        Returns:
            OpenAPI schema dictionary
        """
        cache_path = self._cache_path(endpoint_id)
        
        # Check cache first
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text())
        
        # Fetch from FAL API
        url = f"https://fal.ai/api/openapi/queue/openapi.json?endpoint_id={endpoint_id}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            schema = response.json()
        
        # Save to cache
        cache_path.write_text(json.dumps(schema, indent=2))
        
        return schema

    def get_cached_schema(self, endpoint_id: str) -> Optional[dict[str, Any]]:
        """
        Get cached schema if available.
        
        Args:
            endpoint_id: FAL endpoint ID
            
        Returns:
            Cached schema or None
        """
        cache_path = self._cache_path(endpoint_id)
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        return None
