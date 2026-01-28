"""FAL AI integration for NodeTool."""

# Register the provider
from nodetool.providers.base import register_provider
from nodetool.metadata.types import Provider
from nodetool.fal.fal_provider import FalProvider

# Register as general provider (handles image and video generation)
register_provider(Provider.FalAI)(FalProvider)

__all__ = ["FalProvider"]
