"""FAL AI integration for NodeTool."""

# Register the image provider
from nodetool.image.providers import register_image_provider
from nodetool.fal.fal_image_provider import FalProvider

register_image_provider("fal_ai", lambda: FalProvider())

__all__ = ["FalProvider"]
