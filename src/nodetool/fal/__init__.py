"""FAL AI integration for NodeTool."""

# Register the image provider
from nodetool.image.providers import register_image_provider
from nodetool.fal.fal_image_provider import FalImageProvider

register_image_provider("fal_ai", lambda: FalImageProvider())

__all__ = ["FalImageProvider"]
