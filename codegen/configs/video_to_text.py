"""
Configuration for video_to_text module.

This config file defines overrides and customizations for video-to-text nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "openrouter/router/video/enterprise": {
        "class_name": "OpenrouterRouterVideoEnterprise",
        "docstring": "Run any VLM (Video Language Model) with fal, powered by OpenRouter.",
        "tags": ["video", "transcription", "analysis", "video-understanding"],
        "use_cases": [
            "Video transcription",
            "Video content analysis",
            "Automated captioning",
            "Video understanding",
            "Content indexing",
        ],
    },
    "openrouter/router/video": {
        "class_name": "OpenrouterRouterVideo",
        "docstring": "Run any VLM (Video Language Model) with fal, powered by OpenRouter.",
        "tags": ["video", "transcription", "analysis", "video-understanding"],
        "use_cases": [
            "Video transcription",
            "Video content analysis",
            "Automated captioning",
            "Video understanding",
            "Content indexing",
        ],
    },
}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """
    Get configuration for an endpoint.

    Args:
        endpoint_id: FAL endpoint ID

    Returns:
        Configuration dictionary
    """
    return CONFIGS.get(endpoint_id, {})
