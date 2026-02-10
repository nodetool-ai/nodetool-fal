"""
Configuration for image_to_json module.

This config file defines overrides and customizations for image-to-json nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/bagel/understand": {
        "class_name": "BagelUnderstand",
        "docstring": "Bagel is a 7B parameter multimodal model from Bytedance-Seed that can generate both text and images.",
        "tags": ["vision", "analysis", "json", "image-understanding"],
        "use_cases": [
            "Image analysis to structured data",
            "Visual content understanding",
            "Automated image metadata extraction",
            "Content classification",
            "Image-based data extraction",
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
