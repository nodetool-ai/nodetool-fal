"""
Configuration for text_to_text module.

This config file defines overrides and customizations for text-to-text nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "half-moon-ai/ai-detector/detect-text": {
        "class_name": "HalfMoonAiAiDetectorDetectText",
        "docstring": "AI Detector (Text) is an advanced AI service that analyzes a passage and returns a verdict on whether it was likely written by AI.",
        "tags": ["text", "processing", "transformation", "nlp"],
        "use_cases": [
            "Text transformation",
            "Content analysis",
            "Text classification",
            "Language processing",
            "Content detection",
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
