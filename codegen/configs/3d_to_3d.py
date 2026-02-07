"""
Configuration for 3d-to-3d module.

This config file defines overrides and customizations for 3d-to-3d nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
"fal-ai/ultrashape": {
        "class_name": "Ultrashape",
        "docstring": "Ultrashape",
        "tags": ["3d_to_3d"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/sam-3/3d-align": {
        "class_name": "Sam33DAlign",
        "docstring": "Sam 3",
        "tags": ["3d_to_3d"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

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
