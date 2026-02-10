"""
Configuration for text_to_json module.

This config file defines overrides and customizations for text-to-json nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "bria/fibo-edit/edit/structured_instruction": {
        "class_name": "BriaFiboEditEditStructured_instruction",
        "docstring": "Structured Instructions Generation endpoint for Fibo Edit, Bria's newest editing model.",
        "tags": ["text", "analysis", "json", "extraction"],
        "use_cases": [
            "Text analysis to structured data",
            "Content extraction",
            "Data structuring",
            "Information extraction",
            "Text classification",
        ],
    },
    "bria/fibo-lite/generate/structured_prompt": {
        "class_name": "BriaFiboLiteGenerateStructured_prompt",
        "docstring": "Structured Prompt Generation endpoint for Fibo-Lite, Bria's SOTA Open source model",
        "tags": ["text", "analysis", "json", "extraction"],
        "use_cases": [
            "Text analysis to structured data",
            "Content extraction",
            "Data structuring",
            "Information extraction",
            "Text classification",
        ],
    },
    "bria/fibo-lite/generate/structured_prompt/lite": {
        "class_name": "BriaFiboLiteGenerateStructured_promptLite",
        "docstring": "Structured Prompt Generation endpoint for Fibo-Lite, Bria's SOTA Open source model",
        "tags": ["text", "analysis", "json", "extraction"],
        "use_cases": [
            "Text analysis to structured data",
            "Content extraction",
            "Data structuring",
            "Information extraction",
            "Text classification",
        ],
    },
    "bria/fibo/generate/structured_prompt": {
        "class_name": "BriaFiboGenerateStructured_prompt",
        "docstring": "Structured Prompt Generation endpoint for Fibo, Bria's SOTA Open source model",
        "tags": ["text", "analysis", "json", "extraction"],
        "use_cases": [
            "Text analysis to structured data",
            "Content extraction",
            "Data structuring",
            "Information extraction",
            "Text classification",
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
