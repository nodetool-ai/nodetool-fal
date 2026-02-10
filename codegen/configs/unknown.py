"""
Configuration for unknown module.

This config file defines overrides and customizations for unknown nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/workflow-utilities/interleave-video": {
        "class_name": "WorkflowUtilitiesInterleaveVideo",
        "docstring": "ffmpeg utility to interleave videos",
        "tags": ["utility", "processing", "general"],
        "use_cases": [
            "General media processing",
            "Utility operations",
            "Content manipulation",
            "Automated workflows",
            "Data processing",
        ],
    },
    "fal-ai/qwen-3-tts/clone-voice/1.7b": {
        "class_name": "Qwen3TtsCloneVoice17b",
        "docstring": "Clone your voices using Qwen3-TTS Clone-Voice model with zero shot cloning capabilities and use it on text-to-speech models to create speeches of yours!",
        "tags": ["utility", "processing", "general"],
        "use_cases": [
            "General media processing",
            "Utility operations",
            "Content manipulation",
            "Automated workflows",
            "Data processing",
        ],
    },
    "fal-ai/qwen-3-tts/clone-voice/0.6b": {
        "class_name": "Qwen3TtsCloneVoice06b",
        "docstring": "Clone your voices using Qwen3-TTS Clone-Voice model with zero shot cloning capabilities and use it on text-to-speech models to create speeches of yours!",
        "tags": ["utility", "processing", "general"],
        "use_cases": [
            "General media processing",
            "Utility operations",
            "Content manipulation",
            "Automated workflows",
            "Data processing",
        ],
    },
    "openrouter/router/audio": {
        "class_name": "OpenrouterRouterAudio",
        "docstring": "Run any ALM (Audio Language Model) with fal, powered by OpenRouter.",
        "tags": ["utility", "processing", "general"],
        "use_cases": [
            "General media processing",
            "Utility operations",
            "Content manipulation",
            "Automated workflows",
            "Data processing",
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
