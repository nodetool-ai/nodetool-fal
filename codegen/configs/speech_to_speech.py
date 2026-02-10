"""
Configuration for speech_to_speech module.

This config file defines overrides and customizations for speech-to-speech nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "resemble-ai/chatterboxhd/speech-to-speech": {
        "class_name": "ResembleAiChatterboxhdSpeechToSpeech",
        "docstring": "Transform voices using Resemble AI's Chatterbox. Convert audio to new voices or your own samples, with expressive results and built-in perceptual watermarking.",
        "tags": ["speech", "voice", "transformation", "cloning"],
        "use_cases": [
            "Voice cloning and transformation",
            "Real-time voice conversion",
            "Voice style transfer",
            "Speech enhancement",
            "Accent conversion",
        ],
    },
    "fal-ai/chatterbox/speech-to-speech": {
        "class_name": "ChatterboxSpeechToSpeech",
        "docstring": "Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. Use the first tts from resemble ai.",
        "tags": ["speech", "voice", "transformation", "cloning"],
        "use_cases": [
            "Voice cloning and transformation",
            "Real-time voice conversion",
            "Voice style transfer",
            "Speech enhancement",
            "Accent conversion",
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
