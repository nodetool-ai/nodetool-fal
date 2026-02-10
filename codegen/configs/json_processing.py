"""
Configuration for json_processing module.

This config file defines overrides and customizations for json-processing nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/ffmpeg-api/loudnorm": {
        "class_name": "FfmpegApiLoudnorm",
        "docstring": "Get EBU R128 loudness normalization from audio files using FFmpeg API.",
        "tags": ["json", "processing", "data", "utility"],
        "use_cases": [
            "JSON data processing",
            "Data transformation",
            "Metadata extraction",
            "Audio analysis",
            "Media processing utilities",
        ],
    },
    "fal-ai/ffmpeg-api/waveform": {
        "class_name": "FfmpegApiWaveform",
        "docstring": "Get waveform data from audio files using FFmpeg API.",
        "tags": ["json", "processing", "data", "utility"],
        "use_cases": [
            "JSON data processing",
            "Data transformation",
            "Metadata extraction",
            "Audio analysis",
            "Media processing utilities",
        ],
    },
    "fal-ai/ffmpeg-api/metadata": {
        "class_name": "FfmpegApiMetadata",
        "docstring": "Get encoding metadata from video and audio files using FFmpeg API.",
        "tags": ["json", "processing", "data", "utility"],
        "use_cases": [
            "JSON data processing",
            "Data transformation",
            "Metadata extraction",
            "Audio analysis",
            "Media processing utilities",
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
