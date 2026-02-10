"""
Configuration for video_to_audio module.

This config file defines overrides and customizations for video-to-audio nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/sam-audio/visual-separate": {
        "class_name": "SamAudioVisualSeparate",
        "docstring": "Audio separation with SAM Audio. Isolate any sound using natural language—professional-grade audio editing made simple for creators, researchers, and accessibility applications.",
        "tags": ["audio", "extraction", "video-to-audio", "processing"],
        "use_cases": [
            "Audio extraction from video",
            "Sound separation",
            "Video audio analysis",
            "Music extraction",
            "Sound effect isolation",
        ],
    },
    "mirelo-ai/sfx-v1.5/video-to-audio": {
        "class_name": "MireloAiSfxV15VideoToAudio",
        "docstring": "Generate synced sounds for any video, and return the new sound track (like MMAudio)",
        "tags": ["audio", "extraction", "video-to-audio", "processing"],
        "use_cases": [
            "Audio extraction from video",
            "Sound separation",
            "Video audio analysis",
            "Music extraction",
            "Sound effect isolation",
        ],
    },
    "fal-ai/kling-video/video-to-audio": {
        "class_name": "KlingVideoVideoToAudio",
        "docstring": "Generate audio from input videos using Kling",
        "tags": ["audio", "extraction", "video-to-audio", "processing"],
        "use_cases": [
            "Audio extraction from video",
            "Sound separation",
            "Video audio analysis",
            "Music extraction",
            "Sound effect isolation",
        ],
    },
    "mirelo-ai/sfx-v1/video-to-audio": {
        "class_name": "MireloAiSfxV1VideoToAudio",
        "docstring": "Generate synced sounds for any video, and return the new sound track (like MMAudio)",
        "tags": ["audio", "extraction", "video-to-audio", "processing"],
        "use_cases": [
            "Audio extraction from video",
            "Sound separation",
            "Video audio analysis",
            "Music extraction",
            "Sound effect isolation",
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
