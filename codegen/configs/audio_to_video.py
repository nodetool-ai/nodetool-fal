"""
Configuration for audio-to-video module.

This config file defines overrides and customizations for audio-to-video nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
"fal-ai/ltx-2-19b/distilled/audio-to-video/lora": {
        "class_name": "Ltx219BDistilledAudioToVideoLora",
        "docstring": "LTX-2 19B Distilled",
        "tags": ["video", "generation", "audio-to-video", "visualization", "lora"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/ltx-2-19b/audio-to-video/lora": {
        "class_name": "Ltx219BAudioToVideoLora",
        "docstring": "LTX-2 19B",
        "tags": ["video", "generation", "audio-to-video", "visualization", "lora"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/ltx-2-19b/distilled/audio-to-video": {
        "class_name": "Ltx219BDistilledAudioToVideo",
        "docstring": "LTX-2 19B Distilled",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/ltx-2-19b/audio-to-video": {
        "class_name": "Ltx219BAudioToVideo",
        "docstring": "LTX-2 19B",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/elevenlabs/dubbing": {
        "class_name": "ElevenlabsDubbing",
        "docstring": "ElevenLabs Dubbing",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/longcat-multi-avatar/image-audio-to-video": {
        "class_name": "LongcatMultiAvatarImageAudioToVideo",
        "docstring": "Longcat Multi Avatar",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/longcat-multi-avatar/image-audio-to-video/multi-speaker": {
        "class_name": "LongcatMultiAvatarImageAudioToVideoMultiSpeaker",
        "docstring": "Longcat Multi Avatar",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
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
