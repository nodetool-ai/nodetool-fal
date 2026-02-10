"""
Configuration for audio_to_text module.

This config file defines overrides and customizations for audio-to-text nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/nemotron/asr/stream": {
        "class_name": "NemotronAsrStream",
        "docstring": "Use the fast speed and pin point accuracy of nemotron to transcribe your texts.",
        "tags": ["speech", "recognition", "transcription", "audio-analysis"],
        "use_cases": [
            "Speech recognition",
            "Audio transcription",
            "Speaker diarization",
            "Voice activity detection",
            "Meeting transcription",
        ],
    },
    "fal-ai/nemotron/asr": {
        "class_name": "NemotronAsr",
        "docstring": "Use the fast speed and pin point accuracy of nemotron to transcribe your texts.",
        "tags": ["speech", "recognition", "transcription", "audio-analysis"],
        "use_cases": [
            "Speech recognition",
            "Audio transcription",
            "Speaker diarization",
            "Voice activity detection",
            "Meeting transcription",
        ],
    },
    "fal-ai/silero-vad": {
        "class_name": "SileroVad",
        "docstring": "Detect speech presence and timestamps with accuracy and speed using the ultra-lightweight Silero VAD model",
        "tags": ["speech", "recognition", "transcription", "audio-analysis"],
        "use_cases": [
            "Speech recognition",
            "Audio transcription",
            "Speaker diarization",
            "Voice activity detection",
            "Meeting transcription",
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
