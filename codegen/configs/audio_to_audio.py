"""
Configuration for audio_to_audio module.

This config file defines overrides and customizations for audio-to-audio nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    # ElevenLabs Voice Changer
    "fal-ai/elevenlabs/voice-changer": {
        "class_name": "ElevenlabsVoiceChanger",
        "docstring": "ElevenLabs Voice Changer transforms voice characteristics in audio with AI-powered voice conversion.",
        "tags": ["audio", "voice-change", "elevenlabs", "transformation", "audio-to-audio"],
        "use_cases": [
            "Change voice characteristics in audio",
            "Transform vocal qualities",
            "Create voice variations",
            "Modify speaker identity",
            "Generate voice-changed audio"
        ],
        "basic_fields": ["audio"]
    },

    # Nova SR (Super Resolution)
    "fal-ai/nova-sr": {
        "class_name": "NovaSr",
        "docstring": "Nova SR enhances audio quality through super-resolution processing for clearer and richer sound.",
        "tags": ["audio", "enhancement", "super-resolution", "quality", "audio-to-audio"],
        "use_cases": [
            "Enhance audio quality",
            "Improve sound clarity",
            "Upscale audio resolution",
            "Restore degraded audio",
            "Generate high-quality audio"
        ],
        "basic_fields": ["audio"]
    },

    # DeepFilterNet3
    "fal-ai/deepfilternet3": {
        "class_name": "Deepfilternet3",
        "docstring": "DeepFilterNet3 removes noise and improves audio quality with advanced deep learning filtering.",
        "tags": ["audio", "noise-reduction", "filtering", "cleaning", "audio-to-audio"],
        "use_cases": [
            "Remove noise from audio",
            "Clean audio recordings",
            "Filter unwanted sounds",
            "Improve audio clarity",
            "Generate clean audio"
        ],
        "basic_fields": ["audio"]
    },

    # SAM Audio Separate
    "fal-ai/sam-audio/separate": {
        "class_name": "SamAudioSeparate",
        "docstring": "SAM Audio Separate isolates and extracts different audio sources from mixed recordings.",
        "tags": ["audio", "separation", "source-extraction", "isolation", "audio-to-audio"],
        "use_cases": [
            "Separate audio sources",
            "Extract vocals from music",
            "Isolate instruments",
            "Remove background sounds",
            "Generate separated audio tracks"
        ],
        "basic_fields": ["audio"]
    },

    "fal-ai/sam-audio/span-separate": {
        "class_name": "SamAudioSpanSeparate",
        "docstring": "SAM Audio Span Separate isolates audio sources across time spans with precise temporal control.",
        "tags": ["audio", "separation", "temporal", "span", "audio-to-audio"],
        "use_cases": [
            "Separate audio by time spans",
            "Extract sources in specific periods",
            "Isolate temporal audio segments",
            "Remove sounds in time ranges",
            "Generate time-based separations"
        ],
        "basic_fields": ["audio"]
    },

    # Demucs
    "fal-ai/demucs": {
        "class_name": "Demucs",
        "docstring": "Demucs separates music into vocals, drums, bass, and other instruments with high quality.",
        "tags": ["audio", "music-separation", "stems", "demucs", "audio-to-audio"],
        "use_cases": [
            "Separate music into stems",
            "Extract vocals from songs",
            "Isolate instruments in music",
            "Create karaoke tracks",
            "Generate individual audio stems"
        ],
        "basic_fields": ["audio"]
    },

    # Stable Audio 2.5
    "fal-ai/stable-audio-25/audio-to-audio": {
        "class_name": "StableAudio25AudioToAudio",
        "docstring": "Stable Audio 2.5 transforms and modifies audio with AI-powered processing and effects.",
        "tags": ["audio", "transformation", "stable-audio", "2.5", "audio-to-audio"],
        "use_cases": [
            "Transform audio characteristics",
            "Apply AI-powered audio effects",
            "Modify audio properties",
            "Generate audio variations",
            "Create processed audio"
        ],
        "basic_fields": ["audio", "prompt"]
    },

    # FFmpeg API Merge Audios
    "fal-ai/ffmpeg-api/merge-audios": {
        "class_name": "FfmpegApiMergeAudios",
        "docstring": "FFmpeg API Merge Audios combines multiple audio files into a single output.",
        "tags": ["audio", "processing", "audio-to-audio", "merging", "ffmpeg"],
        "use_cases": [
            "Combine multiple audio tracks",
            "Merge audio segments",
            "Create audio compilations",
            "Join split audio files",
            "Generate combined audio output"
        ],
        "basic_fields": ["audio"]
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
