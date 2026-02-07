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
    "fal-ai/longcat-single-avatar/image-audio-to-video": {
        "class_name": "LongcatSingleAvatarImageAudioToVideo",
        "docstring": "LongCat-Video-Avatar is an audio-driven video generation model that can generates super-realistic, lip-synchronized long video generation with natural dynamics and consistent identity.",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Audio-driven video generation",
            "Music visualization",
            "Talking head animation",
            "Audio-synced content creation",
            "Podcast video generation",
        ],
    },
    "fal-ai/longcat-single-avatar/audio-to-video": {
        "class_name": "LongcatSingleAvatarAudioToVideo",
        "docstring": "LongCat-Video-Avatar is an audio-driven video generation model that can generates super-realistic, lip-synchronized long video generation with natural dynamics and consistent identity.",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Audio-driven video generation",
            "Music visualization",
            "Talking head animation",
            "Audio-synced content creation",
            "Podcast video generation",
        ],
    },
    "argil/avatars/audio-to-video": {
        "class_name": "ArgilAvatarsAudioToVideo",
        "docstring": "High-quality avatar videos that feel real, generated from your audio",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Audio-driven video generation",
            "Music visualization",
            "Talking head animation",
            "Audio-synced content creation",
            "Podcast video generation",
        ],
    },
    "fal-ai/wan/v2.2-14b/speech-to-video": {
        "class_name": "WanV2214bSpeechToVideo",
        "docstring": "Wan-S2V is a video model that generates high-quality videos from static images and audio, with realistic facial expressions, body movements, and professional camera work for film and television applications",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Audio-driven video generation",
            "Music visualization",
            "Talking head animation",
            "Audio-synced content creation",
            "Podcast video generation",
        ],
    },
    "fal-ai/stable-avatar": {
        "class_name": "StableAvatar",
        "docstring": "Stable Avatar generates audio-driven video avatars up to five minutes long",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Audio-driven video generation",
            "Music visualization",
            "Talking head animation",
            "Audio-synced content creation",
            "Podcast video generation",
        ],
    },
    "fal-ai/echomimic-v3": {
        "class_name": "EchomimicV3",
        "docstring": "EchoMimic V3 generates a talking avatar model from a picture, audio and text prompt.",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Audio-driven video generation",
            "Music visualization",
            "Talking head animation",
            "Audio-synced content creation",
            "Podcast video generation",
        ],
    },
    "veed/avatars/audio-to-video": {
        "class_name": "VeedAvatarsAudioToVideo",
        "docstring": "Generate high-quality videos with UGC-like avatars from audio",
        "tags": ["video", "generation", "audio-to-video", "visualization"],
        "use_cases": [
            "Audio-driven video generation",
            "Music visualization",
            "Talking head animation",
            "Audio-synced content creation",
            "Podcast video generation",
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
