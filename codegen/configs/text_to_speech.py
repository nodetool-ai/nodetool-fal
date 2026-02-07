"""
Configuration for text_to_speech module.

This config file defines overrides and customizations for text-to-speech nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    # Qwen-3 TTS Family
    "fal-ai/qwen-3-tts/text-to-speech/1.7b": {
        "class_name": "Qwen3TtsTextToSpeech17B",
        "docstring": "Qwen-3 TTS 1.7B generates natural-sounding speech from text using the large 1.7-billion parameter model.",
        "tags": ["audio", "tts", "qwen", "1.7b", "text-to-speech", "speech-synthesis"],
        "use_cases": [
            "Generate natural-sounding speech from text",
            "Create voice-overs for videos",
            "Produce audiobook narration",
            "Generate spoken content for applications",
            "Create text-to-speech for accessibility"
        ],
        "basic_fields": ["text"]
    },

    "fal-ai/qwen-3-tts/text-to-speech/0.6b": {
        "class_name": "Qwen3TtsTextToSpeech06B",
        "docstring": "Qwen-3 TTS 0.6B generates speech from text efficiently using the compact 600-million parameter model.",
        "tags": ["audio", "tts", "qwen", "0.6b", "efficient", "text-to-speech"],
        "use_cases": [
            "Generate speech efficiently from text",
            "Create fast voice-overs",
            "Produce quick audio narration",
            "Generate spoken content with low latency",
            "Create efficient text-to-speech"
        ],
        "basic_fields": ["text"]
    },

    "fal-ai/qwen-3-tts/voice-design/1.7b": {
        "class_name": "Qwen3TtsVoiceDesign17B",
        "docstring": "Qwen-3 TTS Voice Design 1.7B creates custom voice characteristics for personalized speech synthesis.",
        "tags": ["audio", "tts", "qwen", "voice-design", "custom", "1.7b"],
        "use_cases": [
            "Design custom voice characteristics",
            "Create personalized speech synthesis",
            "Generate unique voice styles",
            "Produce custom voice-overs",
            "Create tailored speech synthesis"
        ],
        "basic_fields": ["text"]
    },

    # VibeVoice
    "fal-ai/vibevoice/0.5b": {
        "class_name": "Vibevoice05B",
        "docstring": "VibeVoice 0.5B generates expressive and emotive speech from text with natural vocal characteristics.",
        "tags": ["audio", "tts", "vibevoice", "0.5b", "expressive", "text-to-speech"],
        "use_cases": [
            "Generate expressive speech from text",
            "Create emotive voice-overs",
            "Produce natural vocal narration",
            "Generate speech with personality",
            "Create engaging audio content"
        ],
        "basic_fields": ["text"]
    },

    # Maya
    "fal-ai/maya": {
        "class_name": "Maya",
        "docstring": "Maya generates high-quality natural speech from text with advanced voice synthesis capabilities.",
        "tags": ["audio", "tts", "maya", "high-quality", "text-to-speech"],
        "use_cases": [
            "Generate high-quality speech from text",
            "Create professional voice-overs",
            "Produce premium audio narration",
            "Generate natural-sounding speech",
            "Create professional audio content"
        ],
        "basic_fields": ["text"]
    },

    # Minimax Speech 2.6
    "fal-ai/minimax/speech-2.6-hd": {
        "class_name": "MinimaxSpeech26Hd",
        "docstring": "Minimax Speech 2.6 HD generates high-definition speech from text with superior audio quality.",
        "tags": ["audio", "tts", "minimax", "2.6", "hd", "high-quality"],
        "use_cases": [
            "Generate HD quality speech from text",
            "Create premium voice-overs",
            "Produce high-fidelity audio narration",
            "Generate superior audio quality speech",
            "Create broadcast-quality audio"
        ],
        "enum_overrides": {
            "OutputFormat": "MinimaxSpeech26HdOutputFormat"
        },
        "basic_fields": ["text"]
    },

    "fal-ai/minimax/speech-2.6-turbo": {
        "class_name": "MinimaxSpeech26Turbo",
        "docstring": "Minimax Speech 2.6 Turbo generates speech from text with optimized speed and good quality.",
        "tags": ["audio", "tts", "minimax", "2.6", "turbo", "fast"],
        "use_cases": [
            "Generate speech quickly from text",
            "Create fast voice-overs",
            "Produce rapid audio narration",
            "Generate speech with turbo speed",
            "Create efficient audio content"
        ],
        "enum_overrides": {
            "OutputFormat": "MinimaxSpeech26TurboOutputFormat"
        },
        "basic_fields": ["text"]
    },

    # Maya Batch
    "fal-ai/maya/batch": {
        "class_name": "MayaBatch",
        "docstring": "Maya Batch TTS generates high-quality speech in batch mode for efficient processing.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts", "batch", "maya"],
        "use_cases": [
            "Generate speech for multiple texts",
            "Batch process narration",
            "Create bulk voice-overs",
            "Efficient audio content creation",
            "Generate multiple speech files"
        ],
        "basic_fields": ["text"]
    },

    # Maya Stream
    "fal-ai/maya/stream": {
        "class_name": "MayaStream",
        "docstring": "Maya Stream TTS generates high-quality speech in streaming mode for real-time applications.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts", "streaming", "maya"],
        "use_cases": [
            "Generate speech in real-time",
            "Stream narration dynamically",
            "Create live voice-overs",
            "Real-time audio synthesis",
            "Generate streaming speech"
        ],
        "basic_fields": ["text"]
    },

    # Index TTS 2
    "fal-ai/index-tts-2/text-to-speech": {
        "class_name": "IndexTts2TextToSpeech",
        "docstring": "Index TTS 2 generates natural-sounding speech from text with advanced neural synthesis.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts", "neural"],
        "use_cases": [
            "Generate natural speech from text",
            "Create voice narration",
            "Produce audio books",
            "Generate voice-overs",
            "Create speech content"
        ],
        "basic_fields": ["text"]
    },

    # IndoVoice
    "fal-ai/indo-voice": {
        "class_name": "IndoVoice",
        "docstring": "IndoVoice TTS generates Indonesian language speech with natural pronunciation and intonation.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts", "indonesian"],
        "use_cases": [
            "Generate Indonesian speech",
            "Create Indonesian narration",
            "Produce Indonesian voice-overs",
            "Generate localized audio content",
            "Create Indonesian audio books"
        ],
        "basic_fields": ["text"]
    },

    # CosyVoice Turbo
    "fal-ai/cosyvoice-turbo": {
        "class_name": "CosyvoiceTurbo",
        "docstring": "CosyVoice Turbo generates high-quality speech with fast processing speed and natural voice.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts", "turbo", "fast"],
        "use_cases": [
            "Generate speech quickly",
            "Create fast voice narration",
            "Produce rapid audio content",
            "Generate speech with turbo speed",
            "Create efficient voice-overs"
        ],
        "basic_fields": ["text"]
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
