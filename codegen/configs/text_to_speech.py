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
    "fal-ai/kling-video/v1/tts": {
        "class_name": "KlingVideoV1Tts",
        "docstring": "Generate speech from text prompts and different voices using the Kling TTS model, which leverages advanced AI techniques to create high-quality text-to-speech.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/chatterbox/text-to-speech/multilingual": {
        "class_name": "ChatterboxTextToSpeechMultilingual",
        "docstring": "Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. Use the first tts from resemble ai.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/vibevoice/7b": {
        "class_name": "Vibevoice7b",
        "docstring": "Generate long, expressive multi-voice speech using Microsoft's powerful TTS",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/vibevoice": {
        "class_name": "Vibevoice",
        "docstring": "Generate long, expressive multi-voice speech using Microsoft's powerful TTS",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/minimax/preview/speech-2.5-hd": {
        "class_name": "MinimaxPreviewSpeech25Hd",
        "docstring": "Generate speech from text prompts and different voices using the MiniMax Speech-02 HD model, which leverages advanced AI techniques to create high-quality text-to-speech.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/minimax/preview/speech-2.5-turbo": {
        "class_name": "MinimaxPreviewSpeech25Turbo",
        "docstring": "Generate fast speech from text prompts and different voices using the MiniMax Speech-02 Turbo model, which leverages advanced AI techniques to create high-quality text-to-speech.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts", "fast"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/minimax/voice-design": {
        "class_name": "MinimaxVoiceDesign",
        "docstring": "Design a personalized voice from a text description, and generate speech from text prompts using the MiniMax model, which leverages advanced AI techniques to create high-quality text-to-speech.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "resemble-ai/chatterboxhd/text-to-speech": {
        "class_name": "ResembleAiChatterboxhdTextToSpeech",
        "docstring": "Generate expressive, natural speech with Resemble AI's Chatterbox. Features unique emotion control, instant voice cloning from short audio, and built-in watermarking.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/chatterbox/text-to-speech": {
        "class_name": "ChatterboxTextToSpeech",
        "docstring": "Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. Use the first tts from resemble ai.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/minimax/voice-clone": {
        "class_name": "MinimaxVoiceClone",
        "docstring": "Clone a voice from a sample audio and generate speech from text prompts using the MiniMax model, which leverages advanced AI techniques to create high-quality text-to-speech.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/minimax/speech-02-turbo": {
        "class_name": "MinimaxSpeech02Turbo",
        "docstring": "Generate fast speech from text prompts and different voices using the MiniMax Speech-02 Turbo model, which leverages advanced AI techniques to create high-quality text-to-speech.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts", "fast"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/minimax/speech-02-hd": {
        "class_name": "MinimaxSpeech02Hd",
        "docstring": "Generate speech from text prompts and different voices using the MiniMax Speech-02 HD model, which leverages advanced AI techniques to create high-quality text-to-speech.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/dia-tts": {
        "class_name": "DiaTts",
        "docstring": "Dia directly generates realistic dialogue from transcripts. Audio conditioning enables emotion control. Produces natural nonverbals like laughter and throat clearing.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/orpheus-tts": {
        "class_name": "OrpheusTts",
        "docstring": "Orpheus TTS is a state-of-the-art, Llama-based Speech-LLM designed for high-quality, empathetic text-to-speech generation. This model has been finetuned to deliver human-level speech synthesis, achieving exceptional clarity, expressiveness, and real-time performances.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
        ],
    },
    "fal-ai/elevenlabs/tts/turbo-v2.5": {
        "class_name": "ElevenlabsTtsTurboV25",
        "docstring": "Generate high-speed text-to-speech audio using ElevenLabs TTS Turbo v2.5.",
        "tags": ["speech", "synthesis", "text-to-speech", "tts", "fast"],
        "use_cases": [
            "Voice synthesis for applications",
            "Audiobook narration",
            "Virtual assistant voices",
            "Accessibility solutions",
            "Content localization",
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
