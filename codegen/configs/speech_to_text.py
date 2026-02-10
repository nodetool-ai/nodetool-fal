"""
Configuration for speech_to_text module.

This config file defines overrides and customizations for speech-to-text nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/elevenlabs/speech-to-text": {
        "class_name": "ElevenLabsSpeechToText",
        "docstring": "ElevenLabs Speech to Text transcribes audio to text with high accuracy.",
        "tags": ["audio", "transcription", "stt", "elevenlabs", "speech-to-text"],
        "use_cases": [
            "Transcribe audio files",
            "Convert speech to text",
            "Generate transcripts from audio",
            "Extract text from recordings",
            "Create captions from audio"
        ],
        "basic_fields": ["audio"]
    },
    
    "fal-ai/elevenlabs/speech-to-text/scribe-v2": {
        "class_name": "ElevenLabsScribeV2",
        "docstring": "ElevenLabs Scribe V2 provides blazingly fast speech-to-text transcription.",
        "tags": ["audio", "transcription", "stt", "fast", "elevenlabs", "speech-to-text"],
        "use_cases": [
            "Fast audio transcription",
            "Real-time speech recognition",
            "Quick transcript generation",
            "High-speed audio processing",
            "Rapid speech-to-text conversion"
        ],
        "basic_fields": ["audio"]
    },
    
    "fal-ai/smart-turn": {
        "class_name": "SmartTurn",
        "docstring": "Pipecat's Smart Turn model provides native audio turn detection for conversations.",
        "tags": ["audio", "turn-detection", "conversation", "pipecat", "speech-analysis"],
        "use_cases": [
            "Detect conversation turns",
            "Identify speaker changes",
            "Analyze dialogue timing",
            "Detect speech boundaries",
            "Process conversational audio"
        ],
        "basic_fields": ["audio"]
    },
    
    "fal-ai/speech-to-text": {
        "class_name": "SpeechToText",
        "docstring": "General-purpose speech-to-text model for accurate audio transcription.",
        "tags": ["audio", "transcription", "stt", "speech-to-text"],
        "use_cases": [
            "General audio transcription",
            "Convert speech recordings to text",
            "Generate audio transcripts",
            "Process voice recordings",
            "Extract text from speech"
        ],
        "basic_fields": ["audio"]
    },
    
    "fal-ai/speech-to-text/stream": {
        "class_name": "SpeechToTextStream",
        "docstring": "Streaming speech-to-text for real-time audio transcription.",
        "tags": ["audio", "transcription", "stt", "streaming", "real-time", "speech-to-text"],
        "use_cases": [
            "Real-time transcription",
            "Live audio captioning",
            "Stream audio processing",
            "Continuous speech recognition",
            "Live speech-to-text conversion"
        ],
        "basic_fields": ["audio_stream"]
    },
    
    "fal-ai/speech-to-text/turbo": {
        "class_name": "SpeechToTextTurbo",
        "docstring": "High-speed speech-to-text model optimized for fast transcription.",
        "tags": ["audio", "transcription", "stt", "turbo", "fast", "speech-to-text"],
        "use_cases": [
            "Fast audio transcription",
            "Quick speech recognition",
            "Rapid transcript generation",
            "High-speed processing",
            "Efficient speech-to-text"
        ],
        "basic_fields": ["audio"]
    },
    
    "fal-ai/speech-to-text/turbo/stream": {
        "class_name": "SpeechToTextTurboStream",
        "docstring": "High-speed streaming speech-to-text for real-time fast transcription.",
        "tags": ["audio", "transcription", "stt", "turbo", "streaming", "fast", "speech-to-text"],
        "use_cases": [
            "Real-time fast transcription",
            "Live fast captioning",
            "High-speed streaming STT",
            "Rapid live transcription",
            "Efficient real-time processing"
        ],
        "basic_fields": ["audio_stream"]
    },
    
    "fal-ai/whisper": {
        "class_name": "Whisper",
        "docstring": "OpenAI's Whisper model for robust multilingual speech recognition.",
        "tags": ["audio", "transcription", "stt", "whisper", "multilingual", "speech-to-text"],
        "use_cases": [
            "Multilingual transcription",
            "Robust speech recognition",
            "Transcribe multiple languages",
            "Handle noisy audio",
            "International audio processing"
        ],
        "basic_fields": ["audio"]
    },
    
    "fal-ai/wizper": {
        "class_name": "Wizper",
        "docstring": "Wizper provides fast and accurate speech-to-text transcription.",
        "tags": ["audio", "transcription", "stt", "wizper", "fast", "speech-to-text"],
        "use_cases": [
            "Fast accurate transcription",
            "Quick speech recognition",
            "Efficient audio processing",
            "Rapid text extraction",
            "Speedy speech-to-text"
        ],
        "basic_fields": ["audio"]
    },
}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """Get config for an endpoint."""
    return CONFIGS.get(endpoint_id, {})
