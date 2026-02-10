"""
Configuration for text_to_audio module.

This config file defines overrides and customizations for text-to-audio and text-to-speech nodes.
"""

from typing import Any


# Map of endpoint_id to config overrides
CONFIGS: dict[str, dict[str, Any]] = {
    "fal-ai/ace-step/prompt-to-audio": {
        "class_name": "ACEStepPromptToAudio",
        "docstring": "ACE-Step generates music from text prompts with high-quality audio synthesis.",
        "tags": ["audio", "generation", "music", "ace-step", "text-to-audio"],
        "use_cases": [
            "Generate music from text descriptions",
            "Create background music for videos",
            "Produce royalty-free music",
            "Generate audio soundtracks",
            "Create custom music compositions"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/ace-step": {
        "class_name": "ACEStep",
        "docstring": "ACE-Step generates music with lyrics from text using advanced audio synthesis.",
        "tags": ["audio", "generation", "music", "lyrics", "ace-step", "text-to-audio"],
        "use_cases": [
            "Generate songs with lyrics",
            "Create music with vocal tracks",
            "Produce complete songs from text",
            "Generate lyrical content",
            "Create vocal music compositions"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/csm-1b": {
        "class_name": "CSM1B",
        "docstring": "CSM (Conversational Speech Model) generates natural conversational speech from text.",
        "tags": ["audio", "speech", "tts", "conversational", "text-to-speech"],
        "use_cases": [
            "Generate natural conversation audio",
            "Create dialogue for characters",
            "Produce conversational voice content",
            "Generate realistic speech",
            "Create interactive voice responses"
        ],
        "basic_fields": ["text"]
    },
    
    "fal-ai/diffrhythm": {
        "class_name": "DiffRhythm",
        "docstring": "DiffRhythm generates rhythmic music and beats using diffusion models.",
        "tags": ["audio", "generation", "rhythm", "beats", "music", "text-to-audio"],
        "use_cases": [
            "Generate rhythmic music",
            "Create drum beats",
            "Produce percussion tracks",
            "Generate rhythm patterns",
            "Create beat sequences"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/elevenlabs/tts/multilingual-v2": {
        "class_name": "ElevenLabsTTSMultilingualV2",
        "docstring": "ElevenLabs Multilingual TTS v2 generates natural speech in multiple languages.",
        "tags": ["audio", "tts", "speech", "multilingual", "elevenlabs", "text-to-speech"],
        "use_cases": [
            "Generate multilingual speech",
            "Create voiceovers in multiple languages",
            "Produce localized audio content",
            "Generate international voice content",
            "Create translated audio"
        ],
        "basic_fields": ["text", "language"]
    },
    
    "fal-ai/elevenlabs/text-to-dialogue/eleven-v3": {
        "class_name": "ElevenLabsTextToDialogueV3",
        "docstring": "ElevenLabs Text to Dialogue v3 generates conversational dialogue with multiple speakers.",
        "tags": ["audio", "dialogue", "conversation", "elevenlabs", "text-to-speech"],
        "use_cases": [
            "Generate multi-speaker dialogue",
            "Create conversational audio",
            "Produce podcast-style content",
            "Generate character conversations",
            "Create interactive dialogues"
        ],
        "basic_fields": ["text"]
    },
    
    "fal-ai/elevenlabs/sound-effects/v2": {
        "class_name": "ElevenLabsSoundEffectsV2",
        "docstring": "ElevenLabs Sound Effects v2 generates custom sound effects from text descriptions.",
        "tags": ["audio", "sound-effects", "sfx", "elevenlabs", "text-to-audio"],
        "use_cases": [
            "Generate custom sound effects",
            "Create audio effects for videos",
            "Produce game sound effects",
            "Generate environmental sounds",
            "Create audio atmosphere"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/elevenlabs/tts/eleven-v3": {
        "class_name": "ElevenLabsTTSV3",
        "docstring": "ElevenLabs TTS v3 generates high-quality natural speech with advanced voice control.",
        "tags": ["audio", "tts", "speech", "elevenlabs", "text-to-speech"],
        "use_cases": [
            "Generate high-quality voiceovers",
            "Create natural speech audio",
            "Produce professional narration",
            "Generate expressive speech",
            "Create audiobook content"
        ],
        "basic_fields": ["text"]
    },
    
    "fal-ai/elevenlabs/music": {
        "class_name": "ElevenLabsMusic",
        "docstring": "ElevenLabs Music generates custom music compositions from text descriptions.",
        "tags": ["audio", "music", "generation", "elevenlabs", "text-to-audio"],
        "use_cases": [
            "Generate custom music",
            "Create background scores",
            "Produce original compositions",
            "Generate mood music",
            "Create cinematic soundtracks"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/f5-tts": {
        "class_name": "F5TTS",
        "docstring": "F5 TTS generates natural speech with fast inference and high quality.",
        "tags": ["audio", "tts", "speech", "fast", "text-to-speech"],
        "use_cases": [
            "Fast speech generation",
            "Real-time TTS applications",
            "Quick voiceover creation",
            "Efficient speech synthesis",
            "Rapid audio production"
        ],
        "basic_fields": ["text"]
    },
    
    "fal-ai/kokoro": {
        "class_name": "Kokoro",
        "docstring": "Kokoro generates expressive and emotional speech with advanced prosody control.",
        "tags": ["audio", "tts", "speech", "expressive", "emotional", "text-to-speech"],
        "use_cases": [
            "Generate expressive speech",
            "Create emotional voiceovers",
            "Produce dramatic narration",
            "Generate character voices",
            "Create emotive audio content"
        ],
        "basic_fields": ["text"]
    },
    
    "fal-ai/lumina-next-music": {
        "class_name": "LuminaNextMusic",
        "docstring": "Lumina Next Music generates advanced music compositions with sophisticated arrangements.",
        "tags": ["audio", "music", "generation", "lumina", "advanced", "text-to-audio"],
        "use_cases": [
            "Generate sophisticated music",
            "Create complex arrangements",
            "Produce advanced compositions",
            "Generate professional music",
            "Create layered soundtracks"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/suno-ai": {
        "class_name": "SunoAI",
        "docstring": "Suno AI generates complete songs with vocals and instrumentals from text.",
        "tags": ["audio", "music", "song", "generation", "suno", "text-to-audio"],
        "use_cases": [
            "Generate complete songs",
            "Create vocal tracks with music",
            "Produce original songs",
            "Generate music with lyrics",
            "Create full audio productions"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/stable-audio": {
        "class_name": "StableAudio",
        "docstring": "Stable Audio generates high-quality audio from text with consistent results.",
        "tags": ["audio", "generation", "stable", "music", "text-to-audio"],
        "use_cases": [
            "Generate consistent audio",
            "Create reliable soundtracks",
            "Produce predictable audio",
            "Generate stable music",
            "Create dependable audio content"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/xtts": {
        "class_name": "XTTS",
        "docstring": "XTTS generates expressive speech with voice cloning capabilities.",
        "tags": ["audio", "tts", "speech", "voice-cloning", "expressive", "text-to-speech"],
        "use_cases": [
            "Clone and generate voices",
            "Create personalized speech",
            "Produce voice-matched content",
            "Generate custom voice audio",
            "Create voice replications"
        ],
        "basic_fields": ["text"]
    },
    
    "fal-ai/joyous": {
        "class_name": "Joyous",
        "docstring": "Joyous generates upbeat and cheerful music from text descriptions.",
        "tags": ["audio", "music", "generation", "upbeat", "cheerful", "text-to-audio"],
        "use_cases": [
            "Generate cheerful music",
            "Create upbeat soundtracks",
            "Produce happy audio content",
            "Generate positive music",
            "Create energetic compositions"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/metavoice": {
        "class_name": "MetaVoice",
        "docstring": "MetaVoice generates natural speech with advanced voice characteristics control.",
        "tags": ["audio", "tts", "speech", "metavoice", "text-to-speech"],
        "use_cases": [
            "Generate natural speech",
            "Control voice characteristics",
            "Create varied voice outputs",
            "Produce customized speech",
            "Generate flexible audio content"
        ],
        "basic_fields": ["text"]
    },
    
    "fal-ai/piper-tts": {
        "class_name": "PiperTTS",
        "docstring": "Piper TTS generates fast, efficient speech with low latency.",
        "tags": ["audio", "tts", "speech", "fast", "efficient", "text-to-speech"],
        "use_cases": [
            "Fast speech generation",
            "Low-latency TTS",
            "Efficient audio production",
            "Real-time speech synthesis",
            "Quick voiceover creation"
        ],
        "basic_fields": ["text"]
    },
    
    "fal-ai/riffusion": {
        "class_name": "Riffusion",
        "docstring": "Riffusion generates music using diffusion models for creative audio synthesis.",
        "tags": ["audio", "music", "generation", "diffusion", "riffusion", "text-to-audio"],
        "use_cases": [
            "Generate creative music",
            "Create experimental audio",
            "Produce unique soundscapes",
            "Generate artistic compositions",
            "Create innovative music"
        ],
        "basic_fields": ["prompt"]
    },
    
    "fal-ai/vocalremover": {
        "class_name": "VocalRemover",
        "docstring": "Vocal Remover separates vocals from music to create instrumental versions.",
        "tags": ["audio", "vocal-separation", "karaoke", "instrumental", "processing"],
        "use_cases": [
            "Create karaoke tracks",
            "Extract instrumentals",
            "Remove vocals from songs",
            "Separate audio stems",
            "Create background music"
        ],
        "basic_fields": ["audio"]
    },

    "fal-ai/minimax-music/v2": {
        "class_name": "MinimaxMusicV2",
        "docstring": "Minimax Music",
        "tags": ["audio", "generation", "text-to-audio", "tts", "professional"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "beatoven/sound-effect-generation": {
        "class_name": "BeatovenSoundEffectGeneration",
        "docstring": "Sound Effect Generation",
        "tags": ["audio", "generation", "text-to-audio", "tts"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "beatoven/music-generation": {
        "class_name": "BeatovenMusicGeneration",
        "docstring": "Music Generation",
        "tags": ["audio", "generation", "text-to-audio", "tts"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/minimax-music/v1.5": {
        "class_name": "MinimaxMusicV15",
        "docstring": "MiniMax (Hailuo AI) Music v1.5",
        "tags": ["audio", "generation", "text-to-audio", "tts", "professional"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "fal-ai/stable-audio-25/text-to-audio": {
        "class_name": "StableAudio25TextToAudio",
        "docstring": "Stable Audio 2.5",
        "tags": ["audio", "generation", "text-to-audio", "tts"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "sonauto/v2/inpaint": {
        "class_name": "SonautoV2Inpaint",
        "docstring": "Sonauto V2",
        "tags": ["audio", "generation", "text-to-audio", "tts"],
        "use_cases": [
            "Automated content generation",
            "Creative workflows",
            "Batch processing",
            "Professional applications",
            "Rapid prototyping"

        ],
    },
    "sonauto/v2/text-to-music": {
        "class_name": "SonautoV2TextToMusic",
        "docstring": "Create full songs in any style",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/lyria2": {
        "class_name": "Lyria2",
        "docstring": "Lyria 2 is Google's latest music generation model, you can generate any type of music with this model.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "cassetteai/sound-effects-generator": {
        "class_name": "CassetteaiSoundEffectsGenerator",
        "docstring": "Create stunningly realistic sound effects in seconds - CassetteAI's Sound Effects Model generates high-quality SFX up to 30 seconds long in just 1 second of processing time",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "cassetteai/music-generator": {
        "class_name": "CassetteaiMusicGenerator",
        "docstring": "CassetteAI’s model generates a 30-second sample in under 2 seconds and a full 3-minute track in under 10 seconds. At 44.1 kHz stereo audio, expect a level of professional consistency with no breaks, no squeaks, and no random interruptions in your creations.  ",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/kokoro/hindi": {
        "class_name": "KokoroHindi",
        "docstring": "A fast and expressive Hindi text-to-speech model with clear pronunciation and accurate intonation.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/kokoro/british-english": {
        "class_name": "KokoroBritishEnglish",
        "docstring": "A high-quality British English text-to-speech model offering natural and expressive voice synthesis.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/kokoro/american-english": {
        "class_name": "KokoroAmericanEnglish",
        "docstring": "Kokoro is a lightweight text-to-speech model that delivers comparable quality to larger models while being significantly faster and more cost-efficient.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/zonos": {
        "class_name": "Zonos",
        "docstring": "Clone voice of any person and speak anything in their voice using zonos' voice cloning.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/kokoro/italian": {
        "class_name": "KokoroItalian",
        "docstring": "A high-quality Italian text-to-speech model delivering smooth and expressive speech synthesis.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/kokoro/brazilian-portuguese": {
        "class_name": "KokoroBrazilianPortuguese",
        "docstring": "A natural and expressive Brazilian Portuguese text-to-speech model optimized for clarity and fluency.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/kokoro/french": {
        "class_name": "KokoroFrench",
        "docstring": "An expressive and natural French text-to-speech model for both European and Canadian French.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/kokoro/japanese": {
        "class_name": "KokoroJapanese",
        "docstring": "A fast and natural-sounding Japanese text-to-speech model optimized for smooth pronunciation.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/kokoro/mandarin-chinese": {
        "class_name": "KokoroMandarinChinese",
        "docstring": "A highly efficient Mandarin Chinese text-to-speech model that captures natural tones and prosody.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/kokoro/spanish": {
        "class_name": "KokoroSpanish",
        "docstring": "A natural-sounding Spanish text-to-speech model optimized for Latin American and European Spanish.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/yue": {
        "class_name": "Yue",
        "docstring": "YuE is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/mmaudio-v2/text-to-audio": {
        "class_name": "MmaudioV2TextToAudio",
        "docstring": "MMAudio generates synchronized audio given text inputs. It can generate sounds described by a prompt.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
    "fal-ai/minimax-music": {
        "class_name": "MinimaxMusic",
        "docstring": "Generate music from text prompts using the MiniMax model, which leverages advanced AI techniques to create high-quality, diverse musical compositions.",
        "tags": ["audio", "generation", "text-to-audio", "sound"],
        "use_cases": [
            "Sound effect generation",
            "Music composition",
            "Audio content creation",
            "Background music generation",
            "Podcast audio production",
        ],
    },
}


def get_config(endpoint_id: str) -> dict[str, Any]:
    """Get config for an endpoint."""
    return CONFIGS.get(endpoint_id, {})
