from enum import Enum
from pydantic import Field
from nodetool.metadata.types import AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class MMAudioV2(FALNode):
    """
    MMAudio V2 generates synchronized audio given text inputs. It can generate sounds described by a prompt.
    audio, generation, synthesis, text-to-audio, synchronization

    Use cases:
    - Generate synchronized audio from text descriptions
    - Create custom sound effects
    - Produce ambient soundscapes
    - Generate audio for multimedia content
    - Create sound design elements
    """

    prompt: str = Field(default="", description="The prompt to generate the audio for")
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to avoid certain elements in the generated audio",
    )
    num_steps: int = Field(
        default=25, ge=1, description="The number of steps to generate the audio for"
    )
    duration: float = Field(
        default=8.0,
        ge=1.0,
        description="The duration of the audio to generate in seconds",
    )
    cfg_strength: float = Field(
        default=4.5, description="The strength of Classifier Free Guidance"
    )
    mask_away_clip: bool = Field(
        default=False, description="Whether to mask away the clip"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same audio every time"
    )

    @classmethod
    def get_title(cls):
        return "MMAudio V2"

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "num_steps": self.num_steps,
            "duration": self.duration,
            "cfg_strength": self.cfg_strength,
            "mask_away_clip": self.mask_away_clip,
        }

        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/mmaudio-v2/text-to-audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration", "num_steps"]


class StableAudio(FALNode):
    """
    Stable Audio generates audio from text prompts. Open source text-to-audio model from fal.ai.
    audio, generation, synthesis, text-to-audio, open-source

    Use cases:
    - Generate custom audio content from text
    - Create background music and sounds
    - Produce audio assets for projects
    - Generate sound effects
    - Create experimental audio content
    """

    prompt: str = Field(default="", description="The prompt to generate the audio from")
    seconds_start: int = Field(
        default=0, description="The start point of the audio clip to generate"
    )
    seconds_total: int = Field(
        default=30, description="The duration of the audio clip to generate in seconds"
    )
    steps: int = Field(
        default=100, description="The number of steps to denoise the audio for"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "seconds_total": self.seconds_total,
            "steps": self.steps,
        }

        if self.seconds_start > 0:
            arguments["seconds_start"] = self.seconds_start

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-audio",
            arguments=arguments,
        )
        assert "audio_file" in res
        return AudioRef(uri=res["audio_file"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "seconds_total", "steps"]


class F5TTS(FALNode):
    """
    F5 TTS (Text-to-Speech) model for generating natural-sounding speech from text with voice cloning capabilities.
    audio, tts, voice-cloning, speech, synthesis, text-to-speech, tts, text-to-audio

    Use cases:
    - Generate natural speech from text
    - Clone and replicate voices
    - Create custom voiceovers
    - Produce multilingual speech content
    - Generate personalized audio content
    """

    gen_text: str = Field(default="", description="The text to be converted to speech")
    ref_audio_url: str = Field(
        default="",
        description="URL of the reference audio file to clone the voice from",
    )
    ref_text: str = Field(
        default="",
        description="Optional reference text. If not provided, ASR will be used",
    )
    model_type: str = Field(
        default="F5-TTS",
        description="Model type to use (F5-TTS or E2-TTS)",
    )
    remove_silence: bool = Field(
        default=True,
        description="Whether to remove silence from the generated audio",
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "gen_text": self.gen_text,
            "ref_audio_url": self.ref_audio_url,
            "model_type": self.model_type,
            "remove_silence": self.remove_silence,
        }

        if self.ref_text:
            arguments["ref_text"] = self.ref_text

        res = await self.submit_request(
            context=context,
            application="fal-ai/f5-tts",
            arguments=arguments,
        )
        assert "audio_url" in res
        return AudioRef(uri=res["audio_url"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["gen_text", "ref_audio_url", "model_type"]

    @classmethod
    def get_title(cls):
        return "F5 TTS"


class PlayAITTSDialog(FALNode):
    """PlayAI Dialog TTS generates speech for multi speaker dialogs.
    audio, tts, dialog, speech, synthesis

    Use cases:
    - Generate interactive conversations
    - Create voice overs with multiple characters
    - Produce spoken dialogs for games
    - Synthesize narration with distinct voices
    - Prototype conversational audio
    """

    text: str = Field(default="", description="Text to convert into speech")
    voice: str = Field(
        default="nova",
        description="Voice preset to use for the spoken dialog",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Playback speed of the generated audio",
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice": self.voice,
            "speed": self.speed,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/playai/tts/dialog/api",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text", "voice"]

    @classmethod
    def get_title(cls):
        return "PlayAI Dialog TTS"


class AudioFormatEnum(str, Enum):
    MP3 = "mp3"
    AAC = "aac"
    M4A = "m4a"
    OGG = "ogg"
    OPUS = "opus"
    FLAC = "flac"
    WAV = "wav"


class NovaSR(FALNode):
    """
    Nova SR enhances muffled 16 kHz speech audio into crystal-clear 48 kHz audio using super-resolution.
    audio, enhancement, super-resolution, speech, upsampling, audio-to-audio, nova-sr

    Use cases:
    - Enhance low-quality speech recordings
    - Upsample audio from 16kHz to 48kHz
    - Improve audio clarity for voice content
    - Prepare speech for downstream processing
    - Recover details from compressed audio
    """

    audio: AudioRef = Field(default=AudioRef(), description="The audio file to enhance")
    audio_format: AudioFormatEnum = Field(
        default=AudioFormatEnum.MP3, description="Output audio format"
    )
    bitrate: str = Field(default="192k", description="Output audio bitrate")
    sync_mode: bool = Field(
        default=False, description="If True, media returned as data URI"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        client = self.get_client(context)
        audio_bytes = await context.asset_to_bytes(self.audio)
        audio_url = await client.upload(audio_bytes, "audio/mp3")

        arguments = {
            "audio_url": audio_url,
            "audio_format": self.audio_format.value,
            "bitrate": self.bitrate,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/nova-sr",
            arguments=arguments,
        )

        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["audio", "audio_format", "bitrate"]


class ElevenLabsTTSV3(FALNode):
    """
    ElevenLabs Eleven V3 Text-to-Speech with high-quality voice synthesis.
    audio, tts, text-to-speech, elevenlabs, voice, synthesis

    Use cases:
    - Generate natural speech from text
    - Create voiceovers
    - Produce audio content
    - Create audiobooks
    - Generate voice notifications
    """

    text: str = Field(default="", description="The text to convert to speech")
    voice_id: str = Field(default="", description="The voice ID to use for synthesis")
    model_id: str = Field(
        default="eleven_multilingual_v2",
        description="The model ID (e.g., eleven_multilingual_v2)",
    )
    stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Voice stability")
    similarity_boost: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Voice similarity boost"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
            },
        }
        if self.voice_id:
            arguments["voice_id"] = self.voice_id

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/tts/eleven-v3",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text", "voice_id"]


class ElevenLabsTTSTurbo(FALNode):
    """
    ElevenLabs Turbo V2.5 Text-to-Speech for fast voice synthesis.
    audio, tts, text-to-speech, elevenlabs, fast, turbo

    Use cases:
    - Quick voice generation
    - Real-time speech synthesis
    - Rapid prototyping
    - Fast audio content
    - Interactive applications
    """

    text: str = Field(default="", description="The text to convert to speech")
    voice_id: str = Field(default="", description="The voice ID to use for synthesis")

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
        }
        if self.voice_id:
            arguments["voice_id"] = self.voice_id

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/tts/turbo-v2.5",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text", "voice_id"]


class ElevenLabsMultilingual(FALNode):
    """
    ElevenLabs Multilingual V2 Text-to-Speech with support for 29 languages.
    audio, tts, text-to-speech, elevenlabs, multilingual

    Use cases:
    - Generate speech in multiple languages
    - Create localized content
    - Produce multilingual voiceovers
    - Create international audio
    - Generate language learning content
    """

    text: str = Field(default="", description="The text to convert to speech")
    voice_id: str = Field(default="", description="The voice ID to use for synthesis")
    language_code: str = Field(
        default="en", description="Language code (e.g., en, es, fr)"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
        }
        if self.voice_id:
            arguments["voice_id"] = self.voice_id
        if self.language_code:
            arguments["language_code"] = self.language_code

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/tts/multilingual-v2",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text", "voice_id", "language_code"]


class KokoroTTS(FALNode):
    """
    Kokoro American English Text-to-Speech with natural voice synthesis.
    audio, tts, text-to-speech, kokoro, english

    Use cases:
    - Generate natural English speech
    - Create voiceovers
    - Produce audio content
    - Create educational material
    - Generate voice notifications
    """

    text: str = Field(default="", description="The text to convert to speech")
    voice: str = Field(default="af_sky", description="The voice to use")
    speed: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice": self.voice,
            "speed": self.speed,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/kokoro/american-english",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text", "voice"]


class DiaTTS(FALNode):
    """
    Dia TTS generates natural speech with emotion and expression control.
    audio, tts, text-to-speech, dia, expressive

    Use cases:
    - Generate expressive speech
    - Create emotional voiceovers
    - Produce dynamic audio content
    - Create character voices
    - Generate storytelling audio
    """

    text: str = Field(default="", description="The text to convert to speech")
    voice: str = Field(default="", description="The voice preset to use")

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
        }
        if self.voice:
            arguments["voice"] = self.voice

        res = await self.submit_request(
            context=context,
            application="fal-ai/dia-tts",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]


class ChatterboxTTS(FALNode):
    """
    Chatterbox Text-to-Speech with conversational voice synthesis.
    audio, tts, text-to-speech, chatterbox, conversational

    Use cases:
    - Generate conversational speech
    - Create chat bot voices
    - Produce dialogue audio
    - Create interactive content
    - Generate voice assistants
    """

    text: str = Field(default="", description="The text to convert to speech")

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/chatterbox/text-to-speech",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]


class OrpheusTTS(FALNode):
    """
    Orpheus TTS generates high-quality speech with natural prosody.
    audio, tts, text-to-speech, orpheus, natural

    Use cases:
    - Generate natural-sounding speech
    - Create professional voiceovers
    - Produce high-quality audio
    - Create audiobooks
    - Generate podcast content
    """

    text: str = Field(default="", description="The text to convert to speech")
    voice: str = Field(default="", description="The voice to use")

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
        }
        if self.voice:
            arguments["voice"] = self.voice

        res = await self.submit_request(
            context=context,
            application="fal-ai/orpheus-tts",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]


class MiniMaxSpeech02HD(FALNode):
    """
    MiniMax Speech 02 HD generates high-quality speech synthesis.
    audio, tts, text-to-speech, minimax, hd

    Use cases:
    - Generate HD quality speech
    - Create professional audio
    - Produce voiceovers
    - Create content narration
    - Generate announcements
    """

    text: str = Field(default="", description="The text to convert to speech")
    voice_id: str = Field(default="", description="The voice ID to use")

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
        }
        if self.voice_id:
            arguments["voice_id"] = self.voice_id

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/speech-02-hd",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text"]


class ElevenLabsMusic(FALNode):
    """
    ElevenLabs Music generates music from text descriptions.
    audio, music, generation, elevenlabs, creative

    Use cases:
    - Generate custom music tracks
    - Create background music
    - Produce jingles
    - Create audio branding
    - Generate ambient music
    """

    prompt: str = Field(
        default="", description="The prompt describing the music to generate"
    )
    duration: float = Field(
        default=30.0, ge=1.0, le=120.0, description="Duration in seconds"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/music",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration"]


class ElevenLabsSoundEffects(FALNode):
    """
    ElevenLabs Sound Effects V2 generates sound effects from text descriptions.
    audio, sound-effects, generation, elevenlabs

    Use cases:
    - Generate custom sound effects
    - Create audio for videos
    - Produce game audio
    - Create ambient sounds
    - Generate UI sounds
    """

    prompt: str = Field(
        default="", description="The prompt describing the sound effect"
    )
    duration: float = Field(
        default=5.0, ge=0.5, le=22.0, description="Duration in seconds"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/sound-effects/v2",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration"]


class MiniMaxMusic(FALNode):
    """
    MiniMax Music generates music tracks from text descriptions.
    audio, music, generation, minimax

    Use cases:
    - Generate custom music
    - Create background tracks
    - Produce audio content
    - Create music for videos
    - Generate jingles
    """

    prompt: str = Field(
        default="", description="The prompt describing the music to generate"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax-music",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]


class ElevenLabsAudioIsolation(FALNode):
    """
    ElevenLabs Audio Isolation separates vocals from audio tracks.
    audio, isolation, separation, elevenlabs

    Use cases:
    - Extract vocals from music
    - Remove background noise
    - Isolate speech
    - Create acapella tracks
    - Clean audio recordings
    """

    audio: AudioRef = Field(default=AudioRef(), description="The audio file to process")

    async def process(self, context: ProcessingContext) -> AudioRef:
        client = await self.get_client(context)
        audio_bytes = await context.asset_to_bytes(self.audio)
        audio_url = await client.upload(audio_bytes, "audio/mp3")

        arguments = {
            "audio_url": audio_url,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/audio-isolation",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]


class Demucs(FALNode):
    """
    Demucs separates audio tracks into stems (vocals, drums, bass, other).
    audio, separation, stems, demucs

    Use cases:
    - Separate music into stems
    - Extract vocals or instruments
    - Create remix material
    - Analyze music components
    - Isolate specific tracks
    """

    audio: AudioRef = Field(
        default=AudioRef(), description="The audio file to separate"
    )

    async def process(self, context: ProcessingContext) -> dict:
        client = await self.get_client(context)
        audio_bytes = await context.asset_to_bytes(self.audio)
        audio_url = await client.upload(audio_bytes, "audio/mp3")

        arguments = {
            "audio_url": audio_url,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/demucs",
            arguments=arguments,
        )

        return {
            "vocals": AudioRef(uri=res["vocals"]["url"]) if "vocals" in res else None,
            "drums": AudioRef(uri=res["drums"]["url"]) if "drums" in res else None,
            "bass": AudioRef(uri=res["bass"]["url"]) if "bass" in res else None,
            "other": AudioRef(uri=res["other"]["url"]) if "other" in res else None,
        }

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

    @classmethod
    def return_type(cls):
        return {
            "vocals": AudioRef,
            "drums": AudioRef,
            "bass": AudioRef,
            "other": AudioRef,
        }


class Qwen3Voice(Enum):
    VIVIAN = "Vivian"
    SERENA = "Serena"
    UNCLE_FU = "Uncle_Fu"
    DYLAN = "Dylan"
    ERIC = "Eric"
    RYAN = "Ryan"
    AIDEN = "Aiden"
    ONO_ANNA = "Ono_Anna"
    SOHEE = "Sohee"


class Qwen3Language(Enum):
    AUTO = "Auto"
    ENGLISH = "English"
    CHINESE = "Chinese"
    SPANISH = "Spanish"
    FRENCH = "French"
    GERMAN = "German"
    ITALIAN = "Italian"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    PORTUGUESE = "Portuguese"
    RUSSIAN = "Russian"


class Qwen3TTS17B(FALNode):
    """
    High-quality text-to-speech synthesis with multiple voice options and language support. Uses the Qwen 3 TTS 1.7B model for natural-sounding speech generation.
    tts, text-to-speech, voice, synthesis, multilingual, qwen

    Use cases:
    - Generate natural-sounding speech from text
    - Create voiceovers in multiple languages
    - Produce audio content with custom voice characteristics
    - Generate speech with specific emotional tones
    - Create multilingual audio content
    """

    text: str = Field(
        default="", description="The text to be converted to speech"
    )
    voice: Qwen3Voice = Field(
        default=Qwen3Voice.VIVIAN,
        description="The voice to be used for speech synthesis",
    )
    language: Qwen3Language = Field(
        default=Qwen3Language.AUTO,
        description="The language of the voice",
    )
    prompt: str = Field(
        default="",
        description="Optional prompt to guide the style of the generated speech",
    )
    speaker_voice_embedding_file_url: str = Field(
        default="",
        description="URL to a speaker embedding file from clone-voice endpoint",
    )
    reference_text: str = Field(
        default="",
        description="Optional reference text used when creating the speaker embedding",
    )
    top_k: int = Field(
        default=50, description="Top-k sampling parameter"
    )
    top_p: float = Field(
        default=1.0, description="Top-p sampling parameter"
    )
    temperature: float = Field(
        default=0.9, description="Sampling temperature; higher => more random"
    )
    repetition_penalty: float = Field(
        default=1.05, description="Penalty to reduce repeated tokens/codes"
    )
    max_new_tokens: int = Field(
        default=200, description="Maximum number of new codec tokens to generate"
    )

    @classmethod
    def get_title(cls):
        return "Qwen 3 TTS 1.7B"

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice": self.voice.value,
            "language": self.language.value,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }

        if self.prompt:
            arguments["prompt"] = self.prompt
        if self.speaker_voice_embedding_file_url:
            arguments["speaker_voice_embedding_file_url"] = self.speaker_voice_embedding_file_url
        if self.reference_text:
            arguments["reference_text"] = self.reference_text

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/text-to-speech/1.7b",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text", "voice", "language"]


class Qwen3TTS06B(FALNode):
    """
    Efficient text-to-speech synthesis with the lighter Qwen 3 TTS 0.6B model. Provides fast speech generation with multiple voice options.
    tts, text-to-speech, voice, synthesis, fast, qwen

    Use cases:
    - Generate quick speech output from text
    - Create fast voiceovers for applications
    - Produce audio content efficiently
    - Generate speech for real-time applications
    - Create lightweight audio content
    """

    text: str = Field(
        default="", description="The text to be converted to speech"
    )
    voice: Qwen3Voice = Field(
        default=Qwen3Voice.VIVIAN,
        description="The voice to be used for speech synthesis",
    )
    language: Qwen3Language = Field(
        default=Qwen3Language.AUTO,
        description="The language of the voice",
    )
    prompt: str = Field(
        default="",
        description="Optional prompt to guide the style of the generated speech",
    )
    speaker_voice_embedding_file_url: str = Field(
        default="",
        description="URL to a speaker embedding file from clone-voice endpoint",
    )
    reference_text: str = Field(
        default="",
        description="Optional reference text used when creating the speaker embedding",
    )
    top_k: int = Field(
        default=50, description="Top-k sampling parameter"
    )
    top_p: float = Field(
        default=1.0, description="Top-p sampling parameter"
    )
    temperature: float = Field(
        default=0.9, description="Sampling temperature; higher => more random"
    )
    repetition_penalty: float = Field(
        default=1.05, description="Penalty to reduce repeated tokens/codes"
    )
    max_new_tokens: int = Field(
        default=200, description="Maximum number of new codec tokens to generate"
    )

    @classmethod
    def get_title(cls):
        return "Qwen 3 TTS 0.6B"

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "voice": self.voice.value,
            "language": self.language.value,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }

        if self.prompt:
            arguments["prompt"] = self.prompt
        if self.speaker_voice_embedding_file_url:
            arguments["speaker_voice_embedding_file_url"] = self.speaker_voice_embedding_file_url
        if self.reference_text:
            arguments["reference_text"] = self.reference_text

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/text-to-speech/0.6b",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text", "voice", "language"]


class Qwen3VoiceDesign17B(FALNode):
    """
    Design custom voice styles and emotions using text prompts. Create expressive speech with specific tones and characteristics using the 1.7B model.
    tts, text-to-speech, voice-design, emotion, synthesis, qwen

    Use cases:
    - Create speech with specific emotional characteristics
    - Design custom voice styles for creative content
    - Generate expressive voiceovers with nuanced tones
    - Produce character voices with distinct personalities
    - Create contextually appropriate speech delivery
    """

    text: str = Field(
        default="", description="The text to be converted to speech"
    )
    prompt: str = Field(
        default="",
        description="Prompt to guide the style and emotion of the generated speech",
    )
    language: Qwen3Language = Field(
        default=Qwen3Language.AUTO,
        description="The language of the voice to be designed",
    )
    top_k: int = Field(
        default=50, description="Top-k sampling parameter"
    )
    top_p: float = Field(
        default=1.0, description="Top-p sampling parameter"
    )
    temperature: float = Field(
        default=0.9, description="Sampling temperature; higher => more random"
    )
    repetition_penalty: float = Field(
        default=1.05, description="Penalty to reduce repeated tokens/codes"
    )
    max_new_tokens: int = Field(
        default=200, description="Maximum number of new codec tokens to generate"
    )

    @classmethod
    def get_title(cls):
        return "Qwen 3 Voice Design 1.7B"

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "prompt": self.prompt,
            "language": self.language.value,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/voice-design/1.7b",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["text", "prompt", "language"]


class Qwen3CloneVoice17B(FALNode):
    """
    Clone a voice from an audio sample for text-to-speech synthesis. Creates a speaker embedding that can be used with other Qwen 3 TTS models.
    tts, voice-cloning, speaker-embedding, voice-synthesis, qwen

    Use cases:
    - Clone custom voices from audio samples
    - Create personalized text-to-speech voices
    - Preserve voice characteristics for content creation
    - Generate speaker embeddings for consistent voices
    - Create voice profiles for character consistency
    """

    audio: AudioRef = Field(
        default=AudioRef(),
        description="Reference audio file used for voice cloning",
    )
    reference_text: str = Field(
        default="",
        description="Optional reference text that corresponds to the audio sample",
    )

    @classmethod
    def get_title(cls):
        return "Qwen 3 Clone Voice 1.7B"

    async def process(self, context: ProcessingContext) -> dict:
        client = await self.get_client(context)
        audio_bytes = await context.asset_to_bytes(self.audio)
        audio_url = await client.upload(audio_bytes, "audio/mp3")

        arguments = {
            "audio_url": audio_url,
        }

        if self.reference_text:
            arguments["reference_text"] = self.reference_text

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/clone-voice/1.7b",
            arguments=arguments,
        )
        assert "speaker_embedding" in res
        
        return {
            "speaker_embedding_url": res["speaker_embedding"]["url"],
        }

    @classmethod
    def get_basic_fields(cls):
        return ["audio", "reference_text"]

    @classmethod
    def return_type(cls):
        return {
            "speaker_embedding_url": str,
        }


class Qwen3CloneVoice06B(FALNode):
    """
    Clone a voice from an audio sample using the efficient 0.6B model. Creates a speaker embedding for fast voice cloning.
    tts, voice-cloning, speaker-embedding, fast, qwen

    Use cases:
    - Quick voice cloning from audio samples
    - Create lightweight speaker embeddings
    - Generate fast voice profiles
    - Clone voices for real-time applications
    - Create efficient voice profiles
    """

    audio: AudioRef = Field(
        default=AudioRef(),
        description="Reference audio file used for voice cloning",
    )
    reference_text: str = Field(
        default="",
        description="Optional reference text that corresponds to the audio sample",
    )

    @classmethod
    def get_title(cls):
        return "Qwen 3 Clone Voice 0.6B"

    async def process(self, context: ProcessingContext) -> dict:
        client = await self.get_client(context)
        audio_bytes = await context.asset_to_bytes(self.audio)
        audio_url = await client.upload(audio_bytes, "audio/mp3")

        arguments = {
            "audio_url": audio_url,
        }

        if self.reference_text:
            arguments["reference_text"] = self.reference_text

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-3-tts/clone-voice/0.6b",
            arguments=arguments,
        )
        assert "speaker_embedding" in res
        
        return {
            "speaker_embedding_url": res["speaker_embedding"]["url"],
        }

    @classmethod
    def get_basic_fields(cls):
        return ["audio", "reference_text"]

    @classmethod
    def return_type(cls):
        return {
            "speaker_embedding_url": str,
        }
