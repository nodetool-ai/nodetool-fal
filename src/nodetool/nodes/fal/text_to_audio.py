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
