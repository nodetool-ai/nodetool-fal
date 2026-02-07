from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class OutputFormat(Enum):
    """
    Output format of the generated audio. Formatted as codec_sample_rate_bitrate.
    """
    MP3_22050_32 = "mp3_22050_32"
    MP3_44100_32 = "mp3_44100_32"
    MP3_44100_64 = "mp3_44100_64"
    MP3_44100_96 = "mp3_44100_96"
    MP3_44100_128 = "mp3_44100_128"
    MP3_44100_192 = "mp3_44100_192"
    PCM_8000 = "pcm_8000"
    PCM_16000 = "pcm_16000"
    PCM_22050 = "pcm_22050"
    PCM_24000 = "pcm_24000"
    PCM_44100 = "pcm_44100"
    PCM_48000 = "pcm_48000"
    ULAW_8000 = "ulaw_8000"
    ALAW_8000 = "alaw_8000"
    OPUS_48000_32 = "opus_48000_32"
    OPUS_48000_64 = "opus_48000_64"
    OPUS_48000_96 = "opus_48000_96"
    OPUS_48000_128 = "opus_48000_128"
    OPUS_48000_192 = "opus_48000_192"


class AudioFormat(Enum):
    """
    The format for the output audio.
    """
    MP3 = "mp3"
    AAC = "aac"
    M4A = "m4a"
    OGG = "ogg"
    OPUS = "opus"
    FLAC = "flac"
    WAV = "wav"


class Acceleration(Enum):
    """
    The acceleration level to use.
    """
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class Model(Enum):
    """
    Demucs model to use for separation
    """
    HTDEMUCS = "htdemucs"
    HTDEMUCS_FT = "htdemucs_ft"
    HTDEMUCS_6S = "htdemucs_6s"
    HDEMUCS_MMI = "hdemucs_mmi"
    MDX = "mdx"
    MDX_EXTRA = "mdx_extra"
    MDX_Q = "mdx_q"
    MDX_EXTRA_Q = "mdx_extra_q"



class ElevenlabsVoiceChanger(FALNode):
    """
    ElevenLabs Voice Changer transforms voice characteristics in audio with AI-powered voice conversion.
    audio, voice-change, elevenlabs, transformation, audio-to-audio

    Use cases:
    - Change voice characteristics in audio
    - Transform vocal qualities
    - Create voice variations
    - Modify speaker identity
    - Generate voice-changed audio
    """

    voice: str = Field(
        default="Rachel", description="The voice to use for speech generation"
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The input audio file"
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP3_44100_128, description="Output format of the generated audio. Formatted as codec_sample_rate_bitrate."
    )
    remove_background_noise: bool = Field(
        default=False, description="If set, will remove the background noise from your audio input using our audio isolation model."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "voice": self.voice,
            "audio_url": self.audio_url,
            "seed": self.seed,
            "output_format": self.output_format.value,
            "remove_background_noise": self.remove_background_noise,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/voice-changer",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]


class NovaSr(FALNode):
    """
    Nova SR enhances audio quality through super-resolution processing for clearer and richer sound.
    audio, enhancement, super-resolution, quality, audio-to-audio

    Use cases:
    - Enhance audio quality
    - Improve sound clarity
    - Upscale audio resolution
    - Restore degraded audio
    - Generate high-quality audio
    """

    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    bitrate: str = Field(
        default="192k", description="The bitrate of the output audio."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file to enhance."
    )
    audio_format: AudioFormat = Field(
        default=AudioFormat.MP3, description="The format for the output audio."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "sync_mode": self.sync_mode,
            "bitrate": self.bitrate,
            "audio_url": self.audio_url,
            "audio_format": self.audio_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/nova-sr",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]


class Deepfilternet3(FALNode):
    """
    DeepFilterNet3 removes noise and improves audio quality with advanced deep learning filtering.
    audio, noise-reduction, filtering, cleaning, audio-to-audio

    Use cases:
    - Remove noise from audio
    - Clean audio recordings
    - Filter unwanted sounds
    - Improve audio clarity
    - Generate clean audio
    """

    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    audio_format: AudioFormat = Field(
        default=AudioFormat.MP3, description="The format for the output audio."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio to enhance."
    )
    bitrate: str = Field(
        default="192k", description="The bitrate of the output audio."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "sync_mode": self.sync_mode,
            "audio_format": self.audio_format.value,
            "audio_url": self.audio_url,
            "bitrate": self.bitrate,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/deepfilternet3",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]



class SamAudioSeparate(FALNode):
    """
    SAM Audio Separate isolates and extracts different audio sources from mixed recordings.
    audio, separation, source-extraction, isolation, audio-to-audio

    Use cases:
    - Separate audio sources
    - Extract vocals from music
    - Isolate instruments
    - Remove background sounds
    - Generate separated audio tracks
    """

    prompt: str = Field(
        default="", description="Text prompt describing the sound to isolate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.BALANCED, description="The acceleration level to use."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to process (WAV, MP3, FLAC supported)"
    )
    predict_spans: bool = Field(
        default=False, description="Automatically predict temporal spans where the target sound occurs."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.WAV, description="Output audio format."
    )
    reranking_candidates: int = Field(
        default=1, description="Number of candidates to generate and rank. Higher improves quality but increases latency and cost."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "audio_url": self.audio_url,
            "predict_spans": self.predict_spans,
            "output_format": self.output_format.value,
            "reranking_candidates": self.reranking_candidates,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-audio/separate",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]



class SamAudioSpanSeparate(FALNode):
    """
    SAM Audio Span Separate isolates audio sources across time spans with precise temporal control.
    audio, separation, temporal, span, audio-to-audio

    Use cases:
    - Separate audio by time spans
    - Extract sources in specific periods
    - Isolate temporal audio segments
    - Remove sounds in time ranges
    - Generate time-based separations
    """

    prompt: str = Field(
        default="", description="Text prompt describing the sound to isolate. Optional but recommended - helps the model identify what type of sound to extract from the span."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.BALANCED, description="The acceleration level to use."
    )
    spans: list[str] = Field(
        default=[], description="Time spans where the target sound occurs which should be isolated."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.WAV, description="Output audio format."
    )
    trim_to_span: bool = Field(
        default=False, description="Trim output audio to only include the specified span time range. If False, returns the full audio length with the target sound isolated throughout."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to process."
    )
    reranking_candidates: int = Field(
        default=1, description="Number of candidates to generate and rank. Higher improves quality but increases latency and cost. Requires text prompt; ignored for span-only separation."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "spans": self.spans,
            "output_format": self.output_format.value,
            "trim_to_span": self.trim_to_span,
            "audio_url": self.audio_url,
            "reranking_candidates": self.reranking_candidates,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-audio/span-separate",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]



class Demucs(FALNode):
    """
    Demucs separates music into vocals, drums, bass, and other instruments with high quality.
    audio, music-separation, stems, demucs, audio-to-audio

    Use cases:
    - Separate music into stems
    - Extract vocals from songs
    - Isolate instruments in music
    - Create karaoke tracks
    - Generate individual audio stems
    """

    segment_length: str = Field(
        default="", description="Length in seconds of each segment for processing. Smaller values use less memory but may reduce quality. Default is model-specific."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.MP3, description="Output audio format for the separated stems"
    )
    stems: str = Field(
        default="", description="Specific stems to extract. If None, extracts all available stems. Available stems depend on model: vocals, drums, bass, other, guitar, piano (for 6s model)"
    )
    overlap: float = Field(
        default=0.25, description="Overlap between segments (0.0 to 1.0). Higher values may improve quality but increase processing time."
    )
    model: Model = Field(
        default=Model.HTDEMUCS_6S, description="Demucs model to use for separation"
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to separate into stems"
    )
    shifts: int = Field(
        default=1, description="Number of random shifts for equivariant stabilization. Higher values improve quality but increase processing time."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "segment_length": self.segment_length,
            "output_format": self.output_format.value,
            "stems": self.stems,
            "overlap": self.overlap,
            "model": self.model.value,
            "audio_url": self.audio_url,
            "shifts": self.shifts,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/demucs",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class StableAudio25AudioToAudio(FALNode):
    """
    Stable Audio 2.5 transforms and modifies audio with AI-powered processing and effects.
    audio, transformation, stable-audio, 2.5, audio-to-audio

    Use cases:
    - Transform audio characteristics
    - Apply AI-powered audio effects
    - Modify audio properties
    - Generate audio variations
    - Create processed audio
    """

    prompt: str = Field(
        default="", description="The prompt to guide the audio generation"
    )
    strength: float = Field(
        default=0.8, description="Sometimes referred to as denoising, this parameter controls how much influence the `audio_url` parameter has on the generated audio. A value of 0 would yield audio that is identical to the input. A value of 1 would be as if you passed in no audio at all."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The audio clip to transform"
    )
    num_inference_steps: int = Field(
        default=8, description="The number of steps to denoise the audio for"
    )
    guidance_scale: int = Field(
        default=1, description="How strictly the diffusion process adheres to the prompt text (higher values make your audio closer to your prompt)."
    )
    seed: int = Field(
        default=0
    )
    total_seconds: int = Field(
        default=0, description="The duration of the audio clip to generate. If not provided, it will be set to the duration of the input audio."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "strength": self.strength,
            "sync_mode": self.sync_mode,
            "audio_url": self.audio_url,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "total_seconds": self.total_seconds,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-audio-25/audio-to-audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio", "prompt"]