from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import VideoRef, AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.nodes.fal.types import AudioTimeSpan  # noqa: F401
from nodetool.workflows.processing_context import ProcessingContext


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

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        FAST = "fast"
        BALANCED = "balanced"
        QUALITY = "quality"

    class OutputFormat(Enum):
        """
        Output audio format.
        """
        WAV = "wav"
        MP3 = "mp3"


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

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        FAST = "fast"
        BALANCED = "balanced"
        QUALITY = "quality"

    class OutputFormat(Enum):
        """
        Output audio format.
        """
        WAV = "wav"
        MP3 = "mp3"


    prompt: str = Field(
        default="", description="Text prompt describing the sound to isolate. Optional but recommended - helps the model identify what type of sound to extract from the span."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.BALANCED, description="The acceleration level to use."
    )
    spans: list[AudioTimeSpan] = Field(
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

    class OutputFormat(Enum):
        """
        Output audio format for the separated stems
        """
        WAV = "wav"
        MP3 = "mp3"

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

class FfmpegApiMergeAudios(FALNode):
    """
    FFmpeg API Merge Audios combines multiple audio files into a single output.
    audio, processing, audio-to-audio, merging, ffmpeg

    Use cases:
    - Combine multiple audio tracks
    - Merge audio segments
    - Create audio compilations
    - Join split audio files
    - Generate combined audio output
    """

    audio_urls: list[str] = Field(
        default=[], description="List of audio URLs to merge in order. The 0th stream of the audio will be considered as the merge candidate."
    )
    output_format: str = Field(
        default="", description="Output format of the combined audio. If not used, will be determined automatically using FFMPEG. Formatted as codec_sample_rate_bitrate."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "audio_urls": self.audio_urls,
            "output_format": self.output_format,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ffmpeg-api/merge-audios",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class KlingVideoCreateVoice(FALNode):
    """
    Create Voices to be used with Kling 2.6 Voice Control
    audio, processing, audio-to-audio, transformation

    Use cases:
    - Audio enhancement and processing
    - Voice transformation
    - Audio style transfer
    - Sound quality improvement
    - Audio effect application
    """

    voice_url: VideoRef = Field(
        default=VideoRef(), description="URL of the voice audio file. Supports .mp3/.wav audio or .mp4/.mov video. Duration must be 5-30 seconds with clean, single-voice audio."
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "voice_url": self.voice_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/create-voice",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class AudioUnderstanding(FALNode):
    """
    A audio understanding model to analyze audio content and answer questions about what's happening in the audio based on user prompts.
    audio, processing, audio-to-audio, transformation

    Use cases:
    - Audio enhancement and processing
    - Voice transformation
    - Audio style transfer
    - Sound quality improvement
    - Audio effect application
    """

    prompt: str = Field(
        default="", description="The question or prompt about the audio content."
    )
    detailed_analysis: bool = Field(
        default=False, description="Whether to request a more detailed analysis of the audio"
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to analyze"
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "prompt": self.prompt,
            "detailed_analysis": self.detailed_analysis,
            "audio_url": self.audio_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/audio-understanding",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class StableAudio25Inpaint(FALNode):
    """
    Generate high quality music and sound effects using Stable Audio 2.5 from StabilityAI
    audio, processing, audio-to-audio, transformation

    Use cases:
    - Audio enhancement and processing
    - Voice transformation
    - Audio style transfer
    - Sound quality improvement
    - Audio effect application
    """

    prompt: str = Field(
        default="", description="The prompt to guide the audio generation"
    )
    guidance_scale: int = Field(
        default=1, description="How strictly the diffusion process adheres to the prompt text (higher values make your audio closer to your prompt)."
    )
    mask_end: int = Field(
        default=190, description="The end point of the audio mask"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The audio clip to inpaint"
    )
    seed: int = Field(
        default=0
    )
    seconds_total: int = Field(
        default=190, description="The duration of the audio clip to generate. If not provided, it will be set to the duration of the input audio."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of steps to denoise the audio for"
    )
    mask_start: int = Field(
        default=30, description="The start point of the audio mask"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "mask_end": self.mask_end,
            "sync_mode": self.sync_mode,
            "audio_url": self.audio_url,
            "seed": self.seed,
            "seconds_total": self.seconds_total,
            "num_inference_steps": self.num_inference_steps,
            "mask_start": self.mask_start,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-audio-25/inpaint",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class SonautoV2Extend(FALNode):
    """
    Extend an existing song
    audio, processing, audio-to-audio, transformation

    Use cases:
    - Audio enhancement and processing
    - Voice transformation
    - Audio style transfer
    - Sound quality improvement
    - Audio effect application
    """

    class OutputFormat(Enum):
        FLAC = "flac"
        MP3 = "mp3"
        WAV = "wav"
        OGG = "ogg"
        M4A = "m4a"

    class Side(Enum):
        """
        Add more to the beginning (left) or end (right) of the song
        """
        LEFT = "left"
        RIGHT = "right"


    prompt: str = Field(
        default="", description="A description of the track you want to generate. This prompt will be used to automatically generate the tags and lyrics unless you manually set them. For example, if you set prompt and tags, then the prompt will be used to generate only the lyrics."
    )
    lyrics_prompt: str = Field(
        default="", description="The lyrics sung in the generated song. An empty string will generate an instrumental track."
    )
    tags: str = Field(
        default="", description="Tags/styles of the music to generate. You can view a list of all available tags at https://sonauto.ai/tag-explorer."
    )
    prompt_strength: float = Field(
        default=1.8, description="Controls how strongly your prompt influences the output. Greater values adhere more to the prompt but sound less natural. (This is CFG.)"
    )
    output_bit_rate: str = Field(
        default="", description="The bit rate to use for mp3 and m4a formats. Not available for other formats."
    )
    num_songs: int = Field(
        default=1, description="Generating 2 songs costs 1.5x the price of generating 1 song. Also, note that using the same seed may not result in identical songs if the number of songs generated is changed."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.WAV
    )
    side: Side = Field(
        default="", description="Add more to the beginning (left) or end (right) of the song"
    )
    balance_strength: float = Field(
        default=0.7, description="Greater means more natural vocals. Lower means sharper instrumentals. We recommend 0.7."
    )
    crop_duration: float = Field(
        default=0, description="Duration in seconds to crop from the selected side before extending from that side."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file to alter. Must be a valid publicly accessible URL."
    )
    seed: str = Field(
        default="", description="The seed to use for generation. Will pick a random seed if not provided. Repeating a request with identical parameters (must use lyrics and tags, not prompt) and the same seed will generate the same song."
    )
    extend_duration: str = Field(
        default="", description="Duration in seconds to extend the song. If not provided, will attempt to automatically determine."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "prompt": self.prompt,
            "lyrics_prompt": self.lyrics_prompt,
            "tags": self.tags,
            "prompt_strength": self.prompt_strength,
            "output_bit_rate": self.output_bit_rate,
            "num_songs": self.num_songs,
            "output_format": self.output_format.value,
            "side": self.side.value,
            "balance_strength": self.balance_strength,
            "crop_duration": self.crop_duration,
            "audio_url": self.audio_url,
            "seed": self.seed,
            "extend_duration": self.extend_duration,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="sonauto/v2/extend",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class AceStepAudioOutpaint(FALNode):
    """
    Extend the beginning or end of provided audio with lyrics and/or style using ACE-Step
    audio, processing, audio-to-audio, transformation

    Use cases:
    - Audio enhancement and processing
    - Voice transformation
    - Audio style transfer
    - Sound quality improvement
    - Audio effect application
    """

    class Scheduler(Enum):
        """
        Scheduler to use for the generation process.
        """
        EULER = "euler"
        HEUN = "heun"

    class GuidanceType(Enum):
        """
        Type of CFG to use for the generation process.
        """
        CFG = "cfg"
        APG = "apg"
        CFG_STAR = "cfg_star"


    number_of_steps: int = Field(
        default=27, description="Number of steps to generate the audio."
    )
    tags: str = Field(
        default="", description="Comma-separated list of genre tags to control the style of the generated audio."
    )
    minimum_guidance_scale: float = Field(
        default=3, description="Minimum guidance scale for the generation after the decay."
    )
    extend_after_duration: float = Field(
        default=30, description="Duration in seconds to extend the audio from the end."
    )
    lyrics: str = Field(
        default="", description="Lyrics to be sung in the audio. If not provided or if [inst] or [instrumental] is the content of this field, no lyrics will be sung. Use control structures like [verse], [chorus] and [bridge] to control the structure of the song."
    )
    tag_guidance_scale: float = Field(
        default=5, description="Tag guidance scale for the generation."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULER, description="Scheduler to use for the generation process."
    )
    extend_before_duration: float = Field(
        default=0, description="Duration in seconds to extend the audio from the start."
    )
    guidance_type: GuidanceType = Field(
        default=GuidanceType.APG, description="Type of CFG to use for the generation process."
    )
    guidance_scale: float = Field(
        default=15, description="Guidance scale for the generation."
    )
    lyric_guidance_scale: float = Field(
        default=1.5, description="Lyric guidance scale for the generation."
    )
    guidance_interval: float = Field(
        default=0.5, description="Guidance interval for the generation. 0.5 means only apply guidance in the middle steps (0.25 * infer_steps to 0.75 * infer_steps)"
    )
    guidance_interval_decay: float = Field(
        default=0, description="Guidance interval decay for the generation. Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to be outpainted."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If not provided, a random seed will be used."
    )
    granularity_scale: int = Field(
        default=10, description="Granularity scale for the generation process. Higher values can reduce artifacts."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "number_of_steps": self.number_of_steps,
            "tags": self.tags,
            "minimum_guidance_scale": self.minimum_guidance_scale,
            "extend_after_duration": self.extend_after_duration,
            "lyrics": self.lyrics,
            "tag_guidance_scale": self.tag_guidance_scale,
            "scheduler": self.scheduler.value,
            "extend_before_duration": self.extend_before_duration,
            "guidance_type": self.guidance_type.value,
            "guidance_scale": self.guidance_scale,
            "lyric_guidance_scale": self.lyric_guidance_scale,
            "guidance_interval": self.guidance_interval,
            "guidance_interval_decay": self.guidance_interval_decay,
            "audio_url": self.audio_url,
            "seed": self.seed,
            "granularity_scale": self.granularity_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ace-step/audio-outpaint",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class AceStepAudioInpaint(FALNode):
    """
    Modify a portion of provided audio with lyrics and/or style using ACE-Step
    audio, processing, audio-to-audio, transformation

    Use cases:
    - Audio enhancement and processing
    - Voice transformation
    - Audio style transfer
    - Sound quality improvement
    - Audio effect application
    """

    class EndTimeRelativeTo(Enum):
        """
        Whether the end time is relative to the start or end of the audio.
        """
        START = "start"
        END = "end"

    class Scheduler(Enum):
        """
        Scheduler to use for the generation process.
        """
        EULER = "euler"
        HEUN = "heun"

    class GuidanceType(Enum):
        """
        Type of CFG to use for the generation process.
        """
        CFG = "cfg"
        APG = "apg"
        CFG_STAR = "cfg_star"

    class StartTimeRelativeTo(Enum):
        """
        Whether the start time is relative to the start or end of the audio.
        """
        START = "start"
        END = "end"


    number_of_steps: int = Field(
        default=27, description="Number of steps to generate the audio."
    )
    start_time: float = Field(
        default=0, description="start time in seconds for the inpainting process."
    )
    tags: str = Field(
        default="", description="Comma-separated list of genre tags to control the style of the generated audio."
    )
    minimum_guidance_scale: float = Field(
        default=3, description="Minimum guidance scale for the generation after the decay."
    )
    lyrics: str = Field(
        default="", description="Lyrics to be sung in the audio. If not provided or if [inst] or [instrumental] is the content of this field, no lyrics will be sung. Use control structures like [verse], [chorus] and [bridge] to control the structure of the song."
    )
    end_time_relative_to: EndTimeRelativeTo = Field(
        default=EndTimeRelativeTo.START, description="Whether the end time is relative to the start or end of the audio."
    )
    tag_guidance_scale: float = Field(
        default=5, description="Tag guidance scale for the generation."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULER, description="Scheduler to use for the generation process."
    )
    end_time: float = Field(
        default=30, description="end time in seconds for the inpainting process."
    )
    guidance_type: GuidanceType = Field(
        default=GuidanceType.APG, description="Type of CFG to use for the generation process."
    )
    guidance_scale: float = Field(
        default=15, description="Guidance scale for the generation."
    )
    lyric_guidance_scale: float = Field(
        default=1.5, description="Lyric guidance scale for the generation."
    )
    guidance_interval: float = Field(
        default=0.5, description="Guidance interval for the generation. 0.5 means only apply guidance in the middle steps (0.25 * infer_steps to 0.75 * infer_steps)"
    )
    variance: float = Field(
        default=0.5, description="Variance for the inpainting process. Higher values can lead to more diverse results."
    )
    guidance_interval_decay: float = Field(
        default=0, description="Guidance interval decay for the generation. Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay."
    )
    start_time_relative_to: StartTimeRelativeTo = Field(
        default=StartTimeRelativeTo.START, description="Whether the start time is relative to the start or end of the audio."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to be inpainted."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If not provided, a random seed will be used."
    )
    granularity_scale: int = Field(
        default=10, description="Granularity scale for the generation process. Higher values can reduce artifacts."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "number_of_steps": self.number_of_steps,
            "start_time": self.start_time,
            "tags": self.tags,
            "minimum_guidance_scale": self.minimum_guidance_scale,
            "lyrics": self.lyrics,
            "end_time_relative_to": self.end_time_relative_to.value,
            "tag_guidance_scale": self.tag_guidance_scale,
            "scheduler": self.scheduler.value,
            "end_time": self.end_time,
            "guidance_type": self.guidance_type.value,
            "guidance_scale": self.guidance_scale,
            "lyric_guidance_scale": self.lyric_guidance_scale,
            "guidance_interval": self.guidance_interval,
            "variance": self.variance,
            "guidance_interval_decay": self.guidance_interval_decay,
            "start_time_relative_to": self.start_time_relative_to.value,
            "audio_url": self.audio_url,
            "seed": self.seed,
            "granularity_scale": self.granularity_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ace-step/audio-inpaint",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class AceStepAudioToAudio(FALNode):
    """
    Generate music from a lyrics and example audio using ACE-Step
    audio, processing, audio-to-audio, transformation

    Use cases:
    - Audio enhancement and processing
    - Voice transformation
    - Audio style transfer
    - Sound quality improvement
    - Audio effect application
    """

    class Scheduler(Enum):
        """
        Scheduler to use for the generation process.
        """
        EULER = "euler"
        HEUN = "heun"

    class GuidanceType(Enum):
        """
        Type of CFG to use for the generation process.
        """
        CFG = "cfg"
        APG = "apg"
        CFG_STAR = "cfg_star"

    class EditMode(Enum):
        """
        Whether to edit the lyrics only or remix the audio.
        """
        LYRICS = "lyrics"
        REMIX = "remix"


    number_of_steps: int = Field(
        default=27, description="Number of steps to generate the audio."
    )
    tags: str = Field(
        default="", description="Comma-separated list of genre tags to control the style of the generated audio."
    )
    minimum_guidance_scale: float = Field(
        default=3, description="Minimum guidance scale for the generation after the decay."
    )
    lyrics: str = Field(
        default="", description="Lyrics to be sung in the audio. If not provided or if [inst] or [instrumental] is the content of this field, no lyrics will be sung. Use control structures like [verse], [chorus] and [bridge] to control the structure of the song."
    )
    tag_guidance_scale: float = Field(
        default=5, description="Tag guidance scale for the generation."
    )
    original_lyrics: str = Field(
        default="", description="Original lyrics of the audio file."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULER, description="Scheduler to use for the generation process."
    )
    guidance_scale: float = Field(
        default=15, description="Guidance scale for the generation."
    )
    guidance_type: GuidanceType = Field(
        default=GuidanceType.APG, description="Type of CFG to use for the generation process."
    )
    lyric_guidance_scale: float = Field(
        default=1.5, description="Lyric guidance scale for the generation."
    )
    guidance_interval: float = Field(
        default=0.5, description="Guidance interval for the generation. 0.5 means only apply guidance in the middle steps (0.25 * infer_steps to 0.75 * infer_steps)"
    )
    edit_mode: EditMode = Field(
        default=EditMode.REMIX, description="Whether to edit the lyrics only or remix the audio."
    )
    guidance_interval_decay: float = Field(
        default=0, description="Guidance interval decay for the generation. Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to be outpainted."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If not provided, a random seed will be used."
    )
    granularity_scale: int = Field(
        default=10, description="Granularity scale for the generation process. Higher values can reduce artifacts."
    )
    original_tags: str = Field(
        default="", description="Original tags of the audio file."
    )
    original_seed: int = Field(
        default=-1, description="Original seed of the audio file."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "number_of_steps": self.number_of_steps,
            "tags": self.tags,
            "minimum_guidance_scale": self.minimum_guidance_scale,
            "lyrics": self.lyrics,
            "tag_guidance_scale": self.tag_guidance_scale,
            "original_lyrics": self.original_lyrics,
            "scheduler": self.scheduler.value,
            "guidance_scale": self.guidance_scale,
            "guidance_type": self.guidance_type.value,
            "lyric_guidance_scale": self.lyric_guidance_scale,
            "guidance_interval": self.guidance_interval,
            "edit_mode": self.edit_mode.value,
            "guidance_interval_decay": self.guidance_interval_decay,
            "audio_url": self.audio_url,
            "seed": self.seed,
            "granularity_scale": self.granularity_scale,
            "original_tags": self.original_tags,
            "original_seed": self.original_seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ace-step/audio-to-audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class DiaTtsVoiceClone(FALNode):
    """
    Clone dialog voices from a sample audio and generate dialogs from text prompts using the Dia TTS which leverages advanced AI techniques to create high-quality text-to-speech.
    audio, processing, audio-to-audio, transformation

    Use cases:
    - Audio enhancement and processing
    - Voice transformation
    - Audio style transfer
    - Sound quality improvement
    - Audio effect application
    """

    text: str = Field(
        default="", description="The text to be converted to speech."
    )
    ref_text: str = Field(
        default="", description="The reference text to be used for TTS."
    )
    ref_audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the reference audio file."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "text": self.text,
            "ref_text": self.ref_text,
            "ref_audio_url": self.ref_audio_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/dia-tts/voice-clone",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio"]

class ElevenlabsAudioIsolation(FALNode):
    """
    Isolate audio tracks using ElevenLabs advanced audio isolation technology.
    audio, processing, audio-to-audio, transformation

    Use cases:
    - Audio enhancement and processing
    - Voice transformation
    - Audio style transfer
    - Sound quality improvement
    - Audio effect application
    """

    video_url: VideoRef = Field(
        default=VideoRef(), description="Video file to use for audio isolation. Either `audio_url` or `video_url` must be provided."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to isolate voice from"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "video_url": self.video_url,
            "audio_url": self.audio_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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