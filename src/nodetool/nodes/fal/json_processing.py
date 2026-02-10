from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import VideoRef, AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class FfmpegApiLoudnorm(FALNode):
    """
    Get EBU R128 loudness normalization from audio files using FFmpeg API.
    json, processing, data, utility

    Use cases:
    - JSON data processing
    - Data transformation
    - Metadata extraction
    - Audio analysis
    - Media processing utilities
    """

    measured_tp: str = Field(
        default="", description="Measured true peak of input file in dBTP. Required for linear mode."
    )
    offset: float = Field(
        default=0, description="Offset gain in dB applied before the true-peak limiter"
    )
    print_summary: bool = Field(
        default=False, description="Return loudness measurement summary with the normalized audio"
    )
    measured_i: str = Field(
        default="", description="Measured integrated loudness of input file in LUFS. Required for linear mode."
    )
    linear: bool = Field(
        default=False, description="Use linear normalization mode (single-pass). If false, uses dynamic mode (two-pass for better quality)."
    )
    measured_lra: str = Field(
        default="", description="Measured loudness range of input file in LU. Required for linear mode."
    )
    dual_mono: bool = Field(
        default=False, description="Treat mono input files as dual-mono for correct EBU R128 measurement on stereo systems"
    )
    measured_thresh: str = Field(
        default="", description="Measured threshold of input file in LUFS. Required for linear mode."
    )
    true_peak: float = Field(
        default=-0.1, description="Maximum true peak in dBTP."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to normalize"
    )
    integrated_loudness: float = Field(
        default=-18, description="Integrated loudness target in LUFS."
    )
    loudness_range: float = Field(
        default=7, description="Loudness range target in LU"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "measured_tp": self.measured_tp,
            "offset": self.offset,
            "print_summary": self.print_summary,
            "measured_i": self.measured_i,
            "linear": self.linear,
            "measured_lra": self.measured_lra,
            "dual_mono": self.dual_mono,
            "measured_thresh": self.measured_thresh,
            "true_peak": self.true_peak,
            "audio_url": self.audio,
            "integrated_loudness": self.integrated_loudness,
            "loudness_range": self.loudness_range,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ffmpeg-api/loudnorm",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["measured_tp", "offset", "print_summary", "measured_i", "linear"]

class FfmpegApiWaveform(FALNode):
    """
    Get waveform data from audio files using FFmpeg API.
    json, processing, data, utility

    Use cases:
    - JSON data processing
    - Data transformation
    - Metadata extraction
    - Audio analysis
    - Media processing utilities
    """

    precision: int = Field(
        default=2, description="Number of decimal places for the waveform values. Higher values provide more precision but increase payload size."
    )
    smoothing_window: int = Field(
        default=3, description="Size of the smoothing window. Higher values create a smoother waveform. Must be an odd number."
    )
    media: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to analyze"
    )
    points_per_second: float = Field(
        default=4, description="Controls how many points are sampled per second of audio. Lower values (e.g. 1-2) create a coarser waveform, higher values (e.g. 4-10) create a more detailed one."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "precision": self.precision,
            "smoothing_window": self.smoothing_window,
            "media_url": self.media,
            "points_per_second": self.points_per_second,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ffmpeg-api/waveform",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["precision", "smoothing_window", "media", "points_per_second"]

class FfmpegApiMetadata(FALNode):
    """
    Get encoding metadata from video and audio files using FFmpeg API.
    json, processing, data, utility

    Use cases:
    - JSON data processing
    - Data transformation
    - Metadata extraction
    - Audio analysis
    - Media processing utilities
    """

    extract_frames: bool = Field(
        default=False, description="Whether to extract the start and end frames for videos. Note that when true the request will be slower."
    )
    media: VideoRef = Field(
        default=VideoRef(), description="URL of the media file (video or audio) to analyze"
    )

    async def process(self, context: ProcessingContext) -> Any:
        arguments = {
            "extract_frames": self.extract_frames,
            "media_url": self.media,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ffmpeg-api/metadata",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["extract_frames", "media"]