from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import VideoRef, AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class SamAudioVisualSeparate(FALNode):
    """
    Audio separation with SAM Audio. Isolate any sound using natural language—professional-grade audio editing made simple for creators, researchers, and accessibility applications.
    audio, extraction, video-to-audio, processing

    Use cases:
    - Audio extraction from video
    - Sound separation
    - Video audio analysis
    - Music extraction
    - Sound effect isolation
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
        default="", description="Text prompt to assist with separation. Use natural language to describe the target sound."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video file to process (MP4, MOV, etc.)"
    )
    acceleration: Acceleration = Field(
        default=Acceleration.BALANCED, description="The acceleration level to use."
    )
    mask_video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the mask video (binary mask indicating target object). Black=target, White=background."
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
            "video_url": self.video_url,
            "acceleration": self.acceleration.value,
            "mask_video_url": self.mask_video_url,
            "output_format": self.output_format.value,
            "reranking_candidates": self.reranking_candidates,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-audio/visual-separate",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "video_url", "acceleration", "mask_video_url", "output_format"]

class MireloAiSfxV15VideoToAudio(FALNode):
    """
    Generate synced sounds for any video, and return the new sound track (like MMAudio)
    audio, extraction, video-to-audio, processing

    Use cases:
    - Audio extraction from video
    - Sound separation
    - Video audio analysis
    - Music extraction
    - Sound effect isolation
    """

    num_samples: str = Field(
        default=2, description="The number of samples to generate from the model"
    )
    duration: str = Field(
        default=10, description="The duration of the generated audio in seconds"
    )
    start_offset: str = Field(
        default=0, description="The start offset in seconds to start the audio generation from"
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="A video url that can accessed from the API to process and add sound effects"
    )
    seed: str = Field(
        default=8069, description="The seed to use for the generation. If not provided, a random seed will be used"
    )
    text_prompt: str = Field(
        default="", description="Additional description to guide the model"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "num_samples": self.num_samples,
            "duration": self.duration,
            "start_offset": self.start_offset,
            "video_url": self.video_url,
            "seed": self.seed,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="mirelo-ai/sfx-v1.5/video-to-audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["num_samples", "duration", "start_offset", "video_url", "seed"]

class KlingVideoVideoToAudio(FALNode):
    """
    Generate audio from input videos using Kling
    audio, extraction, video-to-audio, processing

    Use cases:
    - Audio extraction from video
    - Sound separation
    - Video audio analysis
    - Music extraction
    - Sound effect isolation
    """

    video_url: VideoRef = Field(
        default=VideoRef(), description="The video URL to extract audio from. Only .mp4/.mov formats are supported. File size does not exceed 100MB. Video duration between 3.0s and 20.0s."
    )
    asmr_mode: bool = Field(
        default=False, description="Enable ASMR mode. This mode enhances detailed sound effects and is suitable for highly immersive content scenarios."
    )
    background_music_prompt: str = Field(
        default="intense car race", description="Background music prompt. Cannot exceed 200 characters."
    )
    sound_effect_prompt: str = Field(
        default="Car tires screech as they accelerate in a drag race", description="Sound effect prompt. Cannot exceed 200 characters."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "asmr_mode": self.asmr_mode,
            "background_music_prompt": self.background_music_prompt,
            "sound_effect_prompt": self.sound_effect_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/video-to-audio",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video_url", "asmr_mode", "background_music_prompt", "sound_effect_prompt"]

class MireloAiSfxV1VideoToAudio(FALNode):
    """
    Generate synced sounds for any video, and return the new sound track (like MMAudio)
    audio, extraction, video-to-audio, processing

    Use cases:
    - Audio extraction from video
    - Sound separation
    - Video audio analysis
    - Music extraction
    - Sound effect isolation
    """

    num_samples: str = Field(
        default=2, description="The number of samples to generate from the model"
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="A video url that can accessed from the API to process and add sound effects"
    )
    duration: str = Field(
        default=10, description="The duration of the generated audio in seconds"
    )
    seed: str = Field(
        default=2105, description="The seed to use for the generation. If not provided, a random seed will be used"
    )
    text_prompt: str = Field(
        default="", description="Additional description to guide the model"
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        arguments = {
            "num_samples": self.num_samples,
            "video_url": self.video_url,
            "duration": self.duration,
            "seed": self.seed,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="mirelo-ai/sfx-v1/video-to-audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["num_samples", "video_url", "duration", "seed", "text_prompt"]