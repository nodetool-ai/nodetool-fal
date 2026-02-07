from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class PixverseV56Resolution(Enum):
    """The resolution of the generated video"""
    RES_360P = "360p"
    RES_540P = "540p"
    RES_720P = "720p"
    RES_1080P = "1080p"


class PixverseV56Duration(Enum):
    """The duration of the generated video in seconds. 1080p videos are limited to 5 or 8 seconds"""
    FIVE_SECONDS = "5"
    EIGHT_SECONDS = "8"
    TEN_SECONDS = "10"


class PixverseV56Style(Enum):
    """The style of the generated video"""
    ANIME = "anime"
    ANIMATION_3D = "3d_animation"
    CLAY = "clay"
    COMIC = "comic"
    CYBERPUNK = "cyberpunk"


class PixverseV56ThinkingType(Enum):
    """Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    AUTO = "auto"


class PixverseV56ImageToVideo(FALNode):
    """
    Generate high-quality videos from images with Pixverse v5.6.
    video, generation, pixverse, v5.6, image-to-video, img2vid

    Use cases:
    - Animate photos into professional video clips
    - Create dynamic product showcase videos
    - Generate stylized video content from artwork
    - Produce high-resolution social media animations
    - Transform static images with various visual styles
    """

    prompt: str = Field(
        default="", description="Text prompt describing the desired video motion"
    )
    resolution: PixverseV56Resolution = Field(
        default=PixverseV56Resolution.RES_720P, description="The resolution quality of the output video"
    )
    duration: PixverseV56Duration = Field(
        default=PixverseV56Duration.FIVE_SECONDS, description="The duration of the generated video in seconds"
    )
    style: PixverseV56Style | None = Field(
        default=None, description="Optional visual style for the video"
    )
    thinking_type: PixverseV56ThinkingType | None = Field(
        default=None, description="Thinking mode for video generation"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    generate_audio_switch: bool = Field(
        default=False, description="Whether to generate audio for the video"
    )
    seed: int = Field(
        default=-1, description="Optional seed for reproducible generation"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "image_url": f"data:image/png;base64,{image_base64}",
            "generate_audio_switch": self.generate_audio_switch,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5.6/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "resolution"]

class AspectRatio(Enum):
    """The aspect ratio of the generated video"""
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_21_9 = "21:9"
    RATIO_9_21 = "9:21"


class LumaDreamMachine(FALNode):
    """
    Generate video clips from your images using Luma Dream Machine v1.5. Supports various aspect ratios and optional end-frame blending.
    video, generation, animation, blending, aspect-ratio, img2vid, image-to-video

    Use cases:
    - Create seamless video loops
    - Generate video transitions
    - Transform images into animations
    - Create motion graphics
    - Produce video content
    """

    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    loop: bool = Field(
        default=False, description="Whether the video should loop (end of video is blended with the beginning)"
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="An image to blend the end of the video with"
    )
    image_url: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "loop": self.loop,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "resolution"]