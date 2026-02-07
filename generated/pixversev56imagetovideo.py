from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class Resolution(Enum):
    """The resolution of the generated video"""
    360P = "360p"
    540P = "540p"
    720P = "720p"
    1080P = "1080p"


class Duration(Enum):
    """The duration of the generated video in seconds. 1080p videos are limited to 5 or 8 seconds"""
    VALUE_5 = "5"
    VALUE_8 = "8"
    VALUE_10 = "10"


class Style(Enum):
    """The style of the generated video"""
    ANIME = "anime"
    3D_ANIMATION = "3d_animation"
    CLAY = "clay"
    COMIC = "comic"
    CYBERPUNK = "cyberpunk"


class ThinkingType(Enum):
    """Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    AUTO = "auto"


class Status(Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class PixverseV56ImageToVideo(FALNode):
    """
    PixverseV56ImageToVideo node for fal-ai/pixverse/v5.6/image-to-video
    fal, ai, generation
    """

    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.720P, description="The resolution of the generated video"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds. 1080p videos are limited to 5 or 8 seconds"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    thinking_type: ThinkingType | None = Field(
        default=None, description="Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    generate_audio_switch: bool = Field(
        default=False, description="Enable audio generation (BGM, SFX, dialogue)"
    )
    seed: int = Field(
        default=-1, description="
            The same seed and the same prompt given to the same version of the model
            will output the same video every time.
        "
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "resolution", "duration", "style", "thinking_type"]