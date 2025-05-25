from pydantic import Field

from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.nodes.fal.image_to_video import AspectRatio, KlingDuration
from nodetool.workflows.processing_context import ProcessingContext


class KlingVideoV2(FALNode):
    """
    Generate videos from images using Kling Video V2 Master. Create smooth and realistic animations from a single frame.
    video, generation, animation, img2vid, kling-v2

    Use cases:
    - Convert artwork into animated clips
    - Produce dynamic marketing visuals
    - Generate motion graphics from static scenes
    - Create short cinematic sequences
    - Enhance presentations with video content
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    duration: KlingDuration = Field(
        default=KlingDuration.FIVE_SECONDS,
        description="The duration of the generated video",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video frame",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
        }
        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2/master/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class KlingTextToVideoV2(FALNode):
    """
    Generate videos directly from text prompts using Kling Video V2 Master.
    video, generation, animation, text-to-video, kling-v2

    Use cases:
    - Visualize scripts or storyboards
    - Produce short promotional videos from text
    - Create animated social media content
    - Generate concept previews for film ideas
    - Produce text-driven motion graphics
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    duration: KlingDuration = Field(
        default=KlingDuration.FIVE_SECONDS,
        description="The duration of the generated video",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video frame",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
        }
        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2/master/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration"]
