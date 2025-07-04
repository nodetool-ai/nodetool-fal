from enum import Enum
from typing import Any
from pydantic import Field
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class Veo3AspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"


class Veo3Duration(Enum):
    EIGHT_SECONDS = "8s"


class Veo3(FALNode):
    """
    Generate high-quality videos from text prompts with Google's Veo 3 model.
    video, generation, text-to-video, prompt, audio

    Use cases:
    - Produce short cinematic clips from descriptions
    - Create social media videos
    - Generate visual storyboards
    - Experiment with video concepts
    - Produce marketing content
    """

    prompt: str = Field(
        default="",
        description="The text prompt describing the video you want to generate",
    )
    aspect_ratio: Veo3AspectRatio = Field(
        default=Veo3AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video. If it is not set to 16:9, the video will be outpainted with Luma Ray 2 Reframe functionality.",
    )
    duration: Veo3Duration = Field(
        default=Veo3Duration.EIGHT_SECONDS,
        description="The duration of the generated video in seconds",
    )
    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio for the video. If false, %33 less credits will be used.",
    )
    seed: int = Field(default=-1, description="A seed to use for the video generation")
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation"
    )
    enhance_prompt: bool = Field(
        default=True, description="Whether to enhance the video generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments: dict[str, Any] = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "generate_audio": self.generate_audio,
            "enhance_prompt": self.enhance_prompt,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "duration"]


class PixverseTextToVideo(FALNode):
    """Generate videos from text prompts with Pixverse 4.5 API.
    video, generation, pixverse, text-to-video

    Use cases:
    - Create animated scenes from text
    - Generate marketing clips
    - Produce dynamic social posts
    - Prototype video ideas
    - Explore creative storytelling
    """

    prompt: str = Field(default="", description="The prompt describing the video")
    seed: int = Field(default=-1, description="Optional seed for deterministic output")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments: dict[str, Any] = {"prompt": self.prompt}
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4.5/text-to-video/api",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]


class PixverseTextToVideoFast(FALNode):
    """Generate videos quickly from text prompts with Pixverse 4.5 Fast.
    video, generation, pixverse, text-to-video, fast

    Use cases:
    - Rapid video prototyping
    - Generate quick social posts
    - Produce short marketing clips
    - Test creative ideas fast
    - Create video drafts
    """

    prompt: str = Field(default="", description="The prompt describing the video")
    seed: int = Field(default=-1, description="Optional seed for deterministic output")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {"prompt": self.prompt}
        if self.seed != -1:
            arguments["seed"] = str(self.seed)

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4.5/text-to-video/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]


class PixverseTransition(FALNode):
    """Apply Pixverse transitions between images.
    video, generation, transition, pixverse

    Use cases:
    - Blend between two images
    - Create animated transitions
    - Generate morphing effects
    - Produce smooth scene changes
    - Experiment with visual flows
    """

    start_image: ImageRef = Field(default=ImageRef(), description="The starting image")
    end_image: ImageRef = Field(default=ImageRef(), description="The ending image")
    seed: int = Field(default=-1, description="Optional seed for deterministic output")

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_base64 = await context.image_to_base64(self.start_image)
        end_base64 = await context.image_to_base64(self.end_image)

        arguments: dict[str, Any] = {
            "start_image_url": f"data:image/png;base64,{start_base64}",
            "end_image_url": f"data:image/png;base64,{end_base64}",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4.5/transition",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["start_image", "end_image"]


class PixverseEffects(FALNode):
    """Apply text-driven effects to a video with Pixverse 4.5.
    video, effects, pixverse, text-guided

    Use cases:
    - Stylize existing footage
    - Add visual effects via text
    - Enhance marketing videos
    - Create experimental clips
    - Transform user content
    """

    video: VideoRef = Field(default=VideoRef(), description="The source video")
    prompt: str = Field(default="", description="Text describing the effect")
    seed: int = Field(default=-1, description="Optional seed for deterministic output")

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = self.get_client(context)
        video_bytes = await context.asset_to_bytes(self.video)
        video_url = await client.upload(video_bytes, "video/mp4")

        arguments: dict[str, Any] = {"video_url": video_url, "prompt": self.prompt}
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4.5/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]


class PixverseImageToVideo(FALNode):
    """Animate an image into a video using Pixverse 4.5.
    video, generation, pixverse, image-to-video

    Use cases:
    - Bring photos to life
    - Create moving artwork
    - Generate short clips from images
    - Produce social media animations
    - Experiment with visual storytelling
    """

    image: ImageRef = Field(default=ImageRef(), description="The source image")
    prompt: str = Field(default="", description="Optional style or motion prompt")
    seed: int = Field(default=-1, description="Optional seed for deterministic output")

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments: dict[str, Any] = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4.5/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]
