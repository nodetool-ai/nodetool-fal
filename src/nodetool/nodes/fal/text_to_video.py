from enum import Enum
from typing import Any
from pydantic import Field
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.fal.image_to_video import AspectRatio, KlingDuration



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
        client = await self.get_client(context)
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


class WanProImageToVideo(FALNode):
    """
    Convert an image into a short video clip using Wan Pro.
    video, generation, wan, professional, image-to-video

    Use cases:
    - Create dynamic videos from product photos
    - Generate animations from static artwork
    - Produce short promotional clips
    - Transform images into motion graphics
    - Experiment with visual storytelling
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to animate"
    )
    prompt: str = Field(
        default="", description="Optional prompt describing the desired motion"
    )
    seed: int = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

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
            application="fal-ai/wan-pro/image-to-video/api",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["image", "prompt"]


class WanProTextToVideo(FALNode):
    """
    Generate a short video clip from a text prompt using Wan Pro.
    video, generation, wan, professional, text-to-video

    Use cases:
    - Create animated scenes from descriptions
    - Generate short creative videos
    - Produce promotional content
    - Visualize storyboards
    - Experiment with narrative ideas
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    seed: int = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments: dict[str, Any] = {"prompt": self.prompt}
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-pro/text-to-video/api",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt"]


class WanV2_1_13BTextToVideo(FALNode):
    """
    Create videos from text using WAN v2.1 1.3B, an open-source text-to-video model.
    video, generation, wan, text-to-video

    Use cases:
    - Produce short clips from prompts
    - Generate concept videos
    - Create quick visualizations
    - Iterate on storytelling ideas
    - Experiment with AI video synthesis
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    seed: int = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments: dict[str, Any] = {"prompt": self.prompt}
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.1/1.3b/text-to-video/api",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt"]


class WanT2V(FALNode):
    """
    Generate videos from text using the WAN-T2V model.
    video, generation, wan, text-to-video

    Use cases:
    - Produce creative videos from prompts
    - Experiment with motion concepts
    - Generate quick animated drafts
    - Visualize ideas for stories
    - Create short social media clips
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    seed: int = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments: dict[str, Any] = {"prompt": self.prompt}
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-t2v/api",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt"]


class WanFlf2V(FALNode):
    """
    Generate video loops from text prompts using WAN-FLF2V.
    video, generation, wan, text-to-video

    Use cases:
    - Generate looping videos from descriptions
    - Produce motion graphics from prompts
    - Create abstract video ideas
    - Develop creative transitions
    - Experiment with AI-generated motion
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    seed: int = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments: dict[str, Any] = {"prompt": self.prompt}
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-flf2v/api",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return ["prompt"]

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
