from pydantic import Field

from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


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
        arguments = {
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
        arguments = {"prompt": self.prompt}
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
        arguments = {"prompt": self.prompt}
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
        arguments = {"prompt": self.prompt}
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
        arguments = {"prompt": self.prompt}
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
