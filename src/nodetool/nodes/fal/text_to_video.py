from pydantic import Field
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


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
        arguments = {"prompt": self.prompt}
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
            arguments["seed"] = self.seed

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

        arguments = {
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

        arguments = {"video_url": video_url, "prompt": self.prompt}
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

        arguments = {
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
