from enum import Enum

from pydantic import Field

from nodetool.metadata.types import VideoRef
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
        arguments = {
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
