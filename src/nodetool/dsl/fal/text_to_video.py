from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.fal.text_to_video
import nodetool.nodes.fal.text_to_video


class Veo3(GraphNode):
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

    Veo3AspectRatio: typing.ClassVar[type] = (
        nodetool.nodes.fal.text_to_video.Veo3AspectRatio
    )
    Veo3Duration: typing.ClassVar[type] = nodetool.nodes.fal.text_to_video.Veo3Duration
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="",
        description="The text prompt describing the video you want to generate",
    )
    aspect_ratio: nodetool.nodes.fal.text_to_video.Veo3AspectRatio = Field(
        default=nodetool.nodes.fal.text_to_video.Veo3AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video. If it is not set to 16:9, the video will be outpainted with Luma Ray 2 Reframe functionality.",
    )
    duration: nodetool.nodes.fal.text_to_video.Veo3Duration = Field(
        default=nodetool.nodes.fal.text_to_video.Veo3Duration.EIGHT_SECONDS,
        description="The duration of the generated video in seconds",
    )
    generate_audio: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True,
        description="Whether to generate audio for the video. If false, %33 less credits will be used.",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="A seed to use for the video generation"
    )
    negative_prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="A negative prompt to guide the video generation"
    )
    enhance_prompt: bool | GraphNode | tuple[GraphNode, str] = Field(
        default=True, description="Whether to enhance the video generation"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.Veo3"
