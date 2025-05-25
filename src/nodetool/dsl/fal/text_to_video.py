from pydantic import Field
import typing
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.fal.image_to_video


class KlingTextToVideoV2(GraphNode):
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

    KlingDuration: typing.ClassVar[type] = (
        nodetool.nodes.fal.image_to_video.KlingDuration
    )
    AspectRatio: typing.ClassVar[type] = nodetool.nodes.fal.image_to_video.AspectRatio
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt describing the desired video"
    )
    duration: nodetool.nodes.fal.image_to_video.KlingDuration = Field(
        default=nodetool.nodes.fal.image_to_video.KlingDuration.FIVE_SECONDS,
        description="The duration of the generated video",
    )
    aspect_ratio: nodetool.nodes.fal.image_to_video.AspectRatio = Field(
        default=nodetool.nodes.fal.image_to_video.AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video frame",
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.KlingTextToVideoV2"


class KlingVideoV2(GraphNode):
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

    KlingDuration: typing.ClassVar[type] = (
        nodetool.nodes.fal.image_to_video.KlingDuration
    )
    AspectRatio: typing.ClassVar[type] = nodetool.nodes.fal.image_to_video.AspectRatio
    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The image to transform into a video",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="A description of the desired video motion and style"
    )
    duration: nodetool.nodes.fal.image_to_video.KlingDuration = Field(
        default=nodetool.nodes.fal.image_to_video.KlingDuration.FIVE_SECONDS,
        description="The duration of the generated video",
    )
    aspect_ratio: nodetool.nodes.fal.image_to_video.AspectRatio = Field(
        default=nodetool.nodes.fal.image_to_video.AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video frame",
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.KlingVideoV2"
