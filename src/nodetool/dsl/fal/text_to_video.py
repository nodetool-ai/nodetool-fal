from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class WanFlf2V(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt describing the desired video"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.WanFlf2V"


class WanProImageToVideo(GraphNode):
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

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to animate",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Optional prompt describing the desired motion"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.WanProImageToVideo"


class WanProTextToVideo(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt describing the desired video"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.WanProTextToVideo"


class WanT2V(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt describing the desired video"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.WanT2V"


class WanV2_1_13BTextToVideo(GraphNode):
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

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt describing the desired video"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Randomization seed for reproducible results"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.WanV2_1_13BTextToVideo"
