from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class PixverseEffects(GraphNode):
    """Apply text-driven effects to a video with Pixverse 4.5.
    video, effects, pixverse, text-guided

    Use cases:
    - Stylize existing footage
    - Add visual effects via text
    - Enhance marketing videos
    - Create experimental clips
    - Transform user content
    """

    video: types.VideoRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.VideoRef(
            type="video", uri="", asset_id=None, data=None, duration=None, format=None
        ),
        description="The source video",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Text describing the effect"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Optional seed for deterministic output"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.PixverseEffects"


class PixverseImageToVideo(GraphNode):
    """Animate an image into a video using Pixverse 4.5.
    video, generation, pixverse, image-to-video

    Use cases:
    - Bring photos to life
    - Create moving artwork
    - Generate short clips from images
    - Produce social media animations
    - Experiment with visual storytelling
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The source image",
    )
    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="Optional style or motion prompt"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Optional seed for deterministic output"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.PixverseImageToVideo"


class PixverseTextToVideo(GraphNode):
    """Generate videos from text prompts with Pixverse 4.5 API.
    video, generation, pixverse, text-to-video

    Use cases:
    - Create animated scenes from text
    - Generate marketing clips
    - Produce dynamic social posts
    - Prototype video ideas
    - Explore creative storytelling
    """

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt describing the video"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Optional seed for deterministic output"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.PixverseTextToVideo"


class PixverseTextToVideoFast(GraphNode):
    """Generate videos quickly from text prompts with Pixverse 4.5 Fast.
    video, generation, pixverse, text-to-video, fast

    Use cases:
    - Rapid video prototyping
    - Generate quick social posts
    - Produce short marketing clips
    - Test creative ideas fast
    - Create video drafts
    """

    prompt: str | GraphNode | tuple[GraphNode, str] = Field(
        default="", description="The prompt describing the video"
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Optional seed for deterministic output"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.PixverseTextToVideoFast"


class PixverseTransition(GraphNode):
    """Apply Pixverse transitions between images.
    video, generation, transition, pixverse

    Use cases:
    - Blend between two images
    - Create animated transitions
    - Generate morphing effects
    - Produce smooth scene changes
    - Experiment with visual flows
    """

    start_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The starting image",
    )
    end_image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The ending image",
    )
    seed: int | GraphNode | tuple[GraphNode, str] = Field(
        default=-1, description="Optional seed for deterministic output"
    )

    @classmethod
    def get_node_type(cls):
        return "fal.text_to_video.PixverseTransition"
