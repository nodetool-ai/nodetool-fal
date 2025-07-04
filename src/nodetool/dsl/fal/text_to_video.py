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
