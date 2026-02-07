from enum import Enum
from typing import Any
from pydantic import Field
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.fal.image_to_video import (
    AspectRatio,
    PixverseV56Resolution,
    PixverseV56Duration,
    PixverseV56Style,
    PixverseV56ThinkingType,
)


class KlingDuration(Enum):
    """Duration for Kling video generation"""
    FIVE_SECONDS = "5"
    TEN_SECONDS = "10"


class LumaRay2Resolution(Enum):
    """Resolution for Luma Ray 2 video generation"""
    RES_540P = "540p"
    RES_1080P = "1080p"


class LumaRay2Duration(Enum):
    """Duration for Luma Ray 2 video generation"""
    FIVE_SECONDS = "5"
    TEN_SECONDS = "10"


class PixverseV56AspectRatio(Enum):
    """Aspect ratio for Pixverse V5.6"""
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_21_9 = "21:9"
    RATIO_9_21 = "9:21"



class Veo3AspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"


class Veo3Duration(Enum):
    FOUR_SECONDS = "4s"
    SIX_SECONDS = "6s"
    EIGHT_SECONDS = "8s"


class Veo3Resolution(Enum):
    RES_720P = "720p"
    RES_1080P = "1080p"


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
        description="The aspect ratio of the generated video.",
    )
    duration: Veo3Duration = Field(
        default=Veo3Duration.EIGHT_SECONDS,
        description="The duration of the generated video in seconds",
    )
    resolution: Veo3Resolution = Field(
        default=Veo3Resolution.RES_720P,
        description="The resolution of the generated video",
    )
    generate_audio: bool = Field(
        default=True,
        description="Whether to generate audio for the video. If false, %33 less credits will be used.",
    )
    seed: int = Field(default=-1, description="A seed to use for the video generation")
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation"
    )
    auto_fix: bool = Field(
        default=True, description="Whether to automatically attempt to fix prompts"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments: dict[str, Any] = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
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


class PixverseV56TextToVideo(FALNode):
    """Generate high-quality videos from text prompts with Pixverse v5.6.
    video, generation, pixverse, v5.6, text-to-video, creative

    Use cases:
    - Create professional animated scenes from descriptions
    - Generate marketing and promotional videos
    - Produce dynamic social media content
    - Prototype video concepts with various styles
    - Create stylized video content with anime or cyberpunk themes
    """

    prompt: str = Field(
        default="", description="The text prompt describing the desired video"
    )
    aspect_ratio: PixverseV56AspectRatio = Field(
        default=PixverseV56AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    resolution: PixverseV56Resolution = Field(
        default=PixverseV56Resolution.VALUE_720P,
        description="The resolution quality of the output video",
    )
    duration: PixverseV56Duration = Field(
        default=PixverseV56Duration.FIVE_SECONDS,
        description="The duration of the generated video in seconds",
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated video"
    )
    style: PixverseV56Style | None = Field(
        default=None, description="Optional visual style for the video"
    )
    seed: int = Field(
        default=-1, description="Optional seed for reproducible generation"
    )
    generate_audio_switch: bool | None = Field(
        default=None, description="Whether to generate audio for the video"
    )
    thinking_type: PixverseV56ThinkingType | None = Field(
        default=None, description="Thinking mode for video generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments: dict[str, Any] = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "negative_prompt": self.negative_prompt,
        }
        if self.style is not None:
            arguments["style"] = self.style.value
        if self.seed != -1:
            arguments["seed"] = self.seed
        if self.generate_audio_switch is not None:
            arguments["generate_audio_switch"] = self.generate_audio_switch
        if self.thinking_type is not None:
            arguments["thinking_type"] = self.thinking_type.value

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5.6/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "resolution", "duration"]


class PixverseV56Transition(FALNode):
    """Create smooth transitions between images with Pixverse v5.6.
    video, generation, transition, pixverse, v5.6, morphing

    Use cases:
    - Create seamless transitions between two images
    - Generate morphing effects for presentations
    - Produce smooth scene changes for videos
    - Create animated visual flows
    - Generate creative blending effects
    """

    prompt: str = Field(
        default="", description="Text prompt describing the transition style"
    )
    first_image: ImageRef = Field(
        default=ImageRef(), description="The starting image for the transition"
    )
    end_image: ImageRef | None = Field(
        default=None, description="Optional ending image for the transition"
    )
    aspect_ratio: PixverseV56AspectRatio = Field(
        default=PixverseV56AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    resolution: PixverseV56Resolution = Field(
        default=PixverseV56Resolution.VALUE_720P,
        description="The resolution quality of the output video",
    )
    duration: int = Field(
        default=5, description="Duration in seconds (5 or 8)", ge=5, le=8
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated transition"
    )
    style: PixverseV56Style | None = Field(
        default=None, description="Optional visual style for the transition"
    )
    seed: int = Field(
        default=-1, description="Optional seed for reproducible generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        first_image_base64 = await context.image_to_base64(self.first_image)

        arguments: dict[str, Any] = {
            "prompt": self.prompt,
            "first_image_url": f"data:image/png;base64,{first_image_base64}",
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": self.duration,
            "negative_prompt": self.negative_prompt,
        }
        if self.end_image is not None:
            end_image_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_image_base64}"
        if self.style is not None:
            arguments["style"] = self.style.value
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5.6/transition",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "first_image", "resolution"]


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
    enable_safety_checker: bool = Field(
        default=True,
        description="Whether to enable the safety checker",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments: dict[str, Any] = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "enable_safety_checker": self.enable_safety_checker,
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
    enable_safety_checker: bool = Field(
        default=True,
        description="Whether to enable the safety checker",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments: dict[str, Any] = {
            "prompt": self.prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }
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
    negative_prompt: str = Field(
        default="blur, distort, and low quality",
        description="Negative prompt to be used for the generation",
    )
    cfg_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classifier Free Guidance scale (0.0 to 1.0)",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "duration": self.duration.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
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
    negative_prompt: str = Field(
        default="blur, distort, and low quality",
        description="Negative prompt to be used for the generation",
    )
    cfg_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classifier Free Guidance scale (0.0 to 1.0)",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
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





class LumaRay2TextToVideo(FALNode):
    """
    Luma Ray 2 Text-to-Video generates high-quality videos from text prompts.
    video, generation, luma, ray2, text-to-video, txt2vid

    Use cases:
    - Create videos from descriptions
    - Generate cinematic content
    - Produce creative videos
    - Create marketing clips
    - Generate concept videos
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    loop: bool = Field(default=False, description="Whether the video should loop")
    resolution: LumaRay2Resolution = Field(
        default=LumaRay2Resolution.RES_540P,
        description="The resolution of the generated video",
    )
    duration: LumaRay2Duration = Field(
        default=LumaRay2Duration.FIVE_SECONDS,
        description="The duration of the generated video",
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "loop": self.loop,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "duration", "resolution"]


class LumaRay2FlashTextToVideo(FALNode):
    """
    Luma Ray 2 Flash Text-to-Video is a fast version for quick video generation.
    video, generation, luma, ray2, flash, text-to-video, fast

    Use cases:
    - Quick video prototyping
    - Rapid content creation
    - Fast video iterations
    - Real-time video generation
    - Quick concept tests
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    resolution: LumaRay2Resolution = Field(
        default=LumaRay2Resolution.RES_540P,
        description="The resolution of the generated video",
    )
    duration: LumaRay2Duration = Field(
        default=LumaRay2Duration.FIVE_SECONDS,
        description="The duration of the generated video",
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2-flash",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "duration", "resolution"]


class KlingVideoV21TextToVideo(FALNode):
    """
    Kling Video V2.1 Master Text-to-Video with enhanced quality and motion.
    video, generation, kling, v2.1, text-to-video

    Use cases:
    - Create professional video content
    - Generate high-quality animations
    - Produce cinematic clips
    - Create promotional videos
    - Generate concept previews
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
        description="The aspect ratio of the generated video",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2.1/master/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration"]


class HunyuanVideo(FALNode):
    """
    Hunyuan Video generates videos from text prompts using Tencent's model.
    video, generation, hunyuan, tencent, text-to-video

    Use cases:
    - Create videos from descriptions
    - Generate animated content
    - Produce motion graphics
    - Create promotional clips
    - Generate concept videos
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    num_inference_steps: int = Field(
        default=30, ge=1, description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=7.0, description="How closely to follow the prompt"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "guidance_scale"]


class HunyuanVideoV15TextToVideo(FALNode):
    """
    Hunyuan Video V1.5 Text-to-Video with improved quality and motion.
    video, generation, hunyuan, v1.5, text-to-video

    Use cases:
    - Create high-quality video content
    - Generate smooth animations
    - Produce professional videos
    - Create motion graphics
    - Generate video effects
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    num_inference_steps: int = Field(
        default=30, ge=1, description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=7.0, description="How closely to follow the prompt"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video-v1.5/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "guidance_scale"]


class PikaV22TextToVideo(FALNode):
    """
    Pika V2.2 Text-to-Video generates creative videos from text prompts.
    video, generation, pika, v2.2, text-to-video, creative

    Use cases:
    - Create creative video content
    - Generate artistic animations
    - Produce stylized videos
    - Create unique video clips
    - Generate experimental content
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated video"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/pika/v2.2/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]


class PikaV21TextToVideo(FALNode):
    """
    Pika V2.1 Text-to-Video generates videos from text prompts.
    video, generation, pika, v2.1, text-to-video

    Use cases:
    - Create video content from text
    - Generate animated clips
    - Produce motion graphics
    - Create video effects
    - Generate promotional content
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/pika/v2.1/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]


class Sora2Duration(int, Enum):
    _4s = 4
    _8s = 8
    _12s = 12


class Sora2TextToVideo(FALNode):
    """
    OpenAI Sora 2 Text-to-Video generates high-quality videos from text.
    video, generation, openai, sora, sora2, text-to-video

    Use cases:
    - Create cinematic videos from text
    - Generate realistic motion
    - Produce professional video content
    - Create video narratives
    - Generate concept videos
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    duration: Sora2Duration = Field(
        default=Sora2Duration._4s,
        description="Duration of the video in seconds",
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/sora-2/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration"]


class MiniMaxDuration(Enum):
    SIX_SECONDS = "6"
    TEN_SECONDS = "10"


class MiniMaxHailuo23TextToVideo(FALNode):
    """
    MiniMax Hailuo 2.3 Standard Text-to-Video with improved quality.
    video, generation, minimax, hailuo, 2.3, text-to-video

    Use cases:
    - Create videos from text
    - Generate smooth animations
    - Produce video content
    - Create motion graphics
    - Generate promotional clips
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the prompt optimizer"
    )
    duration: MiniMaxDuration = Field(
        default=MiniMaxDuration.SIX_SECONDS,
        description="The duration of the video in seconds",
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
            "duration": self.duration.value,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-2.3/standard/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration"]


class MochiV1(FALNode):
    """
    Mochi V1 generates creative videos from text prompts with unique style.
    video, generation, mochi, text-to-video, creative

    Use cases:
    - Create creative video content
    - Generate artistic animations
    - Produce stylized videos
    - Create experimental clips
    - Generate unique video effects
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated video"
    )
    num_inference_steps: int = Field(
        default=50, ge=1, description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=4.5, description="How closely to follow the prompt"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/mochi-v1",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "guidance_scale"]


class LTX2TextToVideo(FALNode):
    """
    LTX 2 Text-to-Video generates videos from text with the LTX model.
    video, generation, ltx, text-to-video

    Use cases:
    - Create videos from descriptions
    - Generate animated content
    - Produce motion graphics
    - Create video clips
    - Generate promotional content
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated video"
    )
    num_inference_steps: int = Field(
        default=30, ge=1, description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=3.0, description="How closely to follow the prompt"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "guidance_scale"]


class Kandinsky5TextToVideo(FALNode):
    """
    Kandinsky 5 Text-to-Video generates creative videos from text prompts.
    video, generation, kandinsky, text-to-video, artistic

    Use cases:
    - Create artistic video content
    - Generate creative animations
    - Produce stylized videos
    - Create video art
    - Generate experimental content
    """

    prompt: str = Field(
        default="", description="The prompt describing the desired video"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated video"
    )
    num_inference_steps: int = Field(
        default=50, ge=1, description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=4.0, description="How closely to follow the prompt"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/kandinsky5/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "guidance_scale"]


class Kling3Duration(Enum):
    """Duration for Kling 3.0/O3 videos (3-15 seconds)."""
    THREE_SECONDS = "3"
    FOUR_SECONDS = "4"
    FIVE_SECONDS = "5"
    SIX_SECONDS = "6"
    SEVEN_SECONDS = "7"
    EIGHT_SECONDS = "8"
    NINE_SECONDS = "9"
    TEN_SECONDS = "10"
    ELEVEN_SECONDS = "11"
    TWELVE_SECONDS = "12"
    THIRTEEN_SECONDS = "13"
    FOURTEEN_SECONDS = "14"
    FIFTEEN_SECONDS = "15"


class Kling3AspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"


class Kling3ShotType(Enum):
    """Shot type for multi-shot video generation."""
    CUSTOMIZE = "customize"
    INTELLIGENT = "intelligent"


class KlingV3TextToVideo(FALNode):
    """
    Generate high-quality videos from text prompts using Kling Video 3.0 Standard with improved motion and realistic acting.
    video, generation, kling, v3, text-to-video, cinematic

    Use cases:
    - Create cinematic video clips from descriptions
    - Generate marketing and promotional videos
    - Produce dynamic social media content
    - Visualize creative concepts and storyboards
    - Create professional video content
    """

    prompt: str = Field(
        default="", description="The text prompt describing the desired video"
    )
    duration: Kling3Duration = Field(
        default=Kling3Duration.FIVE_SECONDS,
        description="The duration of the generated video in seconds (3-15)",
    )
    aspect_ratio: Kling3AspectRatio = Field(
        default=Kling3AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate native audio for the video (supports Chinese/English)",
    )
    voice_ids: list[str] = Field(
        default=[],
        description="Voice IDs for audio. Reference in prompt with <<<voice_1>>> (max 2 voices)",
    )
    shot_type: Kling3ShotType = Field(
        default=Kling3ShotType.CUSTOMIZE,
        description="Shot type for multi-shot generation",
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality",
        description="What to avoid in the generated video",
    )
    cfg_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classifier Free Guidance scale (0.0 to 1.0)",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "shot_type": self.shot_type.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }
        if self.voice_ids:
            arguments["voice_ids"] = self.voice_ids

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v3/standard/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration", "aspect_ratio"]


class KlingV3ProTextToVideo(FALNode):
    """
    Generate premium quality videos from text prompts using Kling Video 3.0 Pro with enhanced quality and performance.
    video, generation, kling, v3, pro, text-to-video, premium

    Use cases:
    - Create high-end promotional content
    - Generate professional cinematic sequences
    - Produce premium marketing videos
    - Create detailed visual narratives
    - Generate broadcast-quality content
    """

    prompt: str = Field(
        default="", description="The text prompt describing the desired video"
    )
    duration: Kling3Duration = Field(
        default=Kling3Duration.FIVE_SECONDS,
        description="The duration of the generated video in seconds (3-15)",
    )
    aspect_ratio: Kling3AspectRatio = Field(
        default=Kling3AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate native audio for the video (supports Chinese/English)",
    )
    voice_ids: list[str] = Field(
        default=[],
        description="Voice IDs for audio. Reference in prompt with <<<voice_1>>> (max 2 voices)",
    )
    shot_type: Kling3ShotType = Field(
        default=Kling3ShotType.CUSTOMIZE,
        description="Shot type for multi-shot generation",
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality",
        description="What to avoid in the generated video",
    )
    cfg_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classifier Free Guidance scale (0.0 to 1.0)",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "shot_type": self.shot_type.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }
        if self.voice_ids:
            arguments["voice_ids"] = self.voice_ids

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v3/pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration", "aspect_ratio"]


class KlingO3TextToVideo(FALNode):
    """
    Generate cinematic videos with Kling Video O3 Standard supporting storyboard-first creation and character consistency.
    video, generation, kling, o3, text-to-video, storyboard, cinematic

    Use cases:
    - Create multi-shot video sequences
    - Generate story-driven content
    - Produce character-consistent videos
    - Create structured narrative videos
    - Generate cinematic sequences with continuity
    """

    prompt: str = Field(
        default="", description="The text prompt describing the desired video"
    )
    duration: Kling3Duration = Field(
        default=Kling3Duration.FIVE_SECONDS,
        description="The duration of the generated video in seconds (3-15)",
    )
    aspect_ratio: Kling3AspectRatio = Field(
        default=Kling3AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate native audio for the video",
    )
    voice_ids: list[str] = Field(
        default=[],
        description="Voice IDs for audio. Reference in prompt with <<<voice_1>>> (max 2 voices)",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "shot_type": "customize",
        }
        if self.generate_audio:
            arguments["generate_audio"] = self.generate_audio
        if self.voice_ids:
            arguments["voice_ids"] = self.voice_ids

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/standard/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration", "aspect_ratio"]


class KlingO3ProTextToVideo(FALNode):
    """
    Generate premium cinematic videos with Kling Video O3 Pro featuring higher-end customization and storyboard-first creation.
    video, generation, kling, o3, pro, text-to-video, premium, storyboard

    Use cases:
    - Create professional multi-shot sequences
    - Generate premium story-driven content
    - Produce high-quality character-consistent videos
    - Create broadcast-quality narrative videos
    - Generate cinematic content with advanced controls
    """

    prompt: str = Field(
        default="", description="The text prompt describing the desired video"
    )
    duration: Kling3Duration = Field(
        default=Kling3Duration.FIVE_SECONDS,
        description="The duration of the generated video in seconds (3-15)",
    )
    aspect_ratio: Kling3AspectRatio = Field(
        default=Kling3AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate native audio for the video",
    )
    voice_ids: list[str] = Field(
        default=[],
        description="Voice IDs for audio. Reference in prompt with <<<voice_1>>> (max 2 voices)",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "shot_type": "customize",
        }
        if self.generate_audio:
            arguments["generate_audio"] = self.generate_audio
        if self.voice_ids:
            arguments["voice_ids"] = self.voice_ids

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration", "aspect_ratio"]
