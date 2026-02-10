from pydantic import Field
from typing import Any

from nodetool.metadata.types import AudioRef, ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext
from enum import Enum


class VideoDuration(Enum):
    FOUR_SECONDS = 4
    SIX_SECONDS = 6


class HaiperImageToVideo(FALNode):
    """
    Transform images into hyper-realistic videos with Haiper 2.0. Experience industry-leading resolution, fluid motion, and rapid generation for stunning AI videos.
    video, generation, hyper-realistic, motion, animation, image-to-video, img2vid

    Use cases:
    - Create cinematic animations
    - Generate dynamic video content
    - Transform static images into motion
    - Produce high-resolution videos
    - Create visual effects
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    duration: VideoDuration = Field(
        default=VideoDuration.FOUR_SECONDS,
        description="The duration of the generated video in seconds",
    )
    prompt_enhancer: bool = Field(
        default=True, description="Whether to use the model's prompt enhancer"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same video every time"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "duration": self.duration,
            "prompt_enhancer": self.prompt_enhancer,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/haiper-video-v2/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class AspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_21_9 = "21:9"
    RATIO_9_21 = "9:21"
    RATIO_1_1 = "1:1"


class KlingDuration(Enum):
    FIVE_SECONDS = "5"
    TEN_SECONDS = "10"


class PixverseV56AspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_4_3 = "4:3"
    RATIO_1_1 = "1:1"
    RATIO_3_4 = "3:4"
    RATIO_9_16 = "9:16"


class PixverseV56Resolution(Enum):
    RES_360P = "360p"
    RES_540P = "540p"
    RES_720P = "720p"
    RES_1080P = "1080p"


class PixverseV56Duration(Enum):
    FIVE_SECONDS = "5"
    EIGHT_SECONDS = "8"
    TEN_SECONDS = "10"


class PixverseV56Style(Enum):
    ANIME = "anime"
    ANIMATION_3D = "3d_animation"
    CLAY = "clay"
    COMIC = "comic"
    CYBERPUNK = "cyberpunk"


class PixverseV56ThinkingType(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    AUTO = "auto"


class LumaRay2Resolution(Enum):
    RES_540P = "540p"
    RES_720P = "720p"
    RES_1080P = "1080p"


class LumaRay2Duration(Enum):
    FIVE_SECONDS = "5s"
    NINE_SECONDS = "9s"


class LumaDreamMachine(FALNode):
    """
    Generate video clips from your images using Luma Dream Machine v1.5. Supports various aspect ratios and optional end-frame blending.
    video, generation, animation, blending, aspect-ratio, img2vid, image-to-video

    Use cases:
    - Create seamless video loops
    - Generate video transitions
    - Transform images into animations
    - Create motion graphics
    - Produce video content
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    loop: bool = Field(
        default=False,
        description="Whether the video should loop (end blends with beginning)",
    )
    end_image: ImageRef | None = Field(
        default=None, description="Optional image to blend the end of the video with"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "loop": self.loop,
        }

        if self.end_image:
            end_image_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_image_base64}"

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "aspect_ratio"]


class KlingVideo(FALNode):
    """
    Generate video clips from your images using Kling 1.6. Supports multiple durations and aspect ratios.
    video, generation, animation, duration, aspect-ratio, img2vid, image-to-video

    Use cases:
    - Create custom video content
    - Generate video animations
    - Transform static images
    - Produce motion graphics
    - Create visual presentations
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
            application="fal-ai/kling-video/v1.6/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class KlingVideoPro(FALNode):
    """
    Generate video clips from your images using Kling 1.6 Pro. The professional version offers enhanced quality and performance compared to the standard version.
    video, generation, professional, quality, performance, img2vid, image-to-video

    Use cases:
    - Create professional video content
    - Generate high-quality animations
    - Produce commercial video assets
    - Create advanced motion graphics
    - Generate premium visual content
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
            application="fal-ai/kling-video/v1.6/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class VideoSize(Enum):
    SQUARE_HD = "square_hd"
    SQUARE = "square"
    PORTRAIT_4_3 = "portrait_4_3"
    PORTRAIT_16_9 = "portrait_16_9"
    LANDSCAPE_4_3 = "landscape_4_3"
    LANDSCAPE_16_9 = "landscape_16_9"


class CogVideoX(FALNode):
    """
    Generate videos from images using CogVideoX-5B. Features high-quality motion synthesis with configurable parameters for fine-tuned control over the output.
    video, generation, motion, synthesis, control, img2vid, image-to-video

    Use cases:
    - Create controlled video animations
    - Generate precise motion effects
    - Produce customized video content
    - Create fine-tuned animations
    - Generate motion sequences
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    video_size: VideoSize = Field(
        default=VideoSize.LANDSCAPE_16_9,
        description="The size/aspect ratio of the generated video",
    )
    negative_prompt: str = Field(
        default="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
        description="What to avoid in the generated video",
    )
    num_inference_steps: int = Field(
        default=50,
        description="Number of denoising steps (higher = better quality but slower)",
    )
    guidance_scale: float = Field(
        default=7.0,
        description="How closely to follow the prompt (higher = more faithful but less creative)",
    )
    use_rife: bool = Field(
        default=True, description="Whether to use RIFE for video interpolation"
    )
    export_fps: int = Field(
        default=16, description="Target frames per second for the output video"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same video every time"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "video_size": self.video_size.value,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "use_rife": self.use_rife,
            "export_fps": self.export_fps,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/cogvideox-5b/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "video_size"]


class MiniMaxVideo(FALNode):
    """
    Generate video clips from your images using MiniMax Video model. Transform static art into dynamic masterpieces with enhanced smoothness and vivid motion.
    video, generation, art, motion, smoothness, img2vid, image-to-video

    Use cases:
    - Transform artwork into videos
    - Create smooth animations
    - Generate artistic motion content
    - Produce dynamic visualizations
    - Create video art pieces
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/video-01-live/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class HailuoDuration(Enum):
    SIX_SECONDS = "6"
    TEN_SECONDS = "10"


class MiniMaxHailuoResolution(Enum):
    RES_512P = "512P"
    RES_768P = "768P"


class MiniMaxHailuo02(FALNode):
    """
    Create videos from your images with MiniMax Hailuo-02 Standard. Choose the
    clip length and optionally enhance prompts for sharper results.
    video, generation, minimax, prompt-optimizer, img2vid, image-to-video

    Use cases:
    - Produce social media clips
    - Generate cinematic sequences
    - Visualize storyboards
    - Create promotional videos
    - Animate still graphics
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(default="", description="The prompt describing the video")
    duration: HailuoDuration = Field(
        default=HailuoDuration.SIX_SECONDS,
        description="The duration of the video in seconds. 10 seconds videos are not supported for 1080p resolution.",
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    resolution: MiniMaxHailuoResolution = Field(
        default=MiniMaxHailuoResolution.RES_768P,
        description="The resolution of the generated video",
    )
    end_image: ImageRef | None = Field(
        default=None, description="Optional image to use as the last frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments: dict[str, Any] = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "duration": self.duration.value,
            "prompt_optimizer": self.prompt_optimizer,
            "resolution": self.resolution.value,
        }

        if self.end_image:
            end_image_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_image_base64}"

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-02/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class LTXVideo(FALNode):
    """
    Generate videos from images using LTX Video. Best results with 768x512 images and detailed, chronological descriptions of actions and scenes.
    video, generation, chronological, scenes, actions, img2vid, image-to-video

    Use cases:
    - Create scene-based animations
    - Generate sequential video content
    - Produce narrative videos
    - Create storyboard animations
    - Generate action sequences
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="The image to transform into a video (768x512 recommended)",
    )
    prompt: str = Field(
        default="",
        description="A detailed description of the desired video motion and style",
    )
    negative_prompt: str = Field(
        default="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly",
        description="What to avoid in the generated video",
    )
    num_inference_steps: int = Field(
        default=30,
        description="Number of inference steps (higher = better quality but slower)",
    )
    guidance_scale: float = Field(
        default=3.0,
        description="How closely to follow the prompt (higher = more faithful)",
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same video every time"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class PixVerse(FALNode):
    """
    Generate dynamic videos from images with PixVerse v4.5. Create high-quality motion
    with detailed prompt control and advanced diffusion parameters.
    video, generation, pixverse, motion, diffusion, img2vid, image-to-video

    Use cases:
    - Animate illustrations and photos
    - Produce engaging social media clips
    - Generate short cinematic shots
    - Create motion for product showcases
    - Experiment with creative video effects
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    negative_prompt: str = Field(
        default="low quality, worst quality, distorted, blurred",
        description="What to avoid in the generated video",
    )
    num_inference_steps: int = Field(
        default=50,
        description="Number of inference steps (higher = better quality but slower)",
    )
    guidance_scale: float = Field(
        default=7.5,
        description="How closely to follow the prompt (higher = more faithful)",
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same video every time"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
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
        return ["image", "prompt", "num_inference_steps"]


class PixverseV56ImageToVideo(FALNode):
    """Generate high-quality videos from images with Pixverse v5.6.
    video, generation, pixverse, v5.6, image-to-video, img2vid

    Use cases:
    - Animate photos into professional video clips
    - Create dynamic product showcase videos
    - Generate stylized video content from artwork
    - Produce high-resolution social media animations
    - Transform static images with various visual styles
    """

    prompt: str = Field(
        default="", description="Text prompt describing the desired video motion"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    resolution: PixverseV56Resolution = Field(
        default=PixverseV56Resolution.RES_720P,
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
        image_base64 = await context.image_to_base64(self.image)

        arguments: dict[str, Any] = {
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
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
            application="fal-ai/pixverse/v5.6/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "resolution"]


class StableVideo(FALNode):
    """
    Generate short video clips from your images using Stable Video Diffusion v1.1. Features high-quality motion synthesis with configurable parameters.
    video, generation, diffusion, motion, synthesis, img2vid, image-to-video

    Use cases:
    - Create stable video animations
    - Generate motion content
    - Transform images into videos
    - Produce smooth transitions
    - Create visual effects
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    motion_bucket_id: int = Field(
        default=127, description="Controls motion intensity (higher = more motion)"
    )
    cond_aug: float = Field(
        default=0.02,
        description="Amount of noise added to conditioning (higher = more motion)",
    )
    fps: int = Field(default=25, description="Frames per second of the output video")
    seed: int = Field(
        default=-1, description="The same seed will output the same video every time"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "motion_bucket_id": self.motion_bucket_id,
            "cond_aug": self.cond_aug,
            "fps": self.fps,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "motion_bucket_id", "fps"]


class FastSVD(FALNode):
    """
    Generate short video clips from your images using SVD v1.1 at Lightning Speed. Features high-quality motion synthesis with configurable parameters for rapid video generation.
    video, generation, fast, motion, synthesis, img2vid, image-to-video

    Use cases:
    - Create quick video animations
    - Generate rapid motion content
    - Produce fast video transitions
    - Create instant visual effects
    - Generate quick previews
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    motion_bucket_id: int = Field(
        default=127, description="Controls motion intensity (higher = more motion)"
    )
    cond_aug: float = Field(
        default=0.02,
        description="Amount of noise added to conditioning (higher = more motion)",
    )
    steps: int = Field(
        default=4,
        description="Number of inference steps (higher = better quality but slower)",
    )
    fps: int = Field(
        default=10,
        description="Frames per second of the output video (total length is 25 frames)",
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same video every time"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "motion_bucket_id": self.motion_bucket_id,
            "cond_aug": self.cond_aug,
            "steps": self.steps,
            "fps": self.fps,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-svd-lcm",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "motion_bucket_id", "fps"]


class AMTInterpolation(FALNode):
    """
    Interpolate between image frames to create smooth video transitions. Supports configurable FPS and recursive interpolation passes for higher quality results.
    video, interpolation, transitions, frames, smoothing, img2vid, image-to-video

    Use cases:
    - Create smooth frame transitions
    - Generate fluid animations
    - Enhance video frame rates
    - Produce slow-motion effects
    - Create seamless video blends
    """

    frames: list[ImageRef] = Field(
        default=[ImageRef(), ImageRef()],
        description="List of frames to interpolate between (minimum 2 frames required)",
    )
    output_fps: int = Field(default=24, description="Output frames per second")
    recursive_interpolation_passes: int = Field(
        default=4,
        description="Number of recursive interpolation passes (higher = smoother)",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        frames_base64 = []
        for frame in self.frames:
            frame_base64 = await context.image_to_base64(frame)
            frames_base64.append({"url": f"data:image/png;base64,{frame_base64}"})

        arguments = {
            "frames": frames_base64,
            "output_fps": self.output_fps,
            "recursive_interpolation_passes": self.recursive_interpolation_passes,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/amt-interpolation/frame-interpolation",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["frames", "output_fps"]


class FaceModelResolution(Enum):
    RESOLUTION_256 = "256"
    RESOLUTION_512 = "512"


class PreprocessType(Enum):
    CROP = "crop"
    EXTCROP = "extcrop"
    RESIZE = "resize"
    FULL = "full"
    EXTFULL = "extfull"


class SadTalker(FALNode):
    """
    Generate talking face animations from a single image and audio file. Features configurable face model resolution and expression controls.
    video, animation, face, talking, expression, img2vid, image-to-video, audio-to-video, wav2vid

    Use cases:
    - Create talking head videos
    - Generate lip-sync animations
    - Produce character animations
    - Create video presentations
    - Generate facial expressions
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The source image to animate"
    )
    audio: str = Field(
        default="", description="URL of the audio file to drive the animation"
    )

    face_model_resolution: FaceModelResolution = Field(
        default=FaceModelResolution.RESOLUTION_256,
        description="Resolution of the face model",
    )
    expression_scale: float = Field(
        default=1.0, description="Scale of the expression (1.0 = normal)"
    )
    still_mode: bool = Field(
        default=False, description="Reduce head motion (works with preprocess 'full')"
    )
    preprocess: PreprocessType = Field(
        default=PreprocessType.CROP, description="Type of image preprocessing to apply"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "source_image_url": f"data:image/png;base64,{image_base64}",
            "driven_audio_url": self.audio,
            "face_model_resolution": self.face_model_resolution,
            "expression_scale": self.expression_scale,
            "still_mode": self.still_mode,
            "preprocess": self.preprocess,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/sadtalker",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "audio", "face_model_resolution"]


class MuseTalk(FALNode):
    """
    Real-time high quality audio-driven lip-syncing model. Animate a face video with custom audio for natural-looking speech animation.
    video, lip-sync, animation, speech, real-time, wav2vid, audio-to-video

    Use cases:
    - Create lip-synced videos
    - Generate speech animations
    - Produce dubbed content
    - Create animated presentations
    - Generate voice-over videos
    """

    video: VideoRef = Field(
        default=VideoRef(), description="URL of the source video to animate"
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to drive the lip sync"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_bytes = await context.asset_to_bytes(self.video)
        audio_bytes = await context.asset_to_bytes(self.audio)
        video_url = await client.upload(video_bytes, "video/mp4")
        audio_url = await client.upload(audio_bytes, "audio/mp3")

        arguments = {"source_video_url": video_url, "audio_url": audio_url}

        res = await self.submit_request(
            context=context,
            application="fal-ai/musetalk",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "audio"]


class Veo2(FALNode):
    """
    Generate videos from text prompts using Veo 2. Creates short clips with
    optional control over duration and aspect ratio.
    video, text-to-video, generation, prompt, veo2

    Use cases:
    - Produce cinematic video clips from descriptions
    - Generate marketing or social media footage
    - Create animated scenes from storyboards
    - Experiment with visual concepts rapidly
    """

    prompt: str = Field(default="", description="The prompt to generate a video from")
    duration: VideoDuration = Field(
        default=VideoDuration.FOUR_SECONDS,
        description="The duration of the generated video in seconds",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same video every time"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
            "aspect_ratio": self.aspect_ratio.value,
        }

        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo2/api",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "duration", "aspect_ratio"]


class Veo2ImageToVideo(FALNode):
    """
    Animate a single image into a Veo 2 video clip. Provides control over
    duration and aspect ratio while following an optional prompt.
    video, image-to-video, veo2, animation

    Use cases:
    - Bring still artwork to life
    - Create dynamic social media posts
    - Generate quick product showcase videos
    - Produce animated storyboards
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="Optional description of the desired motion"
    )
    duration: VideoDuration = Field(
        default=VideoDuration.FOUR_SECONDS,
        description="The duration of the generated video in seconds",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated video",
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same video every time"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "duration": self.duration,
            "aspect_ratio": self.aspect_ratio.value,
        }

        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo2/image-to-video/api",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class WanFlf2v(FALNode):
    """
    Generate short video clips from a single image using the WAN FLF2V model. This model converts a still image into an animated clip guided by a text prompt.
    video, generation, animation, image-to-video, wan

    Use cases:
    - Animate still images into short clips
    - Create dynamic content from artwork
    - Produce promotional video snippets
    - Generate visual effects for social posts
    - Explore creative motion ideas
    """

    image: ImageRef = Field(
        default=ImageRef(),
        description="The source image for video generation",
    )
    prompt: str = Field(
        default="",
        description="Description of the desired motion and style",
    )
    num_frames: int = Field(
        default=16,
        ge=1,
        description="Number of frames to generate",
    )
    seed: int = Field(
        default=-1,
        description="The same seed will output the same video every time",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "num_frames": self.num_frames,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-flf2v",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "num_frames"]


class LtxVideoSize(Enum):
    AUTO = "auto"
    SQUARE_HD = "square_hd"
    SQUARE = "square"
    PORTRAIT_4_3 = "portrait_4_3"
    PORTRAIT_16_9 = "portrait_16_9"
    LANDSCAPE_4_3 = "landscape_4_3"
    LANDSCAPE_16_9 = "landscape_16_9"


class LtxAcceleration(Enum):
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"


class LtxCameraLora(Enum):
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"


class LtxVideoOutputType(Enum):
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF = "GIF (.gif)"


class LtxVideoQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class LtxVideoWriteMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"


class LTX219BImageToVideo(FALNode):
    """
    Generate video with audio from images using LTX-2 19B model. A state-of-the-art video generation model with camera motion control and multi-scale generation.
    video, generation, ltx, ltx-2, image-to-video, motion-control, camera, audio

    Use cases:
    - Generate high-quality videos from images
    - Create videos with synchronized audio
    - Control camera movements with LoRA
    - Produce professional video content
    - Animate static images with fluid motion
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to generate the video from"
    )
    prompt: str = Field(
        default="",
        description="The prompt describing the desired video motion and style",
    )
    num_frames: int = Field(
        default=121, ge=1, description="Number of frames to generate"
    )
    video_size: LtxVideoSize = Field(
        default=LtxVideoSize.AUTO, description="Size of the generated video"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video"
    )
    use_multiscale: bool = Field(
        default=True, description="Use multi-scale generation for better coherence"
    )
    fps: float = Field(default=25, description="Frames per second")
    guidance_scale: float = Field(
        default=3, description="Guidance scale for generation"
    )
    num_inference_steps: int = Field(
        default=40, ge=1, description="Number of inference steps"
    )
    acceleration: LtxAcceleration = Field(
        default=LtxAcceleration.REGULAR, description="Acceleration level"
    )
    camera_lora: LtxCameraLora = Field(
        default=LtxCameraLora.NONE, description="Camera movement LoRA"
    )
    camera_lora_scale: float = Field(
        default=1, ge=0, le=1, description="Camera LoRA scale"
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur",
        description="Negative prompt to avoid",
    )
    seed: int = Field(default=-1, description="Random seed for reproducibility")
    enable_prompt_expansion: bool = Field(
        default=False, description="Enable prompt expansion"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker"
    )
    video_output_type: LtxVideoOutputType = Field(
        default=LtxVideoOutputType.X264_MP4, description="Output video format"
    )
    video_quality: LtxVideoQuality = Field(
        default=LtxVideoQuality.HIGH, description="Video quality"
    )
    video_write_mode: LtxVideoWriteMode = Field(
        default=LtxVideoWriteMode.BALANCED, description="Video write mode"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
            "num_frames": self.num_frames,
            "video_size": self.video_size.value,
            "generate_audio": self.generate_audio,
            "use_multiscale": self.use_multiscale,
            "fps": self.fps,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "acceleration": self.acceleration.value,
            "camera_lora": self.camera_lora.value,
            "camera_lora_scale": self.camera_lora_scale,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "enable_safety_checker": self.enable_safety_checker,
            "video_output_type": self.video_output_type.value,
            "video_quality": self.video_quality.value,
            "video_write_mode": self.video_write_mode.value,
        }

        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "generate_audio", "video_size"]


class LumaRay2ImageToVideo(FALNode):
    """
    Luma Ray 2 Image-to-Video generates high-quality videos from images with improved motion.
    video, generation, luma, ray2, image-to-video, img2vid

    Use cases:
    - Create cinematic video from images
    - Generate smooth motion animations
    - Produce high-quality video content
    - Transform photos into videos
    - Create professional video clips
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
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
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
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
            application="fal-ai/luma-dream-machine/ray-2/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "aspect_ratio", "duration"]


class LumaRay2FlashImageToVideo(FALNode):
    """
    Luma Ray 2 Flash Image-to-Video is a fast version for quick video generation.
    video, generation, luma, ray2, flash, image-to-video, fast

    Use cases:
    - Quick video prototyping
    - Rapid content creation
    - Fast video iterations
    - Real-time video generation
    - Quick motion tests
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
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
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2-flash/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "aspect_ratio", "duration"]


class KlingVideoV21Pro(FALNode):
    """
    Kling Video V2.1 Pro Image-to-Video with enhanced quality and motion.
    video, generation, kling, v2.1, pro, image-to-video

    Use cases:
    - Create professional video content
    - Generate high-quality animations
    - Produce cinematic video clips
    - Transform images with smooth motion
    - Create promotional videos
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
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
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2.1/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class HunyuanVideoImageToVideo(FALNode):
    """
    Hunyuan Video Image-to-Video generates videos from images with Tencent's model.
    video, generation, hunyuan, tencent, image-to-video

    Use cases:
    - Create videos from still images
    - Generate motion for photos
    - Produce animated content
    - Transform artwork into video
    - Create video transitions
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
    )
    num_inference_steps: int = Field(
        default=30, ge=1, description="Number of inference steps"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video-image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class HunyuanVideoV15ImageToVideo(FALNode):
    """
    Hunyuan Video V1.5 Image-to-Video with improved quality and motion.
    video, generation, hunyuan, v1.5, image-to-video

    Use cases:
    - Create high-quality video from images
    - Generate smooth animations
    - Produce professional video content
    - Transform photos with motion
    - Create video effects
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
    )
    num_inference_steps: int = Field(
        default=30, ge=1, description="Number of inference steps"
    )
    guidance_scale: float = Field(
        default=7.0, description="How closely to follow the prompt"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video-v1.5/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class PikaV22ImageToVideo(FALNode):
    """
    Pika V2.2 Image-to-Video generates creative videos from images.
    video, generation, pika, v2.2, image-to-video, creative

    Use cases:
    - Create creative video content
    - Generate artistic animations
    - Produce stylized videos
    - Transform images with effects
    - Create unique video clips
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated video"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
        }
        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/pika/v2.2/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class PikaV21ImageToVideo(FALNode):
    """
    Pika V2.1 Image-to-Video generates videos from images with the Pika model.
    video, generation, pika, v2.1, image-to-video

    Use cases:
    - Create video content from images
    - Generate animated clips
    - Produce motion graphics
    - Transform still photos
    - Create video effects
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

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
            application="fal-ai/pika/v2.1/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class ViduQ2ImageToVideo(FALNode):
    """
    Vidu Q2 Image-to-Video Turbo generates fast videos from images.
    video, generation, vidu, q2, turbo, image-to-video, fast

    Use cases:
    - Quick video generation
    - Rapid prototyping
    - Fast content creation
    - Quick motion tests
    - Real-time video production
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

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
            application="fal-ai/vidu/q2/image-to-video/turbo",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class Sora2Duration(int, Enum):
    _4s = 4
    _8s = 8
    _12s = 12

class Sora2ImageToVideo(FALNode):
    """
    OpenAI Sora 2 Image-to-Video generates high-quality videos from images.
    video, generation, openai, sora, sora2, image-to-video

    Use cases:
    - Create cinematic videos from images
    - Generate realistic motion
    - Produce professional video content
    - Transform photos into movies
    - Create video narratives
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
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
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/sora-2/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class SeedanceV15ProImageToVideo(FALNode):
    """
    ByteDance Seedance V1.5 Pro Image-to-Video with high-quality motion.
    video, generation, bytedance, seedance, pro, image-to-video

    Use cases:
    - Create professional video content
    - Generate high-quality animations
    - Produce cinematic clips
    - Transform images with motion
    - Create promotional videos
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

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
            application="fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]


class MiniMaxHailuo23ImageToVideo(FALNode):
    """
    MiniMax Hailuo 2.3 Standard Image-to-Video with improved quality.
    video, generation, minimax, hailuo, 2.3, image-to-video

    Use cases:
    - Create video from images
    - Generate smooth animations
    - Produce video content
    - Transform photos into clips
    - Create motion graphics
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion"
    )
    duration: HailuoDuration = Field(
        default=HailuoDuration.SIX_SECONDS,
        description="The duration of the video in seconds",
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the prompt optimizer"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "duration": self.duration.value,
            "prompt_optimizer": self.prompt_optimizer,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-2.3/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class LTXVideoSize(Enum):
    AUTO = "auto"
    SQUARE_HD = "square_hd"
    SQUARE = "square"
    PORTRAIT_4_3 = "portrait_4_3"
    PORTRAIT_16_9 = "portrait_16_9"
    LANDSCAPE_4_3 = "landscape_4_3"
    LANDSCAPE_16_9 = "landscape_16_9"


class LTXAcceleration(Enum):
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"


class LTXCameraLoRA(Enum):
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"


class LTXVideoOutputType(Enum):
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF = "GIF (.gif)"


class LTXVideoQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class LTXVideoWriteMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"


class LTX219BAudioToVideo(FALNode):
    """
    Generate videos from audio with optional text or image prompts using the LTX-2 19B model. Supports advanced camera controls and high-quality video generation.
    video, audio-to-video, generation, ltx, camera-control, audio-driven

    Use cases:
    - Generate talking head videos from audio
    - Create music visualizations from audio tracks
    - Produce audio-driven animations
    - Generate synchronized video content from podcasts
    - Create video content from voice recordings
    """

    prompt: str = Field(default="", description="The prompt to generate the video from")
    audio: AudioRef = Field(
        default=AudioRef(), description="The audio to generate the video from"
    )
    image: ImageRef | None = Field(
        default=None, description="Optional image to use as the first frame"
    )
    match_audio_length: bool = Field(
        default=True,
        description="Calculate frames based on audio duration and FPS",
    )
    num_frames: int = Field(default=121, description="The number of frames to generate")
    video_size: LTXVideoSize = Field(
        default=LTXVideoSize.LANDSCAPE_4_3,
        description="The size of the generated video",
    )
    use_multiscale: bool = Field(
        default=True,
        description="Use multi-scale generation for better coherence",
    )
    fps: float = Field(
        default=25.0, description="The frames per second of the generated video"
    )
    guidance_scale: float = Field(default=3.0, description="The guidance scale to use")
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps"
    )
    acceleration: LTXAcceleration = Field(
        default=LTXAcceleration.REGULAR, description="The acceleration level to use"
    )
    camera_lora: LTXCameraLoRA = Field(
        default=LTXCameraLoRA.NONE, description="The camera LoRA for movement control"
    )
    camera_lora_scale: float = Field(
        default=1.0, description="The scale of the camera LoRA"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt for video generation"
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator"
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker"
    )
    video_output_type: LTXVideoOutputType = Field(
        default=LTXVideoOutputType.X264_MP4,
        description="The output type of the generated video",
    )
    video_quality: LTXVideoQuality = Field(
        default=LTXVideoQuality.HIGH, description="The quality of the generated video"
    )
    video_write_mode: LTXVideoWriteMode = Field(
        default=LTXVideoWriteMode.BALANCED,
        description="The write mode of the generated video",
    )
    image_strength: float = Field(
        default=1.0, description="The strength of the image for video generation"
    )
    audio_strength: float = Field(
        default=1.0, description="Audio conditioning strength"
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio"
    )

    @classmethod
    def get_title(cls):
        return "LTX-2 19B Audio to Video"

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        audio_bytes = await context.asset_to_bytes(self.audio)
        audio_url = await client.upload(audio_bytes, "audio/mp3")

        arguments = {
            "prompt": self.prompt,
            "audio_url": audio_url,
            "match_audio_length": self.match_audio_length,
            "num_frames": self.num_frames,
            "video_size": self.video_size.value,
            "use_multiscale": self.use_multiscale,
            "fps": self.fps,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "acceleration": self.acceleration.value,
            "camera_lora": self.camera_lora.value,
            "camera_lora_scale": self.camera_lora_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "video_output_type": self.video_output_type.value,
            "video_quality": self.video_quality.value,
            "video_write_mode": self.video_write_mode.value,
            "image_strength": self.image_strength,
            "audio_strength": self.audio_strength,
            "preprocess_audio": self.preprocess_audio,
        }

        if self.image and self.image.uri:
            image_base64 = await context.image_to_base64(self.image)
            arguments["image_url"] = f"data:image/png;base64,{image_base64}"

        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed
        if self.enable_prompt_expansion:
            arguments["enable_prompt_expansion"] = self.enable_prompt_expansion

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "audio", "video_size"]


class LTX219BDistilledAudioToVideo(FALNode):
    """
    Faster audio-to-video generation using the distilled LTX-2 19B model. Provides quicker video generation from audio with optional prompts.
    video, audio-to-video, generation, ltx, distilled, fast

    Use cases:
    - Quick audio-driven video generation
    - Fast talking head video creation
    - Rapid music visualization
    - Time-efficient audio-to-video conversion
    - Fast prototype video generation from audio
    """

    prompt: str = Field(default="", description="The prompt to generate the video from")
    audio: AudioRef = Field(
        default=AudioRef(), description="The audio to generate the video from"
    )
    image: ImageRef | None = Field(
        default=None, description="Optional image to use as the first frame"
    )
    match_audio_length: bool = Field(
        default=True,
        description="Calculate frames based on audio duration and FPS",
    )
    num_frames: int = Field(default=121, description="The number of frames to generate")
    video_size: LTXVideoSize = Field(
        default=LTXVideoSize.LANDSCAPE_4_3,
        description="The size of the generated video",
    )
    use_multiscale: bool = Field(
        default=True,
        description="Use multi-scale generation for better coherence",
    )
    fps: float = Field(
        default=25.0, description="The frames per second of the generated video"
    )
    guidance_scale: float = Field(default=3.0, description="The guidance scale to use")
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps"
    )
    acceleration: LTXAcceleration = Field(
        default=LTXAcceleration.REGULAR, description="The acceleration level to use"
    )
    camera_lora: LTXCameraLoRA = Field(
        default=LTXCameraLoRA.NONE, description="The camera LoRA for movement control"
    )
    camera_lora_scale: float = Field(
        default=1.0, description="The scale of the camera LoRA"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt for video generation"
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator"
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker"
    )
    video_output_type: LTXVideoOutputType = Field(
        default=LTXVideoOutputType.X264_MP4,
        description="The output type of the generated video",
    )
    video_quality: LTXVideoQuality = Field(
        default=LTXVideoQuality.HIGH, description="The quality of the generated video"
    )
    video_write_mode: LTXVideoWriteMode = Field(
        default=LTXVideoWriteMode.BALANCED,
        description="The write mode of the generated video",
    )
    image_strength: float = Field(
        default=1.0, description="The strength of the image for video generation"
    )
    audio_strength: float = Field(
        default=1.0, description="Audio conditioning strength"
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio"
    )

    @classmethod
    def get_title(cls):
        return "LTX-2 19B Distilled Audio to Video"

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        audio_bytes = await context.asset_to_bytes(self.audio)
        audio_url = await client.upload(audio_bytes, "audio/mp3")

        arguments = {
            "prompt": self.prompt,
            "audio_url": audio_url,
            "match_audio_length": self.match_audio_length,
            "num_frames": self.num_frames,
            "video_size": self.video_size.value,
            "use_multiscale": self.use_multiscale,
            "fps": self.fps,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "acceleration": self.acceleration.value,
            "camera_lora": self.camera_lora.value,
            "camera_lora_scale": self.camera_lora_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "video_output_type": self.video_output_type.value,
            "video_quality": self.video_quality.value,
            "video_write_mode": self.video_write_mode.value,
            "image_strength": self.image_strength,
            "audio_strength": self.audio_strength,
            "preprocess_audio": self.preprocess_audio,
        }

        if self.image and self.image.uri:
            image_base64 = await context.image_to_base64(self.image)
            arguments["image_url"] = f"data:image/png;base64,{image_base64}"

        if self.negative_prompt:
            arguments["negative_prompt"] = self.negative_prompt
        if self.seed != -1:
            arguments["seed"] = self.seed
        if self.enable_prompt_expansion:
            arguments["enable_prompt_expansion"] = self.enable_prompt_expansion

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "audio", "video_size"]


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


class KlingV3ImageToVideo(FALNode):
    """
    Transform images into high-quality videos using Kling Video 3.0 Standard with improved motion and realistic acting.
    video, generation, kling, v3, image-to-video, animation, img2vid

    Use cases:
    - Animate still images into cinematic clips
    - Create dynamic product showcase videos
    - Generate motion graphics from static designs
    - Transform artwork into video content
    - Create engaging social media animations
    """

    start_image: ImageRef = Field(
        default=ImageRef(), description="The starting image for the video"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="Optional ending image for the video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
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
    reference_images: list[ImageRef] = Field(
        default=[],
        description="Reference images for character/element consistency. Reference as @Element1, @Element2 in prompt",
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
        start_image_base64 = await context.image_to_base64(self.start_image)

        arguments = {
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        if self.end_image and self.end_image.uri:
            end_image_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_image_base64}"

        if self.voice_ids:
            arguments["voice_ids"] = self.voice_ids

        # Build elements from reference images
        if self.reference_images:
            elements = []
            for ref_image in self.reference_images:
                if ref_image.uri:
                    ref_base64 = await context.image_to_base64(ref_image)
                    ref_data_url = f"data:image/png;base64,{ref_base64}"
                    elements.append(
                        {
                            "frontal_image_url": ref_data_url,
                            "reference_image_urls": [ref_data_url],
                        }
                    )
            if elements:
                arguments["elements"] = elements

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v3/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["start_image", "prompt", "duration"]


class KlingV3ProImageToVideo(FALNode):
    """
    Transform images into premium quality videos using Kling Video 3.0 Pro with enhanced quality and performance.
    video, generation, kling, v3, pro, image-to-video, premium, img2vid

    Use cases:
    - Create high-end video content from images
    - Generate professional product animations
    - Produce broadcast-quality video from stills
    - Create premium visual narratives
    - Generate detailed cinematic sequences
    """

    start_image: ImageRef = Field(
        default=ImageRef(), description="The starting image for the video"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="Optional ending image for the video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
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
    reference_images: list[ImageRef] = Field(
        default=[],
        description="Reference images for character/element consistency. Reference as @Element1, @Element2 in prompt",
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
        start_image_base64 = await context.image_to_base64(self.start_image)

        arguments = {
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        if self.end_image and self.end_image.uri:
            end_image_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_image_base64}"

        if self.voice_ids:
            arguments["voice_ids"] = self.voice_ids

        # Build elements from reference images
        if self.reference_images:
            elements = []
            for ref_image in self.reference_images:
                if ref_image.uri:
                    ref_base64 = await context.image_to_base64(ref_image)
                    ref_data_url = f"data:image/png;base64,{ref_base64}"
                    elements.append(
                        {
                            "frontal_image_url": ref_data_url,
                            "reference_image_urls": [ref_data_url],
                        }
                    )
            if elements:
                arguments["elements"] = elements

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v3/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["start_image", "prompt", "duration"]


class KlingO3ImageToVideo(FALNode):
    """
    Transform images into cinematic videos using Kling Video O3 Standard with storyboard-first creation and character consistency.
    video, generation, kling, o3, image-to-video, storyboard, img2vid

    Use cases:
    - Create story-driven video from images
    - Generate character-consistent animations
    - Produce multi-shot sequences from stills
    - Create structured narrative videos
    - Generate cinematic content with continuity
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The starting image for the video"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="Optional ending image for the video"
    )
    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    duration: Kling3Duration = Field(
        default=Kling3Duration.FIVE_SECONDS,
        description="The duration of the generated video in seconds (3-15)",
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate native audio for the video",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "duration": self.duration.value,
            "shot_type": "customize",
        }

        if self.generate_audio:
            arguments["generate_audio"] = self.generate_audio

        if self.end_image and self.end_image.uri:
            end_image_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_image_base64}"

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class KlingO3ProImageToVideo(FALNode):
    """
    Generate a video by taking a start frame and an end frame, animating the transition between them while following text-driven style and scene guidance (Kling O3 Pro).
    video, generation, kling, o3, pro, image-to-video, start-end-frame, img2vid

    Use cases:
    - Animate between start and end keyframes
    - Create guided transitions with text prompts
    - Generate videos with optional end-frame constraint
    - Multi-shot video with multi_prompt
    - Style and scene-driven motion
    """

    image: ImageRef = Field(
        default=ImageRef(), description="URL of the start frame image"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="Optional end frame image"
    )
    prompt: str = Field(
        default="",
        description="Text prompt for video generation. Either prompt or multi_prompt must be provided.",
    )
    duration: Kling3Duration = Field(
        default=Kling3Duration.FIVE_SECONDS,
        description="Video duration in seconds (3-15)",
    )
    generate_audio: bool = Field(
        default=False,
        description="Whether to generate native audio for the video",
    )
    multi_prompt: list[dict[str, str]] = Field(
        default=[],
        description="List of prompts for multi-shot video. Each item: {prompt, duration?}.",
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        if not (self.prompt or self.multi_prompt):
            raise ValueError(
                "Either prompt or multi_prompt must be provided. "
                "Provide a text prompt or at least one multi-shot prompt."
            )
        if self.prompt and self.multi_prompt:
            raise ValueError(
                "Provide either prompt or multi_prompt, not both."
            )
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "duration": self.duration.value,
        }
        if self.prompt:
            arguments["prompt"] = self.prompt
        if self.generate_audio:
            arguments["generate_audio"] = self.generate_audio
        if self.end_image and self.end_image.uri:
            end_image_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_image_base64}"
        if self.multi_prompt:
            arguments["multi_prompt"] = [
                {"prompt": str(d.get("prompt", "")), "duration": str(d.get("duration", "5"))}
                for d in self.multi_prompt
            ]
            arguments["shot_type"] = "customize"

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "duration"]


class KlingO3ReferenceToVideo(FALNode):
    """
    Generate videos with character consistency using Kling Video O3 reference-to-video with reusable character elements.
    video, generation, kling, o3, reference-to-video, character-consistency

    Use cases:
    - Create videos with consistent character appearances
    - Generate story sequences with recurring characters
    - Produce branded content with consistent subjects
    - Create serialized video content
    - Generate character-driven narratives
    """

    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    start_image: ImageRef = Field(
        default=ImageRef(),
        description="Optional starting image for the video",
    )
    end_image: ImageRef = Field(
        default=ImageRef(),
        description="Optional ending image for the video",
    )
    reference_images: list[ImageRef] = Field(
        default=[],
        description="Reference images for style/appearance (up to 4). Reference as @Image1, @Image2 in prompt",
    )
    element_images: list[ImageRef] = Field(
        default=[],
        description="Character/element images for consistency. Reference as @Element1, @Element2 in prompt",
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "shot_type": "customize",
        }

        if self.generate_audio:
            arguments["generate_audio"] = self.generate_audio

        if self.start_image and self.start_image.uri:
            start_base64 = await context.image_to_base64(self.start_image)
            arguments["start_image_url"] = f"data:image/png;base64,{start_base64}"

        if self.end_image and self.end_image.uri:
            end_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_base64}"

        # Build image_urls from reference images (up to 4)
        if self.reference_images:
            image_urls = []
            for ref_image in self.reference_images[:4]:
                if ref_image.uri:
                    ref_base64 = await context.image_to_base64(ref_image)
                    image_urls.append(f"data:image/png;base64,{ref_base64}")
            if image_urls:
                arguments["image_urls"] = image_urls

        # Build elements from element images
        if self.element_images:
            elements = []
            for elem_image in self.element_images:
                if elem_image.uri:
                    elem_base64 = await context.image_to_base64(elem_image)
                    elem_data_url = f"data:image/png;base64,{elem_base64}"
                    elements.append(
                        {
                            "frontal_image_url": elem_data_url,
                            "reference_image_urls": [elem_data_url],
                        }
                    )
            if elements:
                arguments["elements"] = elements

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/standard/reference-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "reference_images", "duration"]


class KlingO3ProReferenceToVideo(FALNode):
    """
    Generate premium videos with character consistency using Kling Video O3 Pro reference-to-video with enhanced quality.
    video, generation, kling, o3, pro, reference-to-video, character-consistency, premium

    Use cases:
    - Create high-quality videos with consistent characters
    - Generate professional story sequences
    - Produce premium branded content
    - Create broadcast-quality serialized content
    - Generate professional character-driven narratives
    """

    prompt: str = Field(
        default="", description="A description of the desired video motion and style"
    )
    start_image: ImageRef = Field(
        default=ImageRef(),
        description="Optional starting image for the video",
    )
    end_image: ImageRef = Field(
        default=ImageRef(),
        description="Optional ending image for the video",
    )
    reference_images: list[ImageRef] = Field(
        default=[],
        description="Reference images for style/appearance (up to 4). Reference as @Image1, @Image2 in prompt",
    )
    element_images: list[ImageRef] = Field(
        default=[],
        description="Character/element images for consistency. Reference as @Element1, @Element2 in prompt",
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "shot_type": "customize",
        }

        if self.generate_audio:
            arguments["generate_audio"] = self.generate_audio

        if self.start_image and self.start_image.uri:
            start_base64 = await context.image_to_base64(self.start_image)
            arguments["start_image_url"] = f"data:image/png;base64,{start_base64}"

        if self.end_image and self.end_image.uri:
            end_base64 = await context.image_to_base64(self.end_image)
            arguments["end_image_url"] = f"data:image/png;base64,{end_base64}"

        # Build image_urls from reference images (up to 4)
        if self.reference_images:
            image_urls = []
            for ref_image in self.reference_images[:4]:
                if ref_image.uri:
                    ref_base64 = await context.image_to_base64(ref_image)
                    image_urls.append(f"data:image/png;base64,{ref_base64}")
            if image_urls:
                arguments["image_urls"] = image_urls

        # Build elements from element images
        if self.element_images:
            elements = []
            for elem_image in self.element_images:
                if elem_image.uri:
                    elem_base64 = await context.image_to_base64(elem_image)
                    elem_data_url = f"data:image/png;base64,{elem_base64}"
                    elements.append(
                        {
                            "frontal_image_url": elem_data_url,
                            "reference_image_urls": [elem_data_url],
                        }
                    )
            if elements:
                arguments["elements"] = elements

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/pro/reference-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "reference_images", "duration"]
