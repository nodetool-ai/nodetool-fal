from pydantic import Field

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

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "prompt": self.prompt,
            "duration": self.duration.value,
            "prompt_optimizer": self.prompt_optimizer,
        }

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
