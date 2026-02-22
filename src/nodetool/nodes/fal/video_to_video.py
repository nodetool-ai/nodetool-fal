from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.nodes.fal.types import BoxPrompt, BoxPromptBase, Frame, ImageCondition, ImageConditioningInput, KlingV3ImageElementInput, LoRAInput, LoRAWeight, LoraWeight, OmniVideoElementInput, PointPrompt, PointPromptBase, Track, VideoCondition, VideoConditioningInput
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class AMTInterpolation(FALNode):
    """
    AMT (Any-to-Many Temporal) Interpolation creates smooth transitions between video frames.
    video, interpolation, frame-generation, amt, video-to-video

    Use cases:
    - Increase video frame rate smoothly
    - Create slow-motion effects
    - Smooth out choppy video
    - Generate intermediate frames
    - Enhance video playback quality
    """

    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to be processed"
    )
    low_latency_mode: bool = Field(
        default=False, description="Enables low latency mode"
    )
    recursive_interpolation_passes: int = Field(
        default=2, description="Number of recursive interpolation passes"
    )
    output_fps: int = Field(
        default=24, description="Output frames per second"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "low_latency_mode": self.low_latency_mode,
            "recursive_interpolation_passes": self.recursive_interpolation_passes,
            "output_fps": self.output_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/amt-interpolation",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class AIFaceSwapVideo(FALNode):
    """
    AI Face Swap replaces faces in videos with target faces while preserving expressions and movements.
    video, face-swap, deepfake, face-replacement, video-to-video

    Use cases:
    - Replace faces in video content
    - Create personalized video content
    - Swap actors in video scenes
    - Generate face replacement effects
    - Create video with different faces
    """

    enable_occlusion_prevention: bool = Field(
        default=False, description="Enable occlusion prevention for handling faces covered by hands/objects. Warning: Enabling this runs an occlusion-aware model which costs 2x more."
    )
    source_face: ImageRef = Field(
        default=ImageRef(), description="Source face image. Allowed items: bmp, jpeg, png, tiff, webp"
    )
    target_video: VideoRef = Field(
        default=VideoRef(), description="Target video URL (max 25 minutes, will be truncated if longer; FPS capped at 25). Allowed items: avi, m4v, mkv, mp4, mpeg, mov, mxf, webm, wmv"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        source_face_base64 = (
            await context.image_to_base64(self.source_face)
            if not self.source_face.is_empty()
            else None
        )
        target_video_url = (
            await self._upload_asset_to_fal(client, self.target_video, context)
            if not self.target_video.is_empty()
            else None
        )
        arguments = {
            "enable_occlusion_prevention": self.enable_occlusion_prevention,
            "source_face_url": f"data:image/png;base64,{source_face_base64}" if source_face_base64 else None,
            "target_video_url": target_video_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="half-moon-ai/ai-face-swap/faceswapvideo",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "target_face"]

class AnimateDiffVideoToVideo(FALNode):
    """
    AnimateDiff re-animates videos with new styles and effects using diffusion models.
    video, style-transfer, animatediff, re-animation, video-to-video

    Use cases:
    - Restyle existing videos
    - Apply artistic effects to videos
    - Transform video aesthetics
    - Create stylized video versions
    - Generate video variations
    """

    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    first_n_seconds: int = Field(
        default=3, description="The first N number of seconds of video to animate."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video."
    )
    fps: int = Field(
        default=8, description="Number of frames per second to extract from the video."
    )
    strength: float = Field(
        default=0.7, description="The strength of the input video in the final output."
    )
    guidance_scale: float = Field(
        default=7.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=25, description="The number of inference steps to perform."
    )
    negative_prompt: str = Field(
        default="(bad quality, worst quality:1.2), ugly faces, bad anime", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    motions: list[str] = Field(
        default=[], description="The motions to apply to the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "first_n_seconds": self.first_n_seconds,
            "video_url": video_url,
            "fps": self.fps,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "motions": self.motions,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-animatediff/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class AnimateDiffTurboVideoToVideo(FALNode):
    """
    AnimateDiff Turbo re-animates videos quickly with reduced generation time.
    video, style-transfer, animatediff, turbo, fast, video-to-video

    Use cases:
    - Quickly restyle videos
    - Rapid video transformations
    - Fast video effect application
    - Efficient video processing
    - Real-time video styling
    """

    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    first_n_seconds: int = Field(
        default=3, description="The first N number of seconds of video to animate."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video."
    )
    fps: int = Field(
        default=8, description="Number of frames per second to extract from the video."
    )
    strength: float = Field(
        default=0.7, description="The strength of the input video in the final output."
    )
    guidance_scale: float = Field(
        default=2, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=12, description="The number of inference steps to perform. 4-12 is recommended for turbo mode."
    )
    negative_prompt: str = Field(
        default="(bad quality, worst quality:1.2), ugly faces, bad anime", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "first_n_seconds": self.first_n_seconds,
            "video_url": video_url,
            "fps": self.fps,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-animatediff/turbo/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class AutoCaption(FALNode):
    """
    Auto Caption automatically generates and adds captions to videos with speech recognition.
    video, captions, subtitles, speech-to-text, video-to-video

    Use cases:
    - Add subtitles to videos automatically
    - Generate captions for accessibility
    - Create multilingual subtitles
    - Transcribe video speech
    - Add text overlays to videos
    """

    txt_font: str = Field(
        default="Standard", description="Font for generated captions. Choose one in 'Arial','Standard','Garamond', 'Times New Roman','Georgia', or pass a url to a .ttf file"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the .mp4 video with audio. Only videos of size <100MB are allowed."
    )
    top_align: str = Field(
        default="center", description="Top-to-bottom alignment of the text. Can be a string ('top', 'center', 'bottom') or a float (0.0-1.0)"
    )
    txt_color: str = Field(
        default="white", description="Colour of the text. Can be a RGB tuple, a color name, or an hexadecimal notation."
    )
    stroke_width: int = Field(
        default=1, description="Width of the text strokes in pixels"
    )
    refresh_interval: float = Field(
        default=1.5, description="Number of seconds the captions should stay on screen. A higher number will also result in more text being displayed at once."
    )
    font_size: int = Field(
        default=24, description="Size of text in generated captions."
    )
    left_align: str = Field(
        default="center", description="Left-to-right alignment of the text. Can be a string ('left', 'center', 'right') or a float (0.0-1.0)"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "txt_font": self.txt_font,
            "video_url": video_url,
            "top_align": self.top_align,
            "txt_color": self.txt_color,
            "stroke_width": self.stroke_width,
            "refresh_interval": self.refresh_interval,
            "font_size": self.font_size,
            "left_align": self.left_align,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/auto-caption",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class BenV2Video(FALNode):
    """
    Ben v2 Video enhances and processes video content with advanced AI techniques.
    video, enhancement, processing, ben, video-to-video

    Use cases:
    - Enhance video quality
    - Process video content
    - Improve video clarity
    - Apply video enhancements
    - Optimize video output
    """

    video: VideoRef = Field(
        default=VideoRef(), description="URL of video to be used for background removal."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )
    background_color: list[str] = Field(
        default=[], description="Optional RGB values (0-255) for the background color. If not provided, the background will be transparent. For ex: [0, 0, 0]"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "seed": self.seed,
            "background_color": self.background_color,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ben/v2/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class BiRefNetV2VideoVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class BiRefNetV2VideoVideoOutputType(str, Enum):
    """
    The output type of the generated video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class BiRefNetV2VideoOperatingResolution(str, Enum):
    """
    The resolution to operate on. The higher the resolution, the more accurate the output will be for high res input images. The '2304x2304' option is only available for the 'General Use (Dynamic)' model.
    """
    VALUE_1024X1024 = "1024x1024"
    VALUE_2048X2048 = "2048x2048"
    VALUE_2304X2304 = "2304x2304"

class BiRefNetV2VideoVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class BiRefNetV2VideoModel(str, Enum):
    """
    Model to use for background removal.
    The 'General Use (Light)' model is the original model used in the BiRefNet repository.
    The 'General Use (Light 2K)' model is the original model used in the BiRefNet repository but trained with 2K images.
    The 'General Use (Heavy)' model is a slower but more accurate model.
    The 'Matting' model is a model trained specifically for matting images.
    The 'Portrait' model is a model trained specifically for portrait images.
    The 'General Use (Dynamic)' model supports dynamic resolutions from 256x256 to 2304x2304.
    The 'General Use (Light)' model is recommended for most use cases.
    The corresponding models are as follows:
    - 'General Use (Light)': BiRefNet
    - 'General Use (Light 2K)': BiRefNet_lite-2K
    - 'General Use (Heavy)': BiRefNet_lite
    - 'Matting': BiRefNet-matting
    - 'Portrait': BiRefNet-portrait
    - 'General Use (Dynamic)': BiRefNet_dynamic
    """
    GENERAL_USE_LIGHT = "General Use (Light)"
    GENERAL_USE_LIGHT_2K = "General Use (Light 2K)"
    GENERAL_USE_HEAVY = "General Use (Heavy)"
    MATTING = "Matting"
    PORTRAIT = "Portrait"
    GENERAL_USE_DYNAMIC = "General Use (Dynamic)"


class BiRefNetV2Video(FALNode):
    """
    BiRefNet v2 Video performs background removal from videos with high accuracy.
    video, background-removal, segmentation, birefnet, video-to-video

    Use cases:
    - Remove backgrounds from videos
    - Create transparent video backgrounds
    - Isolate video subjects
    - Generate video mattes
    - Prepare videos for compositing
    """

    video_write_mode: BiRefNetV2VideoVideoWriteMode = Field(
        default=BiRefNetV2VideoVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to remove background from"
    )
    video_output_type: BiRefNetV2VideoVideoOutputType = Field(
        default=BiRefNetV2VideoVideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    operating_resolution: BiRefNetV2VideoOperatingResolution = Field(
        default=BiRefNetV2VideoOperatingResolution.VALUE_1024X1024, description="The resolution to operate on. The higher the resolution, the more accurate the output will be for high res input images. The '2304x2304' option is only available for the 'General Use (Dynamic)' model."
    )
    video_quality: BiRefNetV2VideoVideoQuality = Field(
        default=BiRefNetV2VideoVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    model: BiRefNetV2VideoModel = Field(
        default=BiRefNetV2VideoModel.GENERAL_USE_LIGHT, description="Model to use for background removal. The 'General Use (Light)' model is the original model used in the BiRefNet repository. The 'General Use (Light 2K)' model is the original model used in the BiRefNet repository but trained with 2K images. The 'General Use (Heavy)' model is a slower but more accurate model. The 'Matting' model is a model trained specifically for matting images. The 'Portrait' model is a model trained specifically for portrait images. The 'General Use (Dynamic)' model supports dynamic resolutions from 256x256 to 2304x2304. The 'General Use (Light)' model is recommended for most use cases. The corresponding models are as follows: - 'General Use (Light)': BiRefNet - 'General Use (Light 2K)': BiRefNet_lite-2K - 'General Use (Heavy)': BiRefNet_lite - 'Matting': BiRefNet-matting - 'Portrait': BiRefNet-portrait - 'General Use (Dynamic)': BiRefNet_dynamic"
    )
    output_mask: bool = Field(
        default=False, description="Whether to output the mask used to remove the background"
    )
    refine_foreground: bool = Field(
        default=True, description="Whether to refine the foreground using the estimated mask"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "video_url": video_url,
            "video_output_type": self.video_output_type.value,
            "operating_resolution": self.operating_resolution.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "model": self.model.value,
            "output_mask": self.output_mask,
            "refine_foreground": self.refine_foreground,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/birefnet/v2/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class BriaVideoEraserMaskOutputContainerAndCodec(str, Enum):
    """
    Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4.
    """
    MP4_H265 = "mp4_h265"
    MP4_H264 = "mp4_h264"
    WEBM_VP9 = "webm_vp9"
    GIF = "gif"
    MOV_H264 = "mov_h264"
    MOV_H265 = "mov_h265"
    MOV_PRORESKS = "mov_proresks"
    MKV_H264 = "mkv_h264"
    MKV_H265 = "mkv_h265"
    MKV_VP9 = "mkv_vp9"
    MKV_MPEG4 = "mkv_mpeg4"


class BriaVideoEraserMask(FALNode):
    """
    Bria Video Eraser removes objects from videos using mask-based selection.
    video, object-removal, eraser, inpainting, bria, video-to-video

    Use cases:
    - Remove unwanted objects from videos
    - Erase people or items from footage
    - Clean up video backgrounds
    - Remove watermarks from videos
    - Edit video content seamlessly
    """

    preserve_audio: bool = Field(
        default=True, description="If true, audio will be preserved in the output video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    output_container_and_codec: BriaVideoEraserMaskOutputContainerAndCodec = Field(
        default=BriaVideoEraserMaskOutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    mask_video: VideoRef = Field(
        default=VideoRef(), description="Input video to mask erase object from. duration must be less than 5s."
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        mask_video_url = (
            await self._upload_asset_to_fal(client, self.mask_video, context)
            if not self.mask_video.is_empty()
            else None
        )
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": video_url,
            "output_container_and_codec": self.output_container_and_codec.value,
            "mask_video_url": mask_video_url,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/bria_video_eraser/erase/mask",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "mask"]

class BriaVideoEraserKeypointsOutputContainerAndCodec(str, Enum):
    """
    Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4.
    """
    MP4_H265 = "mp4_h265"
    MP4_H264 = "mp4_h264"
    WEBM_VP9 = "webm_vp9"
    GIF = "gif"
    MOV_H264 = "mov_h264"
    MOV_H265 = "mov_h265"
    MOV_PRORESKS = "mov_proresks"
    MKV_H264 = "mkv_h264"
    MKV_H265 = "mkv_h265"
    MKV_VP9 = "mkv_vp9"
    MKV_MPEG4 = "mkv_mpeg4"


class BriaVideoEraserKeypoints(FALNode):
    """
    Bria Video Eraser removes objects from videos using keypoint-based selection.
    video, object-removal, eraser, keypoints, bria, video-to-video

    Use cases:
    - Remove objects using keypoint selection
    - Erase specific areas from videos
    - Targeted video content removal
    - Precision video editing
    - Remove elements with point markers
    """

    preserve_audio: bool = Field(
        default=True, description="If true, audio will be preserved in the output video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    output_container_and_codec: BriaVideoEraserKeypointsOutputContainerAndCodec = Field(
        default=BriaVideoEraserKeypointsOutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    keypoints: list[str] = Field(
        default=[], description="Input keypoints [x,y] to erase or keep from the video. Format like so: {'x':100, 'y':100, 'type':'positive/negative'}"
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": video_url,
            "output_container_and_codec": self.output_container_and_codec.value,
            "keypoints": self.keypoints,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/bria_video_eraser/erase/keypoints",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "keypoints"]

class BriaVideoEraserPromptOutputContainerAndCodec(str, Enum):
    """
    Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4.
    """
    MP4_H265 = "mp4_h265"
    MP4_H264 = "mp4_h264"
    WEBM_VP9 = "webm_vp9"
    GIF = "gif"
    MOV_H264 = "mov_h264"
    MOV_H265 = "mov_h265"
    MOV_PRORESKS = "mov_proresks"
    MKV_H264 = "mkv_h264"
    MKV_H265 = "mkv_h265"
    MKV_VP9 = "mkv_vp9"
    MKV_MPEG4 = "mkv_mpeg4"


class BriaVideoEraserPrompt(FALNode):
    """
    Bria Video Eraser removes objects from videos using text prompt descriptions.
    video, object-removal, eraser, prompt, bria, video-to-video

    Use cases:
    - Remove objects by describing them
    - Text-based video editing
    - Natural language video cleanup
    - Prompt-driven object removal
    - Semantic video editing
    """

    preserve_audio: bool = Field(
        default=True, description="If true, audio will be preserved in the output video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    prompt: str = Field(
        default="", description="Input prompt to detect object to erase"
    )
    output_container_and_codec: BriaVideoEraserPromptOutputContainerAndCodec = Field(
        default=BriaVideoEraserPromptOutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": video_url,
            "prompt": self.prompt,
            "output_container_and_codec": self.output_container_and_codec.value,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/bria_video_eraser/erase/prompt",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class CogVideoX5BVideoToVideo(FALNode):
    """
    CogVideoX-5B transforms existing videos with new styles and effects.
    video, transformation, cogvideo, style-transfer, video-to-video

    Use cases:
    - Transform video styles
    - Apply effects to existing videos
    - Restyle video content
    - Generate video variations
    - Create artistic video versions
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The video to generate the video from."
    )
    use_rife: bool = Field(
        default=True, description="Use RIFE for video interpolation"
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. We currently support one lora."
    )
    video_size: str = Field(
        default="", description="The size of the generated video."
    )
    strength: float = Field(
        default=0.8, description="The strength to use for Video to Video. 1.0 completely remakes the video while 0.0 preserves the original."
    )
    guidance_scale: float = Field(
        default=7, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related video to show you."
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to perform."
    )
    export_fps: int = Field(
        default=16, description="The target FPS of the video"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate video from"
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "use_rife": self.use_rife,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "video_size": self.video_size,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "export_fps": self.export_fps,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/cogvideox-5b/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class HunyuanVideoToVideoAspectRatio(str, Enum):
    """
    The aspect ratio of the video to generate.
    """
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"

class HunyuanVideoToVideoResolution(str, Enum):
    """
    The resolution of the video to generate.
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class HunyuanVideoToVideoNumFrames(str, Enum):
    """
    The number of frames to generate.
    """
    VALUE_129 = "129"
    VALUE_85 = "85"


class HunyuanVideoToVideo(FALNode):
    """
    Hunyuan Video transforms existing videos with advanced AI-powered effects.
    video, transformation, hunyuan, video-to-video

    Use cases:
    - Transform video content
    - Apply AI effects to videos
    - Restyle existing footage
    - Generate video variations
    - Create enhanced video versions
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    aspect_ratio: HunyuanVideoToVideoAspectRatio = Field(
        default=HunyuanVideoToVideoAspectRatio.RATIO_16_9, description="The aspect ratio of the video to generate."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video input."
    )
    resolution: HunyuanVideoToVideoResolution = Field(
        default=HunyuanVideoToVideoResolution.VALUE_720P, description="The resolution of the video to generate."
    )
    strength: float = Field(
        default=0.85, description="Strength for Video-to-Video"
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    num_frames: HunyuanVideoToVideoNumFrames = Field(
        default=129, description="The number of frames to generate."
    )
    seed: str = Field(
        default="", description="The seed to use for generating the video."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to run. Lower gets faster results, higher gets better results."
    )
    pro_mode: bool = Field(
        default=False, description="By default, generations are done with 35 steps. Pro mode does 55 steps which results in higher quality videos but will take more time and cost 2x more billing units."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "video_url": video_url,
            "resolution": self.resolution.value,
            "strength": self.strength,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames.value,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "pro_mode": self.pro_mode,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class VideoUpscaler(FALNode):
    """
    Video Upscaler enhances video resolution and quality using AI.
    video, upscaling, enhancement, resolution, video-to-video

    Use cases:
    - Upscale low resolution videos
    - Enhance video quality
    - Increase video resolution
    - Improve video clarity
    - Restore old video footage
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to upscale"
    )
    scale: float = Field(
        default=2, description="The scale factor"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "scale": self.scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/video-upscaler",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class CCSRColorFixType(str, Enum):
    """
    Type of color correction for samples.
    """
    NONE = "none"
    WAVELET = "wavelet"
    ADAIN = "adain"

class CCSRTileDiffusion(str, Enum):
    """
    If specified, a patch-based sampling strategy will be used for sampling.
    """
    NONE = "none"
    MIX = "mix"
    GAUSSIAN = "gaussian"


class CCSR(FALNode):
    """
    CCSR (Controllable Color Style Restoration) restores and enhances video colors.
    video, color-restoration, enhancement, ccsr, video-to-video

    Use cases:
    - Restore video colors
    - Enhance video color quality
    - Fix color issues in videos
    - Improve video color grading
    - Restore faded video footage
    """

    color_fix_type: CCSRColorFixType = Field(
        default=CCSRColorFixType.ADAIN, description="Type of color correction for samples."
    )
    tile_diffusion_size: int = Field(
        default=1024, description="Size of patch."
    )
    tile_vae_decoder_size: int = Field(
        default=226, description="Size of VAE patch."
    )
    tile_vae_encoder_size: int = Field(
        default=1024, description="Size of latent image"
    )
    t_min: float = Field(
        default=0.3333, description="The starting point of uniform sampling strategy."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL or data URI of the image to upscale."
    )
    tile_diffusion_stride: int = Field(
        default=512, description="Stride of sliding patch."
    )
    tile_vae: bool = Field(
        default=False, description="If specified, a patch-based sampling strategy will be used for VAE decoding."
    )
    scale: float = Field(
        default=2, description="The scale of the output image. The higher the scale, the bigger the output image will be."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducibility. Different seeds will make slightly different results."
    )
    t_max: float = Field(
        default=0.6667, description="The ending point of uniform sampling strategy."
    )
    steps: int = Field(
        default=50, description="The number of steps to run the model for. The higher the number the better the quality and longer it will take to generate."
    )
    tile_diffusion: CCSRTileDiffusion = Field(
        default=CCSRTileDiffusion.NONE, description="If specified, a patch-based sampling strategy will be used for sampling."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "color_fix_type": self.color_fix_type.value,
            "tile_diffusion_size": self.tile_diffusion_size,
            "tile_vae_decoder_size": self.tile_vae_decoder_size,
            "tile_vae_encoder_size": self.tile_vae_encoder_size,
            "t_min": self.t_min,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "tile_diffusion_stride": self.tile_diffusion_stride,
            "tile_vae": self.tile_vae,
            "scale": self.scale,
            "seed": self.seed,
            "t_max": self.t_max,
            "steps": self.steps,
            "tile_diffusion": self.tile_diffusion.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ccsr",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Ltx219BDistilledVideoToVideoLoraVideoOutputType(str, Enum):
    """
    The output type of the generated video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class Ltx219BDistilledVideoToVideoLoraVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Ltx219BDistilledVideoToVideoLoraAcceleration(str, Enum):
    """
    The acceleration level to use.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"

class Ltx219BDistilledVideoToVideoLoraCameraLora(str, Enum):
    """
    The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera.
    """
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"

class Ltx219BDistilledVideoToVideoLoraPreprocessor(str, Enum):
    """
    The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type.
    """
    DEPTH = "depth"
    CANNY = "canny"
    POSE = "pose"
    NONE = "none"

class Ltx219BDistilledVideoToVideoLoraVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Ltx219BDistilledVideoToVideoLoraIcLora(str, Enum):
    """
    The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)
    """
    MATCH_PREPROCESSOR = "match_preprocessor"
    CANNY = "canny"
    DEPTH = "depth"
    POSE = "pose"
    DETAILER = "detailer"
    NONE = "none"


class Ltx219BDistilledVideoToVideoLora(FALNode):
    """
    LTX-2 19B Distilled
    video, editing, video-to-video, vid2vid, lora

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the video from."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    ic_lora_scale: float = Field(
        default=1, description="The scale of the IC-LoRA to use. This allows you to control the strength of the IC-LoRA."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    loras: list[LoRAInput] = Field(
        default=[], description="The LoRAs to use for the generation."
    )
    video_size: str = Field(
        default="auto", description="The size of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: Ltx219BDistilledVideoToVideoLoraVideoOutputType = Field(
        default=Ltx219BDistilledVideoToVideoLoraVideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="An optional URL of an image to use as the first frame of the video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: Ltx219BDistilledVideoToVideoLoraVideoQuality = Field(
        default=Ltx219BDistilledVideoToVideoLoraVideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    match_video_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the video duration and FPS. When disabled, use the specified num_frames."
    )
    acceleration: Ltx219BDistilledVideoToVideoLoraAcceleration = Field(
        default=Ltx219BDistilledVideoToVideoLoraAcceleration.NONE, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: Ltx219BDistilledVideoToVideoLoraCameraLora = Field(
        default=Ltx219BDistilledVideoToVideoLoraCameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    preprocessor: Ltx219BDistilledVideoToVideoLoraPreprocessor = Field(
        default=Ltx219BDistilledVideoToVideoLoraPreprocessor.NONE, description="The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type."
    )
    video_write_mode: Ltx219BDistilledVideoToVideoLoraVideoWriteMode = Field(
        default=Ltx219BDistilledVideoToVideoLoraVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    ic_lora: Ltx219BDistilledVideoToVideoLoraIcLora = Field(
        default=Ltx219BDistilledVideoToVideoLoraIcLora.MATCH_PREPROCESSOR, description="The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)"
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="An optional URL of an audio to use as the audio for the video. If not provided, any audio present in the input video will be used."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        end_image_base64 = (
            await context.image_to_base64(self.end_image)
            if not self.end_image.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        audio_url = (
            await self._upload_asset_to_fal(client, self.audio, context)
            if not self.audio.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "use_multiscale": self.use_multiscale,
            "ic_lora_scale": self.ic_lora_scale,
            "generate_audio": self.generate_audio,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "video_size": self.video_size,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_base64}" if end_image_base64 else None,
            "camera_lora_scale": self.camera_lora_scale,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "match_video_length": self.match_video_length,
            "acceleration": self.acceleration.value,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "enable_safety_checker": self.enable_safety_checker,
            "image_strength": self.image_strength,
            "negative_prompt": self.negative_prompt,
            "preprocessor": self.preprocessor.value,
            "video_write_mode": self.video_write_mode.value,
            "ic_lora": self.ic_lora.value,
            "audio_url": audio_url,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/video-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Ltx219BDistilledVideoToVideoVideoOutputType(str, Enum):
    """
    The output type of the generated video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class Ltx219BDistilledVideoToVideoVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Ltx219BDistilledVideoToVideoAcceleration(str, Enum):
    """
    The acceleration level to use.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"

class Ltx219BDistilledVideoToVideoCameraLora(str, Enum):
    """
    The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera.
    """
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"

class Ltx219BDistilledVideoToVideoPreprocessor(str, Enum):
    """
    The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type.
    """
    DEPTH = "depth"
    CANNY = "canny"
    POSE = "pose"
    NONE = "none"

class Ltx219BDistilledVideoToVideoVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Ltx219BDistilledVideoToVideoIcLora(str, Enum):
    """
    The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)
    """
    MATCH_PREPROCESSOR = "match_preprocessor"
    CANNY = "canny"
    DEPTH = "depth"
    POSE = "pose"
    DETAILER = "detailer"
    NONE = "none"


class Ltx219BDistilledVideoToVideo(FALNode):
    """
    LTX-2 19B Distilled
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the video from."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    ic_lora_scale: float = Field(
        default=1, description="The scale of the IC-LoRA to use. This allows you to control the strength of the IC-LoRA."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    video_size: str = Field(
        default="auto", description="The size of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: Ltx219BDistilledVideoToVideoVideoOutputType = Field(
        default=Ltx219BDistilledVideoToVideoVideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="An optional URL of an image to use as the first frame of the video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: Ltx219BDistilledVideoToVideoVideoQuality = Field(
        default=Ltx219BDistilledVideoToVideoVideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    match_video_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the video duration and FPS. When disabled, use the specified num_frames."
    )
    acceleration: Ltx219BDistilledVideoToVideoAcceleration = Field(
        default=Ltx219BDistilledVideoToVideoAcceleration.NONE, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: Ltx219BDistilledVideoToVideoCameraLora = Field(
        default=Ltx219BDistilledVideoToVideoCameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    preprocessor: Ltx219BDistilledVideoToVideoPreprocessor = Field(
        default=Ltx219BDistilledVideoToVideoPreprocessor.NONE, description="The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type."
    )
    video_write_mode: Ltx219BDistilledVideoToVideoVideoWriteMode = Field(
        default=Ltx219BDistilledVideoToVideoVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    ic_lora: Ltx219BDistilledVideoToVideoIcLora = Field(
        default=Ltx219BDistilledVideoToVideoIcLora.MATCH_PREPROCESSOR, description="The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)"
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="An optional URL of an audio to use as the audio for the video. If not provided, any audio present in the input video will be used."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        end_image_base64 = (
            await context.image_to_base64(self.end_image)
            if not self.end_image.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        audio_url = (
            await self._upload_asset_to_fal(client, self.audio, context)
            if not self.audio.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "use_multiscale": self.use_multiscale,
            "ic_lora_scale": self.ic_lora_scale,
            "generate_audio": self.generate_audio,
            "video_size": self.video_size,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_base64}" if end_image_base64 else None,
            "camera_lora_scale": self.camera_lora_scale,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "match_video_length": self.match_video_length,
            "acceleration": self.acceleration.value,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "enable_safety_checker": self.enable_safety_checker,
            "image_strength": self.image_strength,
            "negative_prompt": self.negative_prompt,
            "preprocessor": self.preprocessor.value,
            "video_write_mode": self.video_write_mode.value,
            "ic_lora": self.ic_lora.value,
            "audio_url": audio_url,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Ltx219BVideoToVideoLoraVideoOutputType(str, Enum):
    """
    The output type of the generated video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class Ltx219BVideoToVideoLoraVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Ltx219BVideoToVideoLoraAcceleration(str, Enum):
    """
    The acceleration level to use.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"

class Ltx219BVideoToVideoLoraCameraLora(str, Enum):
    """
    The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera.
    """
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"

class Ltx219BVideoToVideoLoraPreprocessor(str, Enum):
    """
    The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type.
    """
    DEPTH = "depth"
    CANNY = "canny"
    POSE = "pose"
    NONE = "none"

class Ltx219BVideoToVideoLoraVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Ltx219BVideoToVideoLoraIcLora(str, Enum):
    """
    The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)
    """
    MATCH_PREPROCESSOR = "match_preprocessor"
    CANNY = "canny"
    DEPTH = "depth"
    POSE = "pose"
    DETAILER = "detailer"
    NONE = "none"


class Ltx219BVideoToVideoLora(FALNode):
    """
    LTX-2 19B
    video, editing, video-to-video, vid2vid, lora

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the video from."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    ic_lora_scale: float = Field(
        default=1, description="The scale of the IC-LoRA to use. This allows you to control the strength of the IC-LoRA."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    loras: list[LoRAInput] = Field(
        default=[], description="The LoRAs to use for the generation."
    )
    video_size: str = Field(
        default="auto", description="The size of the generated video."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: Ltx219BVideoToVideoLoraVideoOutputType = Field(
        default=Ltx219BVideoToVideoLoraVideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="An optional URL of an image to use as the first frame of the video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: Ltx219BVideoToVideoLoraVideoQuality = Field(
        default=Ltx219BVideoToVideoLoraVideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    match_video_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the video duration and FPS. When disabled, use the specified num_frames."
    )
    acceleration: Ltx219BVideoToVideoLoraAcceleration = Field(
        default=Ltx219BVideoToVideoLoraAcceleration.REGULAR, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: Ltx219BVideoToVideoLoraCameraLora = Field(
        default=Ltx219BVideoToVideoLoraCameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    preprocessor: Ltx219BVideoToVideoLoraPreprocessor = Field(
        default=Ltx219BVideoToVideoLoraPreprocessor.NONE, description="The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type."
    )
    video_write_mode: Ltx219BVideoToVideoLoraVideoWriteMode = Field(
        default=Ltx219BVideoToVideoLoraVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    ic_lora: Ltx219BVideoToVideoLoraIcLora = Field(
        default=Ltx219BVideoToVideoLoraIcLora.MATCH_PREPROCESSOR, description="The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)"
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="An optional URL of an audio to use as the audio for the video. If not provided, any audio present in the input video will be used."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        end_image_base64 = (
            await context.image_to_base64(self.end_image)
            if not self.end_image.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        audio_url = (
            await self._upload_asset_to_fal(client, self.audio, context)
            if not self.audio.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "use_multiscale": self.use_multiscale,
            "ic_lora_scale": self.ic_lora_scale,
            "generate_audio": self.generate_audio,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_base64}" if end_image_base64 else None,
            "camera_lora_scale": self.camera_lora_scale,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "match_video_length": self.match_video_length,
            "acceleration": self.acceleration.value,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "enable_safety_checker": self.enable_safety_checker,
            "image_strength": self.image_strength,
            "negative_prompt": self.negative_prompt,
            "preprocessor": self.preprocessor.value,
            "video_write_mode": self.video_write_mode.value,
            "ic_lora": self.ic_lora.value,
            "audio_url": audio_url,
            "num_inference_steps": self.num_inference_steps,
            "end_image_strength": self.end_image_strength,
            "audio_strength": self.audio_strength,
            "match_input_fps": self.match_input_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/video-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Ltx219BVideoToVideoVideoOutputType(str, Enum):
    """
    The output type of the generated video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class Ltx219BVideoToVideoVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Ltx219BVideoToVideoAcceleration(str, Enum):
    """
    The acceleration level to use.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"

class Ltx219BVideoToVideoCameraLora(str, Enum):
    """
    The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera.
    """
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"

class Ltx219BVideoToVideoPreprocessor(str, Enum):
    """
    The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type.
    """
    DEPTH = "depth"
    CANNY = "canny"
    POSE = "pose"
    NONE = "none"

class Ltx219BVideoToVideoVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Ltx219BVideoToVideoIcLora(str, Enum):
    """
    The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)
    """
    MATCH_PREPROCESSOR = "match_preprocessor"
    CANNY = "canny"
    DEPTH = "depth"
    POSE = "pose"
    DETAILER = "detailer"
    NONE = "none"


class Ltx219BVideoToVideo(FALNode):
    """
    LTX-2 19B
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the video from."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    ic_lora_scale: float = Field(
        default=1, description="The scale of the IC-LoRA to use. This allows you to control the strength of the IC-LoRA."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    video_size: str = Field(
        default="auto", description="The size of the generated video."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: Ltx219BVideoToVideoVideoOutputType = Field(
        default=Ltx219BVideoToVideoVideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="An optional URL of an image to use as the first frame of the video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: Ltx219BVideoToVideoVideoQuality = Field(
        default=Ltx219BVideoToVideoVideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    match_video_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the video duration and FPS. When disabled, use the specified num_frames."
    )
    acceleration: Ltx219BVideoToVideoAcceleration = Field(
        default=Ltx219BVideoToVideoAcceleration.REGULAR, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: Ltx219BVideoToVideoCameraLora = Field(
        default=Ltx219BVideoToVideoCameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    preprocessor: Ltx219BVideoToVideoPreprocessor = Field(
        default=Ltx219BVideoToVideoPreprocessor.NONE, description="The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type."
    )
    video_write_mode: Ltx219BVideoToVideoVideoWriteMode = Field(
        default=Ltx219BVideoToVideoVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    ic_lora: Ltx219BVideoToVideoIcLora = Field(
        default=Ltx219BVideoToVideoIcLora.MATCH_PREPROCESSOR, description="The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)"
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="An optional URL of an audio to use as the audio for the video. If not provided, any audio present in the input video will be used."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        end_image_base64 = (
            await context.image_to_base64(self.end_image)
            if not self.end_image.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        audio_url = (
            await self._upload_asset_to_fal(client, self.audio, context)
            if not self.audio.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "use_multiscale": self.use_multiscale,
            "ic_lora_scale": self.ic_lora_scale,
            "generate_audio": self.generate_audio,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_base64}" if end_image_base64 else None,
            "camera_lora_scale": self.camera_lora_scale,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "match_video_length": self.match_video_length,
            "acceleration": self.acceleration.value,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "enable_safety_checker": self.enable_safety_checker,
            "image_strength": self.image_strength,
            "negative_prompt": self.negative_prompt,
            "preprocessor": self.preprocessor.value,
            "video_write_mode": self.video_write_mode.value,
            "ic_lora": self.ic_lora.value,
            "audio_url": audio_url,
            "num_inference_steps": self.num_inference_steps,
            "end_image_strength": self.end_image_strength,
            "audio_strength": self.audio_strength,
            "match_input_fps": self.match_input_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Ltx219BDistilledExtendVideoLoraAcceleration(str, Enum):
    """
    The acceleration level to use.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"

class Ltx219BDistilledExtendVideoLoraCameraLora(str, Enum):
    """
    The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera.
    """
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"

class Ltx219BDistilledExtendVideoLoraExtendDirection(str, Enum):
    """
    Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning.
    """
    FORWARD = "forward"
    BACKWARD = "backward"

class Ltx219BDistilledExtendVideoLoraVideoOutputType(str, Enum):
    """
    The output type of the generated video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class Ltx219BDistilledExtendVideoLoraVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Ltx219BDistilledExtendVideoLoraVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class Ltx219BDistilledExtendVideoLora(FALNode):
    """
    LTX-2 19B Distilled
    video, editing, video-to-video, vid2vid, lora

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to extend."
    )
    acceleration: Ltx219BDistilledExtendVideoLoraAcceleration = Field(
        default=Ltx219BDistilledExtendVideoLoraAcceleration.NONE, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    loras: list[LoRAInput] = Field(
        default=[], description="The LoRAs to use for the generation."
    )
    camera_lora: Ltx219BDistilledExtendVideoLoraCameraLora = Field(
        default=Ltx219BDistilledExtendVideoLoraCameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_size: str = Field(
        default="auto", description="The size of the generated video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the extended video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    extend_direction: Ltx219BDistilledExtendVideoLoraExtendDirection = Field(
        default=Ltx219BDistilledExtendVideoLoraExtendDirection.FORWARD, description="Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: Ltx219BDistilledExtendVideoLoraVideoOutputType = Field(
        default=Ltx219BDistilledExtendVideoLoraVideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    video_write_mode: Ltx219BDistilledExtendVideoLoraVideoWriteMode = Field(
        default=Ltx219BDistilledExtendVideoLoraVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    num_context_frames: int = Field(
        default=25, description="The number of frames to use as context for the extension."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: Ltx219BDistilledExtendVideoLoraVideoQuality = Field(
        default=Ltx219BDistilledExtendVideoLoraVideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        end_image_base64 = (
            await context.image_to_base64(self.end_image)
            if not self.end_image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "use_multiscale": self.use_multiscale,
            "fps": self.fps,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}" if end_image_base64 else None,
            "negative_prompt": self.negative_prompt,
            "extend_direction": self.extend_direction.value,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "video_write_mode": self.video_write_mode.value,
            "num_frames": self.num_frames,
            "num_context_frames": self.num_context_frames,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
            "audio_strength": self.audio_strength,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/extend-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Ltx219BDistilledExtendVideoAcceleration(str, Enum):
    """
    The acceleration level to use.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"

class Ltx219BDistilledExtendVideoCameraLora(str, Enum):
    """
    The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera.
    """
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"

class Ltx219BDistilledExtendVideoExtendDirection(str, Enum):
    """
    Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning.
    """
    FORWARD = "forward"
    BACKWARD = "backward"

class Ltx219BDistilledExtendVideoVideoOutputType(str, Enum):
    """
    The output type of the generated video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class Ltx219BDistilledExtendVideoVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Ltx219BDistilledExtendVideoVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class Ltx219BDistilledExtendVideo(FALNode):
    """
    LTX-2 19B Distilled
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to extend."
    )
    acceleration: Ltx219BDistilledExtendVideoAcceleration = Field(
        default=Ltx219BDistilledExtendVideoAcceleration.NONE, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: Ltx219BDistilledExtendVideoCameraLora = Field(
        default=Ltx219BDistilledExtendVideoCameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_size: str = Field(
        default="auto", description="The size of the generated video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the extended video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    extend_direction: Ltx219BDistilledExtendVideoExtendDirection = Field(
        default=Ltx219BDistilledExtendVideoExtendDirection.FORWARD, description="Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: Ltx219BDistilledExtendVideoVideoOutputType = Field(
        default=Ltx219BDistilledExtendVideoVideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    video_write_mode: Ltx219BDistilledExtendVideoVideoWriteMode = Field(
        default=Ltx219BDistilledExtendVideoVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    num_context_frames: int = Field(
        default=25, description="The number of frames to use as context for the extension."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: Ltx219BDistilledExtendVideoVideoQuality = Field(
        default=Ltx219BDistilledExtendVideoVideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        end_image_base64 = (
            await context.image_to_base64(self.end_image)
            if not self.end_image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "use_multiscale": self.use_multiscale,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}" if end_image_base64 else None,
            "negative_prompt": self.negative_prompt,
            "extend_direction": self.extend_direction.value,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "video_write_mode": self.video_write_mode.value,
            "num_frames": self.num_frames,
            "num_context_frames": self.num_context_frames,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
            "audio_strength": self.audio_strength,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/extend-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Ltx219BExtendVideoLoraVideoOutputType(str, Enum):
    """
    The output type of the generated video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class Ltx219BExtendVideoLoraVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Ltx219BExtendVideoLoraAcceleration(str, Enum):
    """
    The acceleration level to use.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"

class Ltx219BExtendVideoLoraCameraLora(str, Enum):
    """
    The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera.
    """
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"

class Ltx219BExtendVideoLoraExtendDirection(str, Enum):
    """
    Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning.
    """
    FORWARD = "forward"
    BACKWARD = "backward"

class Ltx219BExtendVideoLoraVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"


class Ltx219BExtendVideoLora(FALNode):
    """
    LTX-2 19B
    video, editing, video-to-video, vid2vid, lora

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to extend."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    loras: list[LoRAInput] = Field(
        default=[], description="The LoRAs to use for the generation."
    )
    video_size: str = Field(
        default="auto", description="The size of the generated video."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the extended video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: Ltx219BExtendVideoLoraVideoOutputType = Field(
        default=Ltx219BExtendVideoLoraVideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: Ltx219BExtendVideoLoraVideoQuality = Field(
        default=Ltx219BExtendVideoLoraVideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    acceleration: Ltx219BExtendVideoLoraAcceleration = Field(
        default=Ltx219BExtendVideoLoraAcceleration.REGULAR, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: Ltx219BExtendVideoLoraCameraLora = Field(
        default=Ltx219BExtendVideoLoraCameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    extend_direction: Ltx219BExtendVideoLoraExtendDirection = Field(
        default=Ltx219BExtendVideoLoraExtendDirection.FORWARD, description="Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    video_write_mode: Ltx219BExtendVideoLoraVideoWriteMode = Field(
        default=Ltx219BExtendVideoLoraVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    num_context_frames: int = Field(
        default=25, description="The number of frames to use as context for the extension."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        end_image_base64 = (
            await context.image_to_base64(self.end_image)
            if not self.end_image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "use_multiscale": self.use_multiscale,
            "generate_audio": self.generate_audio,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}" if end_image_base64 else None,
            "num_frames": self.num_frames,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "acceleration": self.acceleration.value,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "enable_safety_checker": self.enable_safety_checker,
            "extend_direction": self.extend_direction.value,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "num_context_frames": self.num_context_frames,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/extend-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Ltx219BExtendVideoAcceleration(str, Enum):
    """
    The acceleration level to use.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"

class Ltx219BExtendVideoCameraLora(str, Enum):
    """
    The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera.
    """
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    DOLLY_LEFT = "dolly_left"
    DOLLY_RIGHT = "dolly_right"
    JIB_UP = "jib_up"
    JIB_DOWN = "jib_down"
    STATIC = "static"
    NONE = "none"

class Ltx219BExtendVideoVideoOutputType(str, Enum):
    """
    The output type of the generated video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class Ltx219BExtendVideoVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Ltx219BExtendVideoExtendDirection(str, Enum):
    """
    Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning.
    """
    FORWARD = "forward"
    BACKWARD = "backward"

class Ltx219BExtendVideoVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class Ltx219BExtendVideo(FALNode):
    """
    LTX-2 19B
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to extend."
    )
    acceleration: Ltx219BExtendVideoAcceleration = Field(
        default=Ltx219BExtendVideoAcceleration.REGULAR, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: Ltx219BExtendVideoCameraLora = Field(
        default=Ltx219BExtendVideoCameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_size: str = Field(
        default="auto", description="The size of the generated video."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the extended video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: Ltx219BExtendVideoVideoOutputType = Field(
        default=Ltx219BExtendVideoVideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    video_write_mode: Ltx219BExtendVideoVideoWriteMode = Field(
        default=Ltx219BExtendVideoVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    extend_direction: Ltx219BExtendVideoExtendDirection = Field(
        default=Ltx219BExtendVideoExtendDirection.FORWARD, description="Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    num_context_frames: int = Field(
        default=25, description="The number of frames to use as context for the extension."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: Ltx219BExtendVideoVideoQuality = Field(
        default=Ltx219BExtendVideoVideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        end_image_base64 = (
            await context.image_to_base64(self.end_image)
            if not self.end_image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "audio_strength": self.audio_strength,
            "use_multiscale": self.use_multiscale,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}" if end_image_base64 else None,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "video_write_mode": self.video_write_mode.value,
            "extend_direction": self.extend_direction.value,
            "num_frames": self.num_frames,
            "num_context_frames": self.num_context_frames,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/extend-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class BriaVideoEraseKeypointsOutputContainerAndCodec(str, Enum):
    """
    Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4.
    """
    MP4_H265 = "mp4_h265"
    MP4_H264 = "mp4_h264"
    WEBM_VP9 = "webm_vp9"
    GIF = "gif"
    MOV_H264 = "mov_h264"
    MOV_H265 = "mov_h265"
    MOV_PRORESKS = "mov_proresks"
    MKV_H264 = "mkv_h264"
    MKV_H265 = "mkv_h265"
    MKV_VP9 = "mkv_vp9"
    MKV_MPEG4 = "mkv_mpeg4"


class BriaVideoEraseKeypoints(FALNode):
    """
    Video
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    preserve_audio: bool = Field(
        default=True, description="If true, audio will be preserved in the output video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    output_container_and_codec: BriaVideoEraseKeypointsOutputContainerAndCodec = Field(
        default=BriaVideoEraseKeypointsOutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    keypoints: list[str] = Field(
        default=[], description="Input keypoints [x,y] to erase or keep from the video. Format like so: {'x':100, 'y':100, 'type':'positive/negative'}"
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": video_url,
            "output_container_and_codec": self.output_container_and_codec.value,
            "keypoints": self.keypoints,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/video/erase/keypoints",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class BriaVideoErasePromptOutputContainerAndCodec(str, Enum):
    """
    Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4.
    """
    MP4_H265 = "mp4_h265"
    MP4_H264 = "mp4_h264"
    WEBM_VP9 = "webm_vp9"
    GIF = "gif"
    MOV_H264 = "mov_h264"
    MOV_H265 = "mov_h265"
    MOV_PRORESKS = "mov_proresks"
    MKV_H264 = "mkv_h264"
    MKV_H265 = "mkv_h265"
    MKV_VP9 = "mkv_vp9"
    MKV_MPEG4 = "mkv_mpeg4"


class BriaVideoErasePrompt(FALNode):
    """
    Video
    video, editing, video-to-video, vid2vid, professional

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    preserve_audio: bool = Field(
        default=True, description="If true, audio will be preserved in the output video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    prompt: str = Field(
        default="", description="Input prompt to detect object to erase"
    )
    output_container_and_codec: BriaVideoErasePromptOutputContainerAndCodec = Field(
        default=BriaVideoErasePromptOutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": video_url,
            "prompt": self.prompt,
            "output_container_and_codec": self.output_container_and_codec.value,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/video/erase/prompt",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class BriaVideoEraseMaskOutputContainerAndCodec(str, Enum):
    """
    Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4.
    """
    MP4_H265 = "mp4_h265"
    MP4_H264 = "mp4_h264"
    WEBM_VP9 = "webm_vp9"
    GIF = "gif"
    MOV_H264 = "mov_h264"
    MOV_H265 = "mov_h265"
    MOV_PRORESKS = "mov_proresks"
    MKV_H264 = "mkv_h264"
    MKV_H265 = "mkv_h265"
    MKV_VP9 = "mkv_vp9"
    MKV_MPEG4 = "mkv_mpeg4"


class BriaVideoEraseMask(FALNode):
    """
    Video
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    preserve_audio: bool = Field(
        default=True, description="If true, audio will be preserved in the output video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    output_container_and_codec: BriaVideoEraseMaskOutputContainerAndCodec = Field(
        default=BriaVideoEraseMaskOutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    mask_video: VideoRef = Field(
        default=VideoRef(), description="Input video to mask erase object from. duration must be less than 5s."
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        mask_video_url = (
            await self._upload_asset_to_fal(client, self.mask_video, context)
            if not self.mask_video.is_empty()
            else None
        )
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": video_url,
            "output_container_and_codec": self.output_container_and_codec.value,
            "mask_video_url": mask_video_url,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/video/erase/mask",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class LightxRelightRelitCondType(str, Enum):
    """
    Relight condition type.
    """
    IC = "ic"
    REF = "ref"
    HDR = "hdr"
    BG = "bg"


class LightxRelight(FALNode):
    """
    Lightx
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Optional text prompt. If omitted, Light-X will auto-caption the video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    relight_parameters: str = Field(
        default="", description="Relighting parameters (required for relight_condition_type='ic'). Not used for 'bg' (which expects a background image URL instead)."
    )
    ref_id: int = Field(
        default=0, description="Frame index to use as referencen to relight the video with reference."
    )
    relit_cond_img: ImageRef = Field(
        default=ImageRef(), description="URL of conditioning image. Required for relight_condition_type='ref'/'hdr'. Also required for relight_condition_type='bg' (background image)."
    )
    relit_cond_type: LightxRelightRelitCondType = Field(
        default=LightxRelightRelitCondType.IC, description="Relight condition type."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        relit_cond_img_base64 = (
            await context.image_to_base64(self.relit_cond_img)
            if not self.relit_cond_img.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "relight_parameters": self.relight_parameters,
            "ref_id": self.ref_id,
            "relit_cond_img_url": f"data:image/png;base64,{relit_cond_img_base64}" if relit_cond_img_base64 else None,
            "relit_cond_type": self.relit_cond_type.value,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/lightx/relight",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class LightxRecameraCamera(str, Enum):
    """
    Camera control mode.
    """
    TRAJ = "traj"
    TARGET = "target"

class LightxRecameraMode(str, Enum):
    """
    Camera motion mode.
    """
    GRADUAL = "gradual"
    BULLET = "bullet"
    DIRECT = "direct"
    DOLLY_ZOOM = "dolly-zoom"


class LightxRecamera(FALNode):
    """
    Lightx
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Optional text prompt. If omitted, Light-X will auto-caption the video."
    )
    trajectory: str = Field(
        default="", description="Camera trajectory parameters (required for recamera mode)."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    camera: LightxRecameraCamera = Field(
        default=LightxRecameraCamera.TRAJ, description="Camera control mode."
    )
    target_pose: list[float] = Field(
        default=[], description="Target camera pose [theta, phi, radius, x, y] (required when camera='target')."
    )
    mode: LightxRecameraMode = Field(
        default=LightxRecameraMode.GRADUAL, description="Camera motion mode."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "trajectory": self.trajectory,
            "video_url": video_url,
            "camera": self.camera.value,
            "target_pose": self.target_pose,
            "mode": self.mode.value,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/lightx/recamera",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class KlingVideoV26StandardMotionControlCharacterOrientation(str, Enum):
    """
    Controls whether the output character's orientation matches the reference image or video. 'video': orientation matches reference video - better for complex motions (max 30s). 'image': orientation matches reference image - better for following camera movements (max 10s).
    """
    IMAGE = "image"
    VIDEO = "video"


class KlingVideoV26StandardMotionControl(FALNode):
    """
    Kling Video v2.6 Motion Control [Standard]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default=""
    )
    video: ImageRef = Field(
        default=ImageRef(), description="Reference video URL. The character actions in the generated video will be consistent with this reference video. Should contain a realistic style character with entire body or upper body visible, including head, without obstruction. Duration limit depends on character_orientation: 10s max for 'image', 30s max for 'video'."
    )
    character_orientation: KlingVideoV26StandardMotionControlCharacterOrientation = Field(
        default="", description="Controls whether the output character's orientation matches the reference image or video. 'video': orientation matches reference video - better for complex motions (max 30s). 'image': orientation matches reference image - better for following camera movements (max 10s)."
    )
    keep_original_sound: bool = Field(
        default=True, description="Whether to keep the original sound from the reference video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Reference image URL. The characters, backgrounds, and other elements in the generated video are based on this reference image. Characters should have clear body proportions, avoid occlusion, and occupy more than 5% of the image area."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_base64 = (
            await context.image_to_base64(self.video)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": f"data:image/png;base64,{video_base64}" if video_base64 else None,
            "character_orientation": self.character_orientation.value,
            "keep_original_sound": self.keep_original_sound,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2.6/standard/motion-control",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class KlingVideoV26ProMotionControlCharacterOrientation(str, Enum):
    """
    Controls whether the output character's orientation matches the reference image or video. 'video': orientation matches reference video - better for complex motions (max 30s). 'image': orientation matches reference image - better for following camera movements (max 10s).
    """
    IMAGE = "image"
    VIDEO = "video"


class KlingVideoV26ProMotionControl(FALNode):
    """
    Kling Video v2.6 Motion Control [Pro]
    video, editing, video-to-video, vid2vid, professional

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default=""
    )
    video: ImageRef = Field(
        default=ImageRef(), description="Reference video URL. The character actions in the generated video will be consistent with this reference video. Should contain a realistic style character with entire body or upper body visible, including head, without obstruction. Duration limit depends on character_orientation: 10s max for 'image', 30s max for 'video'."
    )
    character_orientation: KlingVideoV26ProMotionControlCharacterOrientation = Field(
        default="", description="Controls whether the output character's orientation matches the reference image or video. 'video': orientation matches reference video - better for complex motions (max 30s). 'image': orientation matches reference image - better for following camera movements (max 10s)."
    )
    keep_original_sound: bool = Field(
        default=True, description="Whether to keep the original sound from the reference video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Reference image URL. The characters, backgrounds, and other elements in the generated video are based on this reference image. Characters should have clear body proportions, avoid occlusion, and occupy more than 5% of the image area."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_base64 = (
            await context.image_to_base64(self.video)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": f"data:image/png;base64,{video_base64}" if video_base64 else None,
            "character_orientation": self.character_orientation.value,
            "keep_original_sound": self.keep_original_sound,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2.6/pro/motion-control",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class DecartLucyRestyleResolution(str, Enum):
    """
    Resolution of the generated video
    """
    VALUE_720P = "720p"


class DecartLucyRestyle(FALNode):
    """
    Lucy Restyle
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the video to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the video directly in the response without going through the CDN."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to edit"
    )
    resolution: DecartLucyRestyleResolution = Field(
        default=DecartLucyRestyleResolution.VALUE_720P, description="Resolution of the generated video"
    )
    prompt: str = Field(
        default="", description="Text description of the desired video content"
    )
    seed: int = Field(
        default=-1, description="Seed for video generation"
    )
    enhance_prompt: bool = Field(
        default=True, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "sync_mode": self.sync_mode,
            "video_url": video_url,
            "resolution": self.resolution.value,
            "prompt": self.prompt,
            "seed": self.seed,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="decart/lucy-restyle",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class ScailResolution(str, Enum):
    """
    Output resolution. Outputs 896x512 (landscape) or 512x896 (portrait) based on the input image aspect ratio.
    """
    VALUE_512P = "512p"


class Scail(FALNode):
    """
    Scail
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to guide video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use as a reference for the video generation."
    )
    resolution: ScailResolution = Field(
        default=ScailResolution.VALUE_512P, description="Output resolution. Outputs 896x512 (landscape) or 512x896 (portrait) based on the input image aspect ratio."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to use for the video generation."
    )
    multi_character: bool = Field(
        default=False, description="Enable multi-character mode. Use when driving video has multiple people."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "resolution": self.resolution.value,
            "num_inference_steps": self.num_inference_steps,
            "multi_character": self.multi_character,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/scail",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class ClarityaiCrystalVideoUpscaler(FALNode):
    """
    Crystal Upscaler [Video]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="URL to the input video."
    )
    scale_factor: float = Field(
        default=2, description="Scale factor. The scale factor must be chosen such that the upscaled video does not exceed 5K resolution."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "scale_factor": self.scale_factor,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="clarityai/crystal-video-upscaler",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class WanV26ReferenceToVideoAspectRatio(str, Enum):
    """
    The aspect ratio of the generated video.
    """
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"

class WanV26ReferenceToVideoResolution(str, Enum):
    """
    Video resolution tier. R2V only supports 720p and 1080p (no 480p).
    """
    VALUE_720P = "720p"
    VALUE_1080P = "1080p"

class WanV26ReferenceToVideoDuration(str, Enum):
    """
    Duration of the generated video in seconds. R2V supports only 5 or 10 seconds (no 15s).
    """
    VALUE_5 = "5"
    VALUE_10 = "10"


class WanV26ReferenceToVideo(FALNode):
    """
    Wan v2.6 Reference to Video
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Use @Video1, @Video2, @Video3 to reference subjects from your videos. Works for people, animals, or objects. For multi-shot prompts: '[0-3s] Shot 1. [3-6s] Shot 2.' Max 800 characters."
    )
    aspect_ratio: WanV26ReferenceToVideoAspectRatio = Field(
        default=WanV26ReferenceToVideoAspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
    )
    videos: list[str] = Field(
        default=[], description="Reference videos for subject consistency (1-3 videos). Videos' FPS must be at least 16 FPS.Reference in prompt as @Video1, @Video2, @Video3. Works for people, animals, or objects."
    )
    resolution: WanV26ReferenceToVideoResolution = Field(
        default=WanV26ReferenceToVideoResolution.VALUE_1080P, description="Video resolution tier. R2V only supports 720p and 1080p (no 480p)."
    )
    duration: WanV26ReferenceToVideoDuration = Field(
        default=WanV26ReferenceToVideoDuration.VALUE_5, description="Duration of the generated video in seconds. R2V supports only 5 or 10 seconds (no 15s)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    multi_shots: bool = Field(
        default=True, description="When true (default), enables intelligent multi-shot segmentation for coherent narrative videos with multiple shots. When false, generates single continuous shot. Only active when enable_prompt_expansion is True."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to describe content to avoid. Max 500 characters."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt rewriting using LLM."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "video_urls": self.videos,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "multi_shots": self.multi_shots,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="wan/v2.6/reference-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Veo31FastExtendVideoDuration(str, Enum):
    """
    The duration of the generated video.
    """
    VALUE_7S = "7s"

class Veo31FastExtendVideoAspectRatio(str, Enum):
    """
    The aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"

class Veo31FastExtendVideoResolution(str, Enum):
    """
    The resolution of the generated video.
    """
    VALUE_720P = "720p"


class Veo31FastExtendVideo(FALNode):
    """
    Veo 3.1 Fast
    video, editing, video-to-video, vid2vid, fast

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt describing how the video should be extended"
    )
    duration: Veo31FastExtendVideoDuration = Field(
        default=Veo31FastExtendVideoDuration.VALUE_7S, description="The duration of the generated video."
    )
    aspect_ratio: Veo31FastExtendVideoAspectRatio = Field(
        default=Veo31FastExtendVideoAspectRatio.AUTO, description="The aspect ratio of the generated video."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    auto_fix: bool = Field(
        default=False, description="Whether to automatically attempt to fix prompts that fail content policy or other validation checks by rewriting them."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to extend. The video should be 720p or 1080p resolution in 16:9 or 9:16 aspect ratio."
    )
    resolution: Veo31FastExtendVideoResolution = Field(
        default=Veo31FastExtendVideoResolution.VALUE_720P, description="The resolution of the generated video."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "video_url": video_url,
            "resolution": self.resolution.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3.1/fast/extend-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class Veo31ExtendVideoDuration(str, Enum):
    """
    The duration of the generated video.
    """
    VALUE_7S = "7s"

class Veo31ExtendVideoAspectRatio(str, Enum):
    """
    The aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"

class Veo31ExtendVideoResolution(str, Enum):
    """
    The resolution of the generated video.
    """
    VALUE_720P = "720p"


class Veo31ExtendVideo(FALNode):
    """
    Veo 3.1
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt describing how the video should be extended"
    )
    duration: Veo31ExtendVideoDuration = Field(
        default=Veo31ExtendVideoDuration.VALUE_7S, description="The duration of the generated video."
    )
    aspect_ratio: Veo31ExtendVideoAspectRatio = Field(
        default=Veo31ExtendVideoAspectRatio.AUTO, description="The aspect ratio of the generated video."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    auto_fix: bool = Field(
        default=False, description="Whether to automatically attempt to fix prompts that fail content policy or other validation checks by rewriting them."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to extend. The video should be 720p or 1080p resolution in 16:9 or 9:16 aspect ratio."
    )
    resolution: Veo31ExtendVideoResolution = Field(
        default=Veo31ExtendVideoResolution.VALUE_720P, description="The resolution of the generated video."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "video_url": video_url,
            "resolution": self.resolution.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3.1/extend-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class KlingVideoO1StandardVideoToVideoReferenceAspectRatio(str, Enum):
    """
    The aspect ratio of the generated video frame. If 'auto', the aspect ratio will be determined automatically based on the input video, and the closest aspect ratio to the input video will be used.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"

class KlingVideoO1StandardVideoToVideoReferenceDuration(str, Enum):
    """
    Video duration in seconds.
    """
    VALUE_3 = "3"
    VALUE_4 = "4"
    VALUE_5 = "5"
    VALUE_6 = "6"
    VALUE_7 = "7"
    VALUE_8 = "8"
    VALUE_9 = "9"
    VALUE_10 = "10"


class KlingVideoO1StandardVideoToVideoReference(FALNode):
    """
    Kling O1 Reference Video to Video [Standard]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Use @Element1, @Element2 to reference elements and @Image1, @Image2 to reference images in order."
    )
    aspect_ratio: KlingVideoO1StandardVideoToVideoReferenceAspectRatio = Field(
        default=KlingVideoO1StandardVideoToVideoReferenceAspectRatio.AUTO, description="The aspect ratio of the generated video frame. If 'auto', the aspect ratio will be determined automatically based on the input video, and the closest aspect ratio to the input video will be used."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats supported, 3-10 seconds duration, 720-2160px resolution, max 200MB. Max file size: 200.0MB, Min width: 720px, Min height: 720px, Max width: 2160px, Max height: 2160px, Min duration: 3.0s, Max duration: 10.05s, Min FPS: 24.0, Max FPS: 60.0, Timeout: 30.0s"
    )
    duration: KlingVideoO1StandardVideoToVideoReferenceDuration = Field(
        default=KlingVideoO1StandardVideoToVideoReferenceDuration.VALUE_5, description="Video duration in seconds."
    )
    keep_audio: bool = Field(
        default=False, description="Whether to keep the original audio from the video."
    )
    elements: list[OmniVideoElementInput] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    images: list[ImageRef] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        images_data_urls = []
        for image in self.images or []:
            if image.is_empty():
                continue
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "video_url": video_url,
            "duration": self.duration.value,
            "keep_audio": self.keep_audio,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/standard/video-to-video/reference",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class KlingVideoO1StandardVideoToVideoEdit(FALNode):
    """
    Kling O1 Edit Video [Standard]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Use @Element1, @Element2 to reference elements and @Image1, @Image2 to reference images in order."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats supported, 3-10 seconds duration, 720-2160px resolution, max 200MB. Max file size: 200.0MB, Min width: 720px, Min height: 720px, Max width: 2160px, Max height: 2160px, Min duration: 3.0s, Max duration: 10.05s, Min FPS: 24.0, Max FPS: 60.0, Timeout: 30.0s"
    )
    elements: list[OmniVideoElementInput] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    images: list[ImageRef] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    keep_audio: bool = Field(
        default=False, description="Whether to keep the original audio from the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        images_data_urls = []
        for image in self.images or []:
            if image.is_empty():
                continue
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
            "keep_audio": self.keep_audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/standard/video-to-video/edit",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class KlingVideoO3StandardVideoToVideoReferenceAspectRatio(str, Enum):
    """
    Aspect ratio.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"

class KlingVideoO3StandardVideoToVideoReferenceDuration(str, Enum):
    """
    Video duration in seconds (3-15s for reference video).
    """
    VALUE_3 = "3"
    VALUE_4 = "4"
    VALUE_5 = "5"
    VALUE_6 = "6"
    VALUE_7 = "7"
    VALUE_8 = "8"
    VALUE_9 = "9"
    VALUE_10 = "10"
    VALUE_11 = "11"
    VALUE_12 = "12"
    VALUE_13 = "13"
    VALUE_14 = "14"
    VALUE_15 = "15"

class KlingVideoO3StandardVideoToVideoReferenceShotType(str, Enum):
    """
    The type of multi-shot video generation.
    """
    CUSTOMIZE = "customize"


class KlingVideoO3StandardVideoToVideoReference(FALNode):
    """
    Kling O3 Reference Video to Video [Standard]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt for video generation. Reference video as @Video1."
    )
    aspect_ratio: KlingVideoO3StandardVideoToVideoReferenceAspectRatio = Field(
        default=KlingVideoO3StandardVideoToVideoReferenceAspectRatio.AUTO, description="Aspect ratio."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats, 3-10s duration, 720-2160px resolution, max 200MB."
    )
    duration: KlingVideoO3StandardVideoToVideoReferenceDuration | None = Field(
        default=None, description="Video duration in seconds (3-15s for reference video)."
    )
    keep_audio: bool = Field(
        default=True, description="Whether to keep the original audio from the reference video."
    )
    shot_type: KlingVideoO3StandardVideoToVideoReferenceShotType = Field(
        default=KlingVideoO3StandardVideoToVideoReferenceShotType.CUSTOMIZE, description="The type of multi-shot video generation."
    )
    elements: list[KlingV3ImageElementInput] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2."
    )
    images: list[ImageRef] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        images_data_urls = []
        for image in self.images or []:
            if image.is_empty():
                continue
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "video_url": video_url,
            "duration": self.duration.value if self.duration else None,
            "keep_audio": self.keep_audio,
            "shot_type": self.shot_type.value,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/standard/video-to-video/reference",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt", "duration"]

class KlingVideoO3StandardVideoToVideoEditShotType(str, Enum):
    """
    The type of multi-shot video generation.
    """
    CUSTOMIZE = "customize"


class KlingVideoO3StandardVideoToVideoEdit(FALNode):
    """
    Kling O3 Edit Video [Standard]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt for video generation. Reference video as @Video1."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats, 3-10s duration, 720-2160px resolution, max 200MB."
    )
    elements: list[KlingV3ImageElementInput] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2."
    )
    images: list[ImageRef] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    keep_audio: bool = Field(
        default=True, description="Whether to keep the original audio from the reference video."
    )
    shot_type: KlingVideoO3StandardVideoToVideoEditShotType = Field(
        default=KlingVideoO3StandardVideoToVideoEditShotType.CUSTOMIZE, description="The type of multi-shot video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        images_data_urls = []
        for image in self.images or []:
            if image.is_empty():
                continue
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
            "keep_audio": self.keep_audio,
            "shot_type": self.shot_type.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/standard/video-to-video/edit",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class KlingVideoO3ProVideoToVideoReferenceAspectRatio(str, Enum):
    """
    Aspect ratio.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"

class KlingVideoO3ProVideoToVideoReferenceDuration(str, Enum):
    """
    Video duration in seconds (3-15s for reference video).
    """
    VALUE_3 = "3"
    VALUE_4 = "4"
    VALUE_5 = "5"
    VALUE_6 = "6"
    VALUE_7 = "7"
    VALUE_8 = "8"
    VALUE_9 = "9"
    VALUE_10 = "10"
    VALUE_11 = "11"
    VALUE_12 = "12"
    VALUE_13 = "13"
    VALUE_14 = "14"
    VALUE_15 = "15"

class KlingVideoO3ProVideoToVideoReferenceShotType(str, Enum):
    """
    The type of multi-shot video generation.
    """
    CUSTOMIZE = "customize"


class KlingVideoO3ProVideoToVideoReference(FALNode):
    """
    Kling O3 Reference Video to Video [Pro]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt for video generation. Reference video as @Video1."
    )
    aspect_ratio: KlingVideoO3ProVideoToVideoReferenceAspectRatio = Field(
        default=KlingVideoO3ProVideoToVideoReferenceAspectRatio.AUTO, description="Aspect ratio."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats, 3-10s duration, 720-2160px resolution, max 200MB."
    )
    duration: KlingVideoO3ProVideoToVideoReferenceDuration | None = Field(
        default=None, description="Video duration in seconds (3-15s for reference video)."
    )
    keep_audio: bool = Field(
        default=True, description="Whether to keep the original audio from the reference video."
    )
    shot_type: KlingVideoO3ProVideoToVideoReferenceShotType = Field(
        default=KlingVideoO3ProVideoToVideoReferenceShotType.CUSTOMIZE, description="The type of multi-shot video generation."
    )
    elements: list[KlingV3ImageElementInput] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2."
    )
    images: list[ImageRef] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        images_data_urls = []
        for image in self.images or []:
            if image.is_empty():
                continue
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "video_url": video_url,
            "duration": self.duration.value if self.duration else None,
            "keep_audio": self.keep_audio,
            "shot_type": self.shot_type.value,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/pro/video-to-video/reference",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt", "duration"]

class KlingVideoO3ProVideoToVideoEditShotType(str, Enum):
    """
    The type of multi-shot video generation.
    """
    CUSTOMIZE = "customize"


class KlingVideoO3ProVideoToVideoEdit(FALNode):
    """
    Kling O3 Edit Video [Pro]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt for video generation. Reference video as @Video1."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats, 3-10s duration, 720-2160px resolution, max 200MB."
    )
    elements: list[KlingV3ImageElementInput] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2."
    )
    images: list[ImageRef] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    keep_audio: bool = Field(
        default=True, description="Whether to keep the original audio from the reference video."
    )
    shot_type: KlingVideoO3ProVideoToVideoEditShotType = Field(
        default=KlingVideoO3ProVideoToVideoEditShotType.CUSTOMIZE, description="The type of multi-shot video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        images_data_urls = []
        for image in self.images or []:
            if image.is_empty():
                continue
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
            "keep_audio": self.keep_audio,
            "shot_type": self.shot_type.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o3/pro/video-to-video/edit",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class SteadyDancerAcceleration(str, Enum):
    """
    Acceleration levels.
    """
    LIGHT = "light"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class SteadyDancerAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video. If 'auto', will be determined from the reference image.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"

class SteadyDancerResolution(str, Enum):
    """
    Resolution of the generated video. 576p is default, 720p for higher quality. 480p is lower quality.
    """
    VALUE_480P = "480p"
    VALUE_576P = "576p"
    VALUE_720P = "720p"


class SteadyDancer(FALNode):
    """
    Steady Dancer
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="A person dancing with smooth and natural movements.", description="Text prompt describing the desired animation."
    )
    video: ImageRef = Field(
        default=ImageRef(), description="URL of the driving pose video. The motion from this video will be transferred to the reference image."
    )
    acceleration: SteadyDancerAcceleration = Field(
        default=SteadyDancerAcceleration.AGGRESSIVE, description="Acceleration levels."
    )
    pose_guidance_scale: float = Field(
        default=1, description="Pose guidance scale for pose control strength."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    pose_guidance_end: float = Field(
        default=0.4, description="End ratio for pose guidance. Controls when pose guidance ends."
    )
    frames_per_second: int = Field(
        default=0, description="Frames per second of the generated video. Must be between 5 to 24. If not specified, uses the FPS from the input video."
    )
    guidance_scale: float = Field(
        default=1, description="Classifier-free guidance scale for prompt adherence."
    )
    num_frames: int = Field(
        default=0, description="Number of frames to generate. If not specified, uses the frame count from the input video (capped at 241). Will be adjusted to nearest valid value (must satisfy 4k+1 pattern)."
    )
    use_turbo: bool = Field(
        default=False, description="If true, applies quality enhancement for faster generation with improved quality. When enabled, parameters are automatically optimized (num_inference_steps=6, guidance_scale=1.0) and uses the LightX2V distillation LoRA."
    )
    negative_prompt: str = Field(
        default="blurred, distorted face, bad anatomy, extra limbs, poorly drawn hands, poorly drawn feet, disfigured, out of frame, duplicate, watermark, signature, text", description="Negative prompt for video generation."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    aspect_ratio: SteadyDancerAspectRatio = Field(
        default=SteadyDancerAspectRatio.AUTO, description="Aspect ratio of the generated video. If 'auto', will be determined from the reference image."
    )
    pose_guidance_start: float = Field(
        default=0.1, description="Start ratio for pose guidance. Controls when pose guidance begins."
    )
    resolution: SteadyDancerResolution = Field(
        default=SteadyDancerResolution.VALUE_576P, description="Resolution of the generated video. 576p is default, 720p for higher quality. 480p is lower quality."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the reference image to animate. This is the person/character whose appearance will be preserved."
    )
    preserve_audio: bool = Field(
        default=True, description="If enabled, copies audio from the input driving video to the output video."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    num_inference_steps: int = Field(
        default=6, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_base64 = (
            await context.image_to_base64(self.video)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": f"data:image/png;base64,{video_base64}" if video_base64 else None,
            "acceleration": self.acceleration.value,
            "pose_guidance_scale": self.pose_guidance_scale,
            "shift": self.shift,
            "pose_guidance_end": self.pose_guidance_end,
            "frames_per_second": self.frames_per_second,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "use_turbo": self.use_turbo,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "aspect_ratio": self.aspect_ratio.value,
            "pose_guidance_start": self.pose_guidance_start,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "preserve_audio": self.preserve_audio,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/steady-dancer",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class OneToAllAnimation13BResolution(str, Enum):
    """
    The resolution of the video to generate.
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"


class OneToAllAnimation13B(FALNode):
    """
    One To All Animation
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    resolution: OneToAllAnimation13BResolution = Field(
        default=OneToAllAnimation13BResolution.VALUE_480P, description="The resolution of the video to generate."
    )
    image_guidance_scale: float = Field(
        default=2, description="The image guidance scale to use for the video generation."
    )
    pose_guidance_scale: float = Field(
        default=1.5, description="The pose guidance scale to use for the video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use as a reference for the video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to use for the video generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate the video from."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "image_guidance_scale": self.image_guidance_scale,
            "pose_guidance_scale": self.pose_guidance_scale,
            "video_url": video_url,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/one-to-all-animation/1.3b",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class OneToAllAnimation14BResolution(str, Enum):
    """
    The resolution of the video to generate.
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"


class OneToAllAnimation14B(FALNode):
    """
    One To All Animation
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    resolution: OneToAllAnimation14BResolution = Field(
        default=OneToAllAnimation14BResolution.VALUE_480P, description="The resolution of the video to generate."
    )
    image_guidance_scale: float = Field(
        default=2, description="The image guidance scale to use for the video generation."
    )
    pose_guidance_scale: float = Field(
        default=1.5, description="The pose guidance scale to use for the video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use as a reference for the video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to use for the video generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate the video from."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "image_guidance_scale": self.image_guidance_scale,
            "pose_guidance_scale": self.pose_guidance_scale,
            "video_url": video_url,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/one-to-all-animation/14b",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVisionEnhancerTargetResolution(str, Enum):
    """
    Target output resolution for the enhanced video. 720p (native, fast) or 1080p (upscaled, slower). Processing is always done at 720p, then upscaled if 1080p selected.
    """
    VALUE_720P = "720p"
    VALUE_1080P = "1080p"


class WanVisionEnhancer(FALNode):
    """
    Wan Vision Enhancer
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Optional prompt to prepend to the VLM-generated description. Leave empty to use only the auto-generated description from the video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to enhance with Wan Video. Maximum 200MB file size. Videos longer than 500 frames will have only the first 500 frames processed (~8-21 seconds depending on fps)."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If not provided, a random seed will be used."
    )
    target_resolution: WanVisionEnhancerTargetResolution = Field(
        default=WanVisionEnhancerTargetResolution.VALUE_720P, description="Target output resolution for the enhanced video. 720p (native, fast) or 1080p (upscaled, slower). Processing is always done at 720p, then upscaled if 1080p selected."
    )
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still frame, overall gray, worst quality, low quality, JPEG artifacts, ugly, mutated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static motion, cluttered background, three legs, crowded background, walking backwards", description="Negative prompt to avoid unwanted features."
    )
    creativity: int = Field(
        default=1, description="Controls how much the model enhances/changes the video. 0 = Minimal change (preserves original), 1 = Subtle enhancement (default), 2 = Medium enhancement, 3 = Strong enhancement, 4 = Maximum enhancement."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "seed": self.seed,
            "target_resolution": self.target_resolution.value,
            "negative_prompt": self.negative_prompt,
            "creativity": self.creativity,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vision-enhancer",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class SyncLipsyncReact1ModelMode(str, Enum):
    """
    Controls the edit region and movement scope for the model. Available options:
    - `lips`: Only lipsync using react-1 (minimal facial changes).
    - `face`: Lipsync + facial expressions without head movements.
    - `head`: Lipsync + facial expressions + natural talking head movements.
    """
    LIPS = "lips"
    FACE = "face"
    HEAD = "head"

class SyncLipsyncReact1LipsyncMode(str, Enum):
    """
    Lipsync mode when audio and video durations are out of sync.
    """
    CUT_OFF = "cut_off"
    LOOP = "loop"
    BOUNCE = "bounce"
    SILENCE = "silence"
    REMAP = "remap"

class SyncLipsyncReact1Emotion(str, Enum):
    """
    Emotion prompt for the generation. Currently supports single-word emotions only.
    """
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    NEUTRAL = "neutral"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"


class SyncLipsyncReact1(FALNode):
    """
    Sync React-1
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    model_mode: SyncLipsyncReact1ModelMode = Field(
        default=SyncLipsyncReact1ModelMode.FACE, description="Controls the edit region and movement scope for the model. Available options: - `lips`: Only lipsync using react-1 (minimal facial changes). - `face`: Lipsync + facial expressions without head movements. - `head`: Lipsync + facial expressions + natural talking head movements."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the input video. Must be **15 seconds or shorter**."
    )
    lipsync_mode: SyncLipsyncReact1LipsyncMode = Field(
        default=SyncLipsyncReact1LipsyncMode.BOUNCE, description="Lipsync mode when audio and video durations are out of sync."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL to the input audio. Must be **15 seconds or shorter**."
    )
    temperature: float = Field(
        default=0.5, description="Controls the expresiveness of the lipsync."
    )
    emotion: SyncLipsyncReact1Emotion = Field(
        default="", description="Emotion prompt for the generation. Currently supports single-word emotions only."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "model_mode": self.model_mode.value,
            "video_url": video_url,
            "lipsync_mode": self.lipsync_mode.value,
            "audio_url": self.audio,
            "temperature": self.temperature,
            "emotion": self.emotion.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sync-lipsync/react-1",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class VeedVideoBackgroundRemovalFastOutputCodec(str, Enum):
    """
    Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality.
    """
    VP9 = "vp9"
    H264 = "h264"


class VeedVideoBackgroundRemovalFast(FALNode):
    """
    Video Background Removal
    video, editing, video-to-video, vid2vid, fast

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef()
    )
    subject_is_person: bool = Field(
        default=True, description="Set to False if the subject is not a person."
    )
    output_codec: VeedVideoBackgroundRemovalFastOutputCodec = Field(
        default=VeedVideoBackgroundRemovalFastOutputCodec.VP9, description="Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality."
    )
    refine_foreground_edges: bool = Field(
        default=True, description="Improves the quality of the extracted object's edges."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "subject_is_person": self.subject_is_person,
            "output_codec": self.output_codec.value,
            "refine_foreground_edges": self.refine_foreground_edges,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/video-background-removal/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class KlingVideoO1VideoToVideoEdit(FALNode):
    """
    Kling O1 Edit Video [Pro]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Use @Element1, @Element2 to reference elements and @Image1, @Image2 to reference images in order."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats supported, 3-10 seconds duration, 720-2160px resolution, max 200MB. Max file size: 200.0MB, Min width: 720px, Min height: 720px, Max width: 2160px, Max height: 2160px, Min duration: 3.0s, Max duration: 10.05s, Min FPS: 24.0, Max FPS: 60.0, Timeout: 30.0s"
    )
    elements: list[OmniVideoElementInput] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    images: list[ImageRef] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    keep_audio: bool = Field(
        default=False, description="Whether to keep the original audio from the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        images_data_urls = []
        for image in self.images or []:
            if image.is_empty():
                continue
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
            "keep_audio": self.keep_audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/video-to-video/edit",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class KlingVideoO1VideoToVideoReferenceAspectRatio(str, Enum):
    """
    The aspect ratio of the generated video frame. If 'auto', the aspect ratio will be determined automatically based on the input video, and the closest aspect ratio to the input video will be used.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"

class KlingVideoO1VideoToVideoReferenceDuration(str, Enum):
    """
    Video duration in seconds.
    """
    VALUE_3 = "3"
    VALUE_4 = "4"
    VALUE_5 = "5"
    VALUE_6 = "6"
    VALUE_7 = "7"
    VALUE_8 = "8"
    VALUE_9 = "9"
    VALUE_10 = "10"


class KlingVideoO1VideoToVideoReference(FALNode):
    """
    Kling O1 Reference Video to Video [Pro]
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Use @Element1, @Element2 to reference elements and @Image1, @Image2 to reference images in order."
    )
    aspect_ratio: KlingVideoO1VideoToVideoReferenceAspectRatio = Field(
        default=KlingVideoO1VideoToVideoReferenceAspectRatio.AUTO, description="The aspect ratio of the generated video frame. If 'auto', the aspect ratio will be determined automatically based on the input video, and the closest aspect ratio to the input video will be used."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats supported, 3-10 seconds duration, 720-2160px resolution, max 200MB. Max file size: 200.0MB, Min width: 720px, Min height: 720px, Max width: 2160px, Max height: 2160px, Min duration: 3.0s, Max duration: 10.05s, Min FPS: 24.0, Max FPS: 60.0, Timeout: 30.0s"
    )
    duration: KlingVideoO1VideoToVideoReferenceDuration = Field(
        default=KlingVideoO1VideoToVideoReferenceDuration.VALUE_5, description="Video duration in seconds."
    )
    keep_audio: bool = Field(
        default=False, description="Whether to keep the original audio from the video."
    )
    elements: list[OmniVideoElementInput] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    images: list[ImageRef] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        images_data_urls = []
        for image in self.images or []:
            if image.is_empty():
                continue
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "video_url": video_url,
            "duration": self.duration.value,
            "keep_audio": self.keep_audio,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/video-to-video/reference",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class VeedVideoBackgroundRemovalOutputCodec(str, Enum):
    """
    Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality.
    """
    VP9 = "vp9"
    H264 = "h264"


class VeedVideoBackgroundRemoval(FALNode):
    """
    Video Background Removal
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef()
    )
    subject_is_person: bool = Field(
        default=True, description="Set to False if the subject is not a person."
    )
    output_codec: VeedVideoBackgroundRemovalOutputCodec = Field(
        default=VeedVideoBackgroundRemovalOutputCodec.VP9, description="Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality."
    )
    refine_foreground_edges: bool = Field(
        default=True, description="Improves the quality of the extracted object's edges."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "subject_is_person": self.subject_is_person,
            "output_codec": self.output_codec.value,
            "refine_foreground_edges": self.refine_foreground_edges,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/video-background-removal",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class VeedVideoBackgroundRemovalGreenScreenOutputCodec(str, Enum):
    """
    Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality.
    """
    VP9 = "vp9"
    H264 = "h264"


class VeedVideoBackgroundRemovalGreenScreen(FALNode):
    """
    Video Background Removal
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef()
    )
    output_codec: VeedVideoBackgroundRemovalGreenScreenOutputCodec = Field(
        default=VeedVideoBackgroundRemovalGreenScreenOutputCodec.VP9, description="Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality."
    )
    spill_suppression_strength: str = Field(
        default=0.8, description="Increase the value if green spots remain in the video, decrease if color changes are noticed on the extracted subject."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "output_codec": self.output_codec.value,
            "spill_suppression_strength": self.spill_suppression_strength,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/video-background-removal/green-screen",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Ltx2RetakeVideoRetakeMode(str, Enum):
    """
    The retake mode to use for the retake
    """
    REPLACE_AUDIO = "replace_audio"
    REPLACE_VIDEO = "replace_video"
    REPLACE_AUDIO_AND_VIDEO = "replace_audio_and_video"


class Ltx2RetakeVideo(FALNode):
    """
    LTX Video 2.0 Retake
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to retake the video with"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to retake"
    )
    start_time: float = Field(
        default=0, description="The start time of the video to retake in seconds"
    )
    duration: float = Field(
        default=5, description="The duration of the video to retake in seconds"
    )
    retake_mode: Ltx2RetakeVideoRetakeMode = Field(
        default=Ltx2RetakeVideoRetakeMode.REPLACE_AUDIO_AND_VIDEO, description="The retake mode to use for the retake"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "start_time": self.start_time,
            "duration": self.duration,
            "retake_mode": self.retake_mode.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2/retake-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class DecartLucyEditFast(FALNode):
    """
    Lucy Edit [Fast]
    video, editing, video-to-video, vid2vid, fast

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the video to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the video directly in the response without going through the CDN."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to edit"
    )
    prompt: str = Field(
        default="", description="Text description of the desired video content"
    )
    enhance_prompt: bool = Field(
        default=True, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "sync_mode": self.sync_mode,
            "video_url": video_url,
            "prompt": self.prompt,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="decart/lucy-edit/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Sam3VideoRleRleReturnMode(str, Enum):
    """
    Mode of returning rles.
    """
    URL = "url"
    LIST = "list"


class Sam3VideoRle(FALNode):
    """
    Sam 3
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt for segmentation. Use commas to track multiple objects (e.g., 'person, cloth')."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to be segmented."
    )
    detection_threshold: float = Field(
        default=0.5, description="Detection confidence threshold (0.0-1.0). Lower = more detections but less precise. Defaults: 0.5 for existing, 0.7 for new objects. Try 0.2-0.3 if text prompts fail."
    )
    rle_return_mode: Sam3VideoRleRleReturnMode = Field(
        default=Sam3VideoRleRleReturnMode.LIST, description="Mode of returning rles."
    )
    mask_url: str = Field(
        default="", description="The URL of the mask to be applied initially."
    )
    box_prompts: list[BoxPrompt] = Field(
        default=[], description="List of box prompts with optional frame_index."
    )
    point_prompts: list[PointPrompt] = Field(
        default=[], description="List of point prompts with frame indices."
    )
    boundingbox_zip: bool = Field(
        default=False, description="Return per-frame bounding box overlays as a zip archive."
    )
    frame_index: int = Field(
        default=0, description="Frame index used for initial interaction when mask_url is provided."
    )
    return_rle: bool = Field(
        default=False, description="Return the Run Length Encoding of the mask."
    )
    apply_mask: bool = Field(
        default=False, description="Apply the mask on the video."
    )
    text_prompt: str = Field(
        default="", description="[DEPRECATED] Use 'prompt' instead. Kept for backward compatibility."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "detection_threshold": self.detection_threshold,
            "rle_return_mode": self.rle_return_mode.value,
            "mask_url": self.mask_url,
            "box_prompts": [item.model_dump(exclude={"type"}) for item in self.box_prompts],
            "point_prompts": [item.model_dump(exclude={"type"}) for item in self.point_prompts],
            "boundingbox_zip": self.boundingbox_zip,
            "frame_index": self.frame_index,
            "return_rle": self.return_rle,
            "apply_mask": self.apply_mask,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/video-rle",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Sam3Video(FALNode):
    """
    Sam 3
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt for segmentation. Use commas to track multiple objects (e.g., 'person, cloth')."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to be segmented."
    )
    detection_threshold: float = Field(
        default=0.5, description="Detection confidence threshold (0.0-1.0). Lower = more detections but less precise."
    )
    box_prompts: list[BoxPromptBase] = Field(
        default=[], description="List of box prompt coordinates (x_min, y_min, x_max, y_max)."
    )
    point_prompts: list[PointPromptBase] = Field(
        default=[], description="List of point prompts"
    )
    apply_mask: bool = Field(
        default=True, description="Apply the mask on the video."
    )
    text_prompt: str = Field(
        default="", description="[DEPRECATED] Use 'prompt' instead. Kept for backward compatibility."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "detection_threshold": self.detection_threshold,
            "box_prompts": [item.model_dump(exclude={"type"}) for item in self.box_prompts],
            "point_prompts": [item.model_dump(exclude={"type"}) for item in self.point_prompts],
            "apply_mask": self.apply_mask,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class EdittoSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class EdittoVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class EdittoResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class EdittoAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"

class EdittoVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class Editto(FALNode):
    """
    Editto
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. Required for inpainting."
    )
    acceleration: str = Field(
        default="regular", description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    frames_per_second: str = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    sampler: EdittoSampler = Field(
        default=EdittoSampler.UNIPC, description="Sampler to use for video generation."
    )
    video_write_mode: EdittoVideoWriteMode = Field(
        default=EdittoVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    resolution: EdittoResolution = Field(
        default=EdittoResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: EdittoAspectRatio = Field(
        default=EdittoAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    video_quality: EdittoVideoQuality = Field(
        default=EdittoVideoQuality.HIGH, description="The quality of the generated video."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "acceleration": self.acceleration,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "shift": self.shift,
            "frames_per_second": self.frames_per_second,
            "match_input_num_frames": self.match_input_num_frames,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "sampler": self.sampler.value,
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "return_frames_zip": self.return_frames_zip,
            "video_quality": self.video_quality.value,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "sync_mode": self.sync_mode,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "enable_auto_downsample": self.enable_auto_downsample,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/editto",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class FlashvsrUpscaleVideoAcceleration(str, Enum):
    """
    Acceleration mode for VAE decoding. Options: regular (best quality), high (balanced), full (fastest). More accerleation means longer duration videos can be processed too.
    """
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"

class FlashvsrUpscaleVideoOutputWriteMode(str, Enum):
    """
    The write mode of the output video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class FlashvsrUpscaleVideoOutputFormat(str, Enum):
    """
    The format of the output video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class FlashvsrUpscaleVideoOutputQuality(str, Enum):
    """
    The quality of the output video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class FlashvsrUpscaleVideo(FALNode):
    """
    Flashvsr
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The input video to be upscaled"
    )
    acceleration: FlashvsrUpscaleVideoAcceleration = Field(
        default=FlashvsrUpscaleVideoAcceleration.REGULAR, description="Acceleration mode for VAE decoding. Options: regular (best quality), high (balanced), full (fastest). More accerleation means longer duration videos can be processed too."
    )
    quality: int = Field(
        default=70, description="Quality level for tile blending (0-100). Controls overlap between tiles to prevent grid artifacts. Higher values provide better quality with more overlap. Recommended: 70-85 for high-res videos, 50-70 for faster processing."
    )
    output_write_mode: FlashvsrUpscaleVideoOutputWriteMode = Field(
        default=FlashvsrUpscaleVideoOutputWriteMode.BALANCED, description="The write mode of the output video."
    )
    output_format: FlashvsrUpscaleVideoOutputFormat = Field(
        default=FlashvsrUpscaleVideoOutputFormat.X264_MP4, description="The format of the output video."
    )
    color_fix: bool = Field(
        default=True, description="Color correction enabled."
    )
    preserve_audio: bool = Field(
        default=False, description="Copy the original audio tracks into the upscaled video using FFmpeg when possible."
    )
    output_quality: FlashvsrUpscaleVideoOutputQuality = Field(
        default=FlashvsrUpscaleVideoOutputQuality.HIGH, description="The quality of the output video."
    )
    upscale_factor: float = Field(
        default=2, description="Upscaling factor to be used."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned inline and not stored in history."
    )
    seed: int = Field(
        default=-1, description="The random seed used for the generation process."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "acceleration": self.acceleration.value,
            "quality": self.quality,
            "output_write_mode": self.output_write_mode.value,
            "output_format": self.output_format.value,
            "color_fix": self.color_fix,
            "preserve_audio": self.preserve_audio,
            "output_quality": self.output_quality.value,
            "upscale_factor": self.upscale_factor,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flashvsr/upscale/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WorkflowUtilitiesAutoSubtitleFontWeight(str, Enum):
    """
    Font weight (TikTok style typically uses bold or black)
    """
    NORMAL = "normal"
    BOLD = "bold"
    BLACK = "black"

class WorkflowUtilitiesAutoSubtitleFontColor(str, Enum):
    """
    Subtitle text color for non-active words
    """
    WHITE = "white"
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    GRAY = "gray"
    CYAN = "cyan"
    MAGENTA = "magenta"

class WorkflowUtilitiesAutoSubtitleStrokeColor(str, Enum):
    """
    Text stroke/outline color
    """
    BLACK = "black"
    WHITE = "white"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    GRAY = "gray"
    CYAN = "cyan"
    MAGENTA = "magenta"

class WorkflowUtilitiesAutoSubtitleHighlightColor(str, Enum):
    """
    Color for the currently speaking word (karaoke-style highlight)
    """
    WHITE = "white"
    BLACK = "black"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    GRAY = "gray"
    CYAN = "cyan"
    MAGENTA = "magenta"

class WorkflowUtilitiesAutoSubtitlePosition(str, Enum):
    """
    Vertical position of subtitles
    """
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"

class WorkflowUtilitiesAutoSubtitleBackgroundColor(str, Enum):
    """
    Background color behind text ('none' or 'transparent' for no background)
    """
    BLACK = "black"
    WHITE = "white"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    GRAY = "gray"
    CYAN = "cyan"
    MAGENTA = "magenta"
    NONE = "none"
    TRANSPARENT = "transparent"


class WorkflowUtilitiesAutoSubtitle(FALNode):
    """
    Workflow Utilities
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    font_weight: WorkflowUtilitiesAutoSubtitleFontWeight = Field(
        default=WorkflowUtilitiesAutoSubtitleFontWeight.BOLD, description="Font weight (TikTok style typically uses bold or black)"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video file to add automatic subtitles to Max file size: 95.4MB, Timeout: 30.0s"
    )
    stroke_width: int = Field(
        default=3, description="Text stroke/outline width in pixels (0 for no stroke)"
    )
    font_color: WorkflowUtilitiesAutoSubtitleFontColor = Field(
        default=WorkflowUtilitiesAutoSubtitleFontColor.WHITE, description="Subtitle text color for non-active words"
    )
    font_size: int = Field(
        default=100, description="Font size for subtitles (TikTok style uses larger text)"
    )
    language: str = Field(
        default="en", description="Language code for transcription (e.g., 'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'ja', 'zh', 'ko') or 3-letter ISO code (e.g., 'eng', 'spa', 'fra')"
    )
    y_offset: int = Field(
        default=75, description="Vertical offset in pixels (positive = move down, negative = move up)"
    )
    background_opacity: float = Field(
        default=0, description="Background opacity (0.0 = fully transparent, 1.0 = fully opaque)"
    )
    stroke_color: WorkflowUtilitiesAutoSubtitleStrokeColor = Field(
        default=WorkflowUtilitiesAutoSubtitleStrokeColor.BLACK, description="Text stroke/outline color"
    )
    highlight_color: WorkflowUtilitiesAutoSubtitleHighlightColor = Field(
        default=WorkflowUtilitiesAutoSubtitleHighlightColor.PURPLE, description="Color for the currently speaking word (karaoke-style highlight)"
    )
    enable_animation: bool = Field(
        default=True, description="Enable animation effects for subtitles (bounce style entrance)"
    )
    font_name: str = Field(
        default="Montserrat", description="Any Google Font name from fonts.google.com (e.g., 'Montserrat', 'Poppins', 'BBH Sans Hegarty')"
    )
    position: WorkflowUtilitiesAutoSubtitlePosition = Field(
        default=WorkflowUtilitiesAutoSubtitlePosition.BOTTOM, description="Vertical position of subtitles"
    )
    words_per_subtitle: int = Field(
        default=3, description="Maximum number of words per subtitle segment. Use 1 for single-word display, 2-3 for short phrases, or 8-12 for full sentences."
    )
    background_color: WorkflowUtilitiesAutoSubtitleBackgroundColor = Field(
        default=WorkflowUtilitiesAutoSubtitleBackgroundColor.NONE, description="Background color behind text ('none' or 'transparent' for no background)"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "font_weight": self.font_weight.value,
            "video_url": video_url,
            "stroke_width": self.stroke_width,
            "font_color": self.font_color.value,
            "font_size": self.font_size,
            "language": self.language,
            "y_offset": self.y_offset,
            "background_opacity": self.background_opacity,
            "stroke_color": self.stroke_color.value,
            "highlight_color": self.highlight_color.value,
            "enable_animation": self.enable_animation,
            "font_name": self.font_name,
            "position": self.position.value,
            "words_per_subtitle": self.words_per_subtitle,
            "background_color": self.background_color.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/workflow-utilities/auto-subtitle",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class BytedanceUpscalerUpscaleVideoTargetFps(str, Enum):
    """
    The target FPS of the video to upscale.
    """
    VALUE_30FPS = "30fps"
    VALUE_60FPS = "60fps"

class BytedanceUpscalerUpscaleVideoTargetResolution(str, Enum):
    """
    The target resolution of the video to upscale.
    """
    VALUE_1080P = "1080p"
    VALUE_2K = "2k"
    VALUE_4K = "4k"


class BytedanceUpscalerUpscaleVideo(FALNode):
    """
    Bytedance Upscaler
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to upscale."
    )
    target_fps: BytedanceUpscalerUpscaleVideoTargetFps = Field(
        default=BytedanceUpscalerUpscaleVideoTargetFps.VALUE_30FPS, description="The target FPS of the video to upscale."
    )
    target_resolution: BytedanceUpscalerUpscaleVideoTargetResolution = Field(
        default=BytedanceUpscalerUpscaleVideoTargetResolution.VALUE_1080P, description="The target resolution of the video to upscale."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "target_fps": self.target_fps.value,
            "target_resolution": self.target_resolution.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance-upscaler/upscale/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class VideoAsPromptAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"

class VideoAsPromptResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"


class VideoAsPrompt(FALNode):
    """
    Video As Prompt
    video, editing, video-to-video, vid2vid, professional

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    aspect_ratio: VideoAsPromptAspectRatio = Field(
        default=VideoAsPromptAspectRatio.RATIO_9_16, description="Aspect ratio of the generated video."
    )
    resolution: VideoAsPromptResolution = Field(
        default=VideoAsPromptResolution.VALUE_480P, description="Resolution of the generated video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="reference video to generate effect video from."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Input image to generate the effect video for."
    )
    fps: int = Field(
        default=16, description="Frames per second for the output video. Only applicable if output_type is 'video'."
    )
    video_description: str = Field(
        default="", description="A brief description of the input video content."
    )
    seed: str = Field(
        default="", description="Random seed for reproducible generation. If set none, a random seed will be used."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for generation."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_frames: int = Field(
        default=49, description="The number of frames to generate."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "video_url": video_url,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "fps": self.fps,
            "video_description": self.video_description,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/video-as-prompt",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class MireloAiSfxV15VideoToVideo(FALNode):
    """
    Mirelo SFX V1.5
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    num_samples: str = Field(
        default=2, description="The number of samples to generate from the model"
    )
    duration: str = Field(
        default=10, description="The duration of the generated audio in seconds"
    )
    start_offset: str = Field(
        default=0, description="The start offset in seconds to start the audio generation from"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="A video url that can accessed from the API to process and add sound effects"
    )
    seed: str = Field(
        default=8069, description="The seed to use for the generation. If not provided, a random seed will be used"
    )
    text_prompt: str = Field(
        default="", description="Additional description to guide the model"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "num_samples": self.num_samples,
            "duration": self.duration,
            "start_offset": self.start_offset,
            "video_url": video_url,
            "seed": self.seed,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="mirelo-ai/sfx-v1.5/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class KreaWan14BVideoToVideo(FALNode):
    """
    Krea Wan 14B
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Prompt for the video-to-video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video. Currently, only outputs of 16:9 aspect ratio and 480p resolution are supported. Video duration should be less than 1000 frames at 16fps, and output frames will be 6 plus a multiple of 12, for example 18, 30, 42, etc."
    )
    strength: float = Field(
        default=0.85, description="Denoising strength for the video-to-video generation. 0.0 preserves the original, 1.0 completely remakes the video."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )
    seed: str = Field(
        default="", description="Seed for the video-to-video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "strength": self.strength,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/krea-wan-14b/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Sora2VideoToVideoRemix(FALNode):
    """
    Sora 2
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Updated text prompt that directs the remix generation"
    )
    video_id: str = Field(
        default="", description="The video_id from a previous Sora 2 generation. Note: You can only remix videos that were generated by Sora (via text-to-video or image-to-video endpoints), not arbitrary uploaded videos."
    )
    delete_video: bool = Field(
        default=True, description="Whether to delete the video after generation for privacy reasons. If True, the video cannot be used for remixing and will be permanently deleted."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_id": self.video_id,
            "delete_video": self.delete_video,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sora-2/video-to-video/remix",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVaceAppsLongReframeAcceleration(str, Enum):
    """
    Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster.
    """
    NONE = "none"
    LOW = "low"
    REGULAR = "regular"

class WanVaceAppsLongReframeSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class WanVaceAppsLongReframeVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanVaceAppsLongReframeAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"

class WanVaceAppsLongReframeResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanVaceAppsLongReframeTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class WanVaceAppsLongReframeVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class WanVaceAppsLongReframeInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"


class WanVaceAppsLongReframe(FALNode):
    """
    Wan 2.1 VACE Long Reframe
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. This video will be used as a reference for the reframe task."
    )
    acceleration: WanVaceAppsLongReframeAcceleration = Field(
        default=WanVaceAppsLongReframeAcceleration.REGULAR, description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    paste_back: bool = Field(
        default=True, description="Whether to paste back the reframed scene to the original video."
    )
    zoom_factor: float = Field(
        default=0, description="Zoom factor for the video. When this value is greater than 0, the video will be zoomed in by this factor (in relation to the canvas size,) cutting off the edges of the video. A value of 0 means no zoom."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation. Optional for reframing."
    )
    scene_threshold: float = Field(
        default=30, description="Threshold for scene detection sensitivity (0-100). Lower values detect more scenes."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    auto_downsample_min_fps: float = Field(
        default=6, description="Minimum FPS for auto downsample."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    sampler: WanVaceAppsLongReframeSampler = Field(
        default=WanVaceAppsLongReframeSampler.UNIPC, description="Sampler to use for video generation."
    )
    video_write_mode: WanVaceAppsLongReframeVideoWriteMode = Field(
        default=WanVaceAppsLongReframeVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    aspect_ratio: WanVaceAppsLongReframeAspectRatio = Field(
        default=WanVaceAppsLongReframeAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    resolution: WanVaceAppsLongReframeResolution = Field(
        default=WanVaceAppsLongReframeResolution.AUTO, description="Resolution of the generated video."
    )
    transparency_mode: WanVaceAppsLongReframeTransparencyMode = Field(
        default=WanVaceAppsLongReframeTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    trim_borders: bool = Field(
        default=True, description="Whether to trim borders from the video."
    )
    video_quality: WanVaceAppsLongReframeVideoQuality = Field(
        default=WanVaceAppsLongReframeVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    interpolator_model: WanVaceAppsLongReframeInterpolatorModel = Field(
        default=WanVaceAppsLongReframeInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=True, description="Whether to enable auto downsample."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "shift": self.shift,
            "video_url": video_url,
            "acceleration": self.acceleration.value,
            "paste_back": self.paste_back,
            "zoom_factor": self.zoom_factor,
            "prompt": self.prompt,
            "scene_threshold": self.scene_threshold,
            "enable_safety_checker": self.enable_safety_checker,
            "guidance_scale": self.guidance_scale,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "negative_prompt": self.negative_prompt,
            "sampler": self.sampler.value,
            "video_write_mode": self.video_write_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "transparency_mode": self.transparency_mode.value,
            "return_frames_zip": self.return_frames_zip,
            "trim_borders": self.trim_borders,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "num_inference_steps": self.num_inference_steps,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-apps/long-reframe",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class InfinitalkVideoToVideoResolution(str, Enum):
    """
    Resolution of the video to generate. Must be either 480p or 720p.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"

class InfinitalkVideoToVideoAcceleration(str, Enum):
    """
    The acceleration level to use for generation.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"


class InfinitalkVideoToVideo(FALNode):
    """
    Infinitalk
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: InfinitalkVideoToVideoResolution = Field(
        default=InfinitalkVideoToVideoResolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    acceleration: InfinitalkVideoToVideoAcceleration = Field(
        default=InfinitalkVideoToVideoAcceleration.REGULAR, description="The acceleration level to use for generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    num_frames: int = Field(
        default=145, description="Number of frames to generate. Must be between 81 to 129 (inclusive). If the number of frames is greater than 81, the video will be generated with 1.25x more billing units."
    )
    seed: int = Field(
        default=42, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "video_url": video_url,
            "audio_url": self.audio,
            "num_frames": self.num_frames,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/infinitalk/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class SeedvrUpscaleVideoUpscaleMode(str, Enum):
    """
    The mode to use for the upscale. If 'target', the upscale factor will be calculated based on the target resolution. If 'factor', the upscale factor will be used directly.
    """
    TARGET = "target"
    FACTOR = "factor"

class SeedvrUpscaleVideoOutputFormat(str, Enum):
    """
    The format of the output video.
    """
    X264_MP4 = "X264 (.mp4)"
    VP9_WEBM = "VP9 (.webm)"
    PRORES4444_MOV = "PRORES4444 (.mov)"
    GIF_GIF = "GIF (.gif)"

class SeedvrUpscaleVideoOutputWriteMode(str, Enum):
    """
    The write mode of the output video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class SeedvrUpscaleVideoTargetResolution(str, Enum):
    """
    The target resolution to upscale to when `upscale_mode` is `target`.
    """
    VALUE_720P = "720p"
    VALUE_1080P = "1080p"
    VALUE_1440P = "1440p"
    VALUE_2160P = "2160p"

class SeedvrUpscaleVideoOutputQuality(str, Enum):
    """
    The quality of the output video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class SeedvrUpscaleVideo(FALNode):
    """
    SeedVR2
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    upscale_mode: SeedvrUpscaleVideoUpscaleMode = Field(
        default=SeedvrUpscaleVideoUpscaleMode.FACTOR, description="The mode to use for the upscale. If 'target', the upscale factor will be calculated based on the target resolution. If 'factor', the upscale factor will be used directly."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The input video to be processed"
    )
    noise_scale: float = Field(
        default=0.1, description="The noise scale to use for the generation process."
    )
    output_format: SeedvrUpscaleVideoOutputFormat = Field(
        default=SeedvrUpscaleVideoOutputFormat.X264_MP4, description="The format of the output video."
    )
    output_write_mode: SeedvrUpscaleVideoOutputWriteMode = Field(
        default=SeedvrUpscaleVideoOutputWriteMode.BALANCED, description="The write mode of the output video."
    )
    target_resolution: SeedvrUpscaleVideoTargetResolution = Field(
        default=SeedvrUpscaleVideoTargetResolution.VALUE_1080P, description="The target resolution to upscale to when `upscale_mode` is `target`."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    output_quality: SeedvrUpscaleVideoOutputQuality = Field(
        default=SeedvrUpscaleVideoOutputQuality.HIGH, description="The quality of the output video."
    )
    upscale_factor: float = Field(
        default=2, description="Upscaling factor to be used. Will multiply the dimensions with this factor when `upscale_mode` is `factor`."
    )
    seed: str = Field(
        default="", description="The random seed used for the generation process."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "upscale_mode": self.upscale_mode.value,
            "video_url": video_url,
            "noise_scale": self.noise_scale,
            "output_format": self.output_format.value,
            "output_write_mode": self.output_write_mode.value,
            "target_resolution": self.target_resolution.value,
            "sync_mode": self.sync_mode,
            "output_quality": self.output_quality.value,
            "upscale_factor": self.upscale_factor,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/seedvr/upscale/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVaceAppsVideoEditResolution(str, Enum):
    """
    Resolution of the edited video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanVaceAppsVideoEditAcceleration(str, Enum):
    """
    Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster.
    """
    NONE = "none"
    LOW = "low"
    REGULAR = "regular"

class WanVaceAppsVideoEditAspectRatio(str, Enum):
    """
    Aspect ratio of the edited video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"

class WanVaceAppsVideoEditVideoType(str, Enum):
    """
    The type of video you're editing. Use 'general' for most videos, and 'human' for videos emphasizing human subjects and motions. The default value 'auto' means the model will guess based on the first frame of the video.
    """
    AUTO = "auto"
    GENERAL = "general"
    HUMAN = "human"


class WanVaceAppsVideoEdit(FALNode):
    """
    Wan VACE Video Edit
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Prompt to edit the video."
    )
    resolution: WanVaceAppsVideoEditResolution = Field(
        default=WanVaceAppsVideoEditResolution.AUTO, description="Resolution of the edited video."
    )
    acceleration: WanVaceAppsVideoEditAcceleration = Field(
        default=WanVaceAppsVideoEditAcceleration.REGULAR, description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    return_frames_zip: bool = Field(
        default=False, description="Whether to include a ZIP archive containing all generated frames."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    aspect_ratio: WanVaceAppsVideoEditAspectRatio = Field(
        default=WanVaceAppsVideoEditAspectRatio.AUTO, description="Aspect ratio of the edited video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    video_type: WanVaceAppsVideoEditVideoType = Field(
        default=WanVaceAppsVideoEditVideoType.AUTO, description="The type of video you're editing. Use 'general' for most videos, and 'human' for videos emphasizing human subjects and motions. The default value 'auto' means the model will guess based on the first frame of the video."
    )
    images: list[ImageRef] = Field(
        default=[], description="URLs of the input images to use as a reference for the generation."
    )
    enable_auto_downsample: bool = Field(
        default=True, description="Whether to enable automatic downsampling. If your video has a high frame rate or is long, enabling longer sequences to be generated. The video will be interpolated back to the original frame rate after generation."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        images_data_urls = []
        for image in self.images or []:
            if image.is_empty():
                continue
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "return_frames_zip": self.return_frames_zip,
            "video_url": video_url,
            "aspect_ratio": self.aspect_ratio.value,
            "enable_safety_checker": self.enable_safety_checker,
            "video_type": self.video_type.value,
            "image_urls": images_data_urls,
            "enable_auto_downsample": self.enable_auto_downsample,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-apps/video-edit",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanV2214bAnimateReplaceResolution(str, Enum):
    """
    Resolution of the generated video (480p, 580p, or 720p).
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanV2214bAnimateReplaceVideoWriteMode(str, Enum):
    """
    The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanV2214bAnimateReplaceVideoQuality(str, Enum):
    """
    The quality of the output video. Higher quality means better visual quality but larger file size.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class WanV2214bAnimateReplace(FALNode):
    """
    Wan-Animate Replace is a model that can integrate animated characters into reference videos, replacing the original character while preserving the scene’s lighting and color tone for seamless environmental integration.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    shift: float = Field(
        default=5, description="Shift value for the video. Must be between 1.0 and 10.0."
    )
    resolution: WanV2214bAnimateReplaceResolution = Field(
        default=WanV2214bAnimateReplaceResolution.VALUE_480P, description="Resolution of the generated video (480p, 580p, or 720p)."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP archive containing per-frame images generated on GPU (lossless)."
    )
    video_write_mode: WanV2214bAnimateReplaceVideoWriteMode = Field(
        default=WanV2214bAnimateReplaceVideoWriteMode.BALANCED, description="The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    video_quality: WanV2214bAnimateReplaceVideoQuality = Field(
        default=WanV2214bAnimateReplaceVideoQuality.HIGH, description="The quality of the output video. Higher quality means better visual quality but larger file size."
    )
    guidance_scale: float = Field(
        default=1, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_inference_steps: int = Field(
        default=20, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    use_turbo: bool = Field(
        default=False, description="If true, applies quality enhancement for faster generation with improved quality. When enabled, parameters are automatically optimized for best results."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "shift": self.shift,
            "resolution": self.resolution.value,
            "video_url": video_url,
            "return_frames_zip": self.return_frames_zip,
            "video_write_mode": self.video_write_mode.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "video_quality": self.video_quality.value,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "use_turbo": self.use_turbo,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-14b/animate/replace",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanV2214bAnimateMoveResolution(str, Enum):
    """
    Resolution of the generated video (480p, 580p, or 720p).
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanV2214bAnimateMoveVideoWriteMode(str, Enum):
    """
    The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanV2214bAnimateMoveVideoQuality(str, Enum):
    """
    The quality of the output video. Higher quality means better visual quality but larger file size.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class WanV2214bAnimateMove(FALNode):
    """
    Wan-Animate is a video model that generates high-fidelity character videos by replicating the expressions and movements of characters from reference videos.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    shift: float = Field(
        default=5, description="Shift value for the video. Must be between 1.0 and 10.0."
    )
    resolution: WanV2214bAnimateMoveResolution = Field(
        default=WanV2214bAnimateMoveResolution.VALUE_480P, description="Resolution of the generated video (480p, 580p, or 720p)."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP archive containing per-frame images generated on GPU (lossless)."
    )
    video_write_mode: WanV2214bAnimateMoveVideoWriteMode = Field(
        default=WanV2214bAnimateMoveVideoWriteMode.BALANCED, description="The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    video_quality: WanV2214bAnimateMoveVideoQuality = Field(
        default=WanV2214bAnimateMoveVideoQuality.HIGH, description="The quality of the output video. Higher quality means better visual quality but larger file size."
    )
    guidance_scale: float = Field(
        default=1, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_inference_steps: int = Field(
        default=20, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    use_turbo: bool = Field(
        default=False, description="If true, applies quality enhancement for faster generation with improved quality. When enabled, parameters are automatically optimized for best results."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "shift": self.shift,
            "resolution": self.resolution.value,
            "video_url": video_url,
            "return_frames_zip": self.return_frames_zip,
            "video_write_mode": self.video_write_mode.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "video_quality": self.video_quality.value,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "use_turbo": self.use_turbo,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-14b/animate/move",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class DecartLucyEditProResolution(str, Enum):
    """
    Resolution of the generated video
    """
    VALUE_720P = "720p"


class DecartLucyEditPro(FALNode):
    """
    Edit outfits, objects, faces, or restyle your video - all with maximum detail retention.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    sync_mode: bool = Field(
        default=True, description="If set to true, the function will wait for the video to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the video directly in the response without going through the CDN."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to edit"
    )
    prompt: str = Field(
        default="", description="Text description of the desired video content"
    )
    resolution: DecartLucyEditProResolution = Field(
        default=DecartLucyEditProResolution.VALUE_720P, description="Resolution of the generated video"
    )
    enhance_prompt: bool = Field(
        default=True, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "sync_mode": self.sync_mode,
            "video_url": video_url,
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="decart/lucy-edit/pro",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class DecartLucyEditDev(FALNode):
    """
    Edit outfits, objects, faces, or restyle your video - all with maximum detail retention.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    sync_mode: bool = Field(
        default=True, description="If set to true, the function will wait for the video to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the video directly in the response without going through the CDN."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to edit"
    )
    prompt: str = Field(
        default="", description="Text description of the desired video content"
    )
    enhance_prompt: bool = Field(
        default=True, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "sync_mode": self.sync_mode,
            "video_url": video_url,
            "prompt": self.prompt,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="decart/lucy-edit/dev",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Wan22VaceFunA14bReframeTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class Wan22VaceFunA14bReframeSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class Wan22VaceFunA14bReframeVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Wan22VaceFunA14bReframeInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class Wan22VaceFunA14bReframeAcceleration(str, Enum):
    """
    Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster.
    """
    NONE = "none"
    LOW = "low"
    REGULAR = "regular"

class Wan22VaceFunA14bReframeVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Wan22VaceFunA14bReframeResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class Wan22VaceFunA14bReframeAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class Wan22VaceFunA14bReframe(FALNode):
    """
    VACE Fun for Wan 2.2 A14B from Alibaba-PAI
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation. Optional for reframing."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. This video will be used as a reference for the reframe task."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    transparency_mode: Wan22VaceFunA14bReframeTransparencyMode = Field(
        default=Wan22VaceFunA14bReframeTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    sampler: Wan22VaceFunA14bReframeSampler = Field(
        default=Wan22VaceFunA14bReframeSampler.UNIPC, description="Sampler to use for video generation."
    )
    trim_borders: bool = Field(
        default=True, description="Whether to trim borders from the video."
    )
    video_quality: Wan22VaceFunA14bReframeVideoQuality = Field(
        default=Wan22VaceFunA14bReframeVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: Wan22VaceFunA14bReframeInterpolatorModel = Field(
        default=Wan22VaceFunA14bReframeInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    acceleration: Wan22VaceFunA14bReframeAcceleration = Field(
        default=Wan22VaceFunA14bReframeAcceleration.REGULAR, description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    zoom_factor: float = Field(
        default=0, description="Zoom factor for the video. When this value is greater than 0, the video will be zoomed in by this factor (in relation to the canvas size,) cutting off the edges of the video. A value of 0 means no zoom."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    match_input_num_frames: bool = Field(
        default=True, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: Wan22VaceFunA14bReframeVideoWriteMode = Field(
        default=Wan22VaceFunA14bReframeVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    resolution: Wan22VaceFunA14bReframeResolution = Field(
        default=Wan22VaceFunA14bReframeResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: Wan22VaceFunA14bReframeAspectRatio = Field(
        default=Wan22VaceFunA14bReframeAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    match_input_frames_per_second: bool = Field(
        default=True, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "transparency_mode": self.transparency_mode.value,
            "sampler": self.sampler.value,
            "trim_borders": self.trim_borders,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "shift": self.shift,
            "acceleration": self.acceleration.value,
            "zoom_factor": self.zoom_factor,
            "frames_per_second": self.frames_per_second,
            "match_input_num_frames": self.match_input_num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "return_frames_zip": self.return_frames_zip,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-22-vace-fun-a14b/reframe",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Wan22VaceFunA14bOutpaintingSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class Wan22VaceFunA14bOutpaintingTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class Wan22VaceFunA14bOutpaintingVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Wan22VaceFunA14bOutpaintingInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class Wan22VaceFunA14bOutpaintingAcceleration(str, Enum):
    """
    Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster.
    """
    NONE = "none"
    LOW = "low"
    REGULAR = "regular"

class Wan22VaceFunA14bOutpaintingVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Wan22VaceFunA14bOutpaintingResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class Wan22VaceFunA14bOutpaintingAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class Wan22VaceFunA14bOutpainting(FALNode):
    """
    VACE Fun for Wan 2.2 A14B from Alibaba-PAI
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. Required for outpainting."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    ref_images: list[str] = Field(
        default=[], description="URLs to source reference image. If provided, the model will use this image as reference."
    )
    expand_ratio: float = Field(
        default=0.25, description="Amount of expansion. This is a float value between 0 and 1, where 0.25 adds 25% to the original video size on the specified sides."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    expand_bottom: bool = Field(
        default=False, description="Whether to expand the video to the bottom."
    )
    sampler: Wan22VaceFunA14bOutpaintingSampler = Field(
        default=Wan22VaceFunA14bOutpaintingSampler.UNIPC, description="Sampler to use for video generation."
    )
    transparency_mode: Wan22VaceFunA14bOutpaintingTransparencyMode = Field(
        default=Wan22VaceFunA14bOutpaintingTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    video_quality: Wan22VaceFunA14bOutpaintingVideoQuality = Field(
        default=Wan22VaceFunA14bOutpaintingVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: Wan22VaceFunA14bOutpaintingInterpolatorModel = Field(
        default=Wan22VaceFunA14bOutpaintingInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    expand_top: bool = Field(
        default=False, description="Whether to expand the video to the top."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    expand_left: bool = Field(
        default=False, description="Whether to expand the video to the left."
    )
    acceleration: Wan22VaceFunA14bOutpaintingAcceleration = Field(
        default=Wan22VaceFunA14bOutpaintingAcceleration.REGULAR, description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: Wan22VaceFunA14bOutpaintingVideoWriteMode = Field(
        default=Wan22VaceFunA14bOutpaintingVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    resolution: Wan22VaceFunA14bOutpaintingResolution = Field(
        default=Wan22VaceFunA14bOutpaintingResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: Wan22VaceFunA14bOutpaintingAspectRatio = Field(
        default=Wan22VaceFunA14bOutpaintingAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    expand_right: bool = Field(
        default=False, description="Whether to expand the video to the right."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "ref_image_urls": self.ref_images,
            "expand_ratio": self.expand_ratio,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "expand_bottom": self.expand_bottom,
            "sampler": self.sampler.value,
            "transparency_mode": self.transparency_mode.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "expand_top": self.expand_top,
            "shift": self.shift,
            "expand_left": self.expand_left,
            "acceleration": self.acceleration.value,
            "frames_per_second": self.frames_per_second,
            "match_input_num_frames": self.match_input_num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "expand_right": self.expand_right,
            "return_frames_zip": self.return_frames_zip,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-22-vace-fun-a14b/outpainting",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Wan22VaceFunA14bInpaintingTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class Wan22VaceFunA14bInpaintingSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class Wan22VaceFunA14bInpaintingVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Wan22VaceFunA14bInpaintingInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class Wan22VaceFunA14bInpaintingAcceleration(str, Enum):
    """
    Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster.
    """
    NONE = "none"
    LOW = "low"
    REGULAR = "regular"

class Wan22VaceFunA14bInpaintingVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Wan22VaceFunA14bInpaintingResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class Wan22VaceFunA14bInpaintingAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class Wan22VaceFunA14bInpainting(FALNode):
    """
    VACE Fun for Wan 2.2 A14B from Alibaba-PAI
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. Required for inpainting."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    ref_images: list[str] = Field(
        default=[], description="Urls to source reference image. If provided, the model will use this image as reference."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    transparency_mode: Wan22VaceFunA14bInpaintingTransparencyMode = Field(
        default=Wan22VaceFunA14bInpaintingTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    sampler: Wan22VaceFunA14bInpaintingSampler = Field(
        default=Wan22VaceFunA14bInpaintingSampler.UNIPC, description="Sampler to use for video generation."
    )
    video_quality: Wan22VaceFunA14bInpaintingVideoQuality = Field(
        default=Wan22VaceFunA14bInpaintingVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    mask_video: VideoRef = Field(
        default=VideoRef(), description="URL to the source mask file. Required for inpainting."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: Wan22VaceFunA14bInpaintingInterpolatorModel = Field(
        default=Wan22VaceFunA14bInpaintingInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the input video."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    acceleration: Wan22VaceFunA14bInpaintingAcceleration = Field(
        default=Wan22VaceFunA14bInpaintingAcceleration.REGULAR, description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    mask_image: ImageRef = Field(
        default=ImageRef(), description="URL to the guiding mask file. If provided, the model will use this mask as a reference to create masked video using salient mask tracking. Will be ignored if mask_video_url is provided."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: Wan22VaceFunA14bInpaintingVideoWriteMode = Field(
        default=Wan22VaceFunA14bInpaintingVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    resolution: Wan22VaceFunA14bInpaintingResolution = Field(
        default=Wan22VaceFunA14bInpaintingResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: Wan22VaceFunA14bInpaintingAspectRatio = Field(
        default=Wan22VaceFunA14bInpaintingAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        mask_video_url = (
            await self._upload_asset_to_fal(client, self.mask_video, context)
            if not self.mask_video.is_empty()
            else None
        )
        mask_image_base64 = (
            await context.image_to_base64(self.mask_image)
            if not self.mask_image.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "ref_image_urls": self.ref_images,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "transparency_mode": self.transparency_mode.value,
            "sampler": self.sampler.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "mask_video_url": mask_video_url,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "preprocess": self.preprocess,
            "enable_auto_downsample": self.enable_auto_downsample,
            "shift": self.shift,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "acceleration": self.acceleration.value,
            "mask_image_url": f"data:image/png;base64,{mask_image_base64}" if mask_image_base64 else None,
            "frames_per_second": self.frames_per_second,
            "match_input_num_frames": self.match_input_num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "return_frames_zip": self.return_frames_zip,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-22-vace-fun-a14b/inpainting",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Wan22VaceFunA14bDepthTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class Wan22VaceFunA14bDepthSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class Wan22VaceFunA14bDepthVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Wan22VaceFunA14bDepthInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class Wan22VaceFunA14bDepthAcceleration(str, Enum):
    """
    Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster.
    """
    NONE = "none"
    LOW = "low"
    REGULAR = "regular"

class Wan22VaceFunA14bDepthVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Wan22VaceFunA14bDepthResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class Wan22VaceFunA14bDepthAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class Wan22VaceFunA14bDepth(FALNode):
    """
    VACE Fun for Wan 2.2 A14B from Alibaba-PAI
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. Required for depth task."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    ref_images: list[str] = Field(
        default=[], description="URLs to source reference image. If provided, the model will use this image as reference."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    transparency_mode: Wan22VaceFunA14bDepthTransparencyMode = Field(
        default=Wan22VaceFunA14bDepthTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    sampler: Wan22VaceFunA14bDepthSampler = Field(
        default=Wan22VaceFunA14bDepthSampler.UNIPC, description="Sampler to use for video generation."
    )
    video_quality: Wan22VaceFunA14bDepthVideoQuality = Field(
        default=Wan22VaceFunA14bDepthVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: Wan22VaceFunA14bDepthInterpolatorModel = Field(
        default=Wan22VaceFunA14bDepthInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the input video."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    acceleration: Wan22VaceFunA14bDepthAcceleration = Field(
        default=Wan22VaceFunA14bDepthAcceleration.REGULAR, description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: Wan22VaceFunA14bDepthVideoWriteMode = Field(
        default=Wan22VaceFunA14bDepthVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    resolution: Wan22VaceFunA14bDepthResolution = Field(
        default=Wan22VaceFunA14bDepthResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: Wan22VaceFunA14bDepthAspectRatio = Field(
        default=Wan22VaceFunA14bDepthAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "ref_image_urls": self.ref_images,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "transparency_mode": self.transparency_mode.value,
            "sampler": self.sampler.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "preprocess": self.preprocess,
            "shift": self.shift,
            "acceleration": self.acceleration.value,
            "frames_per_second": self.frames_per_second,
            "match_input_num_frames": self.match_input_num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "return_frames_zip": self.return_frames_zip,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-22-vace-fun-a14b/depth",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Wan22VaceFunA14bPoseTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class Wan22VaceFunA14bPoseSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class Wan22VaceFunA14bPoseVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class Wan22VaceFunA14bPoseInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class Wan22VaceFunA14bPoseAcceleration(str, Enum):
    """
    Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster.
    """
    NONE = "none"
    LOW = "low"
    REGULAR = "regular"

class Wan22VaceFunA14bPoseVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class Wan22VaceFunA14bPoseResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class Wan22VaceFunA14bPoseAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class Wan22VaceFunA14bPose(FALNode):
    """
    VACE Fun for Wan 2.2 A14B from Alibaba-PAI
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation. For pose task, the prompt should describe the desired pose and action of the subject in the video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. Required for pose task."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    ref_images: list[str] = Field(
        default=[], description="URLs to source reference image. If provided, the model will use this image as reference."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    transparency_mode: Wan22VaceFunA14bPoseTransparencyMode = Field(
        default=Wan22VaceFunA14bPoseTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    sampler: Wan22VaceFunA14bPoseSampler = Field(
        default=Wan22VaceFunA14bPoseSampler.UNIPC, description="Sampler to use for video generation."
    )
    video_quality: Wan22VaceFunA14bPoseVideoQuality = Field(
        default=Wan22VaceFunA14bPoseVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: Wan22VaceFunA14bPoseInterpolatorModel = Field(
        default=Wan22VaceFunA14bPoseInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the input video."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    acceleration: Wan22VaceFunA14bPoseAcceleration = Field(
        default=Wan22VaceFunA14bPoseAcceleration.REGULAR, description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: Wan22VaceFunA14bPoseVideoWriteMode = Field(
        default=Wan22VaceFunA14bPoseVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    resolution: Wan22VaceFunA14bPoseResolution = Field(
        default=Wan22VaceFunA14bPoseResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: Wan22VaceFunA14bPoseAspectRatio = Field(
        default=Wan22VaceFunA14bPoseAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "ref_image_urls": self.ref_images,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "transparency_mode": self.transparency_mode.value,
            "sampler": self.sampler.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "preprocess": self.preprocess,
            "shift": self.shift,
            "acceleration": self.acceleration.value,
            "frames_per_second": self.frames_per_second,
            "match_input_num_frames": self.match_input_num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "return_frames_zip": self.return_frames_zip,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-22-vace-fun-a14b/pose",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class HunyuanVideoFoley(FALNode):
    """
    Use the capabilities of the hunyuan foley model to bring life to your videos by adding sound effect to them.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate audio for."
    )
    guidance_scale: float = Field(
        default=4.5, description="Guidance scale for audio generation."
    )
    num_inference_steps: int = Field(
        default=50, description="Number of inference steps for generation."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )
    negative_prompt: str = Field(
        default="noisy, harsh", description="Negative prompt to avoid certain audio characteristics."
    )
    text_prompt: str = Field(
        default="", description="Text description of the desired audio (optional)."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video-foley",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class SyncLipsyncV2ProSyncMode(str, Enum):
    """
    Lipsync mode when audio and video durations are out of sync.
    """
    CUT_OFF = "cut_off"
    LOOP = "loop"
    BOUNCE = "bounce"
    SILENCE = "silence"
    REMAP = "remap"


class SyncLipsyncV2Pro(FALNode):
    """
    Generate high-quality realistic lipsync animations from audio while preserving unique details like natural teeth and unique facial features using the state-of-the-art Sync Lipsync 2 Pro model.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    sync_mode: SyncLipsyncV2ProSyncMode = Field(
        default=SyncLipsyncV2ProSyncMode.CUT_OFF, description="Lipsync mode when audio and video durations are out of sync."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video"
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the input audio"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "sync_mode": self.sync_mode.value,
            "video_url": video_url,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sync-lipsync/v2/pro",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanFunControlPreprocessType(str, Enum):
    """
    The type of preprocess to apply to the video. Only used when preprocess_video is True.
    """
    DEPTH = "depth"
    POSE = "pose"


class WanFunControl(FALNode):
    """
    Generate pose or depth controlled video using Alibaba-PAI's Wan 2.2 Fun
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video."
    )
    shift: float = Field(
        default=5, description="The shift for the scheduler."
    )
    preprocess_video: bool = Field(
        default=False, description="Whether to preprocess the video. If True, the video will be preprocessed to depth or pose."
    )
    reference_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the reference image to use as a reference for the video generation."
    )
    fps: int = Field(
        default=16, description="The fps to generate. Only used when match_input_fps is False."
    )
    match_input_num_frames: bool = Field(
        default=True, description="Whether to match the number of frames in the input video."
    )
    guidance_scale: float = Field(
        default=6, description="The guidance scale."
    )
    preprocess_type: WanFunControlPreprocessType = Field(
        default=WanFunControlPreprocessType.DEPTH, description="The type of preprocess to apply to the video. Only used when preprocess_video is True."
    )
    control_video: VideoRef = Field(
        default=VideoRef(), description="The URL of the control video to use as a reference for the video generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate the video."
    )
    num_frames: int = Field(
        default=81, description="The number of frames to generate. Only used when match_input_num_frames is False."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    num_inference_steps: int = Field(
        default=27, description="The number of inference steps."
    )
    match_input_fps: bool = Field(
        default=True, description="Whether to match the fps in the input video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        reference_image_base64 = (
            await context.image_to_base64(self.reference_image)
            if not self.reference_image.is_empty()
            else None
        )
        control_video_url = (
            await self._upload_asset_to_fal(client, self.control_video, context)
            if not self.control_video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "shift": self.shift,
            "preprocess_video": self.preprocess_video,
            "reference_image_url": f"data:image/png;base64,{reference_image_base64}" if reference_image_base64 else None,
            "fps": self.fps,
            "match_input_num_frames": self.match_input_num_frames,
            "guidance_scale": self.guidance_scale,
            "preprocess_type": self.preprocess_type.value,
            "control_video_url": control_video_url,
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "match_input_fps": self.match_input_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-fun-control",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class BriaVideoIncreaseResolutionOutputContainerAndCodec(str, Enum):
    """
    Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, mov_h265, mov_proresks, mkv_h265, mkv_h264, mkv_vp9, gif.
    """
    MP4_H265 = "mp4_h265"
    MP4_H264 = "mp4_h264"
    WEBM_VP9 = "webm_vp9"
    MOV_H265 = "mov_h265"
    MOV_PRORESKS = "mov_proresks"
    MKV_H265 = "mkv_h265"
    MKV_H264 = "mkv_h264"
    MKV_VP9 = "mkv_vp9"
    GIF = "gif"

class BriaVideoIncreaseResolutionDesiredIncrease(str, Enum):
    """
    desired_increase factor. Options: 2x, 4x.
    """
    VALUE_2 = "2"
    VALUE_4 = "4"


class BriaVideoIncreaseResolution(FALNode):
    """
    Upscale videos up to 8K output resolution. Trained on fully licensed and commercially safe data.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="Input video to increase resolution. Size should be less than 7680,4320 and duration less than 30s."
    )
    output_container_and_codec: BriaVideoIncreaseResolutionOutputContainerAndCodec = Field(
        default=BriaVideoIncreaseResolutionOutputContainerAndCodec.WEBM_VP9, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, mov_h265, mov_proresks, mkv_h265, mkv_h264, mkv_vp9, gif."
    )
    desired_increase: BriaVideoIncreaseResolutionDesiredIncrease = Field(
        default=BriaVideoIncreaseResolutionDesiredIncrease.VALUE_2, description="desired_increase factor. Options: 2x, 4x."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "output_container_and_codec": self.output_container_and_codec.value,
            "desired_increase": self.desired_increase.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/video/increase-resolution",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class InfinitalkResolution(str, Enum):
    """
    Resolution of the video to generate. Must be either 480p or 720p.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"

class InfinitalkAcceleration(str, Enum):
    """
    The acceleration level to use for generation.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"


class Infinitalk(FALNode):
    """
    Infinitalk model generates a talking avatar video from an image and audio file. The avatar lip-syncs to the provided audio with natural facial expressions.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: InfinitalkResolution = Field(
        default=InfinitalkResolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    acceleration: InfinitalkAcceleration = Field(
        default=InfinitalkAcceleration.REGULAR, description="The acceleration level to use for generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    num_frames: int = Field(
        default=145, description="Number of frames to generate. Must be between 41 to 721."
    )
    seed: int = Field(
        default=42, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "audio_url": self.audio,
            "num_frames": self.num_frames,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/infinitalk",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class MireloAiSfxV1VideoToVideo(FALNode):
    """
    Generate synced sounds for any video, and return it with its new sound track (like MMAudio) 
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    num_samples: str = Field(
        default=2, description="The number of samples to generate from the model"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="A video url that can accessed from the API to process and add sound effects"
    )
    duration: str = Field(
        default=10, description="The duration of the generated audio in seconds"
    )
    seed: str = Field(
        default=2105, description="The seed to use for the generation. If not provided, a random seed will be used"
    )
    text_prompt: str = Field(
        default="", description="Additional description to guide the model"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "num_samples": self.num_samples,
            "video_url": video_url,
            "duration": self.duration,
            "seed": self.seed,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="mirelo-ai/sfx-v1/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class MoonvalleyMareyPoseTransfer(FALNode):
    """
    Ideal for matching human movement. Your input video determines human poses, gestures, and body movements that will appear in the generated video.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate a video from"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use as the control video."
    )
    seed: str = Field(
        default=-1, description="Seed for random number generation. Use -1 for random seed each run."
    )
    reference_image: ImageRef = Field(
        default=ImageRef(), description="Optional reference image URL to use for pose control or as a starting frame"
    )
    negative_prompt: str = Field(
        default="<synthetic> <scene cut> low-poly, flat shader, bad rigging, stiff animation, uncanny eyes, low-quality textures, looping glitch, cheap effect, overbloom, bloom spam, default lighting, game asset, stiff face, ugly specular, AI artifacts", description="Negative prompt used to guide the model away from undesirable features."
    )
    first_frame_image: ImageRef = Field(
        default=ImageRef(), description="Optional first frame image URL to use as the first frame of the generated video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        reference_image_base64 = (
            await context.image_to_base64(self.reference_image)
            if not self.reference_image.is_empty()
            else None
        )
        first_frame_image_base64 = (
            await context.image_to_base64(self.first_frame_image)
            if not self.first_frame_image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "seed": self.seed,
            "reference_image_url": f"data:image/png;base64,{reference_image_base64}" if reference_image_base64 else None,
            "negative_prompt": self.negative_prompt,
            "first_frame_image_url": f"data:image/png;base64,{first_frame_image_base64}" if first_frame_image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="moonvalley/marey/pose-transfer",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class MoonvalleyMareyMotionTransfer(FALNode):
    """
    Pull motion from a reference video and apply it to new subjects or scenes.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate a video from"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use as the control video."
    )
    seed: str = Field(
        default=-1, description="Seed for random number generation. Use -1 for random seed each run."
    )
    reference_image: ImageRef = Field(
        default=ImageRef(), description="Optional reference image URL to use for pose control or as a starting frame"
    )
    negative_prompt: str = Field(
        default="<synthetic> <scene cut> low-poly, flat shader, bad rigging, stiff animation, uncanny eyes, low-quality textures, looping glitch, cheap effect, overbloom, bloom spam, default lighting, game asset, stiff face, ugly specular, AI artifacts", description="Negative prompt used to guide the model away from undesirable features."
    )
    first_frame_image: ImageRef = Field(
        default=ImageRef(), description="Optional first frame image URL to use as the first frame of the generated video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        reference_image_base64 = (
            await context.image_to_base64(self.reference_image)
            if not self.reference_image.is_empty()
            else None
        )
        first_frame_image_base64 = (
            await context.image_to_base64(self.first_frame_image)
            if not self.first_frame_image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "seed": self.seed,
            "reference_image_url": f"data:image/png;base64,{reference_image_base64}" if reference_image_base64 else None,
            "negative_prompt": self.negative_prompt,
            "first_frame_image_url": f"data:image/png;base64,{first_frame_image_base64}" if first_frame_image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="moonvalley/marey/motion-transfer",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class FfmpegApiMergeVideos(FALNode):
    """
    Use ffmpeg capabilities to merge 2 or more videos.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    resolution: str = Field(
        default="", description="Resolution of the final video. Width and height must be between 512 and 2048."
    )
    videos: list[str] = Field(
        default=[], description="List of video URLs to merge in order"
    )
    target_fps: str = Field(
        default="", description="Target FPS for the output video. If not provided, uses the lowest FPS from input videos."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "resolution": self.resolution,
            "video_urls": self.videos,
            "target_fps": self.target_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ffmpeg-api/merge-videos",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanV22A14bVideoToVideoAcceleration(str, Enum):
    """
    Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'.
    """
    NONE = "none"
    REGULAR = "regular"

class WanV22A14bVideoToVideoVideoWriteMode(str, Enum):
    """
    The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanV22A14bVideoToVideoAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"

class WanV22A14bVideoToVideoResolution(str, Enum):
    """
    Resolution of the generated video (480p, 580p, or 720p).
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanV22A14bVideoToVideoVideoQuality(str, Enum):
    """
    The quality of the output video. Higher quality means better visual quality but larger file size.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class WanV22A14bVideoToVideoInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. If None, no interpolation is applied.
    """
    NONE = "none"
    FILM = "film"
    RIFE = "rife"


class WanV22A14bVideoToVideo(FALNode):
    """
    Wan-2.2 video-to-video is a video model that generates high-quality videos with high visual quality and motion diversity from text prompts and source videos.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    shift: float = Field(
        default=5, description="Shift value for the video. Must be between 1.0 and 10.0."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    num_interpolated_frames: int = Field(
        default=1, description="Number of frames to interpolate between each pair of generated frames. Must be between 0 and 4."
    )
    acceleration: WanV22A14bVideoToVideoAcceleration = Field(
        default=WanV22A14bVideoToVideoAcceleration.REGULAR, description="Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resample_fps: bool = Field(
        default=False, description="If true, the video will be resampled to the passed frames per second. If false, the video will not be resampled."
    )
    frames_per_second: str = Field(
        default=16, description="Frames per second of the generated video. Must be between 4 to 60. When using interpolation and `adjust_fps_for_interpolation` is set to true (default true,) the final FPS will be multiplied by the number of interpolated frames plus one. For example, if the generated frames per second is 16 and the number of interpolated frames is 1, the final frames per second will be 32. If `adjust_fps_for_interpolation` is set to false, this value will be used as-is."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 17 to 161 (inclusive)."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    video_write_mode: WanV22A14bVideoToVideoVideoWriteMode = Field(
        default=WanV22A14bVideoToVideoVideoWriteMode.BALANCED, description="The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size."
    )
    aspect_ratio: WanV22A14bVideoToVideoAspectRatio = Field(
        default=WanV22A14bVideoToVideoAspectRatio.AUTO, description="Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input video."
    )
    resolution: WanV22A14bVideoToVideoResolution = Field(
        default=WanV22A14bVideoToVideoResolution.VALUE_720P, description="Resolution of the generated video (480p, 580p, or 720p)."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    guidance_scale_2: float = Field(
        default=4, description="Guidance scale for the second stage of the model. This is used to control the adherence to the prompt in the second stage of the model."
    )
    video_quality: WanV22A14bVideoToVideoVideoQuality = Field(
        default=WanV22A14bVideoToVideoVideoQuality.HIGH, description="The quality of the output video. Higher quality means better visual quality but larger file size."
    )
    strength: float = Field(
        default=0.9, description="Strength of the video transformation. A value of 1.0 means the output will be completely based on the prompt, while a value of 0.0 means the output will be identical to the input video."
    )
    adjust_fps_for_interpolation: bool = Field(
        default=True, description="If true, the number of frames per second will be multiplied by the number of interpolated frames plus one. For example, if the generated frames per second is 16 and the number of interpolated frames is 1, the final frames per second will be 32. If false, the passed frames per second will be used as-is."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: WanV22A14bVideoToVideoInterpolatorModel = Field(
        default=WanV22A14bVideoToVideoInterpolatorModel.FILM, description="The model to use for frame interpolation. If None, no interpolation is applied."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )
    num_inference_steps: int = Field(
        default=27, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "shift": self.shift,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "acceleration": self.acceleration.value,
            "prompt": self.prompt,
            "resample_fps": self.resample_fps,
            "frames_per_second": self.frames_per_second,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "guidance_scale_2": self.guidance_scale_2,
            "video_quality": self.video_quality.value,
            "strength": self.strength,
            "adjust_fps_for_interpolation": self.adjust_fps_for_interpolation,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-a14b/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Ltxv13b098DistilledExtendResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"

class Ltxv13b098DistilledExtendAspectRatio(str, Enum):
    """
    The aspect ratio of the video.
    """
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_16_9 = "16:9"
    AUTO = "auto"


class Ltxv13b098DistilledExtend(FALNode):
    """
    Extend videos using LTX Video-0.9.8 13B Distilled and custom LoRA
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    second_pass_skip_initial_steps: int = Field(
        default=5, description="The number of inference steps to skip in the initial steps of the second pass. By skipping some steps at the beginning, the second pass can focus on smaller details instead of larger changes."
    )
    first_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the first pass."
    )
    frame_rate: int = Field(
        default=24, description="The frame rate of the video."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video."
    )
    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    expand_prompt: bool = Field(
        default=False, description="Whether to expand the prompt using a language model."
    )
    temporal_adain_factor: float = Field(
        default=0.5, description="The factor for adaptive instance normalization (AdaIN) applied to generated video chunks after the first. This can help deal with a gradual increase in saturation/contrast in the generated video by normalizing the color distribution across the video. A high value will ensure the color distribution is more consistent across the video, while a low value will allow for more variation in color distribution."
    )
    loras: list[LoRAWeight] = Field(
        default=[], description="LoRA weights to use for generation"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    num_frames: int = Field(
        default=121, description="The number of frames in the video."
    )
    second_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the second pass."
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Video to be extended."
    )
    start_frame_num: int = Field(
        default=0, description="Frame number of the video from which the conditioning starts. Must be a multiple of 8."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video. This is useful for tasks where the video conditioning should be applied in reverse order."
    )
    limit_num_frames: bool = Field(
        default=False, description="Whether to limit the number of frames used from the video. If True, the `max_num_frames` parameter will be used to limit the number of frames."
    )
    resample_fps: bool = Field(
        default=False, description="Whether to resample the video to a specific FPS. If True, the `target_fps` parameter will be used to resample the video."
    )
    strength: float = Field(
        default=0.0, description="Strength of the conditioning. 0.0 means no conditioning, 1.0 means full conditioning."
    )
    target_fps: int = Field(
        default=0, description="Target FPS to resample the video to. Only relevant if `resample_fps` is True."
    )
    max_num_frames: int = Field(
        default=0, description="Maximum number of frames to use from the video. If None, all frames will be used."
    )
    enable_detail_pass: bool = Field(
        default=False, description="Whether to use a detail pass. If True, the model will perform a second pass to refine the video and enhance details. This incurs a 2.0x cost multiplier on the base price."
    )
    resolution: Ltxv13b098DistilledExtendResolution = Field(
        default=Ltxv13b098DistilledExtendResolution.VALUE_720P, description="Resolution of the generated video."
    )
    aspect_ratio: Ltxv13b098DistilledExtendAspectRatio = Field(
        default=Ltxv13b098DistilledExtendAspectRatio.AUTO, description="The aspect ratio of the video."
    )
    tone_map_compression_ratio: float = Field(
        default=0, description="The compression ratio for tone mapping. This is used to compress the dynamic range of the video to improve visual quality. A value of 0.0 means no compression, while a value of 1.0 means maximum compression."
    )
    constant_rate_factor: int = Field(
        default=29, description="The constant rate factor (CRF) to compress input media with. Compressed input media more closely matches the model's training data, which can improve motion quality."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "second_pass_skip_initial_steps": self.second_pass_skip_initial_steps,
            "first_pass_num_inference_steps": self.first_pass_num_inference_steps,
            "frame_rate": self.frame_rate,
            "prompt": self.prompt,
            "expand_prompt": self.expand_prompt,
            "temporal_adain_factor": self.temporal_adain_factor,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "video": {
                "video_url": video_url,
                "start_frame_num": self.start_frame_num,
                "reverse_video": self.reverse_video,
                "limit_num_frames": self.limit_num_frames,
                "resample_fps": self.resample_fps,
                "strength": self.strength,
                "target_fps": self.target_fps,
                "max_num_frames": self.max_num_frames,
            },
            "enable_detail_pass": self.enable_detail_pass,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "tone_map_compression_ratio": self.tone_map_compression_ratio,
            "constant_rate_factor": self.constant_rate_factor,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltxv-13b-098-distilled/extend",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class RifeVideo(FALNode):
    """
    Interpolate videos with RIFE - Real-Time Intermediate Flow Estimation
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use for interpolation."
    )
    use_scene_detection: bool = Field(
        default=False, description="If True, the input video will be split into scenes before interpolation. This removes smear frames between scenes, but can result in false positives if the scene detection is not accurate. If False, the entire video will be treated as a single scene."
    )
    loop: bool = Field(
        default=False, description="If True, the final frame will be looped back to the first frame to create a seamless loop. If False, the final frame will not loop back."
    )
    num_frames: int = Field(
        default=1, description="The number of frames to generate between the input video frames."
    )
    use_calculated_fps: bool = Field(
        default=True, description="If True, the function will use the calculated FPS of the input video multiplied by the number of frames to determine the output FPS. If False, the passed FPS will be used."
    )
    fps: int = Field(
        default=8, description="Frames per second for the output video. Only applicable if use_calculated_fps is False."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "use_scene_detection": self.use_scene_detection,
            "loop": self.loop,
            "num_frames": self.num_frames,
            "use_calculated_fps": self.use_calculated_fps,
            "fps": self.fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/rife/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class FilmVideoVideoWriteMode(str, Enum):
    """
    The write mode of the output video. Only applicable if output_type is 'video'.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class FilmVideoVideoQuality(str, Enum):
    """
    The quality of the output video. Only applicable if output_type is 'video'.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class FilmVideo(FALNode):
    """
    Interpolate videos with FILM - Frame Interpolation for Large Motion
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video_write_mode: FilmVideoVideoWriteMode = Field(
        default=FilmVideoVideoWriteMode.BALANCED, description="The write mode of the output video. Only applicable if output_type is 'video'."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use for interpolation."
    )
    use_calculated_fps: bool = Field(
        default=True, description="If True, the function will use the calculated FPS of the input video multiplied by the number of frames to determine the output FPS. If False, the passed FPS will be used."
    )
    loop: bool = Field(
        default=False, description="If True, the final frame will be looped back to the first frame to create a seamless loop. If False, the final frame will not loop back."
    )
    fps: int = Field(
        default=8, description="Frames per second for the output video. Only applicable if use_calculated_fps is False."
    )
    video_quality: FilmVideoVideoQuality = Field(
        default=FilmVideoVideoQuality.HIGH, description="The quality of the output video. Only applicable if output_type is 'video'."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    use_scene_detection: bool = Field(
        default=False, description="If True, the input video will be split into scenes before interpolation. This removes smear frames between scenes, but can result in false positives if the scene detection is not accurate. If False, the entire video will be treated as a single scene."
    )
    num_frames: int = Field(
        default=1, description="The number of frames to generate between the input video frames."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "video_url": video_url,
            "use_calculated_fps": self.use_calculated_fps,
            "loop": self.loop,
            "fps": self.fps,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "use_scene_detection": self.use_scene_detection,
            "num_frames": self.num_frames,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/film/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LumaDreamMachineRay2FlashModifyMode(str, Enum):
    """
    Amount of modification to apply to the video, adhere_1 is the least amount of modification, reimagine_3 is the most
    """
    ADHERE_1 = "adhere_1"
    ADHERE_2 = "adhere_2"
    ADHERE_3 = "adhere_3"
    FLEX_1 = "flex_1"
    FLEX_2 = "flex_2"
    FLEX_3 = "flex_3"
    REIMAGINE_1 = "reimagine_1"
    REIMAGINE_2 = "reimagine_2"
    REIMAGINE_3 = "reimagine_3"


class LumaDreamMachineRay2FlashModify(FALNode):
    """
    Ray2 Flash Modify is a video generative model capable of restyling or retexturing the entire shot, from turning live-action into CG or stylized animation, to changing wardrobe, props, or the overall aesthetic and swap environments or time periods, giving you control over background, location, or even weather.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Instruction for modifying the video"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video to modify"
    )
    mode: LumaDreamMachineRay2FlashModifyMode = Field(
        default=LumaDreamMachineRay2FlashModifyMode.FLEX_1, description="Amount of modification to apply to the video, adhere_1 is the least amount of modification, reimagine_3 is the most"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the first frame image for modification"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "mode": self.mode.value,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2-flash/modify",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Ltxv13b098DistilledMulticonditioningResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"

class Ltxv13b098DistilledMulticonditioningAspectRatio(str, Enum):
    """
    The aspect ratio of the video.
    """
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_16_9 = "16:9"
    AUTO = "auto"


class Ltxv13b098DistilledMulticonditioning(FALNode):
    """
    Generate long videos from prompts, images, and videos using LTX Video-0.9.8 13B Distilled and custom LoRA
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    second_pass_skip_initial_steps: int = Field(
        default=5, description="The number of inference steps to skip in the initial steps of the second pass. By skipping some steps at the beginning, the second pass can focus on smaller details instead of larger changes."
    )
    first_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the first pass."
    )
    frame_rate: int = Field(
        default=24, description="The frame rate of the video."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video."
    )
    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    expand_prompt: bool = Field(
        default=False, description="Whether to expand the prompt using a language model."
    )
    temporal_adain_factor: float = Field(
        default=0.5, description="The factor for adaptive instance normalization (AdaIN) applied to generated video chunks after the first. This can help deal with a gradual increase in saturation/contrast in the generated video by normalizing the color distribution across the video. A high value will ensure the color distribution is more consistent across the video, while a low value will allow for more variation in color distribution."
    )
    loras: list[LoRAWeight] = Field(
        default=[], description="LoRA weights to use for generation"
    )
    images: list[ImageConditioningInput] = Field(
        default=[], description="URL of images to use as conditioning"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    num_frames: int = Field(
        default=121, description="The number of frames in the video."
    )
    second_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the second pass."
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )
    enable_detail_pass: bool = Field(
        default=False, description="Whether to use a detail pass. If True, the model will perform a second pass to refine the video and enhance details. This incurs a 2.0x cost multiplier on the base price."
    )
    resolution: Ltxv13b098DistilledMulticonditioningResolution = Field(
        default=Ltxv13b098DistilledMulticonditioningResolution.VALUE_720P, description="Resolution of the generated video."
    )
    aspect_ratio: Ltxv13b098DistilledMulticonditioningAspectRatio = Field(
        default=Ltxv13b098DistilledMulticonditioningAspectRatio.AUTO, description="The aspect ratio of the video."
    )
    tone_map_compression_ratio: float = Field(
        default=0, description="The compression ratio for tone mapping. This is used to compress the dynamic range of the video to improve visual quality. A value of 0.0 means no compression, while a value of 1.0 means maximum compression."
    )
    videos: list[VideoConditioningInput] = Field(
        default=[], description="Videos to use as conditioning"
    )
    constant_rate_factor: int = Field(
        default=29, description="The constant rate factor (CRF) to compress input media with. Compressed input media more closely matches the model's training data, which can improve motion quality."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "second_pass_skip_initial_steps": self.second_pass_skip_initial_steps,
            "first_pass_num_inference_steps": self.first_pass_num_inference_steps,
            "frame_rate": self.frame_rate,
            "reverse_video": self.reverse_video,
            "prompt": self.prompt,
            "expand_prompt": self.expand_prompt,
            "temporal_adain_factor": self.temporal_adain_factor,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "images": [item.model_dump(exclude={"type"}) for item in self.images],
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "enable_detail_pass": self.enable_detail_pass,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "tone_map_compression_ratio": self.tone_map_compression_ratio,
            "videos": [item.model_dump(exclude={"type"}) for item in self.videos],
            "constant_rate_factor": self.constant_rate_factor,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltxv-13b-098-distilled/multiconditioning",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class PixverseSoundEffects(FALNode):
    """
    Add immersive sound effects and background music to your videos using PixVerse sound effects  generation
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Description of the sound effect to generate. If empty, a random sound effect will be generated"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video to add sound effects to"
    )
    original_sound_switch: bool = Field(
        default=False, description="Whether to keep the original audio from the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "original_sound_switch": self.original_sound_switch,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/sound-effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class ThinksoundAudio(FALNode):
    """
    Generate realistic audio from a video with an optional text prompt
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="A prompt to guide the audio generation. If not provided, it will be extracted from the video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the audio for."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator"
    )
    num_inference_steps: int = Field(
        default=24, description="The number of inference steps for audio generation."
    )
    cfg_scale: float = Field(
        default=5, description="The classifier-free guidance scale for audio generation."
    )

    async def process(self, context: ProcessingContext) -> AudioRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/thinksound/audio",
            arguments=arguments,
        )
        assert "audio" in res
        return AudioRef(uri=res["audio"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Thinksound(FALNode):
    """
    Generate realistic audio for a video with an optional text prompt and combine
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="A prompt to guide the audio generation. If not provided, it will be extracted from the video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the audio for."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator"
    )
    num_inference_steps: int = Field(
        default=24, description="The number of inference steps for audio generation."
    )
    cfg_scale: float = Field(
        default=5, description="The classifier-free guidance scale for audio generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/thinksound",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class PixverseExtendFastResolution(str, Enum):
    """
    The resolution of the generated video. Fast mode doesn't support 1080p
    """
    VALUE_360P = "360p"
    VALUE_540P = "540p"
    VALUE_720P = "720p"

class PixverseExtendFastStyle(str, Enum):
    """
    The style of the extended video
    """
    ANIME = "anime"
    ANIMATION_3D = "3d_animation"
    CLAY = "clay"
    COMIC = "comic"
    CYBERPUNK = "cyberpunk"

class PixverseExtendFastModel(str, Enum):
    """
    The model version to use for generation
    """
    V3_5 = "v3.5"
    V4 = "v4"
    V4_5 = "v4.5"
    V5 = "v5"
    V5_5 = "v5.5"
    V5_6 = "v5.6"


class PixverseExtendFast(FALNode):
    """
    PixVerse Extend model is a video extending tool for your videos using with high-quality video extending techniques 
    video, editing, video-to-video, vid2vid, fast

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Prompt describing how to extend the video"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video to extend"
    )
    resolution: PixverseExtendFastResolution = Field(
        default=PixverseExtendFastResolution.VALUE_720P, description="The resolution of the generated video. Fast mode doesn't support 1080p"
    )
    style: PixverseExtendFastStyle | None = Field(
        default=None, description="The style of the extended video"
    )
    model: PixverseExtendFastModel = Field(
        default=PixverseExtendFastModel.V4_5, description="The model version to use for generation"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "model": self.model.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/extend/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class PixverseExtendResolution(str, Enum):
    """
    The resolution of the generated video
    """
    VALUE_360P = "360p"
    VALUE_540P = "540p"
    VALUE_720P = "720p"
    VALUE_1080P = "1080p"

class PixverseExtendStyle(str, Enum):
    """
    The style of the extended video
    """
    ANIME = "anime"
    ANIMATION_3D = "3d_animation"
    CLAY = "clay"
    COMIC = "comic"
    CYBERPUNK = "cyberpunk"

class PixverseExtendDuration(str, Enum):
    """
    The duration of the generated video in seconds. 1080p videos are limited to 5 seconds
    """
    VALUE_5 = "5"
    VALUE_8 = "8"

class PixverseExtendModel(str, Enum):
    """
    The model version to use for generation
    """
    V3_5 = "v3.5"
    V4 = "v4"
    V4_5 = "v4.5"
    V5 = "v5"
    V5_5 = "v5.5"
    V5_6 = "v5.6"


class PixverseExtend(FALNode):
    """
    PixVerse Extend model is a video extending tool for your videos using with high-quality video extending techniques 
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Prompt describing how to extend the video"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video to extend"
    )
    resolution: PixverseExtendResolution = Field(
        default=PixverseExtendResolution.VALUE_720P, description="The resolution of the generated video"
    )
    style: PixverseExtendStyle | None = Field(
        default=None, description="The style of the extended video"
    )
    duration: PixverseExtendDuration = Field(
        default=PixverseExtendDuration.VALUE_5, description="The duration of the generated video in seconds. 1080p videos are limited to 5 seconds"
    )
    model: PixverseExtendModel = Field(
        default=PixverseExtendModel.V4_5, description="The model version to use for generation"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "duration": self.duration.value,
            "model": self.model.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/extend",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class PixverseLipsyncVoiceId(str, Enum):
    """
    Voice to use for TTS when audio_url is not provided
    """
    EMILY = "Emily"
    JAMES = "James"
    ISABELLA = "Isabella"
    LIAM = "Liam"
    CHLOE = "Chloe"
    ADRIAN = "Adrian"
    HARPER = "Harper"
    AVA = "Ava"
    SOPHIA = "Sophia"
    JULIA = "Julia"
    MASON = "Mason"
    JACK = "Jack"
    OLIVER = "Oliver"
    ETHAN = "Ethan"
    AUTO = "Auto"


class PixverseLipsync(FALNode):
    """
    Generate realistic lipsync animations from audio using advanced algorithms for high-quality synchronization with PixVerse Lipsync model
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    text: str = Field(
        default="", description="Text content for TTS when audio_url is not provided"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video"
    )
    voice_id: PixverseLipsyncVoiceId = Field(
        default=PixverseLipsyncVoiceId.AUTO, description="Voice to use for TTS when audio_url is not provided"
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the input audio. If not provided, TTS will be used."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "text": self.text,
            "video_url": video_url,
            "voice_id": self.voice_id.value,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/lipsync",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LumaDreamMachineRay2ModifyMode(str, Enum):
    """
    Amount of modification to apply to the video, adhere_1 is the least amount of modification, reimagine_3 is the most
    """
    ADHERE_1 = "adhere_1"
    ADHERE_2 = "adhere_2"
    ADHERE_3 = "adhere_3"
    FLEX_1 = "flex_1"
    FLEX_2 = "flex_2"
    FLEX_3 = "flex_3"
    REIMAGINE_1 = "reimagine_1"
    REIMAGINE_2 = "reimagine_2"
    REIMAGINE_3 = "reimagine_3"


class LumaDreamMachineRay2Modify(FALNode):
    """
    Ray2 Modify is a video generative model capable of restyling or retexturing the entire shot, from turning live-action into CG or stylized animation, to changing wardrobe, props, or the overall aesthetic and swap environments or time periods, giving you control over background, location, or even weather.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Instruction for modifying the video"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video to modify"
    )
    mode: LumaDreamMachineRay2ModifyMode = Field(
        default=LumaDreamMachineRay2ModifyMode.FLEX_1, description="Amount of modification to apply to the video, adhere_1 is the least amount of modification, reimagine_3 is the most"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the first frame image for modification"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "mode": self.mode.value,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2/modify",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVace14bReframeTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class WanVace14bReframeSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class WanVace14bReframeVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class WanVace14bReframeInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class WanVace14bReframeVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanVace14bReframeResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanVace14bReframeAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class WanVace14bReframe(FALNode):
    """
    VACE is a video generation model that uses a source image, mask, and video to create prompted videos with controllable sources.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation. Optional for reframing."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. This video will be used as a reference for the reframe task."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    transparency_mode: WanVace14bReframeTransparencyMode = Field(
        default=WanVace14bReframeTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    trim_borders: bool = Field(
        default=True, description="Whether to trim borders from the video."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    sampler: WanVace14bReframeSampler = Field(
        default=WanVace14bReframeSampler.UNIPC, description="Sampler to use for video generation."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    video_quality: WanVace14bReframeVideoQuality = Field(
        default=WanVace14bReframeVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: WanVace14bReframeInterpolatorModel = Field(
        default=WanVace14bReframeInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    acceleration: str = Field(
        default="regular", description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    zoom_factor: float = Field(
        default=0, description="Zoom factor for the video. When this value is greater than 0, the video will be zoomed in by this factor (in relation to the canvas size,) cutting off the edges of the video. A value of 0 means no zoom."
    )
    match_input_num_frames: bool = Field(
        default=True, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    frames_per_second: str = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: WanVace14bReframeVideoWriteMode = Field(
        default=WanVace14bReframeVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    resolution: WanVace14bReframeResolution = Field(
        default=WanVace14bReframeResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: WanVace14bReframeAspectRatio = Field(
        default=WanVace14bReframeAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    match_input_frames_per_second: bool = Field(
        default=True, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "transparency_mode": self.transparency_mode.value,
            "num_frames": self.num_frames,
            "trim_borders": self.trim_borders,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "sampler": self.sampler.value,
            "guidance_scale": self.guidance_scale,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "shift": self.shift,
            "acceleration": self.acceleration,
            "zoom_factor": self.zoom_factor,
            "match_input_num_frames": self.match_input_num_frames,
            "frames_per_second": self.frames_per_second,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "return_frames_zip": self.return_frames_zip,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-14b/reframe",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVace14bOutpaintingTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class WanVace14bOutpaintingSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class WanVace14bOutpaintingVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class WanVace14bOutpaintingInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class WanVace14bOutpaintingVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanVace14bOutpaintingResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanVace14bOutpaintingAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class WanVace14bOutpainting(FALNode):
    """
    VACE is a video generation model that uses a source image, mask, and video to create prompted videos with controllable sources.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. Required for outpainting."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    ref_images: list[str] = Field(
        default=[], description="URLs to source reference image. If provided, the model will use this image as reference."
    )
    expand_ratio: float = Field(
        default=0.25, description="Amount of expansion. This is a float value between 0 and 1, where 0.25 adds 25% to the original video size on the specified sides."
    )
    transparency_mode: WanVace14bOutpaintingTransparencyMode = Field(
        default=WanVace14bOutpaintingTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    expand_bottom: bool = Field(
        default=False, description="Whether to expand the video to the bottom."
    )
    sampler: WanVace14bOutpaintingSampler = Field(
        default=WanVace14bOutpaintingSampler.UNIPC, description="Sampler to use for video generation."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    video_quality: WanVace14bOutpaintingVideoQuality = Field(
        default=WanVace14bOutpaintingVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: WanVace14bOutpaintingInterpolatorModel = Field(
        default=WanVace14bOutpaintingInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    expand_top: bool = Field(
        default=False, description="Whether to expand the video to the top."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    expand_left: bool = Field(
        default=False, description="Whether to expand the video to the left."
    )
    acceleration: str = Field(
        default="regular", description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    frames_per_second: str = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: WanVace14bOutpaintingVideoWriteMode = Field(
        default=WanVace14bOutpaintingVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    resolution: WanVace14bOutpaintingResolution = Field(
        default=WanVace14bOutpaintingResolution.AUTO, description="Resolution of the generated video."
    )
    expand_right: bool = Field(
        default=False, description="Whether to expand the video to the right."
    )
    aspect_ratio: WanVace14bOutpaintingAspectRatio = Field(
        default=WanVace14bOutpaintingAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "ref_image_urls": self.ref_images,
            "expand_ratio": self.expand_ratio,
            "transparency_mode": self.transparency_mode.value,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "expand_bottom": self.expand_bottom,
            "sampler": self.sampler.value,
            "guidance_scale": self.guidance_scale,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "expand_top": self.expand_top,
            "shift": self.shift,
            "expand_left": self.expand_left,
            "acceleration": self.acceleration,
            "match_input_num_frames": self.match_input_num_frames,
            "frames_per_second": self.frames_per_second,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "return_frames_zip": self.return_frames_zip,
            "resolution": self.resolution.value,
            "expand_right": self.expand_right,
            "aspect_ratio": self.aspect_ratio.value,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-14b/outpainting",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVace14bInpaintingTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class WanVace14bInpaintingSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class WanVace14bInpaintingVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class WanVace14bInpaintingInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class WanVace14bInpaintingVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanVace14bInpaintingResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanVace14bInpaintingAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class WanVace14bInpainting(FALNode):
    """
    VACE is a video generation model that uses a source image, mask, and video to create prompted videos with controllable sources.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. Required for inpainting."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    ref_images: list[str] = Field(
        default=[], description="Urls to source reference image. If provided, the model will use this image as reference."
    )
    transparency_mode: WanVace14bInpaintingTransparencyMode = Field(
        default=WanVace14bInpaintingTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    sampler: WanVace14bInpaintingSampler = Field(
        default=WanVace14bInpaintingSampler.UNIPC, description="Sampler to use for video generation."
    )
    video_quality: WanVace14bInpaintingVideoQuality = Field(
        default=WanVace14bInpaintingVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    mask_video: VideoRef = Field(
        default=VideoRef(), description="URL to the source mask file. Required for inpainting."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: WanVace14bInpaintingInterpolatorModel = Field(
        default=WanVace14bInpaintingInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the input video."
    )
    acceleration: str = Field(
        default="regular", description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    mask_image: ImageRef = Field(
        default=ImageRef(), description="URL to the guiding mask file. If provided, the model will use this mask as a reference to create masked video using salient mask tracking. Will be ignored if mask_video_url is provided."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    frames_per_second: str = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: WanVace14bInpaintingVideoWriteMode = Field(
        default=WanVace14bInpaintingVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    resolution: WanVace14bInpaintingResolution = Field(
        default=WanVace14bInpaintingResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: WanVace14bInpaintingAspectRatio = Field(
        default=WanVace14bInpaintingAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        mask_video_url = (
            await self._upload_asset_to_fal(client, self.mask_video, context)
            if not self.mask_video.is_empty()
            else None
        )
        mask_image_base64 = (
            await context.image_to_base64(self.mask_image)
            if not self.mask_image.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "ref_image_urls": self.ref_images,
            "transparency_mode": self.transparency_mode.value,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "guidance_scale": self.guidance_scale,
            "sampler": self.sampler.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "mask_video_url": mask_video_url,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "shift": self.shift,
            "preprocess": self.preprocess,
            "acceleration": self.acceleration,
            "mask_image_url": f"data:image/png;base64,{mask_image_base64}" if mask_image_base64 else None,
            "match_input_num_frames": self.match_input_num_frames,
            "frames_per_second": self.frames_per_second,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "return_frames_zip": self.return_frames_zip,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-14b/inpainting",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVace14bPoseTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class WanVace14bPoseSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class WanVace14bPoseVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class WanVace14bPoseInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class WanVace14bPoseVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanVace14bPoseResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanVace14bPoseAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class WanVace14bPose(FALNode):
    """
    VACE is a video generation model that uses a source image, mask, and video to create prompted videos with controllable sources.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation. For pose task, the prompt should describe the desired pose and action of the subject in the video."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. Required for pose task."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    ref_images: list[str] = Field(
        default=[], description="URLs to source reference image. If provided, the model will use this image as reference."
    )
    transparency_mode: WanVace14bPoseTransparencyMode = Field(
        default=WanVace14bPoseTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    sampler: WanVace14bPoseSampler = Field(
        default=WanVace14bPoseSampler.UNIPC, description="Sampler to use for video generation."
    )
    video_quality: WanVace14bPoseVideoQuality = Field(
        default=WanVace14bPoseVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: WanVace14bPoseInterpolatorModel = Field(
        default=WanVace14bPoseInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the input video."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    acceleration: str = Field(
        default="regular", description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    frames_per_second: str = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: WanVace14bPoseVideoWriteMode = Field(
        default=WanVace14bPoseVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    resolution: WanVace14bPoseResolution = Field(
        default=WanVace14bPoseResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: WanVace14bPoseAspectRatio = Field(
        default=WanVace14bPoseAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "ref_image_urls": self.ref_images,
            "transparency_mode": self.transparency_mode.value,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "guidance_scale": self.guidance_scale,
            "sampler": self.sampler.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "preprocess": self.preprocess,
            "shift": self.shift,
            "acceleration": self.acceleration,
            "match_input_num_frames": self.match_input_num_frames,
            "frames_per_second": self.frames_per_second,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "return_frames_zip": self.return_frames_zip,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-14b/pose",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVace14bDepthTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class WanVace14bDepthSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class WanVace14bDepthVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class WanVace14bDepthInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class WanVace14bDepthVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanVace14bDepthResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanVace14bDepthAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class WanVace14bDepth(FALNode):
    """
    VACE is a video generation model that uses a source image, mask, and video to create prompted videos with controllable sources.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. Required for depth task."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    ref_images: list[str] = Field(
        default=[], description="URLs to source reference image. If provided, the model will use this image as reference."
    )
    transparency_mode: WanVace14bDepthTransparencyMode = Field(
        default=WanVace14bDepthTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    sampler: WanVace14bDepthSampler = Field(
        default=WanVace14bDepthSampler.UNIPC, description="Sampler to use for video generation."
    )
    video_quality: WanVace14bDepthVideoQuality = Field(
        default=WanVace14bDepthVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: WanVace14bDepthInterpolatorModel = Field(
        default=WanVace14bDepthInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the input video."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    acceleration: str = Field(
        default="regular", description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    frames_per_second: str = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: WanVace14bDepthVideoWriteMode = Field(
        default=WanVace14bDepthVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    resolution: WanVace14bDepthResolution = Field(
        default=WanVace14bDepthResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: WanVace14bDepthAspectRatio = Field(
        default=WanVace14bDepthAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "ref_image_urls": self.ref_images,
            "transparency_mode": self.transparency_mode.value,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "guidance_scale": self.guidance_scale,
            "sampler": self.sampler.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "preprocess": self.preprocess,
            "shift": self.shift,
            "acceleration": self.acceleration,
            "match_input_num_frames": self.match_input_num_frames,
            "frames_per_second": self.frames_per_second,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "return_frames_zip": self.return_frames_zip,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-14b/depth",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class DwposeVideoDrawMode(str, Enum):
    """
    Mode of drawing the pose on the video. Options are: 'full-pose', 'body-pose', 'face-pose', 'hand-pose', 'face-hand-mask', 'face-mask', 'hand-mask'.
    """
    FULL_POSE = "full-pose"
    BODY_POSE = "body-pose"
    FACE_POSE = "face-pose"
    HAND_POSE = "hand-pose"
    FACE_HAND_MASK = "face-hand-mask"
    FACE_MASK = "face-mask"
    HAND_MASK = "hand-mask"


class DwposeVideo(FALNode):
    """
    Predict poses from videos.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="URL of video to be used for pose estimation"
    )
    draw_mode: DwposeVideoDrawMode = Field(
        default=DwposeVideoDrawMode.BODY_POSE, description="Mode of drawing the pose on the video. Options are: 'full-pose', 'body-pose', 'face-pose', 'hand-pose', 'face-hand-mask', 'face-mask', 'hand-mask'."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "draw_mode": self.draw_mode.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/dwpose/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class FfmpegApiMergeAudioVideo(FALNode):
    """
    Merge videos with standalone audio files or audio from video files.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    start_offset: float = Field(
        default=0, description="Offset in seconds for when the audio should start relative to the video"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video file to use as the video track"
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio file to use as the audio track"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "start_offset": self.start_offset,
            "video_url": video_url,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ffmpeg-api/merge-audio-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVace13bTask(str, Enum):
    """
    Task type for the model.
    """
    DEPTH = "depth"
    INPAINTING = "inpainting"
    POSE = "pose"

class WanVace13bResolution(str, Enum):
    """
    Resolution of the generated video (480p,580p, or 720p).
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanVace13bAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video (16:9 or 9:16).
    """
    AUTO = "auto"
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"


class WanVace13b(FALNode):
    """
    Vace a video generation model that uses a source image, mask, and video to create prompted videos with controllable sources.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. If provided, the model will use this video as a reference."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    mask_image: ImageRef = Field(
        default=ImageRef(), description="URL to the guiding mask file. If provided, the model will use this mask as a reference to create masked video. If provided mask video url will be ignored."
    )
    task: WanVace13bTask = Field(
        default=WanVace13bTask.DEPTH, description="Task type for the model."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 24."
    )
    ref_images: list[str] = Field(
        default=[], description="Urls to source reference image. If provided, the model will use this image as reference."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 100 (inclusive). Works only with only reference images as input if source video or mask video is provided output len would be same as source up to 241 frames"
    )
    negative_prompt: str = Field(
        default="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    resolution: WanVace13bResolution = Field(
        default=WanVace13bResolution.VALUE_720P, description="Resolution of the generated video (480p,580p, or 720p)."
    )
    aspect_ratio: WanVace13bAspectRatio = Field(
        default=WanVace13bAspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
    )
    mask_video: VideoRef = Field(
        default=VideoRef(), description="URL to the source mask file. If provided, the model will use this mask as a reference."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the input video."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        mask_image_base64 = (
            await context.image_to_base64(self.mask_image)
            if not self.mask_image.is_empty()
            else None
        )
        mask_video_url = (
            await self._upload_asset_to_fal(client, self.mask_video, context)
            if not self.mask_video.is_empty()
            else None
        )
        arguments = {
            "shift": self.shift,
            "video_url": video_url,
            "prompt": self.prompt,
            "mask_image_url": f"data:image/png;base64,{mask_image_base64}" if mask_image_base64 else None,
            "task": self.task.value,
            "frames_per_second": self.frames_per_second,
            "ref_image_urls": self.ref_images,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "negative_prompt": self.negative_prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "mask_video_url": mask_video_url,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "preprocess": self.preprocess,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-1-3b",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LumaDreamMachineRay2FlashReframeAspectRatio(str, Enum):
    """
    The aspect ratio of the reframed video
    """
    RATIO_1_1 = "1:1"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_21_9 = "21:9"
    RATIO_9_21 = "9:21"


class LumaDreamMachineRay2FlashReframe(FALNode):
    """
    Adjust and enhance videos with Ray-2 Reframe. This advanced tool seamlessly reframes videos to your desired aspect ratio, intelligently inpainting missing regions to ensure realistic visuals and coherent motion, delivering exceptional quality and creative flexibility.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Optional prompt for reframing"
    )
    aspect_ratio: LumaDreamMachineRay2FlashReframeAspectRatio = Field(
        default="", description="The aspect ratio of the reframed video"
    )
    y_start: int = Field(
        default=0, description="Start Y coordinate for reframing"
    )
    x_end: int = Field(
        default=0, description="End X coordinate for reframing"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video to reframe"
    )
    y_end: int = Field(
        default=0, description="End Y coordinate for reframing"
    )
    x_start: int = Field(
        default=0, description="Start X coordinate for reframing"
    )
    grid_position_y: int = Field(
        default=0, description="Y position of the grid for reframing"
    )
    grid_position_x: int = Field(
        default=0, description="X position of the grid for reframing"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the first frame image for reframing"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "y_start": self.y_start,
            "x_end": self.x_end,
            "video_url": video_url,
            "y_end": self.y_end,
            "x_start": self.x_start,
            "grid_position_y": self.grid_position_y,
            "grid_position_x": self.grid_position_x,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2-flash/reframe",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LumaDreamMachineRay2ReframeAspectRatio(str, Enum):
    """
    The aspect ratio of the reframed video
    """
    RATIO_1_1 = "1:1"
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_21_9 = "21:9"
    RATIO_9_21 = "9:21"


class LumaDreamMachineRay2Reframe(FALNode):
    """
    Adjust and enhance videos with Ray-2 Reframe. This advanced tool seamlessly reframes videos to your desired aspect ratio, intelligently inpainting missing regions to ensure realistic visuals and coherent motion, delivering exceptional quality and creative flexibility.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Optional prompt for reframing"
    )
    aspect_ratio: LumaDreamMachineRay2ReframeAspectRatio = Field(
        default="", description="The aspect ratio of the reframed video"
    )
    y_start: int = Field(
        default=0, description="Start Y coordinate for reframing"
    )
    x_end: int = Field(
        default=0, description="End X coordinate for reframing"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video to reframe"
    )
    y_end: int = Field(
        default=0, description="End Y coordinate for reframing"
    )
    x_start: int = Field(
        default=0, description="Start X coordinate for reframing"
    )
    grid_position_y: int = Field(
        default=0, description="Y position of the grid for reframing"
    )
    grid_position_x: int = Field(
        default=0, description="X position of the grid for reframing"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the first frame image for reframing"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "y_start": self.y_start,
            "x_end": self.x_end,
            "video_url": video_url,
            "y_end": self.y_end,
            "x_start": self.x_start,
            "grid_position_y": self.grid_position_y,
            "grid_position_x": self.grid_position_x,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2/reframe",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class VeedLipsync(FALNode):
    """
    Generate realistic lipsync from any audio using VEED's latest model
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef()
    )
    audio: AudioRef = Field(
        default=AudioRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/lipsync",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVace14bTransparencyMode(str, Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"

class WanVace14bSampler(str, Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPM_PLUS_PLUS = "dpm++"
    EULER = "euler"

class WanVace14bVideoQuality(str, Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"

class WanVace14bInterpolatorModel(str, Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"

class WanVace14bTask(str, Enum):
    """
    Task type for the model.
    """
    DEPTH = "depth"
    POSE = "pose"
    INPAINTING = "inpainting"
    OUTPAINTING = "outpainting"
    REFRAME = "reframe"

class WanVace14bVideoWriteMode(str, Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"

class WanVace14bResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    AUTO = "auto"
    VALUE_240P = "240p"
    VALUE_360P = "360p"
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class WanVace14bAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video.
    """
    AUTO = "auto"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"


class WanVace14b(FALNode):
    """
    VACE is a video generation model that uses a source image, mask, and video to create prompted videos with controllable sources.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. If provided, the model will use this video as a reference."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between the original frames. A value of 0 means no interpolation."
    )
    temporal_downsample_factor: int = Field(
        default=0, description="Temporal downsample factor for the video. This is an integer value that determines how many frames to skip in the video. A value of 0 means no downsampling. For each downsample factor, one upsample factor will automatically be applied."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the first frame of the video. If provided, the model will use this frame as a reference."
    )
    ref_images: list[str] = Field(
        default=[], description="URLs to source reference image. If provided, the model will use this image as reference."
    )
    transparency_mode: WanVace14bTransparencyMode = Field(
        default=WanVace14bTransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 241 (inclusive)."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to. This is used to help determine the auto downsample factor to try and find the lowest detail-preserving downsample factor. The default value is appropriate for most videos, if you are using a video with very fast motion, you may need to increase this value. If your video has a very low amount of motion, you could decrease this value to allow for higher downsampling and thus longer sequences."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    sampler: WanVace14bSampler = Field(
        default=WanVace14bSampler.UNIPC, description="Sampler to use for video generation."
    )
    video_quality: WanVace14bVideoQuality = Field(
        default=WanVace14bVideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    mask_video: VideoRef = Field(
        default=VideoRef(), description="URL to the source mask file. If provided, the model will use this mask as a reference."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: WanVace14bInterpolatorModel = Field(
        default=WanVace14bInterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=False, description="If true, the model will automatically temporally downsample the video to an appropriate frame length for the model, then will interpolate it back to the original frame length."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the input video."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    acceleration: str = Field(
        default="regular", description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    mask_image: ImageRef = Field(
        default=ImageRef(), description="URL to the guiding mask file. If provided, the model will use this mask as a reference to create masked video. If provided mask video url will be ignored."
    )
    task: WanVace14bTask = Field(
        default=WanVace14bTask.DEPTH, description="Task type for the model."
    )
    match_input_num_frames: bool = Field(
        default=False, description="If true, the number of frames in the generated video will match the number of frames in the input video. If false, the number of frames will be determined by the num_frames parameter."
    )
    frames_per_second: str = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30. Ignored if match_input_frames_per_second is true."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    video_write_mode: WanVace14bVideoWriteMode = Field(
        default=WanVace14bVideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    resolution: WanVace14bResolution = Field(
        default=WanVace14bResolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: WanVace14bAspectRatio = Field(
        default=WanVace14bAspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    match_input_frames_per_second: bool = Field(
        default=False, description="If true, the frames per second of the generated video will match the input video. If false, the frames per second will be determined by the frames_per_second parameter."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL to the last frame of the video. If provided, the model will use this frame as a reference."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        first_frame_url = (
            await self._upload_asset_to_fal(client, self.first_frame, context)
            if not self.first_frame.is_empty()
            else None
        )
        mask_video_url = (
            await self._upload_asset_to_fal(client, self.mask_video, context)
            if not self.mask_video.is_empty()
            else None
        )
        mask_image_base64 = (
            await context.image_to_base64(self.mask_image)
            if not self.mask_image.is_empty()
            else None
        )
        last_frame_url = (
            await self._upload_asset_to_fal(client, self.last_frame, context)
            if not self.last_frame.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "num_interpolated_frames": self.num_interpolated_frames,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "first_frame_url": first_frame_url,
            "ref_image_urls": self.ref_images,
            "transparency_mode": self.transparency_mode.value,
            "num_frames": self.num_frames,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "guidance_scale": self.guidance_scale,
            "sampler": self.sampler.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "mask_video_url": mask_video_url,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "preprocess": self.preprocess,
            "shift": self.shift,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "acceleration": self.acceleration,
            "mask_image_url": f"data:image/png;base64,{mask_image_base64}" if mask_image_base64 else None,
            "task": self.task.value,
            "match_input_num_frames": self.match_input_num_frames,
            "frames_per_second": self.frames_per_second,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "return_frames_zip": self.return_frames_zip,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "match_input_frames_per_second": self.match_input_frames_per_second,
            "num_inference_steps": self.num_inference_steps,
            "last_frame_url": last_frame_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-14b",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LtxVideo13bDistilledExtendResolution(str, Enum):
    """
    Resolution of the generated video (480p or 720p).
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"

class LtxVideo13bDistilledExtendAspectRatio(str, Enum):
    """
    The aspect ratio of the video.
    """
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_16_9 = "16:9"
    AUTO = "auto"


class LtxVideo13bDistilledExtend(FALNode):
    """
    Extend videos using LTX Video-0.9.7 13B Distilled and custom LoRA
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    second_pass_skip_initial_steps: int = Field(
        default=5, description="The number of inference steps to skip in the initial steps of the second pass. By skipping some steps at the beginning, the second pass can focus on smaller details instead of larger changes."
    )
    first_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the first pass."
    )
    frame_rate: int = Field(
        default=30, description="The frame rate of the video."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video."
    )
    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    expand_prompt: bool = Field(
        default=False, description="Whether to expand the prompt using a language model."
    )
    loras: list[LoRAWeight] = Field(
        default=[], description="LoRA weights to use for generation"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    num_frames: int = Field(
        default=121, description="The number of frames in the video."
    )
    second_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the second pass."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Video to be extended."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video. This is useful for tasks where the video conditioning should be applied in reverse order."
    )
    start_frame_num: int = Field(
        default=0, description="Frame number of the video from which the conditioning starts. Must be a multiple of 8."
    )
    limit_num_frames: bool = Field(
        default=False, description="Whether to limit the number of frames used from the video. If True, the `max_num_frames` parameter will be used to limit the number of frames."
    )
    resample_fps: bool = Field(
        default=False, description="Whether to resample the video to a specific FPS. If True, the `target_fps` parameter will be used to resample the video."
    )
    strength: float = Field(
        default=0.0, description="Strength of the conditioning. 0.0 means no conditioning, 1.0 means full conditioning."
    )
    target_fps: int = Field(
        default=0, description="Target FPS to resample the video to. Only relevant if `resample_fps` is True."
    )
    max_num_frames: int = Field(
        default=0, description="Maximum number of frames to use from the video. If None, all frames will be used."
    )
    conditioning_type: str = Field(
        default="", description="Type of conditioning this video provides. This is relevant to ensure in-context LoRA weights are applied correctly, as well as selecting the correct preprocessing pipeline, when enabled."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the video. If True, the video will be preprocessed to match the conditioning type. This is a no-op for RGB conditioning."
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )
    resolution: LtxVideo13bDistilledExtendResolution = Field(
        default=LtxVideo13bDistilledExtendResolution.VALUE_720P, description="Resolution of the generated video (480p or 720p)."
    )
    aspect_ratio: LtxVideo13bDistilledExtendAspectRatio = Field(
        default=LtxVideo13bDistilledExtendAspectRatio.AUTO, description="The aspect ratio of the video."
    )
    constant_rate_factor: int = Field(
        default=35, description="The constant rate factor (CRF) to compress input media with. Compressed input media more closely matches the model's training data, which can improve motion quality."
    )
    first_pass_skip_final_steps: int = Field(
        default=1, description="Number of inference steps to skip in the final steps of the first pass. By skipping some steps at the end, the first pass can focus on larger changes instead of smaller details."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "second_pass_skip_initial_steps": self.second_pass_skip_initial_steps,
            "first_pass_num_inference_steps": self.first_pass_num_inference_steps,
            "frame_rate": self.frame_rate,
            "prompt": self.prompt,
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "video": {
                "video_url": video_url,
                "reverse_video": self.reverse_video,
                "start_frame_num": self.start_frame_num,
                "limit_num_frames": self.limit_num_frames,
                "resample_fps": self.resample_fps,
                "strength": self.strength,
                "target_fps": self.target_fps,
                "max_num_frames": self.max_num_frames,
                "conditioning_type": self.conditioning_type,
                "preprocess": self.preprocess,
            },
            "negative_prompt": self.negative_prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "constant_rate_factor": self.constant_rate_factor,
            "first_pass_skip_final_steps": self.first_pass_skip_final_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-13b-distilled/extend",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LtxVideo13bDistilledMulticonditioningResolution(str, Enum):
    """
    Resolution of the generated video (480p or 720p).
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"

class LtxVideo13bDistilledMulticonditioningAspectRatio(str, Enum):
    """
    The aspect ratio of the video.
    """
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_16_9 = "16:9"
    AUTO = "auto"


class LtxVideo13bDistilledMulticonditioning(FALNode):
    """
    Generate videos from prompts, images, and videos using LTX Video-0.9.7 13B Distilled and custom LoRA
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    second_pass_skip_initial_steps: int = Field(
        default=5, description="The number of inference steps to skip in the initial steps of the second pass. By skipping some steps at the beginning, the second pass can focus on smaller details instead of larger changes."
    )
    first_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the first pass."
    )
    frame_rate: int = Field(
        default=30, description="The frame rate of the video."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video."
    )
    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    expand_prompt: bool = Field(
        default=False, description="Whether to expand the prompt using a language model."
    )
    loras: list[LoRAWeight] = Field(
        default=[], description="LoRA weights to use for generation"
    )
    images: list[ImageConditioningInput] = Field(
        default=[], description="URL of images to use as conditioning"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    num_frames: int = Field(
        default=121, description="The number of frames in the video."
    )
    second_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the second pass."
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )
    resolution: LtxVideo13bDistilledMulticonditioningResolution = Field(
        default=LtxVideo13bDistilledMulticonditioningResolution.VALUE_720P, description="Resolution of the generated video (480p or 720p)."
    )
    aspect_ratio: LtxVideo13bDistilledMulticonditioningAspectRatio = Field(
        default=LtxVideo13bDistilledMulticonditioningAspectRatio.AUTO, description="The aspect ratio of the video."
    )
    constant_rate_factor: int = Field(
        default=35, description="The constant rate factor (CRF) to compress input media with. Compressed input media more closely matches the model's training data, which can improve motion quality."
    )
    videos: list[VideoConditioningInput] = Field(
        default=[], description="Videos to use as conditioning"
    )
    first_pass_skip_final_steps: int = Field(
        default=1, description="Number of inference steps to skip in the final steps of the first pass. By skipping some steps at the end, the first pass can focus on larger changes instead of smaller details."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "second_pass_skip_initial_steps": self.second_pass_skip_initial_steps,
            "first_pass_num_inference_steps": self.first_pass_num_inference_steps,
            "frame_rate": self.frame_rate,
            "reverse_video": self.reverse_video,
            "prompt": self.prompt,
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "images": [item.model_dump(exclude={"type"}) for item in self.images],
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "constant_rate_factor": self.constant_rate_factor,
            "videos": [item.model_dump(exclude={"type"}) for item in self.videos],
            "first_pass_skip_final_steps": self.first_pass_skip_final_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-13b-distilled/multiconditioning",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LtxVideo13bDevMulticonditioningAspectRatio(str, Enum):
    """
    The aspect ratio of the video.
    """
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_16_9 = "16:9"
    AUTO = "auto"

class LtxVideo13bDevMulticonditioningResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"


class LtxVideo13bDevMulticonditioning(FALNode):
    """
    Generate videos from prompts, images, and videos using LTX Video-0.9.7 13B and custom LoRA
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    second_pass_skip_initial_steps: int = Field(
        default=17, description="The number of inference steps to skip in the initial steps of the second pass. By skipping some steps at the beginning, the second pass can focus on smaller details instead of larger changes."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video."
    )
    enable_detail_pass_autoregression: bool = Field(
        default=True, description="Whether to use autoregression in the detail pass. If True, the model will use the previous frame as input for the next frame in the detail pass."
    )
    frame_rate: int = Field(
        default=24, description="The frame rate of the video."
    )
    temporal_adain_factor: float = Field(
        default=0.5, description="The factor for adaptive instance normalization (AdaIN) applied to generated video chunks after the first. This can help deal with a gradual increase in saturation/contrast in the generated video by normalizing the color distribution across the video. A high value will ensure the color distribution is more consistent across the video, while a low value will allow for more variation in color distribution."
    )
    first_pass_number_of_steps: str = Field(
        default="", description="Number of inference steps during the first pass. Deprecated. Use `first_pass_num_inference_steps` instead."
    )
    number_of_frames: str = Field(
        default="", description="The number of frames in the video. Deprecated. Use `num_frames` instead."
    )
    loras: list[LoRAWeight] = Field(
        default=[], description="LoRA weights to use for generation"
    )
    images: list[ImageConditioningInput] = Field(
        default=[], description="URL of images to use as conditioning"
    )
    second_pass_num_inference_steps: int = Field(
        default=30, description="Number of inference steps during the second pass."
    )
    num_frames: int = Field(
        default=121, description="The number of frames in the video."
    )
    guidance_scale: str = Field(
        default="", description="Deprecated, not used."
    )
    enable_detail_pass: bool = Field(
        default=False, description="Whether to use a detail pass. If True, the model will perform a second pass to refine the video and enhance details. This incurs a 2.0x cost multiplier on the base price."
    )
    tone_map_compression_ratio: float = Field(
        default=0, description="The compression ratio for tone mapping. This is used to compress the dynamic range of the video to improve visual quality. A value of 0.0 means no compression, while a value of 1.0 means maximum compression."
    )
    seed: str = Field(
        default="", description="Random seed for generation"
    )
    detail_pass_skip_initial_steps: int = Field(
        default=7, description="The number of inference steps to skip in the initial steps of the detail pass. By skipping some steps at the beginning, the detail pass can focus on smaller details instead of larger changes."
    )
    second_pass_number_of_steps: str = Field(
        default="", description="Number of inference steps during the second pass. Deprecated. Use `second_pass_num_inference_steps` instead."
    )
    detail_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the detail pass."
    )
    detail_pass_noise_scale: float = Field(
        default=0.125, description="The noise scale for the detail pass. This controls the amount of noise added to the generated video during the detail pass. A value of 0.0 means no noise, while a value of 1.0 means maximum noise."
    )
    first_pass_num_inference_steps: int = Field(
        default=30, description="Number of inference steps during the first pass."
    )
    expand_prompt: bool = Field(
        default=False, description="Whether to expand the prompt using a language model."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )
    aspect_ratio: LtxVideo13bDevMulticonditioningAspectRatio = Field(
        default=LtxVideo13bDevMulticonditioningAspectRatio.AUTO, description="The aspect ratio of the video."
    )
    resolution: LtxVideo13bDevMulticonditioningResolution = Field(
        default=LtxVideo13bDevMulticonditioningResolution.VALUE_720P, description="Resolution of the generated video."
    )
    videos: list[VideoConditioningInput] = Field(
        default=[], description="Videos to use as conditioning"
    )
    constant_rate_factor: int = Field(
        default=29, description="The constant rate factor (CRF) to compress input media with. Compressed input media more closely matches the model's training data, which can improve motion quality."
    )
    first_pass_skip_final_steps: str = Field(
        default="", description="Deprecated. No longer used."
    )
    num_inference_steps: str = Field(
        default="", description="Number of inference steps. Deprecated. Use `first_pass_num_inference_steps` instead."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "second_pass_skip_initial_steps": self.second_pass_skip_initial_steps,
            "reverse_video": self.reverse_video,
            "enable_detail_pass_autoregression": self.enable_detail_pass_autoregression,
            "frame_rate": self.frame_rate,
            "temporal_adain_factor": self.temporal_adain_factor,
            "first_pass_number_of_steps": self.first_pass_number_of_steps,
            "number_of_frames": self.number_of_frames,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "images": [item.model_dump(exclude={"type"}) for item in self.images],
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "num_frames": self.num_frames,
            "guidance_scale": self.guidance_scale,
            "enable_detail_pass": self.enable_detail_pass,
            "tone_map_compression_ratio": self.tone_map_compression_ratio,
            "seed": self.seed,
            "detail_pass_skip_initial_steps": self.detail_pass_skip_initial_steps,
            "second_pass_number_of_steps": self.second_pass_number_of_steps,
            "detail_pass_num_inference_steps": self.detail_pass_num_inference_steps,
            "detail_pass_noise_scale": self.detail_pass_noise_scale,
            "first_pass_num_inference_steps": self.first_pass_num_inference_steps,
            "expand_prompt": self.expand_prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "videos": [item.model_dump(exclude={"type"}) for item in self.videos],
            "constant_rate_factor": self.constant_rate_factor,
            "first_pass_skip_final_steps": self.first_pass_skip_final_steps,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-13b-dev/multiconditioning",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LtxVideo13bDevExtendAspectRatio(str, Enum):
    """
    The aspect ratio of the video.
    """
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_16_9 = "16:9"
    AUTO = "auto"

class LtxVideo13bDevExtendResolution(str, Enum):
    """
    Resolution of the generated video.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"


class LtxVideo13bDevExtend(FALNode):
    """
    Extend videos using LTX Video-0.9.7 13B and custom LoRA
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    second_pass_skip_initial_steps: int = Field(
        default=17, description="The number of inference steps to skip in the initial steps of the second pass. By skipping some steps at the beginning, the second pass can focus on smaller details instead of larger changes."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video."
    )
    enable_detail_pass_autoregression: bool = Field(
        default=True, description="Whether to use autoregression in the detail pass. If True, the model will use the previous frame as input for the next frame in the detail pass."
    )
    frame_rate: int = Field(
        default=24, description="The frame rate of the video."
    )
    temporal_adain_factor: float = Field(
        default=0.5, description="The factor for adaptive instance normalization (AdaIN) applied to generated video chunks after the first. This can help deal with a gradual increase in saturation/contrast in the generated video by normalizing the color distribution across the video. A high value will ensure the color distribution is more consistent across the video, while a low value will allow for more variation in color distribution."
    )
    first_pass_number_of_steps: str = Field(
        default="", description="Number of inference steps during the first pass. Deprecated. Use `first_pass_num_inference_steps` instead."
    )
    number_of_frames: str = Field(
        default="", description="The number of frames in the video. Deprecated. Use `num_frames` instead."
    )
    loras: list[LoRAWeight] = Field(
        default=[], description="LoRA weights to use for generation"
    )
    second_pass_num_inference_steps: int = Field(
        default=30, description="Number of inference steps during the second pass."
    )
    num_frames: int = Field(
        default=121, description="The number of frames in the video."
    )
    guidance_scale: str = Field(
        default="", description="Deprecated, not used."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Video to be extended."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video. This is useful for tasks where the video conditioning should be applied in reverse order."
    )
    start_frame_num: int = Field(
        default=0, description="Frame number of the video from which the conditioning starts. Must be a multiple of 8."
    )
    limit_num_frames: bool = Field(
        default=False, description="Whether to limit the number of frames used from the video. If True, the `max_num_frames` parameter will be used to limit the number of frames."
    )
    resample_fps: bool = Field(
        default=False, description="Whether to resample the video to a specific FPS. If True, the `target_fps` parameter will be used to resample the video."
    )
    strength: float = Field(
        default=0.0, description="Strength of the conditioning. 0.0 means no conditioning, 1.0 means full conditioning."
    )
    target_fps: int = Field(
        default=0, description="Target FPS to resample the video to. Only relevant if `resample_fps` is True."
    )
    max_num_frames: int = Field(
        default=0, description="Maximum number of frames to use from the video. If None, all frames will be used."
    )
    conditioning_type: str = Field(
        default="", description="Type of conditioning this video provides. This is relevant to ensure in-context LoRA weights are applied correctly, as well as selecting the correct preprocessing pipeline, when enabled."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the video. If True, the video will be preprocessed to match the conditioning type. This is a no-op for RGB conditioning."
    )
    enable_detail_pass: bool = Field(
        default=False, description="Whether to use a detail pass. If True, the model will perform a second pass to refine the video and enhance details. This incurs a 2.0x cost multiplier on the base price."
    )
    tone_map_compression_ratio: float = Field(
        default=0, description="The compression ratio for tone mapping. This is used to compress the dynamic range of the video to improve visual quality. A value of 0.0 means no compression, while a value of 1.0 means maximum compression."
    )
    seed: str = Field(
        default="", description="Random seed for generation"
    )
    detail_pass_skip_initial_steps: int = Field(
        default=7, description="The number of inference steps to skip in the initial steps of the detail pass. By skipping some steps at the beginning, the detail pass can focus on smaller details instead of larger changes."
    )
    second_pass_number_of_steps: str = Field(
        default="", description="Number of inference steps during the second pass. Deprecated. Use `second_pass_num_inference_steps` instead."
    )
    detail_pass_num_inference_steps: int = Field(
        default=8, description="Number of inference steps during the detail pass."
    )
    detail_pass_noise_scale: float = Field(
        default=0.125, description="The noise scale for the detail pass. This controls the amount of noise added to the generated video during the detail pass. A value of 0.0 means no noise, while a value of 1.0 means maximum noise."
    )
    first_pass_num_inference_steps: int = Field(
        default=30, description="Number of inference steps during the first pass."
    )
    expand_prompt: bool = Field(
        default=False, description="Whether to expand the prompt using a language model."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )
    aspect_ratio: LtxVideo13bDevExtendAspectRatio = Field(
        default=LtxVideo13bDevExtendAspectRatio.AUTO, description="The aspect ratio of the video."
    )
    resolution: LtxVideo13bDevExtendResolution = Field(
        default=LtxVideo13bDevExtendResolution.VALUE_720P, description="Resolution of the generated video."
    )
    constant_rate_factor: int = Field(
        default=29, description="The constant rate factor (CRF) to compress input media with. Compressed input media more closely matches the model's training data, which can improve motion quality."
    )
    first_pass_skip_final_steps: str = Field(
        default="", description="Deprecated. No longer used."
    )
    num_inference_steps: str = Field(
        default="", description="Number of inference steps. Deprecated. Use `first_pass_num_inference_steps` instead."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "second_pass_skip_initial_steps": self.second_pass_skip_initial_steps,
            "enable_detail_pass_autoregression": self.enable_detail_pass_autoregression,
            "frame_rate": self.frame_rate,
            "temporal_adain_factor": self.temporal_adain_factor,
            "first_pass_number_of_steps": self.first_pass_number_of_steps,
            "number_of_frames": self.number_of_frames,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "num_frames": self.num_frames,
            "guidance_scale": self.guidance_scale,
            "video": {
                "video_url": video_url,
                "reverse_video": self.reverse_video,
                "start_frame_num": self.start_frame_num,
                "limit_num_frames": self.limit_num_frames,
                "resample_fps": self.resample_fps,
                "strength": self.strength,
                "target_fps": self.target_fps,
                "max_num_frames": self.max_num_frames,
                "conditioning_type": self.conditioning_type,
                "preprocess": self.preprocess,
            },
            "enable_detail_pass": self.enable_detail_pass,
            "tone_map_compression_ratio": self.tone_map_compression_ratio,
            "seed": self.seed,
            "detail_pass_skip_initial_steps": self.detail_pass_skip_initial_steps,
            "second_pass_number_of_steps": self.second_pass_number_of_steps,
            "detail_pass_num_inference_steps": self.detail_pass_num_inference_steps,
            "detail_pass_noise_scale": self.detail_pass_noise_scale,
            "first_pass_num_inference_steps": self.first_pass_num_inference_steps,
            "expand_prompt": self.expand_prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "constant_rate_factor": self.constant_rate_factor,
            "first_pass_skip_final_steps": self.first_pass_skip_final_steps,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-13b-dev/extend",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LtxVideoLoraMulticonditioningAspectRatio(str, Enum):
    """
    The aspect ratio of the video.
    """
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"
    AUTO = "auto"

class LtxVideoLoraMulticonditioningResolution(str, Enum):
    """
    The resolution of the video.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"


class LtxVideoLoraMulticonditioning(FALNode):
    """
    Generate videos from prompts, images, and videos using LTX Video-0.9.7 and custom LoRA
    video, editing, video-to-video, vid2vid, lora

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    number_of_steps: int = Field(
        default=30, description="The number of inference steps to use."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video."
    )
    frame_rate: int = Field(
        default=25, description="The frame rate of the video."
    )
    expand_prompt: bool = Field(
        default=False, description="Whether to expand the prompt using the LLM."
    )
    number_of_frames: int = Field(
        default=89, description="The number of frames in the video."
    )
    loras: list[LoRAWeight] = Field(
        default=[], description="The LoRA weights to use for generation."
    )
    images: list[ImageCondition] = Field(
        default=[], description="The image conditions to use for generation."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    negative_prompt: str = Field(
        default="blurry, low quality, low resolution, inconsistent motion, jittery, distorted", description="The negative prompt to use."
    )
    aspect_ratio: LtxVideoLoraMulticonditioningAspectRatio = Field(
        default=LtxVideoLoraMulticonditioningAspectRatio.AUTO, description="The aspect ratio of the video."
    )
    resolution: LtxVideoLoraMulticonditioningResolution = Field(
        default=LtxVideoLoraMulticonditioningResolution.VALUE_720P, description="The resolution of the video."
    )
    videos: list[VideoCondition] = Field(
        default=[], description="The video conditions to use for generation."
    )
    seed: int = Field(
        default=-1, description="The seed to use for generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "number_of_steps": self.number_of_steps,
            "prompt": self.prompt,
            "reverse_video": self.reverse_video,
            "frame_rate": self.frame_rate,
            "expand_prompt": self.expand_prompt,
            "number_of_frames": self.number_of_frames,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "images": [item.model_dump(exclude={"type"}) for item in self.images],
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "videos": [item.model_dump(exclude={"type"}) for item in self.videos],
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-lora/multiconditioning",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class WanVaceTask(str, Enum):
    """
    Task type for the model.
    """
    DEPTH = "depth"
    INPAINTING = "inpainting"

class WanVaceAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video (16:9 or 9:16).
    """
    AUTO = "auto"
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"

class WanVaceResolution(str, Enum):
    """
    Resolution of the generated video (480p,580p, or 720p).
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"


class WanVace(FALNode):
    """
    Vace a video generation model that uses a source image, mask, and video to create prompted videos with controllable sources.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. If provided, the model will use this video as a reference."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    ref_images: list[str] = Field(
        default=[], description="Urls to source reference image. If provided, the model will use this image as reference."
    )
    task: WanVaceTask = Field(
        default=WanVaceTask.DEPTH, description="Task type for the model."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 24."
    )
    mask_image: ImageRef = Field(
        default=ImageRef(), description="URL to the guiding mask file. If provided, the model will use this mask as a reference to create masked video. If provided mask video url will be ignored."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 100 (inclusive). Works only with only reference images as input if source video or mask video is provided output len would be same as source up to 241 frames"
    )
    negative_prompt: str = Field(
        default="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    aspect_ratio: WanVaceAspectRatio = Field(
        default=WanVaceAspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
    )
    resolution: WanVaceResolution = Field(
        default=WanVaceResolution.VALUE_720P, description="Resolution of the generated video (480p,580p, or 720p)."
    )
    mask_video: VideoRef = Field(
        default=VideoRef(), description="URL to the source mask file. If provided, the model will use this mask as a reference."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    preprocess: bool = Field(
        default=False, description="Whether to preprocess the input video."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        mask_image_base64 = (
            await context.image_to_base64(self.mask_image)
            if not self.mask_image.is_empty()
            else None
        )
        mask_video_url = (
            await self._upload_asset_to_fal(client, self.mask_video, context)
            if not self.mask_video.is_empty()
            else None
        )
        arguments = {
            "shift": self.shift,
            "video_url": video_url,
            "prompt": self.prompt,
            "ref_image_urls": self.ref_images,
            "task": self.task.value,
            "frames_per_second": self.frames_per_second,
            "mask_image_url": f"data:image/png;base64,{mask_image_base64}" if mask_image_base64 else None,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "mask_video_url": mask_video_url,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "preprocess": self.preprocess,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class CassetteaiVideoSoundEffectsGenerator(FALNode):
    """
    Add sound effects to your videos
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="A video file to analyze & re-sound with generated SFX."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="cassetteai/video-sound-effects-generator",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class SyncLipsyncV2SyncMode(str, Enum):
    """
    Lipsync mode when audio and video durations are out of sync.
    """
    CUT_OFF = "cut_off"
    LOOP = "loop"
    BOUNCE = "bounce"
    SILENCE = "silence"
    REMAP = "remap"

class SyncLipsyncV2Model(str, Enum):
    """
    The model to use for lipsyncing. `lipsync-2-pro` will cost roughly 1.67 times as much as `lipsync-2` for the same duration.
    """
    LIPSYNC_2 = "lipsync-2"
    LIPSYNC_2_PRO = "lipsync-2-pro"


class SyncLipsyncV2(FALNode):
    """
    Generate realistic lipsync animations from audio using advanced algorithms for high-quality synchronization with Sync Lipsync 2.0 model
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    sync_mode: SyncLipsyncV2SyncMode = Field(
        default=SyncLipsyncV2SyncMode.CUT_OFF, description="Lipsync mode when audio and video durations are out of sync."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video"
    )
    model: SyncLipsyncV2Model = Field(
        default=SyncLipsyncV2Model.LIPSYNC_2, description="The model to use for lipsyncing. `lipsync-2-pro` will cost roughly 1.67 times as much as `lipsync-2` for the same duration."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the input audio"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "sync_mode": self.sync_mode.value,
            "video_url": video_url,
            "model": self.model.value,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sync-lipsync/v2",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LatentsyncLoopMode(str, Enum):
    """
    Video loop mode when audio is longer than video. Options: pingpong, loop
    """
    PINGPONG = "pingpong"
    LOOP = "loop"


class Latentsync(FALNode):
    """
    LatentSync is a video-to-video model that generates lip sync animations from audio using advanced algorithms for high-quality synchronization.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the lip sync for."
    )
    guidance_scale: float = Field(
        default=1, description="Guidance scale for the model inference"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation. If None, a random seed will be used."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio to generate the lip sync for."
    )
    loop_mode: LatentsyncLoopMode | None = Field(
        default=None, description="Video loop mode when audio is longer than video. Options: pingpong, loop"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "audio_url": self.audio,
            "loop_mode": self.loop_mode.value if self.loop_mode else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/latentsync",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class PikaV2Pikadditions(FALNode):
    """
    Pikadditions is a powerful video-to-video AI model that allows you to add anyone or anything to any video with seamless integration.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt describing what to add"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video"
    )
    seed: str = Field(
        default="", description="The seed for the random number generator"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to guide the model"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to add"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "video_url": video_url,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pika/v2/pikadditions",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LtxVideoV095ExtendResolution(str, Enum):
    """
    Resolution of the generated video (480p or 720p).
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"

class LtxVideoV095ExtendAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video (16:9 or 9:16).
    """
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"


class LtxVideoV095Extend(FALNode):
    """
    Generate videos from prompts and videos using LTX Video-0.9.5
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    resolution: LtxVideoV095ExtendResolution = Field(
        default=LtxVideoV095ExtendResolution.VALUE_720P, description="Resolution of the generated video (480p or 720p)."
    )
    aspect_ratio: LtxVideoV095ExtendAspectRatio = Field(
        default=LtxVideoV095ExtendAspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to expand the prompt using the model's own capabilities."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )
    num_inference_steps: int = Field(
        default=40, description="Number of inference steps"
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Video to be extended."
    )
    start_frame_num: int = Field(
        default=0, description="Frame number of the video from which the conditioning starts. Must be a multiple of 8."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "expand_prompt": self.expand_prompt,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "video": {
                "video_url": video_url,
                "start_frame_num": self.start_frame_num,
            },
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-v095/extend",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class LtxVideoV095MulticonditioningResolution(str, Enum):
    """
    Resolution of the generated video (480p or 720p).
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"

class LtxVideoV095MulticonditioningAspectRatio(str, Enum):
    """
    Aspect ratio of the generated video (16:9 or 9:16).
    """
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"


class LtxVideoV095Multiconditioning(FALNode):
    """
    Generate videos from prompts,images, and videos using LTX Video-0.9.5
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    resolution: LtxVideoV095MulticonditioningResolution = Field(
        default=LtxVideoV095MulticonditioningResolution.VALUE_720P, description="Resolution of the generated video (480p or 720p)."
    )
    aspect_ratio: LtxVideoV095MulticonditioningAspectRatio = Field(
        default=LtxVideoV095MulticonditioningAspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to expand the prompt using the model's own capabilities."
    )
    images: list[ImageConditioningInput] = Field(
        default=[], description="URL of images to use as conditioning"
    )
    videos: list[VideoConditioningInput] = Field(
        default=[], description="Videos to use as conditioning"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )
    num_inference_steps: int = Field(
        default=40, description="Number of inference steps"
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "expand_prompt": self.expand_prompt,
            "images": [item.model_dump(exclude={"type"}) for item in self.images],
            "videos": [item.model_dump(exclude={"type"}) for item in self.videos],
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-v095/multiconditioning",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class TopazUpscaleVideo(FALNode):
    """
    Professional-grade video upscaling using Topaz technology. Enhance your videos with high-quality upscaling.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    H264_output: bool = Field(
        default=False, description="Whether to use H264 codec for output video. Default is H265."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to upscale"
    )
    upscale_factor: float = Field(
        default=2, description="Factor to upscale the video by (e.g. 2.0 doubles width and height)"
    )
    target_fps: int = Field(
        default=0, description="Target FPS for frame interpolation. If set, frame interpolation will be enabled."
    )
    focus: float = Field(
        default=0.2, description="Focus/sharpening level (0.0-1.0, default 0.2)"
    )
    sharpen: float = Field(
        default=0.6, description="Sharpening level (0.0-1.0, default 0.6)"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "H264_output": self.H264_output,
            "video_url": video_url,
            "upscale_factor": self.upscale_factor,
            "target_fps": self.target_fps,
            "focus": self.focus,
            "sharpen": self.sharpen,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/topaz/upscale/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class HunyuanVideoLoraVideoToVideoAspectRatio(str, Enum):
    """
    The aspect ratio of the video to generate.
    """
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"

class HunyuanVideoLoraVideoToVideoResolution(str, Enum):
    """
    The resolution of the video to generate.
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"

class HunyuanVideoLoraVideoToVideoNumFrames(str, Enum):
    """
    The number of frames to generate.
    """
    VALUE_129 = "129"
    VALUE_85 = "85"


class HunyuanVideoLoraVideoToVideo(FALNode):
    """
    Hunyuan Video is an Open video generation model with high visual quality, motion diversity, text-video alignment, and generation stability. Use this endpoint to generate videos from videos.
    video, editing, video-to-video, vid2vid, lora

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    aspect_ratio: HunyuanVideoLoraVideoToVideoAspectRatio = Field(
        default=HunyuanVideoLoraVideoToVideoAspectRatio.RATIO_16_9, description="The aspect ratio of the video to generate."
    )
    resolution: HunyuanVideoLoraVideoToVideoResolution = Field(
        default=HunyuanVideoLoraVideoToVideoResolution.VALUE_720P, description="The resolution of the video to generate."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video"
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    strength: float = Field(
        default=0.75, description="Strength of video-to-video"
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The seed to use for generating the video."
    )
    num_frames: HunyuanVideoLoraVideoToVideoNumFrames = Field(
        default=129, description="The number of frames to generate."
    )
    pro_mode: bool = Field(
        default=False, description="By default, generations are done with 35 steps. Pro mode does 55 steps which results in higher quality videos but will take more time and cost 2x more billing units."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "video_url": video_url,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "strength": self.strength,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "num_frames": self.num_frames.value,
            "pro_mode": self.pro_mode,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video-lora/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class FfmpegApiCompose(FALNode):
    """
    Compose videos from multiple media sources using FFmpeg API.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    tracks: list[Track] = Field(
        default=[], description="List of tracks to be combined into the final media"
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "tracks": [item.model_dump(exclude={"type"}) for item in self.tracks],
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ffmpeg-api/compose",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class SyncLipsyncSyncMode(str, Enum):
    """
    Lipsync mode when audio and video durations are out of sync.
    """
    CUT_OFF = "cut_off"
    LOOP = "loop"
    BOUNCE = "bounce"
    SILENCE = "silence"
    REMAP = "remap"

class SyncLipsyncModel(str, Enum):
    """
    The model to use for lipsyncing
    """
    LIPSYNC_1_8_0 = "lipsync-1.8.0"
    LIPSYNC_1_7_1 = "lipsync-1.7.1"
    LIPSYNC_1_9_0_BETA = "lipsync-1.9.0-beta"


class SyncLipsync(FALNode):
    """
    Generate realistic lipsync animations from audio using advanced algorithms for high-quality synchronization.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    sync_mode: SyncLipsyncSyncMode = Field(
        default=SyncLipsyncSyncMode.CUT_OFF, description="Lipsync mode when audio and video durations are out of sync."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video"
    )
    model: SyncLipsyncModel = Field(
        default=SyncLipsyncModel.LIPSYNC_1_9_0_BETA, description="The model to use for lipsyncing"
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the input audio"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "sync_mode": self.sync_mode.value,
            "video_url": video_url,
            "model": self.model.value,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sync-lipsync",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class DubbingTargetLanguage(str, Enum):
    """
    Target language to dub the video to
    """
    HINDI = "hindi"
    TURKISH = "turkish"
    ENGLISH = "english"


class Dubbing(FALNode):
    """
    This endpoint delivers seamlessly localized videos by generating lip-synced dubs in multiple languages, ensuring natural and immersive multilingual experiences
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    do_lipsync: bool = Field(
        default=True, description="Whether to lip sync the audio to the video"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="Input video URL to be dubbed."
    )
    target_language: DubbingTargetLanguage = Field(
        default=DubbingTargetLanguage.HINDI, description="Target language to dub the video to"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "do_lipsync": self.do_lipsync,
            "video_url": video_url,
            "target_language": self.target_language.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/dubbing",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Controlnext(FALNode):
    """
    Animate a reference image with a driving video using ControlNeXt.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    controlnext_cond_scale: float = Field(
        default=1, description="Condition scale for ControlNeXt."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    fps: int = Field(
        default=7, description="Frames per second for the output video."
    )
    max_frame_num: int = Field(
        default=240, description="Maximum number of frames to process."
    )
    width: int = Field(
        default=576, description="Width of the output video."
    )
    overlap: int = Field(
        default=6, description="Number of overlapping frames between batches."
    )
    guidance_scale: float = Field(
        default=3, description="Guidance scale for the diffusion process."
    )
    batch_frames: int = Field(
        default=24, description="Number of frames to process in each batch."
    )
    height: int = Field(
        default=1024, description="Height of the output video."
    )
    sample_stride: int = Field(
        default=2, description="Stride for sampling frames from the input video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the reference image."
    )
    decode_chunk_size: int = Field(
        default=2, description="Chunk size for decoding frames."
    )
    motion_bucket_id: float = Field(
        default=127, description="Motion bucket ID for the pipeline."
    )
    num_inference_steps: int = Field(
        default=25, description="Number of inference steps."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        image_base64 = (
            await context.image_to_base64(self.image)
            if not self.image.is_empty()
            else None
        )
        arguments = {
            "controlnext_cond_scale": self.controlnext_cond_scale,
            "video_url": video_url,
            "fps": self.fps,
            "max_frame_num": self.max_frame_num,
            "width": self.width,
            "overlap": self.overlap,
            "guidance_scale": self.guidance_scale,
            "batch_frames": self.batch_frames,
            "height": self.height,
            "sample_stride": self.sample_stride,
            "image_url": f"data:image/png;base64,{image_base64}" if image_base64 else None,
            "decode_chunk_size": self.decode_chunk_size,
            "motion_bucket_id": self.motion_bucket_id,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/controlnext",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]

class Sam2Video(FALNode):
    """
    SAM 2 is a model for segmenting images and videos in real-time.
    video, editing, video-to-video, vid2vid

    Use cases:
    - Video style transfer
    - Video enhancement and restoration
    - Automated video editing
    - Special effects generation
    - Content repurposing
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to be segmented."
    )
    prompts: list[PointPrompt] = Field(
        default=[], description="List of prompts to segment the video"
    )
    boundingbox_zip: bool = Field(
        default=False, description="Return per-frame bounding box overlays as a zip archive."
    )
    box_prompts: list[BoxPrompt] = Field(
        default=[], description="Coordinates for boxes"
    )
    apply_mask: bool = Field(
        default=False, description="Apply the mask on the video."
    )
    mask_url: str = Field(
        default="", description="The URL of the mask to be applied initially."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        client = await self.get_client(context)
        video_url = (
            await self._upload_asset_to_fal(client, self.video, context)
            if not self.video.is_empty()
            else None
        )
        arguments = {
            "video_url": video_url,
            "prompts": [item.model_dump(exclude={"type"}) for item in self.prompts],
            "boundingbox_zip": self.boundingbox_zip,
            "box_prompts": [item.model_dump(exclude={"type"}) for item in self.box_prompts],
            "apply_mask": self.apply_mask,
            "mask_url": self.mask_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}
        # Also filter nested dicts
        for key in arguments:
            if isinstance(arguments[key], dict):
                arguments[key] = {k: v for k, v in arguments[key].items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam2/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video", "prompt"]