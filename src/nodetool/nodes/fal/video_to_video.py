from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class VideoWriteMode(Enum):
    """
    The write mode of the generated video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"


class VideoOutputType(Enum):
    """
    The output type of the generated video.
    """
    X264__MP4 = "X264 (.mp4)"
    VP9__WEBM = "VP9 (.webm)"
    PRORES4444__MOV = "PRORES4444 (.mov)"
    GIF__GIF = "GIF (.gif)"


class OperatingResolution(Enum):
    """
    The resolution to operate on. The higher the resolution, the more accurate the output will be for high res input images. The '2304x2304' option is only available for the 'General Use (Dynamic)' model.
    """
    VALUE_1024X1024 = "1024x1024"
    VALUE_2048X2048 = "2048x2048"
    VALUE_2304X2304 = "2304x2304"


class Model(Enum):
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


class VideoQuality(Enum):
    """
    The quality of the generated video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class OutputContainerAndCodec(Enum):
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


class AspectRatio(Enum):
    """
    The aspect ratio of the video to generate.
    """
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"


class Resolution(Enum):
    """
    The resolution of the video to generate.
    """
    VALUE_480P = "480p"
    VALUE_580P = "580p"
    VALUE_720P = "720p"


class NumFrames(Enum):
    """
    The number of frames to generate.
    """
    VALUE_129 = "129"
    VALUE_85 = "85"


class ColorFixType(Enum):
    """
    Type of color correction for samples.
    """
    NONE = "none"
    WAVELET = "wavelet"
    ADAIN = "adain"


class TileDiffusion(Enum):
    """
    If specified, a patch-based sampling strategy will be used for sampling.
    """
    NONE = "none"
    MIX = "mix"
    GAUSSIAN = "gaussian"


class Acceleration(Enum):
    """
    The acceleration level to use.
    """
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"
    FULL = "full"


class CameraLora(Enum):
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


class Preprocessor(Enum):
    """
    The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type.
    """
    DEPTH = "depth"
    CANNY = "canny"
    POSE = "pose"
    NONE = "none"


class IcLora(Enum):
    """
    The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)
    """
    MATCH_PREPROCESSOR = "match_preprocessor"
    CANNY = "canny"
    DEPTH = "depth"
    POSE = "pose"
    DETAILER = "detailer"
    NONE = "none"


class ExtendDirection(Enum):
    """
    Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning.
    """
    FORWARD = "forward"
    BACKWARD = "backward"


class RelitCondType(Enum):
    """
    Relight condition type.
    """
    IC = "ic"
    REF = "ref"
    HDR = "hdr"
    BG = "bg"


class Camera(Enum):
    """
    Camera control mode.
    """
    TRAJ = "traj"
    TARGET = "target"


class Mode(Enum):
    """
    Camera motion mode.
    """
    GRADUAL = "gradual"
    BULLET = "bullet"
    DIRECT = "direct"
    DOLLY_ZOOM = "dolly-zoom"


class CharacterOrientation(Enum):
    """
    Controls whether the output character's orientation matches the reference image or video. 'video': orientation matches reference video - better for complex motions (max 30s). 'image': orientation matches reference image - better for following camera movements (max 10s).
    """
    IMAGE = "image"
    VIDEO = "video"


class Duration(Enum):
    """
    Duration of the generated video in seconds. R2V supports only 5 or 10 seconds (no 15s).
    """
    VALUE_5 = "5"
    VALUE_10 = "10"


class TargetResolution(Enum):
    """
    Target output resolution for the enhanced video. 720p (native, fast) or 1080p (upscaled, slower). Processing is always done at 720p, then upscaled if 1080p selected.
    """
    VALUE_720P = "720p"
    VALUE_1080P = "1080p"


class Emotion(Enum):
    """
    Emotion prompt for the generation. Currently supports single-word emotions only.
    """
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    NEUTRAL = "neutral"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"


class LipsyncMode(Enum):
    """
    Lipsync mode when audio and video durations are out of sync.
    """
    CUT_OFF = "cut_off"
    LOOP = "loop"
    BOUNCE = "bounce"
    SILENCE = "silence"
    REMAP = "remap"


class ModelMode(Enum):
    """
    Controls the edit region and movement scope for the model. Available options:
    - `lips`: Only lipsync using react-1 (minimal facial changes).
    - `face`: Lipsync + facial expressions without head movements.
    - `head`: Lipsync + facial expressions + natural talking head movements.
    """
    LIPS = "lips"
    FACE = "face"
    HEAD = "head"


class OutputCodec(Enum):
    """
    Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality.
    """
    VP9 = "vp9"
    H264 = "h264"


class RetakeMode(Enum):
    """
    The retake mode to use for the retake
    """
    REPLACE_AUDIO = "replace_audio"
    REPLACE_VIDEO = "replace_video"
    REPLACE_AUDIO_AND_VIDEO = "replace_audio_and_video"


class Sampler(Enum):
    """
    Sampler to use for video generation.
    """
    UNIPC = "unipc"
    DPMPP = "dpmPP"
    EULER = "euler"


class OutputFormat(Enum):
    """
    The format of the output video.
    """
    X264__MP4 = "X264 (.mp4)"
    VP9__WEBM = "VP9 (.webm)"
    PRORES4444__MOV = "PRORES4444 (.mov)"
    GIF__GIF = "GIF (.gif)"


class OutputWriteMode(Enum):
    """
    The write mode of the output video.
    """
    FAST = "fast"
    BALANCED = "balanced"
    SMALL = "small"


class OutputQuality(Enum):
    """
    The quality of the output video.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class FontWeight(Enum):
    """
    Font weight (TikTok style typically uses bold or black)
    """
    NORMAL = "normal"
    BOLD = "bold"
    BLACK = "black"


class FontColor(Enum):
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


class StrokeColor(Enum):
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


class HighlightColor(Enum):
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


class Position(Enum):
    """
    Vertical position of subtitles
    """
    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"


class BackgroundColor(Enum):
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


class TargetFps(Enum):
    """
    The target FPS of the video to upscale.
    """
    VALUE_30FPS = "30fps"
    VALUE_60FPS = "60fps"


class TransparencyMode(Enum):
    """
    The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled.
    """
    CONTENT_AWARE = "content_aware"
    WHITE = "white"
    BLACK = "black"


class InterpolatorModel(Enum):
    """
    The model to use for frame interpolation. Options are 'rife' or 'film'.
    """
    RIFE = "rife"
    FILM = "film"


class UpscaleMode(Enum):
    """
    The mode to use for the upscale. If 'target', the upscale factor will be calculated based on the target resolution. If 'factor', the upscale factor will be used directly.
    """
    TARGET = "target"
    FACTOR = "factor"


class VideoType(Enum):
    """
    The type of video you're editing. Use 'general' for most videos, and 'human' for videos emphasizing human subjects and motions. The default value 'auto' means the model will guess based on the first frame of the video.
    """
    AUTO = "auto"
    GENERAL = "general"
    HUMAN = "human"


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

    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video to be processed"
    )
    recursive_interpolation_passes: int = Field(
        default=2, description="Number of recursive interpolation passes"
    )
    output_fps: int = Field(
        default=24, description="Output frames per second"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "recursive_interpolation_passes": self.recursive_interpolation_passes,
            "output_fps": self.output_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    source_face_url: ImageRef = Field(
        default=ImageRef(), description="Source face image. Allowed items: bmp, jpeg, png, tiff, webp"
    )
    target_video_url: VideoRef = Field(
        default=VideoRef(), description="Target video URL (max 25 minutes, will be truncated if longer; FPS capped at 25). Allowed items: avi, m4v, mkv, mp4, mpeg, mov, mxf, webm, wmv"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        source_face_url_base64 = await context.image_to_base64(self.source_face_url)
        arguments = {
            "enable_occlusion_prevention": self.enable_occlusion_prevention,
            "source_face_url": f"data:image/png;base64,{source_face_url_base64}",
            "target_video_url": self.target_video_url,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video."
    )
    first_n_seconds: int = Field(
        default=3, description="The first N number of seconds of video to animate."
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
    num_inference_steps: int = Field(
        default=25, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    negative_prompt: str = Field(
        default="(bad quality, worst quality:1.2), ugly faces, bad anime", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    motions: list[str] = Field(
        default=[], description="The motions to apply to the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "first_n_seconds": self.first_n_seconds,
            "fps": self.fps,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "motions": self.motions,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video."
    )
    first_n_seconds: int = Field(
        default=3, description="The first N number of seconds of video to animate."
    )
    fps: int = Field(
        default=8, description="Number of frames per second to extract from the video."
    )
    strength: float = Field(
        default=0.7, description="The strength of the input video in the final output."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform. 4-12 is recommended for turbo mode."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    negative_prompt: str = Field(
        default="(bad quality, worst quality:1.2), ugly faces, bad anime", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    motions: list[str] = Field(
        default=[], description="The motions to apply to the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "first_n_seconds": self.first_n_seconds,
            "fps": self.fps,
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "motions": self.motions,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
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
        arguments = {
            "txt_font": self.txt_font,
            "video_url": self.video_url,
            "top_align": self.top_align,
            "txt_color": self.txt_color,
            "stroke_width": self.stroke_width,
            "refresh_interval": self.refresh_interval,
            "font_size": self.font_size,
            "left_align": self.left_align,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of video to be used for background removal."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )
    background_color: list[str] = Field(
        default=[], description="Optional RGB values (0-255) for the background color. If not provided, the background will be transparent. For ex: [0, 0, 0]"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "seed": self.seed,
            "background_color": self.background_color,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    operating_resolution: OperatingResolution = Field(
        default=OperatingResolution.VALUE_1024X1024, description="The resolution to operate on. The higher the resolution, the more accurate the output will be for high res input images. The '2304x2304' option is only available for the 'General Use (Dynamic)' model."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video to remove background from"
    )
    model: Model = Field(
        default=Model.GENERAL_USE_LIGHT, description="Model to use for background removal. The 'General Use (Light)' model is the original model used in the BiRefNet repository. The 'General Use (Light 2K)' model is the original model used in the BiRefNet repository but trained with 2K images. The 'General Use (Heavy)' model is a slower but more accurate model. The 'Matting' model is a model trained specifically for matting images. The 'Portrait' model is a model trained specifically for portrait images. The 'General Use (Dynamic)' model supports dynamic resolutions from 256x256 to 2304x2304. The 'General Use (Light)' model is recommended for most use cases. The corresponding models are as follows: - 'General Use (Light)': BiRefNet - 'General Use (Light 2K)': BiRefNet_lite-2K - 'General Use (Heavy)': BiRefNet_lite - 'Matting': BiRefNet-matting - 'Portrait': BiRefNet-portrait - 'General Use (Dynamic)': BiRefNet_dynamic"
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    output_mask: bool = Field(
        default=False, description="Whether to output the mask used to remove the background"
    )
    refine_foreground: bool = Field(
        default=True, description="Whether to refine the foreground using the estimated mask"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "operating_resolution": self.operating_resolution.value,
            "video_url": self.video_url,
            "model": self.model.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "output_mask": self.output_mask,
            "refine_foreground": self.refine_foreground,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    output_container_and_codec: OutputContainerAndCodec = Field(
        default=OutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    mask_video_url: VideoRef = Field(
        default=VideoRef(), description="Input video to mask erase object from. duration must be less than 5s."
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": self.video_url,
            "output_container_and_codec": self.output_container_and_codec.value,
            "mask_video_url": self.mask_video_url,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    output_container_and_codec: OutputContainerAndCodec = Field(
        default=OutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    keypoints: list[str] = Field(
        default=[], description="Input keypoints [x,y] to erase or keep from the video. Format like so: {'x':100, 'y':100, 'type':'positive/negative'}"
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": self.video_url,
            "output_container_and_codec": self.output_container_and_codec.value,
            "keypoints": self.keypoints,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    prompt: str = Field(
        default="", description="Input prompt to detect object to erase"
    )
    output_container_and_codec: OutputContainerAndCodec = Field(
        default=OutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": self.video_url,
            "prompt": self.prompt,
            "output_container_and_codec": self.output_container_and_codec.value,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="The video to generate the video from."
    )
    use_rife: bool = Field(
        default=True, description="Use RIFE for video interpolation"
    )
    loras: list[str] = Field(
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
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "use_rife": self.use_rife,
            "loras": self.loras,
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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video to generate."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the video to generate."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video input."
    )
    strength: float = Field(
        default=0.85, description="Strength for Video-to-Video"
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to run. Lower gets faster results, higher gets better results."
    )
    seed: int = Field(
        default=-1, description="The seed to use for generating the video."
    )
    num_frames: NumFrames = Field(
        default=129, description="The number of frames to generate."
    )
    pro_mode: bool = Field(
        default=False, description="By default, generations are done with 35 steps. Pro mode does 55 steps which results in higher quality videos but will take more time and cost 2x more billing units."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "video_url": self.video_url,
            "strength": self.strength,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "num_frames": self.num_frames.value,
            "pro_mode": self.pro_mode,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to upscale"
    )
    scale: float = Field(
        default=2, description="The scale factor"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "scale": self.scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    color_fix_type: ColorFixType = Field(
        default=ColorFixType.ADAIN, description="Type of color correction for samples."
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
    image_url: ImageRef = Field(
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
    tile_diffusion: TileDiffusion = Field(
        default=TileDiffusion.NONE, description="If specified, a patch-based sampling strategy will be used for sampling."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "color_fix_type": self.color_fix_type.value,
            "tile_diffusion_size": self.tile_diffusion_size,
            "tile_vae_decoder_size": self.tile_vae_decoder_size,
            "tile_vae_encoder_size": self.tile_vae_encoder_size,
            "t_min": self.t_min,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

        res = await self.submit_request(
            context=context,
            application="fal-ai/ccsr",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["video"]








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
    video_url: VideoRef = Field(
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
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the generation."
    )
    video_size: str = Field(
        default="auto", description="The size of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="An optional URL of an image to use as the first frame of the video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )
    match_video_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the video duration and FPS. When disabled, use the specified num_frames."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
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
    preprocessor: Preprocessor = Field(
        default=Preprocessor.NONE, description="The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    ic_lora: IcLora = Field(
        default=IcLora.MATCH_PREPROCESSOR, description="The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)"
    )
    audio_url: VideoRef = Field(
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
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "use_multiscale": self.use_multiscale,
            "ic_lora_scale": self.ic_lora_scale,
            "generate_audio": self.generate_audio,
            "loras": self.loras,
            "video_size": self.video_size,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "camera_lora_scale": self.camera_lora_scale,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            "audio_url": self.audio_url,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
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
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="An optional URL of an image to use as the first frame of the video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )
    match_video_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the video duration and FPS. When disabled, use the specified num_frames."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
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
    preprocessor: Preprocessor = Field(
        default=Preprocessor.NONE, description="The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    ic_lora: IcLora = Field(
        default=IcLora.MATCH_PREPROCESSOR, description="The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)"
    )
    audio_url: VideoRef = Field(
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
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "use_multiscale": self.use_multiscale,
            "ic_lora_scale": self.ic_lora_scale,
            "generate_audio": self.generate_audio,
            "video_size": self.video_size,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "camera_lora_scale": self.camera_lora_scale,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            "audio_url": self.audio_url,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the video from."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    ic_lora_scale: float = Field(
        default=1, description="The scale of the IC-LoRA to use. This allows you to control the strength of the IC-LoRA."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    loras: list[str] = Field(
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
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="An optional URL of an image to use as the first frame of the video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )
    match_video_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the video duration and FPS. When disabled, use the specified num_frames."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
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
    preprocessor: Preprocessor = Field(
        default=Preprocessor.NONE, description="The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    ic_lora: IcLora = Field(
        default=IcLora.MATCH_PREPROCESSOR, description="The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)"
    )
    audio_url: VideoRef = Field(
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
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "video_url": self.video_url,
            "prompt": self.prompt,
            "ic_lora_scale": self.ic_lora_scale,
            "generate_audio": self.generate_audio,
            "loras": self.loras,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "camera_lora_scale": self.camera_lora_scale,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            "audio_url": self.audio_url,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the video from."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
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
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="An optional URL of an image to use as the first frame of the video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )
    match_video_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the video duration and FPS. When disabled, use the specified num_frames."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
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
    preprocessor: Preprocessor = Field(
        default=Preprocessor.NONE, description="The preprocessor to use for the video. When a preprocessor is used and `ic_lora_type` is set to `match_preprocessor`, the IC-LoRA will be loaded based on the preprocessor type."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    ic_lora: IcLora = Field(
        default=IcLora.MATCH_PREPROCESSOR, description="The type of IC-LoRA to load. In-Context LoRA weights are used to condition the video based on edge, depth, or pose videos. Only change this from `match_preprocessor` if your videos are already preprocessed (or you are using the detailer.)"
    )
    audio_url: VideoRef = Field(
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
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "video_url": self.video_url,
            "prompt": self.prompt,
            "ic_lora_scale": self.ic_lora_scale,
            "generate_audio": self.generate_audio,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "camera_lora_scale": self.camera_lora_scale,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
            "audio_url": self.audio_url,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to extend."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the generation."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
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
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the extended video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    extend_direction: ExtendDirection = Field(
        default=ExtendDirection.FORWARD, description="Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    num_context_frames: int = Field(
        default=25, description="The number of frames to use as context for the extension."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "video_url": self.video_url,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "prompt": self.prompt,
            "fps": self.fps,
            "loras": self.loras,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "negative_prompt": self.negative_prompt,
            "extend_direction": self.extend_direction.value,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "video_strength": self.video_strength,
            "num_context_frames": self.num_context_frames,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "match_input_fps": self.match_input_fps,
            "end_image_strength": self.end_image_strength,
            "audio_strength": self.audio_strength,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to extend."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
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
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the extended video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    extend_direction: ExtendDirection = Field(
        default=ExtendDirection.FORWARD, description="Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    num_context_frames: int = Field(
        default=25, description="The number of frames to use as context for the extension."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "video_url": self.video_url,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "prompt": self.prompt,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "negative_prompt": self.negative_prompt,
            "extend_direction": self.extend_direction.value,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "video_strength": self.video_strength,
            "num_context_frames": self.num_context_frames,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "match_input_fps": self.match_input_fps,
            "end_image_strength": self.end_image_strength,
            "audio_strength": self.audio_strength,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to extend."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    loras: list[str] = Field(
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
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the extended video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    extend_direction: ExtendDirection = Field(
        default=ExtendDirection.FORWARD, description="Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    num_context_frames: int = Field(
        default=25, description="The number of frames to use as context for the extension."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
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
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "video_url": self.video_url,
            "prompt": self.prompt,
            "generate_audio": self.generate_audio,
            "loras": self.loras,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "num_frames": self.num_frames,
            "video_strength": self.video_strength,
            "video_output_type": self.video_output_type.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "acceleration": self.acceleration.value,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "extend_direction": self.extend_direction.value,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "num_context_frames": self.num_context_frames,
            "num_inference_steps": self.num_inference_steps,
            "end_image_strength": self.end_image_strength,
            "match_input_fps": self.match_input_fps,
            "audio_strength": self.audio_strength,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to extend."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
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
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the extended video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    extend_direction: ExtendDirection = Field(
        default=ExtendDirection.FORWARD, description="Direction to extend the video. 'forward' extends from the end of the video, 'backward' extends from the beginning."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    video_strength: float = Field(
        default=1, description="Video conditioning strength. Lower values represent more freedom given to the model to change the video content."
    )
    num_context_frames: int = Field(
        default=25, description="The number of frames to use as context for the extension."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    match_input_fps: bool = Field(
        default=True, description="When true, match the output FPS to the input video's FPS instead of using the default target FPS."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Lower values represent more freedom given to the model to change the audio content."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "video_url": self.video_url,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "negative_prompt": self.negative_prompt,
            "extend_direction": self.extend_direction.value,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "video_strength": self.video_strength,
            "num_context_frames": self.num_context_frames,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "match_input_fps": self.match_input_fps,
            "end_image_strength": self.end_image_strength,
            "audio_strength": self.audio_strength,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    output_container_and_codec: OutputContainerAndCodec = Field(
        default=OutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    keypoints: list[str] = Field(
        default=[], description="Input keypoints [x,y] to erase or keep from the video. Format like so: {'x':100, 'y':100, 'type':'positive/negative'}"
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": self.video_url,
            "output_container_and_codec": self.output_container_and_codec.value,
            "keypoints": self.keypoints,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    prompt: str = Field(
        default="", description="Input prompt to detect object to erase"
    )
    output_container_and_codec: OutputContainerAndCodec = Field(
        default=OutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": self.video_url,
            "prompt": self.prompt,
            "output_container_and_codec": self.output_container_and_codec.value,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="Input video to erase object from. duration must be less than 5s."
    )
    output_container_and_codec: OutputContainerAndCodec = Field(
        default=OutputContainerAndCodec.MP4_H264, description="Output container and codec. Options: mp4_h265, mp4_h264, webm_vp9, gif, mov_h264, mov_h265, mov_proresks, mkv_h264, mkv_h265, mkv_vp9, mkv_mpeg4."
    )
    mask_video_url: VideoRef = Field(
        default=VideoRef(), description="Input video to mask erase object from. duration must be less than 5s."
    )
    auto_trim: bool = Field(
        default=True, description="auto trim the video, to working duration ( 5s )"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "preserve_audio": self.preserve_audio,
            "video_url": self.video_url,
            "output_container_and_codec": self.output_container_and_codec.value,
            "mask_video_url": self.mask_video_url,
            "auto_trim": self.auto_trim,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    relight_parameters: str = Field(
        default="", description="Relighting parameters (required for relight_condition_type='ic'). Not used for 'bg' (which expects a background image URL instead)."
    )
    ref_id: int = Field(
        default=0, description="Frame index to use as referencen to relight the video with reference."
    )
    relit_cond_img_url: ImageRef = Field(
        default=ImageRef(), description="URL of conditioning image. Required for relight_condition_type='ref'/'hdr'. Also required for relight_condition_type='bg' (background image)."
    )
    relit_cond_type: RelitCondType = Field(
        default=RelitCondType.IC, description="Relight condition type."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        relit_cond_img_url_base64 = await context.image_to_base64(self.relit_cond_img_url)
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "relight_parameters": self.relight_parameters,
            "ref_id": self.ref_id,
            "relit_cond_img_url": f"data:image/png;base64,{relit_cond_img_url_base64}",
            "relit_cond_type": self.relit_cond_type.value,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    camera: Camera = Field(
        default=Camera.TRAJ, description="Camera control mode."
    )
    target_pose: list[float] = Field(
        default=[], description="Target camera pose [theta, phi, radius, x, y] (required when camera='target')."
    )
    mode: Mode = Field(
        default=Mode.GRADUAL, description="Camera motion mode."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "trajectory": self.trajectory,
            "video_url": self.video_url,
            "camera": self.camera.value,
            "target_pose": self.target_pose,
            "mode": self.mode.value,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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


class KlingVideoV2_6StandardMotionControl(FALNode):
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
    video_url: ImageRef = Field(
        default=ImageRef(), description="Reference video URL. The character actions in the generated video will be consistent with this reference video. Should contain a realistic style character with entire body or upper body visible, including head, without obstruction. Duration limit depends on character_orientation: 10s max for 'image', 30s max for 'video'."
    )
    character_orientation: CharacterOrientation = Field(
        default="", description="Controls whether the output character's orientation matches the reference image or video. 'video': orientation matches reference video - better for complex motions (max 30s). 'image': orientation matches reference image - better for following camera movements (max 10s)."
    )
    keep_original_sound: bool = Field(
        default=True, description="Whether to keep the original sound from the reference video."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Reference image URL. The characters, backgrounds, and other elements in the generated video are based on this reference image. Characters should have clear body proportions, avoid occlusion, and occupy more than 5% of the image area."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_url_base64 = await context.image_to_base64(self.video_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "video_url": f"data:image/png;base64,{video_url_base64}",
            "character_orientation": self.character_orientation.value,
            "keep_original_sound": self.keep_original_sound,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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


class KlingVideoV2_6ProMotionControl(FALNode):
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
    video_url: ImageRef = Field(
        default=ImageRef(), description="Reference video URL. The character actions in the generated video will be consistent with this reference video. Should contain a realistic style character with entire body or upper body visible, including head, without obstruction. Duration limit depends on character_orientation: 10s max for 'image', 30s max for 'video'."
    )
    character_orientation: CharacterOrientation = Field(
        default="", description="Controls whether the output character's orientation matches the reference image or video. 'video': orientation matches reference video - better for complex motions (max 30s). 'image': orientation matches reference image - better for following camera movements (max 10s)."
    )
    keep_original_sound: bool = Field(
        default=True, description="Whether to keep the original sound from the reference video."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Reference image URL. The characters, backgrounds, and other elements in the generated video are based on this reference image. Characters should have clear body proportions, avoid occlusion, and occupy more than 5% of the image area."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        video_url_base64 = await context.image_to_base64(self.video_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "video_url": f"data:image/png;base64,{video_url_base64}",
            "character_orientation": self.character_orientation.value,
            "keep_original_sound": self.keep_original_sound,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video to edit"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video"
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
        arguments = {
            "sync_mode": self.sync_mode,
            "video_url": self.video_url,
            "resolution": self.resolution.value,
            "prompt": self.prompt,
            "seed": self.seed,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use as a reference for the video generation."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_512P, description="Output resolution. Outputs 896x512 (landscape) or 512x896 (portrait) based on the input image aspect ratio."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to use for the video generation."
    )
    multi_character: bool = Field(
        default=False, description="Enable multi-character mode. Use when driving video has multiple people."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "resolution": self.resolution.value,
            "num_inference_steps": self.num_inference_steps,
            "multi_character": self.multi_character,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    video_url: VideoRef = Field(
        default=VideoRef(), description="URL to the input video."
    )
    scale_factor: float = Field(
        default=2, description="Scale factor. The scale factor must be chosen such that the upscaled video does not exceed 5K resolution."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "scale_factor": self.scale_factor,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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




class WanV2_6ReferenceToVideo(FALNode):
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
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the generated video in seconds. R2V supports only 5 or 10 seconds (no 15s)."
    )
    video_urls: list[str] = Field(
        default=[], description="Reference videos for subject consistency (1-3 videos). Videos' FPS must be at least 16 FPS.Reference in prompt as @Video1, @Video2, @Video3. Works for people, animals, or objects."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1080P, description="Video resolution tier. R2V only supports 720p and 1080p (no 480p)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt rewriting using LLM."
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
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "video_urls": self.video_urls,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "multi_shots": self.multi_shots,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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




class Veo3_1FastExtendVideo(FALNode):
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
    duration: Duration = Field(
        default=Duration.VALUE_7S, description="The duration of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the generated video."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    auto_fix: bool = Field(
        default=False, description="Whether to automatically attempt to fix prompts that fail content policy or other validation checks by rewriting them."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video to extend. The video should be 720p or 1080p resolution in 16:9 or 9:16 aspect ratio."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "video_url": self.video_url,
            "resolution": self.resolution.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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




class Veo3_1ExtendVideo(FALNode):
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
    duration: Duration = Field(
        default=Duration.VALUE_7S, description="The duration of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the generated video."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    auto_fix: bool = Field(
        default=False, description="Whether to automatically attempt to fix prompts that fail content policy or other validation checks by rewriting them."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video to extend. The video should be 720p or 1080p resolution in 16:9 or 9:16 aspect ratio."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "video_url": self.video_url,
            "resolution": self.resolution.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the generated video frame. If 'auto', the aspect ratio will be determined automatically based on the input video, and the closest aspect ratio to the input video will be used."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats supported, 3-10 seconds duration, 720-2160px resolution, max 200MB. Max file size: 200.0MB, Min width: 720px, Min height: 720px, Max width: 2160px, Max height: 2160px, Min duration: 3.0s, Max duration: 10.05s, Min FPS: 24.0, Max FPS: 60.0, Timeout: 30.0s"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Video duration in seconds."
    )
    keep_audio: bool = Field(
        default=False, description="Whether to keep the original audio from the video."
    )
    elements: list[str] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    image_urls: list[str] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "video_url": self.video_url,
            "duration": self.duration.value,
            "keep_audio": self.keep_audio,
            "elements": self.elements,
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats supported, 3-10 seconds duration, 720-2160px resolution, max 200MB. Max file size: 200.0MB, Min width: 720px, Min height: 720px, Max width: 2160px, Max height: 2160px, Min duration: 3.0s, Max duration: 10.05s, Min FPS: 24.0, Max FPS: 60.0, Timeout: 30.0s"
    )
    elements: list[str] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    image_urls: list[str] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    keep_audio: bool = Field(
        default=False, description="Whether to keep the original audio from the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "elements": self.elements,
            "image_urls": self.image_urls,
            "keep_audio": self.keep_audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
    video_url: ImageRef = Field(
        default="https://v3b.fal.media/files/b/0a84de68/jXDWywjhagRfR-GuZjoRs_video.mp4", description="URL of the driving pose video. The motion from this video will be transferred to the reference image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.AGGRESSIVE, description="Acceleration levels."
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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the generated video. If 'auto', will be determined from the reference image."
    )
    pose_guidance_start: float = Field(
        default=0.1, description="Start ratio for pose guidance. Controls when pose guidance begins."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_576P, description="Resolution of the generated video. 576p is default, 720p for higher quality. 480p is lower quality."
    )
    image_url: ImageRef = Field(
        default="https://v3b.fal.media/files/b/0a85edaa/GDUCMPrdvOMcI5JpEcU7f.png", description="URL of the reference image to animate. This is the person/character whose appearance will be preserved."
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
        video_url_base64 = await context.image_to_base64(self.video_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "video_url": f"data:image/png;base64,{video_url_base64}",
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
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "preserve_audio": self.preserve_audio,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/steady-dancer",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]


class OneToAllAnimation1_3B(FALNode):
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
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the video to generate."
    )
    image_guidance_scale: float = Field(
        default=2, description="The image guidance scale to use for the video generation."
    )
    pose_guidance_scale: float = Field(
        default=1.5, description="The pose guidance scale to use for the video generation."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use as a reference for the video generation."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to use for the video generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate the video from."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "image_guidance_scale": self.image_guidance_scale,
            "pose_guidance_scale": self.pose_guidance_scale,
            "video_url": self.video_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/one-to-all-animation/1.3b",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]


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
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the video to generate."
    )
    image_guidance_scale: float = Field(
        default=2, description="The image guidance scale to use for the video generation."
    )
    pose_guidance_scale: float = Field(
        default=1.5, description="The pose guidance scale to use for the video generation."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to use as a reference for the video generation."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to use for the video generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate the video from."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "image_guidance_scale": self.image_guidance_scale,
            "pose_guidance_scale": self.pose_guidance_scale,
            "video_url": self.video_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/one-to-all-animation/14b",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]


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
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to enhance with Wan Video. Maximum 200MB file size. Videos longer than 500 frames will have only the first 500 frames processed (~8-21 seconds depending on fps)."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If not provided, a random seed will be used."
    )
    target_resolution: TargetResolution = Field(
        default=TargetResolution.VALUE_720P, description="Target output resolution for the enhanced video. 720p (native, fast) or 1080p (upscaled, slower). Processing is always done at 720p, then upscaled if 1080p selected."
    )
    negative_prompt: str = Field(
        default="oversaturated, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still frame, overall gray, worst quality, low quality, JPEG artifacts, ugly, mutated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static motion, cluttered background, three legs, crowded background, walking backwards", description="Negative prompt to avoid unwanted features."
    )
    creativity: int = Field(
        default=1, description="Controls how much the model enhances/changes the video. 0 = Minimal change (preserves original), 1 = Subtle enhancement (default), 2 = Medium enhancement, 3 = Strong enhancement, 4 = Maximum enhancement."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "seed": self.seed,
            "target_resolution": self.target_resolution.value,
            "negative_prompt": self.negative_prompt,
            "creativity": self.creativity,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vision-enhancer",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]




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

    emotion: Emotion = Field(
        default="", description="Emotion prompt for the generation. Currently supports single-word emotions only."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL to the input video. Must be **15 seconds or shorter**."
    )
    lipsync_mode: LipsyncMode = Field(
        default=LipsyncMode.BOUNCE, description="Lipsync mode when audio and video durations are out of sync."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="URL to the input audio. Must be **15 seconds or shorter**."
    )
    temperature: float = Field(
        default=0.5, description="Controls the expresiveness of the lipsync."
    )
    model_mode: ModelMode = Field(
        default=ModelMode.FACE, description="Controls the edit region and movement scope for the model. Available options: - `lips`: Only lipsync using react-1 (minimal facial changes). - `face`: Lipsync + facial expressions without head movements. - `head`: Lipsync + facial expressions + natural talking head movements."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "emotion": self.emotion.value,
            "video_url": self.video_url,
            "lipsync_mode": self.lipsync_mode.value,
            "audio_url": self.audio_url,
            "temperature": self.temperature,
            "model_mode": self.model_mode.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sync-lipsync/react-1",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]


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

    video_url: VideoRef = Field(
        default=VideoRef()
    )
    subject_is_person: bool = Field(
        default=True, description="Set to False if the subject is not a person."
    )
    output_codec: OutputCodec = Field(
        default=OutputCodec.VP9, description="Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality."
    )
    refine_foreground_edges: bool = Field(
        default=True, description="Improves the quality of the extracted object's edges."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "subject_is_person": self.subject_is_person,
            "output_codec": self.output_codec.value,
            "refine_foreground_edges": self.refine_foreground_edges,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/video-background-removal/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats supported, 3-10 seconds duration, 720-2160px resolution, max 200MB. Max file size: 200.0MB, Min width: 720px, Min height: 720px, Max width: 2160px, Max height: 2160px, Min duration: 3.0s, Max duration: 10.05s, Min FPS: 24.0, Max FPS: 60.0, Timeout: 30.0s"
    )
    elements: list[str] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    image_urls: list[str] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    keep_audio: bool = Field(
        default=False, description="Whether to keep the original audio from the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "elements": self.elements,
            "image_urls": self.image_urls,
            "keep_audio": self.keep_audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/video-to-video/edit",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]



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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the generated video frame. If 'auto', the aspect ratio will be determined automatically based on the input video, and the closest aspect ratio to the input video will be used."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="Reference video URL. Only .mp4/.mov formats supported, 3-10 seconds duration, 720-2160px resolution, max 200MB. Max file size: 200.0MB, Min width: 720px, Min height: 720px, Max width: 2160px, Max height: 2160px, Min duration: 3.0s, Max duration: 10.05s, Min FPS: 24.0, Max FPS: 60.0, Timeout: 30.0s"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Video duration in seconds."
    )
    keep_audio: bool = Field(
        default=False, description="Whether to keep the original audio from the video."
    )
    elements: list[str] = Field(
        default=[], description="Elements (characters/objects) to include. Reference in prompt as @Element1, @Element2, etc. Maximum 4 total (elements + reference images) when using video."
    )
    image_urls: list[str] = Field(
        default=[], description="Reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 4 total (elements + reference images) when using video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "video_url": self.video_url,
            "duration": self.duration.value,
            "keep_audio": self.keep_audio,
            "elements": self.elements,
            "image_urls": self.image_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/video-to-video/reference",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]


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

    video_url: VideoRef = Field(
        default=VideoRef()
    )
    subject_is_person: bool = Field(
        default=True, description="Set to False if the subject is not a person."
    )
    output_codec: OutputCodec = Field(
        default=OutputCodec.VP9, description="Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality."
    )
    refine_foreground_edges: bool = Field(
        default=True, description="Improves the quality of the extracted object's edges."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "subject_is_person": self.subject_is_person,
            "output_codec": self.output_codec.value,
            "refine_foreground_edges": self.refine_foreground_edges,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/video-background-removal",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]


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

    video_url: VideoRef = Field(
        default=VideoRef()
    )
    output_codec: OutputCodec = Field(
        default=OutputCodec.VP9, description="Single VP9 video with alpha channel or two videos (rgb and alpha) in H264 format. H264 is recommended for better RGB quality."
    )
    spill_suppression_strength: str = Field(
        default=0.8, description="Increase the value if green spots remain in the video, decrease if color changes are noticed on the extracted subject."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "output_codec": self.output_codec.value,
            "spill_suppression_strength": self.spill_suppression_strength,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/video-background-removal/green-screen",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]


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
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to retake"
    )
    start_time: float = Field(
        default=0, description="The start time of the video to retake in seconds"
    )
    duration: float = Field(
        default=5, description="The duration of the video to retake in seconds"
    )
    retake_mode: RetakeMode = Field(
        default=RetakeMode.REPLACE_AUDIO_AND_VIDEO, description="The retake mode to use for the retake"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "start_time": self.start_time,
            "duration": self.duration,
            "retake_mode": self.retake_mode.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2/retake-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video to edit"
    )
    prompt: str = Field(
        default="", description="Text description of the desired video content"
    )
    enhance_prompt: bool = Field(
        default=True, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "sync_mode": self.sync_mode,
            "video_url": self.video_url,
            "prompt": self.prompt,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="decart/lucy-edit/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to be segmented."
    )
    detection_threshold: float = Field(
        default=0.5, description="Detection confidence threshold (0.0-1.0). Lower = more detections but less precise. Defaults: 0.5 for existing, 0.7 for new objects. Try 0.2-0.3 if text prompts fail."
    )
    box_prompts: list[str] = Field(
        default=[], description="List of box prompts with optional frame_index."
    )
    point_prompts: list[str] = Field(
        default=[], description="List of point prompts with frame indices."
    )
    boundingbox_zip: bool = Field(
        default=False, description="Return per-frame bounding box overlays as a zip archive."
    )
    frame_index: int = Field(
        default=0, description="Frame index used for initial interaction when mask_url is provided."
    )
    mask_url: str = Field(
        default="", description="The URL of the mask to be applied initially."
    )
    apply_mask: bool = Field(
        default=False, description="Apply the mask on the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "detection_threshold": self.detection_threshold,
            "box_prompts": self.box_prompts,
            "point_prompts": self.point_prompts,
            "boundingbox_zip": self.boundingbox_zip,
            "frame_index": self.frame_index,
            "mask_url": self.mask_url,
            "apply_mask": self.apply_mask,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/video-rle",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

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
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to be segmented."
    )
    detection_threshold: float = Field(
        default=0.5, description="Detection confidence threshold (0.0-1.0). Lower = more detections but less precise."
    )
    box_prompts: list[str] = Field(
        default=[], description="List of box prompt coordinates (x_min, y_min, x_max, y_max)."
    )
    point_prompts: list[str] = Field(
        default=[], description="List of point prompts"
    )
    apply_mask: bool = Field(
        default=True, description="Apply the mask on the video."
    )
    text_prompt: str = Field(
        default="", description="[DEPRECATED] Use 'prompt' instead. Kept for backward compatibility."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "detection_threshold": self.detection_threshold,
            "box_prompts": self.box_prompts,
            "point_prompts": self.point_prompts,
            "apply_mask": self.apply_mask,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sam-3/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]






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
    video_url: VideoRef = Field(
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
    sampler: Sampler = Field(
        default=Sampler.UNIPC, description="Sampler to use for video generation."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    resolution: Resolution = Field(
        default=Resolution.AUTO, description="Resolution of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
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
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
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

        res = await self.submit_request(
            context=context,
            application="fal-ai/editto",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]





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

    video_url: VideoRef = Field(
        default=VideoRef(), description="The input video to be upscaled"
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration mode for VAE decoding. Options: regular (best quality), high (balanced), full (fastest). More accerleation means longer duration videos can be processed too."
    )
    quality: int = Field(
        default=70, description="Quality level for tile blending (0-100). Controls overlap between tiles to prevent grid artifacts. Higher values provide better quality with more overlap. Recommended: 70-85 for high-res videos, 50-70 for faster processing."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.X264__MP4, description="The format of the output video."
    )
    color_fix: bool = Field(
        default=True, description="Color correction enabled."
    )
    output_write_mode: OutputWriteMode = Field(
        default=OutputWriteMode.BALANCED, description="The write mode of the output video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned inline and not stored in history."
    )
    output_quality: OutputQuality = Field(
        default=OutputQuality.HIGH, description="The quality of the output video."
    )
    upscale_factor: float = Field(
        default=2, description="Upscaling factor to be used."
    )
    preserve_audio: bool = Field(
        default=False, description="Copy the original audio tracks into the upscaled video using FFmpeg when possible."
    )
    seed: int = Field(
        default=-1, description="The random seed used for the generation process."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "acceleration": self.acceleration.value,
            "quality": self.quality,
            "output_format": self.output_format.value,
            "color_fix": self.color_fix,
            "output_write_mode": self.output_write_mode.value,
            "sync_mode": self.sync_mode,
            "output_quality": self.output_quality.value,
            "upscale_factor": self.upscale_factor,
            "preserve_audio": self.preserve_audio,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flashvsr/upscale/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]







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

    font_weight: FontWeight = Field(
        default=FontWeight.BOLD, description="Font weight (TikTok style typically uses bold or black)"
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video file to add automatic subtitles to Max file size: 95.4MB, Timeout: 30.0s"
    )
    stroke_width: int = Field(
        default=3, description="Text stroke/outline width in pixels (0 for no stroke)"
    )
    font_color: FontColor = Field(
        default=FontColor.WHITE, description="Subtitle text color for non-active words"
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
    stroke_color: StrokeColor = Field(
        default=StrokeColor.BLACK, description="Text stroke/outline color"
    )
    highlight_color: HighlightColor = Field(
        default=HighlightColor.PURPLE, description="Color for the currently speaking word (karaoke-style highlight)"
    )
    enable_animation: bool = Field(
        default=True, description="Enable animation effects for subtitles (bounce style entrance)"
    )
    font_name: str = Field(
        default="Montserrat", description="Any Google Font name from fonts.google.com (e.g., 'Montserrat', 'Poppins', 'BBH Sans Hegarty')"
    )
    position: Position = Field(
        default=Position.BOTTOM, description="Vertical position of subtitles"
    )
    words_per_subtitle: int = Field(
        default=3, description="Maximum number of words per subtitle segment. Use 1 for single-word display, 2-3 for short phrases, or 8-12 for full sentences."
    )
    background_color: BackgroundColor = Field(
        default=BackgroundColor.NONE, description="Background color behind text ('none' or 'transparent' for no background)"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "font_weight": self.font_weight.value,
            "video_url": self.video_url,
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

        res = await self.submit_request(
            context=context,
            application="fal-ai/workflow-utilities/auto-subtitle",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]



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

    target_fps: TargetFps = Field(
        default=TargetFps.VALUE_30FPS, description="The target FPS of the video to upscale."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to upscale."
    )
    target_resolution: TargetResolution = Field(
        default=TargetResolution.VALUE_1080P, description="The target resolution of the video to upscale."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "target_fps": self.target_fps.value,
            "video_url": self.video_url,
            "target_resolution": self.target_resolution.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance-upscaler/upscale/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]



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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_9_16, description="Aspect ratio of the generated video."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the generated video."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="reference video to generate effect video from."
    )
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "video_url": self.video_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "fps": self.fps,
            "video_description": self.video_description,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/video-as-prompt",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

class MireloAiSfxV1_5VideoToVideo(FALNode):
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
    video_url: VideoRef = Field(
        default=VideoRef(), description="A video url that can accessed from the API to process and add sound effects"
    )
    seed: str = Field(
        default=8069, description="The seed to use for the generation. If not provided, a random seed will be used"
    )
    text_prompt: str = Field(
        default="", description="Additional description to guide the model"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "num_samples": self.num_samples,
            "duration": self.duration,
            "start_offset": self.start_offset,
            "video_url": self.video_url,
            "seed": self.seed,
            "text_prompt": self.text_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="mirelo-ai/sfx-v1.5/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

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
    video_url: VideoRef = Field(
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
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "strength": self.strength,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/krea-wan-14b/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]

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

        res = await self.submit_request(
            context=context,
            application="fal-ai/sora-2/video-to-video/remix",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]









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
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL to the source video file. This video will be used as a reference for the reframe task."
    )
    zoom_factor: float = Field(
        default=0, description="Zoom factor for the video. When this value is greater than 0, the video will be zoomed in by this factor (in relation to the canvas size,) cutting off the edges of the video. A value of 0 means no zoom."
    )
    paste_back: bool = Field(
        default=True, description="Whether to paste back the reframed scene to the original video."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation. Optional for reframing."
    )
    scene_threshold: float = Field(
        default=30, description="Threshold for scene detection sensitivity (0-100). Lower values detect more scenes."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance. Higher values encourage the model to generate images closely related to the text prompt."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    auto_downsample_min_fps: float = Field(
        default=6, description="Minimum FPS for auto downsample."
    )
    negative_prompt: str = Field(
        default="letterboxing, borders, black bars, bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    sampler: Sampler = Field(
        default=Sampler.UNIPC, description="Sampler to use for video generation."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    return_frames_zip: bool = Field(
        default=False, description="If true, also return a ZIP file containing all generated frames."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the generated video."
    )
    resolution: Resolution = Field(
        default=Resolution.AUTO, description="Resolution of the generated video."
    )
    transparency_mode: TransparencyMode = Field(
        default=TransparencyMode.CONTENT_AWARE, description="The transparency mode to apply to the first and last frames. This controls how the transparent areas of the first and last frames are filled."
    )
    trim_borders: bool = Field(
        default=True, description="Whether to trim borders from the video."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: InterpolatorModel = Field(
        default=InterpolatorModel.FILM, description="The model to use for frame interpolation. Options are 'rife' or 'film'."
    )
    enable_auto_downsample: bool = Field(
        default=True, description="Whether to enable auto downsample."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "shift": self.shift,
            "video_url": self.video_url,
            "zoom_factor": self.zoom_factor,
            "paste_back": self.paste_back,
            "acceleration": self.acceleration.value,
            "prompt": self.prompt,
            "scene_threshold": self.scene_threshold,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
            "negative_prompt": self.negative_prompt,
            "sampler": self.sampler.value,
            "video_write_mode": self.video_write_mode.value,
            "return_frames_zip": self.return_frames_zip,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "transparency_mode": self.transparency_mode.value,
            "trim_borders": self.trim_borders,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_auto_downsample": self.enable_auto_downsample,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-apps/long-reframe",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]



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
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for generation."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    num_frames: int = Field(
        default=145, description="Number of frames to generate. Must be between 81 to 129 (inclusive). If the number of frames is greater than 81, the video will be generated with 1.25x more billing units."
    )
    seed: int = Field(
        default=42, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "video_url": self.video_url,
            "audio_url": self.audio_url,
            "num_frames": self.num_frames,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/infinitalk/video-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]






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

    upscale_mode: UpscaleMode = Field(
        default=UpscaleMode.FACTOR, description="The mode to use for the upscale. If 'target', the upscale factor will be calculated based on the target resolution. If 'factor', the upscale factor will be used directly."
    )
    video_url: VideoRef = Field(
        default=VideoRef(), description="The input video to be processed"
    )
    noise_scale: float = Field(
        default=0.1, description="The noise scale to use for the generation process."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.X264__MP4, description="The format of the output video."
    )
    output_write_mode: OutputWriteMode = Field(
        default=OutputWriteMode.BALANCED, description="The write mode of the output video."
    )
    target_resolution: TargetResolution = Field(
        default=TargetResolution.VALUE_1080P, description="The target resolution to upscale to when `upscale_mode` is `target`."
    )
    output_quality: OutputQuality = Field(
        default=OutputQuality.HIGH, description="The quality of the output video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    upscale_factor: float = Field(
        default=2, description="Upscaling factor to be used. Will multiply the dimensions with this factor when `upscale_mode` is `factor`."
    )
    seed: str = Field(
        default="", description="The random seed used for the generation process."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "upscale_mode": self.upscale_mode.value,
            "video_url": self.video_url,
            "noise_scale": self.noise_scale,
            "output_format": self.output_format.value,
            "output_write_mode": self.output_write_mode.value,
            "target_resolution": self.target_resolution.value,
            "output_quality": self.output_quality.value,
            "sync_mode": self.sync_mode,
            "upscale_factor": self.upscale_factor,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/seedvr/upscale/video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]





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
    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the input video."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration to use for inference. Options are 'none' or 'regular'. Accelerated inference will very slightly affect output, but will be significantly faster."
    )
    resolution: Resolution = Field(
        default=Resolution.AUTO, description="Resolution of the edited video."
    )
    return_frames_zip: bool = Field(
        default=False, description="Whether to include a ZIP archive containing all generated frames."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the edited video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    video_type: VideoType = Field(
        default=VideoType.AUTO, description="The type of video you're editing. Use 'general' for most videos, and 'human' for videos emphasizing human subjects and motions. The default value 'auto' means the model will guess based on the first frame of the video."
    )
    image_urls: list[str] = Field(
        default=[], description="URLs of the input images to use as a reference for the generation."
    )
    enable_auto_downsample: bool = Field(
        default=True, description="Whether to enable automatic downsampling. If your video has a high frame rate or is long, enabling longer sequences to be generated. The video will be interpolated back to the original frame rate after generation."
    )
    auto_downsample_min_fps: float = Field(
        default=15, description="The minimum frames per second to downsample the video to."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "video_url": self.video_url,
            "acceleration": self.acceleration.value,
            "resolution": self.resolution.value,
            "return_frames_zip": self.return_frames_zip,
            "aspect_ratio": self.aspect_ratio.value,
            "enable_safety_checker": self.enable_safety_checker,
            "video_type": self.video_type.value,
            "image_urls": self.image_urls,
            "enable_auto_downsample": self.enable_auto_downsample,
            "auto_downsample_min_fps": self.auto_downsample_min_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-vace-apps/video-edit",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video"]