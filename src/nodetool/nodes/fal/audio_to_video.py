from enum import Enum
from pydantic import Field
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class Ltx219BDistilledAudioToVideoLora(FALNode):
    """
    LTX-2 19B Distilled
    video, generation, audio-to-video, visualization, lora

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

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

    class VideoQuality(Enum):
        """
        The quality of the generated video.
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        MAXIMUM = "maximum"


    match_audio_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the audio duration and FPS. When disabled, use the specified num_frames."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
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
        default="landscape_4_3", description="The size of the generated video. Use 'auto' to match the input image dimensions if provided."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio before using it as conditioning."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Optional URL of an image to use as the first frame of the video."
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
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Values below 1.0 will allow the model to change the audio, while a value of exactly 1.0 will use the input audio without modification."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to generate the video from."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "match_audio_length": self.match_audio_length,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "use_multiscale": self.use_multiscale,
            "fps": self.fps,
            "loras": self.loras,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "image_strength": self.image_strength,
            "preprocess_audio": self.preprocess_audio,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "audio_url": self.audio_url,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/audio-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["match_audio_length", "prompt", "acceleration", "use_multiscale", "fps"]

class Ltx219BAudioToVideoLora(FALNode):
    """
    LTX-2 19B
    video, generation, audio-to-video, visualization, lora

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

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

    class VideoQuality(Enum):
        """
        The quality of the generated video.
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        MAXIMUM = "maximum"


    match_audio_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the audio duration and FPS. When disabled, use the specified num_frames."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
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
        default="landscape_4_3", description="The size of the generated video. Use 'auto' to match the input image dimensions if provided."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio before using it as conditioning."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Optional URL of an image to use as the first frame of the video."
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
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Values below 1.0 will allow the model to change the audio, while a value of exactly 1.0 will use the input audio without modification."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to generate the video from."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "match_audio_length": self.match_audio_length,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "use_multiscale": self.use_multiscale,
            "num_inference_steps": self.num_inference_steps,
            "fps": self.fps,
            "loras": self.loras,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "image_strength": self.image_strength,
            "preprocess_audio": self.preprocess_audio,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "audio_url": self.audio_url,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/audio-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["match_audio_length", "prompt", "acceleration", "use_multiscale", "num_inference_steps"]

class Ltx219BDistilledAudioToVideo(FALNode):
    """
    LTX-2 19B Distilled
    video, generation, audio-to-video, visualization

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

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

    class VideoQuality(Enum):
        """
        The quality of the generated video.
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        MAXIMUM = "maximum"


    match_audio_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the audio duration and FPS. When disabled, use the specified num_frames."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_size: str = Field(
        default="landscape_4_3", description="The size of the generated video. Use 'auto' to match the input image dimensions if provided."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio before using it as conditioning."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Optional URL of an image to use as the first frame of the video."
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
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Values below 1.0 will allow the model to change the audio, while a value of exactly 1.0 will use the input audio without modification."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to generate the video from."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "match_audio_length": self.match_audio_length,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "use_multiscale": self.use_multiscale,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "image_strength": self.image_strength,
            "preprocess_audio": self.preprocess_audio,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "audio_url": self.audio_url,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["match_audio_length", "prompt", "acceleration", "use_multiscale", "fps"]

class Ltx219BAudioToVideo(FALNode):
    """
    LTX-2 19B
    video, generation, audio-to-video, visualization

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

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

    class VideoQuality(Enum):
        """
        The quality of the generated video.
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        MAXIMUM = "maximum"


    match_audio_length: bool = Field(
        default=True, description="When enabled, the number of frames will be calculated based on the audio duration and FPS. When disabled, use the specified num_frames."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
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
        default="landscape_4_3", description="The size of the generated video. Use 'auto' to match the input image dimensions if provided."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio before using it as conditioning."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="Optional URL of an image to use as the first frame of the video."
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
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Values below 1.0 will allow the model to change the audio, while a value of exactly 1.0 will use the input audio without modification."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to generate the video from."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "match_audio_length": self.match_audio_length,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "use_multiscale": self.use_multiscale,
            "num_inference_steps": self.num_inference_steps,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "image_strength": self.image_strength,
            "preprocess_audio": self.preprocess_audio,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "audio_strength": self.audio_strength,
            "end_image_strength": self.end_image_strength,
            "audio_url": self.audio_url,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["match_audio_length", "prompt", "acceleration", "use_multiscale", "num_inference_steps"]

class ElevenlabsDubbing(FALNode):
    """
    ElevenLabs Dubbing
    video, generation, audio-to-video, visualization

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    video_url: VideoRef = Field(
        default=VideoRef(), description="URL of the video file to dub. Either audio_url or video_url must be provided. If both are provided, video_url takes priority."
    )
    highest_resolution: bool = Field(
        default=True, description="Whether to use the highest resolution for dubbing."
    )
    audio_url: VideoRef = Field(
        default=VideoRef(), description="URL of the audio file to dub. Either audio_url or video_url must be provided."
    )
    target_lang: str = Field(
        default="", description="Target language code for dubbing (ISO 639-1)"
    )
    num_speakers: str = Field(
        default="", description="Number of speakers in the audio. If not provided, will be auto-detected."
    )
    source_lang: str = Field(
        default="", description="Source language code. If not provided, will be auto-detected."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video_url,
            "highest_resolution": self.highest_resolution,
            "audio_url": self.audio_url,
            "target_lang": self.target_lang,
            "num_speakers": self.num_speakers,
            "source_lang": self.source_lang,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/elevenlabs/dubbing",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["video_url", "highest_resolution", "audio_url", "target_lang", "num_speakers"]

class LongcatMultiAvatarImageAudioToVideo(FALNode):
    """
    Longcat Multi Avatar
    video, generation, audio-to-video, visualization

    Use cases:
    - Automated content generation
    - Creative workflows
    - Batch processing
    - Professional applications
    - Rapid prototyping
    """

    class Resolution(Enum):
        """
        Resolution of the generated video (480p or 720p). Billing is per video-second (16 frames): 480p is 1 unit per second and 720p is 4 units per second.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AudioType(Enum):
        """
        How to combine the two audio tracks. 'para' (parallel) plays both simultaneously, 'add' (sequential) plays person 1 first then person 2.
        """
        PARA = "para"
        ADD = "add"


    prompt: str = Field(
        default="Two people are having a conversation with natural expressions and movements.", description="The prompt to guide the video generation."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to use."
    )
    audio_url_person2: str = Field(
        default="https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/refs/heads/main/assets/avatar/multi/sing_woman.WAV", description="The URL of the audio file for person 2 (right side)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    bbox_person1: str = Field(
        default="", description="Bounding box for person 1. If not provided, defaults to left half of image."
    )
    negative_prompt: str = Field(
        default="Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", description="The negative prompt to avoid in the video generation."
    )
    text_guidance_scale: float = Field(
        default=4, description="The text guidance scale for classifier-free guidance."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the generated video (480p or 720p). Billing is per video-second (16 frames): 480p is 1 unit per second and 720p is 4 units per second."
    )
    audio_type: AudioType = Field(
        default=AudioType.PARA, description="How to combine the two audio tracks. 'para' (parallel) plays both simultaneously, 'add' (sequential) plays person 1 first then person 2."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image containing two speakers."
    )
    audio_url_person1: str = Field(
        default="https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/refs/heads/main/assets/avatar/multi/sing_man.WAV", description="The URL of the audio file for person 1 (left side)."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    audio_guidance_scale: float = Field(
        default=4, description="The audio guidance scale. Higher values may lead to exaggerated mouth movements."
    )
    bbox_person2: str = Field(
        default="", description="Bounding box for person 2. If not provided, defaults to right half of image."
    )
    num_segments: int = Field(
        default=1, description="Number of video segments to generate. Each segment adds ~5 seconds of video. First segment is ~5.8s, additional segments are 5s each."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_inference_steps": self.num_inference_steps,
            "audio_url_person2": self.audio_url_person2,
            "enable_safety_checker": self.enable_safety_checker,
            "bbox_person1": self.bbox_person1,
            "negative_prompt": self.negative_prompt,
            "text_guidance_scale": self.text_guidance_scale,
            "resolution": self.resolution.value,
            "audio_type": self.audio_type.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "audio_url_person1": self.audio_url_person1,
            "seed": self.seed,
            "audio_guidance_scale": self.audio_guidance_scale,
            "bbox_person2": self.bbox_person2,
            "num_segments": self.num_segments,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-multi-avatar/image-audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "num_inference_steps", "audio_url_person2", "enable_safety_checker", "bbox_person1"]