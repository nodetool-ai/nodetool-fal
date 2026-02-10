from enum import Enum
from pydantic import Field
from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.nodes.fal.types import LoRAInput
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
        X264_MP4 = "X264 (.mp4)"
        VP9_WEBM = "VP9 (.webm)"
        PRORES4444_MOV = "PRORES4444 (.mov)"
        GIF_GIF = "GIF (.gif)"

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
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    loras: list[LoRAInput] = Field(
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
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio before using it as conditioning."
    )
    image: ImageRef = Field(
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
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to generate the video from."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Values below 1.0 will allow the model to change the audio, while a value of exactly 1.0 will use the input audio without modification."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "match_audio_length": self.match_audio_length,
            "use_multiscale": self.use_multiscale,
            "acceleration": self.acceleration.value,
            "prompt": self.prompt,
            "fps": self.fps,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "image_strength": self.image_strength,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "preprocess_audio": self.preprocess_audio,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "audio_url": self.audio,
            "audio_strength": self.audio_strength,
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
        return ["match_audio_length", "use_multiscale", "acceleration", "prompt", "fps"]

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
        X264_MP4 = "X264 (.mp4)"
        VP9_WEBM = "VP9 (.webm)"
        PRORES4444_MOV = "PRORES4444 (.mov)"
        GIF_GIF = "GIF (.gif)"

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
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Values below 1.0 will allow the model to change the audio, while a value of exactly 1.0 will use the input audio without modification."
    )
    fps: float = Field(
        default=25, description="The frames per second of the generated video."
    )
    loras: list[LoRAInput] = Field(
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
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio before using it as conditioning."
    )
    image: ImageRef = Field(
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
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to generate the video from."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "match_audio_length": self.match_audio_length,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "use_multiscale": self.use_multiscale,
            "audio_strength": self.audio_strength,
            "fps": self.fps,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "image_strength": self.image_strength,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "preprocess_audio": self.preprocess_audio,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "audio_url": self.audio,
            "num_inference_steps": self.num_inference_steps,
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
        return ["match_audio_length", "prompt", "acceleration", "use_multiscale", "audio_strength"]

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
        X264_MP4 = "X264 (.mp4)"
        VP9_WEBM = "VP9 (.webm)"
        PRORES4444_MOV = "PRORES4444 (.mov)"
        GIF_GIF = "GIF (.gif)"

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
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
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
        default="landscape_4_3", description="The size of the generated video. Use 'auto' to match the input image dimensions if provided."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio before using it as conditioning."
    )
    image: ImageRef = Field(
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
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to generate the video from."
    )
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Values below 1.0 will allow the model to change the audio, while a value of exactly 1.0 will use the input audio without modification."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "match_audio_length": self.match_audio_length,
            "use_multiscale": self.use_multiscale,
            "acceleration": self.acceleration.value,
            "prompt": self.prompt,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "image_strength": self.image_strength,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "preprocess_audio": self.preprocess_audio,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "audio_url": self.audio,
            "audio_strength": self.audio_strength,
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
        return ["match_audio_length", "use_multiscale", "acceleration", "prompt", "fps"]

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
        X264_MP4 = "X264 (.mp4)"
        VP9_WEBM = "VP9 (.webm)"
        PRORES4444_MOV = "PRORES4444 (.mov)"
        GIF_GIF = "GIF (.gif)"

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
    audio_strength: float = Field(
        default=1, description="Audio conditioning strength. Values below 1.0 will allow the model to change the audio, while a value of exactly 1.0 will use the input audio without modification."
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
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    preprocess_audio: bool = Field(
        default=True, description="Whether to preprocess the audio before using it as conditioning."
    )
    image: ImageRef = Field(
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
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to generate the video from."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "match_audio_length": self.match_audio_length,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "use_multiscale": self.use_multiscale,
            "audio_strength": self.audio_strength,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "image_strength": self.image_strength,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "preprocess_audio": self.preprocess_audio,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "audio_url": self.audio,
            "num_inference_steps": self.num_inference_steps,
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
        return ["match_audio_length", "prompt", "acceleration", "use_multiscale", "audio_strength"]

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

    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video file to dub. Either audio_url or video_url must be provided. If both are provided, video_url takes priority."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="URL of the audio file to dub. Either audio_url or video_url must be provided."
    )
    highest_resolution: bool = Field(
        default=True, description="Whether to use the highest resolution for dubbing."
    )
    num_speakers: str = Field(
        default="", description="Number of speakers in the audio. If not provided, will be auto-detected."
    )
    target_lang: str = Field(
        default="", description="Target language code for dubbing (ISO 639-1)"
    )
    source_lang: str = Field(
        default="", description="Source language code. If not provided, will be auto-detected."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video,
            "audio_url": self.audio,
            "highest_resolution": self.highest_resolution,
            "num_speakers": self.num_speakers,
            "target_lang": self.target_lang,
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
        return ["video", "audio", "highest_resolution", "num_speakers", "target_lang"]

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
    image: ImageRef = Field(
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
        image_base64 = await context.image_to_base64(self.image)
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
            "image_url": f"data:image/png;base64,{image_base64}",
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

class LongcatSingleAvatarImageAudioToVideo(FALNode):
    """
    LongCat-Video-Avatar is an audio-driven video generation model that can generates super-realistic, lip-synchronized long video generation with natural dynamics and consistent identity.
    video, generation, audio-to-video, visualization

    Use cases:
    - Audio-driven video generation
    - Music visualization
    - Talking head animation
    - Audio-synced content creation
    - Podcast video generation
    """

    class Resolution(Enum):
        """
        Resolution of the generated video (480p or 720p). Billing is per video-second (16 frames): 480p is 1 unit per second and 720p is 4 units per second.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"


    prompt: str = Field(
        default="", description="The prompt to guide the video generation."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the generated video (480p or 720p). Billing is per video-second (16 frames): 480p is 1 unit per second and 720p is 4 units per second."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    audio_guidance_scale: float = Field(
        default=4, description="The audio guidance scale. Higher values may lead to exaggerated mouth movements."
    )
    num_segments: int = Field(
        default=1, description="Number of video segments to generate. Each segment adds ~5 seconds of video. First segment is ~5.8s, additional segments are 5s each."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to animate."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file to drive the avatar."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to use."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", description="The negative prompt to avoid in the video generation."
    )
    text_guidance_scale: float = Field(
        default=4, description="The text guidance scale for classifier-free guidance."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "audio_guidance_scale": self.audio_guidance_scale,
            "num_segments": self.num_segments,
            "image_url": f"data:image/png;base64,{image_base64}",
            "audio_url": self.audio,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "text_guidance_scale": self.text_guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-single-avatar/image-audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "resolution", "enable_safety_checker", "audio_guidance_scale", "num_segments"]

class LongcatSingleAvatarAudioToVideo(FALNode):
    """
    LongCat-Video-Avatar is an audio-driven video generation model that can generates super-realistic, lip-synchronized long video generation with natural dynamics and consistent identity.
    video, generation, audio-to-video, visualization

    Use cases:
    - Audio-driven video generation
    - Music visualization
    - Talking head animation
    - Audio-synced content creation
    - Podcast video generation
    """

    class Resolution(Enum):
        """
        Resolution of the generated video (480p or 720p). Billing is per video-second (16 frames): 480p is 1 unit per second and 720p is 4 units per second.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"


    prompt: str = Field(
        default="A person is talking naturally with natural expressions and movements.", description="The prompt to guide the video generation."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the generated video (480p or 720p). Billing is per video-second (16 frames): 480p is 1 unit per second and 720p is 4 units per second."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    audio_guidance_scale: float = Field(
        default=4, description="The audio guidance scale. Higher values may lead to exaggerated mouth movements."
    )
    num_segments: int = Field(
        default=1, description="Number of video segments to generate. Each segment adds ~5 seconds of video. First segment is ~5.8s, additional segments are 5s each."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file to drive the avatar."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to use."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", description="The negative prompt to avoid in the video generation."
    )
    text_guidance_scale: float = Field(
        default=4, description="The text guidance scale for classifier-free guidance."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "audio_guidance_scale": self.audio_guidance_scale,
            "num_segments": self.num_segments,
            "audio_url": self.audio,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "text_guidance_scale": self.text_guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-single-avatar/audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "resolution", "enable_safety_checker", "audio_guidance_scale", "num_segments"]

class ArgilAvatarsAudioToVideo(FALNode):
    """
    High-quality avatar videos that feel real, generated from your audio
    video, generation, audio-to-video, visualization

    Use cases:
    - Audio-driven video generation
    - Music visualization
    - Talking head animation
    - Audio-synced content creation
    - Podcast video generation
    """

    class Avatar(Enum):
        MIA_OUTDOOR_UGC = "Mia outdoor (UGC)"
        LARA_MASTERCLASS = "Lara (Masterclass)"
        INES_UGC = "Ines (UGC)"
        MARIA_MASTERCLASS = "Maria (Masterclass)"
        EMMA_UGC = "Emma (UGC)"
        SIENNA_MASTERCLASS = "Sienna (Masterclass)"
        ELENA_UGC = "Elena (UGC)"
        JASMINE_MASTERCLASS = "Jasmine (Masterclass)"
        AMARA_MASTERCLASS = "Amara (Masterclass)"
        RYAN_PODCAST_UGC = "Ryan podcast (UGC)"
        TYLER_MASTERCLASS = "Tyler (Masterclass)"
        JAYSE_MASTERCLASS = "Jayse (Masterclass)"
        PAUL_MASTERCLASS = "Paul (Masterclass)"
        MATTEO_UGC = "Matteo (UGC)"
        DANIEL_CAR_UGC = "Daniel car (UGC)"
        DARIO_MASTERCLASS = "Dario (Masterclass)"
        VIVA_MASTERCLASS = "Viva (Masterclass)"
        CHEN_MASTERCLASS = "Chen (Masterclass)"
        ALEX_MASTERCLASS = "Alex (Masterclass)"
        VANESSA_UGC = "Vanessa (UGC)"
        LAURENT_UGC = "Laurent (UGC)"
        NOEMIE_CAR_UGC = "Noemie car (UGC)"
        BRANDON_UGC = "Brandon (UGC)"
        BYRON_MASTERCLASS = "Byron (Masterclass)"
        CALISTA_MASTERCLASS = "Calista (Masterclass)"
        MILO_MASTERCLASS = "Milo (Masterclass)"
        FABIEN_MASTERCLASS = "Fabien (Masterclass)"
        ROSE_UGC = "Rose (UGC)"


    avatar: Avatar = Field(
        default=""
    )
    remove_background: bool = Field(
        default=False, description="Enabling the remove background feature will result in a 50% increase in the price."
    )
    audio: AudioRef = Field(
        default=AudioRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "avatar": self.avatar.value,
            "remove_background": self.remove_background,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="argil/avatars/audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["avatar", "remove_background", "audio"]

class WanV2214bSpeechToVideo(FALNode):
    """
    Wan-S2V is a video model that generates high-quality videos from static images and audio, with realistic facial expressions, body movements, and professional camera work for film and television applications
    video, generation, audio-to-video, visualization

    Use cases:
    - Audio-driven video generation
    - Music visualization
    - Talking head animation
    - Audio-synced content creation
    - Podcast video generation
    """

    class VideoWriteMode(Enum):
        """
        The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class Resolution(Enum):
        """
        Resolution of the generated video (480p, 580p, or 720p).
        """
        VALUE_480P = "480p"
        VALUE_580P = "580p"
        VALUE_720P = "720p"

    class VideoQuality(Enum):
        """
        The quality of the output video. Higher quality means better visual quality but larger file size.
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        MAXIMUM = "maximum"


    shift: float = Field(
        default=5, description="Shift value for the video. Must be between 1.0 and 10.0."
    )
    prompt: str = Field(
        default="", description="The text prompt used for video generation."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 4 to 60. When using interpolation and `adjust_fps_for_interpolation` is set to true (default true,) the final FPS will be multiplied by the number of interpolated frames plus one. For example, if the generated frames per second is 16 and the number of interpolated frames is 1, the final frames per second will be 32. If `adjust_fps_for_interpolation` is set to false, this value will be used as-is."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    num_frames: int = Field(
        default=80, description="Number of frames to generate. Must be between 40 to 120, (must be multiple of 4)."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the generated video (480p, 580p, or 720p)."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the output video. Higher quality means better visual quality but larger file size."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    num_inference_steps: int = Field(
        default=27, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "shift": self.shift,
            "prompt": self.prompt,
            "frames_per_second": self.frames_per_second,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "audio_url": self.audio,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-14b/speech-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["shift", "prompt", "frames_per_second", "enable_safety_checker", "num_frames"]

class StableAvatar(FALNode):
    """
    Stable Avatar generates audio-driven video avatars up to five minutes long
    video, generation, audio-to-video, visualization

    Use cases:
    - Audio-driven video generation
    - Music visualization
    - Talking head animation
    - Audio-synced content creation
    - Podcast video generation
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the video to generate. If 'auto', the aspect ratio will be determined by the reference image.
        """
        RATIO_16_9 = "16:9"
        RATIO_1_1 = "1:1"
        RATIO_9_16 = "9:16"
        AUTO = "auto"


    prompt: str = Field(
        default="", description="The prompt to use for the video generation."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the video to generate. If 'auto', the aspect ratio will be determined by the reference image."
    )
    perturbation: float = Field(
        default=0.1, description="The amount of perturbation to use for the video generation. 0.0 means no perturbation, 1.0 means full perturbation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )
    guidance_scale: float = Field(
        default=5, description="The guidance scale to use for the video generation."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the video generation."
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to use for the video generation."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to use as a reference for the video generation."
    )
    audio_guidance_scale: float = Field(
        default=4, description="The audio guidance scale to use for the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "perturbation": self.perturbation,
            "image_url": f"data:image/png;base64,{image_base64}",
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "audio_url": self.audio,
            "audio_guidance_scale": self.audio_guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-avatar",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "perturbation", "image", "guidance_scale"]

class EchomimicV3(FALNode):
    """
    EchoMimic V3 generates a talking avatar model from a picture, audio and text prompt.
    video, generation, audio-to-video, visualization

    Use cases:
    - Audio-driven video generation
    - Music visualization
    - Talking head animation
    - Audio-synced content creation
    - Podcast video generation
    """

    prompt: str = Field(
        default="", description="The prompt to use for the video generation."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio to use as a reference for the video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )
    guidance_scale: float = Field(
        default=4.5, description="The guidance scale to use for the video generation."
    )
    audio_guidance_scale: float = Field(
        default=2.5, description="The audio guidance scale to use for the video generation."
    )
    num_frames_per_generation: int = Field(
        default=121, description="The number of frames to generate at once."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use for the video generation."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "audio_url": self.audio,
            "image_url": f"data:image/png;base64,{image_base64}",
            "guidance_scale": self.guidance_scale,
            "audio_guidance_scale": self.audio_guidance_scale,
            "num_frames_per_generation": self.num_frames_per_generation,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/echomimic-v3",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "audio", "image", "guidance_scale", "audio_guidance_scale"]

class VeedAvatarsAudioToVideo(FALNode):
    """
    Generate high-quality videos with UGC-like avatars from audio
    video, generation, audio-to-video, visualization

    Use cases:
    - Audio-driven video generation
    - Music visualization
    - Talking head animation
    - Audio-synced content creation
    - Podcast video generation
    """

    class AvatarId(Enum):
        """
        The avatar to use for the video
        """
        EMILY_VERTICAL_PRIMARY = "emily_vertical_primary"
        EMILY_VERTICAL_SECONDARY = "emily_vertical_secondary"
        MARCUS_VERTICAL_PRIMARY = "marcus_vertical_primary"
        MARCUS_VERTICAL_SECONDARY = "marcus_vertical_secondary"
        MIRA_VERTICAL_PRIMARY = "mira_vertical_primary"
        MIRA_VERTICAL_SECONDARY = "mira_vertical_secondary"
        JASMINE_VERTICAL_PRIMARY = "jasmine_vertical_primary"
        JASMINE_VERTICAL_SECONDARY = "jasmine_vertical_secondary"
        JASMINE_VERTICAL_WALKING = "jasmine_vertical_walking"
        AISHA_VERTICAL_WALKING = "aisha_vertical_walking"
        ELENA_VERTICAL_PRIMARY = "elena_vertical_primary"
        ELENA_VERTICAL_SECONDARY = "elena_vertical_secondary"
        ANY_MALE_VERTICAL_PRIMARY = "any_male_vertical_primary"
        ANY_FEMALE_VERTICAL_PRIMARY = "any_female_vertical_primary"
        ANY_MALE_VERTICAL_SECONDARY = "any_male_vertical_secondary"
        ANY_FEMALE_VERTICAL_SECONDARY = "any_female_vertical_secondary"
        ANY_FEMALE_VERTICAL_WALKING = "any_female_vertical_walking"
        EMILY_PRIMARY = "emily_primary"
        EMILY_SIDE = "emily_side"
        MARCUS_PRIMARY = "marcus_primary"
        MARCUS_SIDE = "marcus_side"
        AISHA_WALKING = "aisha_walking"
        ELENA_PRIMARY = "elena_primary"
        ELENA_SIDE = "elena_side"
        ANY_MALE_PRIMARY = "any_male_primary"
        ANY_FEMALE_PRIMARY = "any_female_primary"
        ANY_MALE_SIDE = "any_male_side"
        ANY_FEMALE_SIDE = "any_female_side"


    audio: AudioRef = Field(
        default=AudioRef()
    )
    avatar_id: AvatarId = Field(
        default="", description="The avatar to use for the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "audio_url": self.audio,
            "avatar_id": self.avatar_id.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/avatars/audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["audio", "avatar_id"]