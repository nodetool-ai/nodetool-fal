from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.nodes.fal.types import DynamicMask, Frame, KeyframeTransition, KlingV3ComboElementInput, KlingV3MultiPromptElement, LoRAInput, LoRAWeight, LoraWeight, OmniVideoElementInput, Turn
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class PixverseV56ImageToVideo(FALNode):
    """
    Generate high-quality videos from images with Pixverse v5.6.
    video, generation, pixverse, v5.6, image-to-video, img2vid

    Use cases:
    - Animate photos into professional video clips
    - Create dynamic product showcase videos
    - Generate stylized video content from artwork
    - Produce high-resolution social media animations
    - Transform static images with various visual styles
    """

    class PixverseV56Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class PixverseV56Duration(Enum):
        """
        The duration of the generated video in seconds. 1080p videos are limited to 5 or 8 seconds
        """
        FIVE_SECONDS = "5"
        EIGHT_SECONDS = "8"
        TEN_SECONDS = "10"

    class PixverseV56Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class PixverseV56ThinkingType(Enum):
        """
        Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision
        """
        ENABLED = "enabled"
        DISABLED = "disabled"
        AUTO = "auto"


    prompt: str = Field(
        default="", description="Text prompt describing the desired video motion"
    )
    resolution: PixverseV56Resolution = Field(
        default=PixverseV56Resolution.VALUE_720P, description="The resolution quality of the output video"
    )
    duration: PixverseV56Duration = Field(
        default=PixverseV56Duration.FIVE_SECONDS, description="The duration of the generated video in seconds"
    )
    style: PixverseV56Style | None = Field(
        default=None, description="Optional visual style for the video"
    )
    thinking_type: PixverseV56ThinkingType | None = Field(
        default=None, description="Thinking mode for video generation"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The image to transform into a video"
    )
    generate_audio_switch: bool = Field(
        default=False, description="Whether to generate audio for the video"
    )
    seed: int = Field(
        default=-1, description="Optional seed for reproducible generation"
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "image_url": f"data:image/png;base64,{image_base64}",
            "generate_audio_switch": self.generate_audio_switch,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"
        RATIO_21_9 = "21:9"
        RATIO_9_21 = "9:21"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    loop: bool = Field(
        default=False, description="Whether the video should loop (end of video is blended with the beginning)"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="An image to blend the end of the video with"
    )
    image: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "loop": self.loop,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt", "resolution"]

class AMTFrameInterpolation(FALNode):
    """
    AMT Frame Interpolation creates smooth transitions between image frames.
    video, interpolation, frame-generation, amt, image-to-video

    Use cases:
    - Create smooth transitions between images
    - Generate intermediate frames
    - Animate image sequences
    - Create video from image pairs
    - Produce smooth motion effects
    """

    frames: list[Frame] = Field(
        default=[], description="Frames to interpolate"
    )
    recursive_interpolation_passes: int = Field(
        default=4, description="Number of recursive interpolation passes"
    )
    output_fps: int = Field(
        default=24, description="Output frames per second"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "frames": [item.model_dump(exclude={"type"}) for item in self.frames],
            "recursive_interpolation_passes": self.recursive_interpolation_passes,
            "output_fps": self.output_fps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/amt-interpolation/frame-interpolation",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]

class AIAvatar(FALNode):
    """
    MultiTalk generates talking avatar videos from images and audio files.
    video, avatar, talking-head, multitalk, image-to-video

    Use cases:
    - Create talking avatar videos
    - Animate portrait photos with audio
    - Generate spokesperson videos
    - Produce avatar presentations
    - Create personalized video messages
    """

    class AIAvatarResolution(Enum):
        """
        Resolution of the video to generate. Must be either 480p or 720p.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class Acceleration(Enum):
        """
        The acceleration level to use for generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: AIAvatarResolution = Field(
        default=AIAvatarResolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
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
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "audio_url": self.audio,
            "num_frames": self.num_frames,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ai-avatar",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "audio"]

class AIAvatarSingleText(FALNode):
    """
    MultiTalk generates talking avatar videos from an image and text input.
    video, avatar, talking-head, text-to-speech, image-to-video

    Use cases:
    - Create avatar videos from text
    - Generate talking heads with TTS
    - Produce text-driven avatars
    - Create virtual presenters
    - Generate automated spokesperson videos
    """

    class AIAvatarSingleTextResolution(Enum):
        """
        Resolution of the video to generate. Must be either 480p or 720p.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class Acceleration(Enum):
        """
        The acceleration level to use for generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class Voice(Enum):
        """
        The voice to use for speech generation
        """
        ARIA = "Aria"
        ROGER = "Roger"
        SARAH = "Sarah"
        LAURA = "Laura"
        CHARLIE = "Charlie"
        GEORGE = "George"
        CALLUM = "Callum"
        RIVER = "River"
        LIAM = "Liam"
        CHARLOTTE = "Charlotte"
        ALICE = "Alice"
        MATILDA = "Matilda"
        WILL = "Will"
        JESSICA = "Jessica"
        ERIC = "Eric"
        CHRIS = "Chris"
        BRIAN = "Brian"
        DANIEL = "Daniel"
        LILY = "Lily"
        BILL = "Bill"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: AIAvatarSingleTextResolution = Field(
        default=AIAvatarSingleTextResolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for generation."
    )
    text_input: str = Field(
        default="", description="The text input to guide video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    voice: Voice = Field(
        default="", description="The voice to use for speech generation"
    )
    seed: int = Field(
        default=42, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    num_frames: int = Field(
        default=136, description="Number of frames to generate. Must be between 81 to 129 (inclusive). If the number of frames is greater than 81, the video will be generated with 1.25x more billing units."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "text_input": self.text_input,
            "image_url": f"data:image/png;base64,{image_base64}",
            "voice": self.voice.value,
            "seed": self.seed,
            "num_frames": self.num_frames,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ai-avatar/single-text",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "text"]

class AIAvatarMultiText(FALNode):
    """
    MultiTalk generates multi-speaker avatar videos from images and text.
    video, avatar, multi-speaker, talking-head, image-to-video

    Use cases:
    - Create multi-speaker conversations
    - Generate dialogue between avatars
    - Produce interactive presentations
    - Create conversational content
    - Generate multi-character scenes
    """

    class Acceleration(Enum):
        """
        The acceleration level to use for generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class AIAvatarMultiTextResolution(Enum):
        """
        Resolution of the video to generate. Must be either 480p or 720p.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class Voice2(Enum):
        """
        The second person's voice to use for speech generation
        """
        ARIA = "Aria"
        ROGER = "Roger"
        SARAH = "Sarah"
        LAURA = "Laura"
        CHARLIE = "Charlie"
        GEORGE = "George"
        CALLUM = "Callum"
        RIVER = "River"
        LIAM = "Liam"
        CHARLOTTE = "Charlotte"
        ALICE = "Alice"
        MATILDA = "Matilda"
        WILL = "Will"
        JESSICA = "Jessica"
        ERIC = "Eric"
        CHRIS = "Chris"
        BRIAN = "Brian"
        DANIEL = "Daniel"
        LILY = "Lily"
        BILL = "Bill"

    class Voice1(Enum):
        """
        The first person's voice to use for speech generation
        """
        ARIA = "Aria"
        ROGER = "Roger"
        SARAH = "Sarah"
        LAURA = "Laura"
        CHARLIE = "Charlie"
        GEORGE = "George"
        CALLUM = "Callum"
        RIVER = "River"
        LIAM = "Liam"
        CHARLOTTE = "Charlotte"
        ALICE = "Alice"
        MATILDA = "Matilda"
        WILL = "Will"
        JESSICA = "Jessica"
        ERIC = "Eric"
        CHRIS = "Chris"
        BRIAN = "Brian"
        DANIEL = "Daniel"
        LILY = "Lily"
        BILL = "Bill"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    second_text_input: str = Field(
        default="", description="The text input to guide video generation."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for generation."
    )
    resolution: AIAvatarMultiTextResolution = Field(
        default=AIAvatarMultiTextResolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    first_text_input: str = Field(
        default="", description="The text input to guide video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    voice2: Voice2 = Field(
        default=Voice2.ROGER, description="The second person's voice to use for speech generation"
    )
    voice1: Voice1 = Field(
        default=Voice1.SARAH, description="The first person's voice to use for speech generation"
    )
    seed: int = Field(
        default=81, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    num_frames: int = Field(
        default=191, description="Number of frames to generate. Must be between 81 to 129 (inclusive). If the number of frames is greater than 81, the video will be generated with 1.25x more billing units."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "second_text_input": self.second_text_input,
            "acceleration": self.acceleration.value,
            "resolution": self.resolution.value,
            "first_text_input": self.first_text_input,
            "image_url": f"data:image/png;base64,{image_base64}",
            "voice2": self.voice2.value,
            "voice1": self.voice1.value,
            "seed": self.seed,
            "num_frames": self.num_frames,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ai-avatar/multi-text",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["images", "texts"]

class AIAvatarMulti(FALNode):
    """
    MultiTalk generates multi-speaker avatar videos with audio synchronization.
    video, avatar, multi-speaker, talking-head, image-to-video

    Use cases:
    - Create multi-speaker videos with audio
    - Generate synchronized dialogue
    - Produce conversation videos
    - Create interactive characters
    - Generate multi-avatar content
    """

    class AIAvatarMultiResolution(Enum):
        """
        Resolution of the video to generate. Must be either 480p or 720p.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class Acceleration(Enum):
        """
        The acceleration level to use for generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: AIAvatarMultiResolution = Field(
        default=AIAvatarMultiResolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for generation."
    )
    first_audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the Person 1 audio file."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    second_audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the Person 2 audio file."
    )
    seed: int = Field(
        default=81, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    use_only_first_audio: bool = Field(
        default=False, description="Whether to use only the first audio file."
    )
    num_frames: int = Field(
        default=181, description="Number of frames to generate. Must be between 81 to 129 (inclusive). If the number of frames is greater than 81, the video will be generated with 1.25x more billing units."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "first_audio_url": self.first_audio,
            "image_url": f"data:image/png;base64,{image_base64}",
            "second_audio_url": self.second_audio,
            "seed": self.seed,
            "use_only_first_audio": self.use_only_first_audio,
            "num_frames": self.num_frames,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ai-avatar/multi",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["images", "audio"]

class SeeDanceV15ProImageToVideo(FALNode):
    """
    SeeDance v1.5 Pro generates high-quality dance videos from images.
    video, dance, animation, seedance, bytedance, image-to-video

    Use cases:
    - Animate photos into dance videos
    - Create dance choreography from images
    - Generate dance performances
    - Produce music video content
    - Create dance training materials
    """

    class SeeDanceV15ProAspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"

    class SeeDanceV15ProDuration(Enum):
        """
        Duration of the video in seconds
        """
        VALUE_4 = "4"
        VALUE_5 = "5"
        VALUE_6 = "6"
        VALUE_7 = "7"
        VALUE_8 = "8"
        VALUE_9 = "9"
        VALUE_10 = "10"
        VALUE_11 = "11"
        VALUE_12 = "12"

    class SeeDanceV15ProResolution(Enum):
        """
        Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    aspect_ratio: SeeDanceV15ProAspectRatio = Field(
        default=SeeDanceV15ProAspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    duration: SeeDanceV15ProDuration = Field(
        default=SeeDanceV15ProDuration.VALUE_5, description="Duration of the video in seconds"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video"
    )
    resolution: SeeDanceV15ProResolution = Field(
        default=SeeDanceV15ProResolution.VALUE_720P, description="Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image used to generate video"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    camera_fixed: bool = Field(
        default=False, description="Whether to fix the camera position"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image the video ends with. Defaults to None."
    )
    seed: int = Field(
        default=-1, description="Random seed to control video generation. Use -1 for random."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "generate_audio": self.generate_audio,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "camera_fixed": self.camera_fixed,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class SeeDanceV1ProFastImageToVideo(FALNode):
    """
    SeeDance v1 Pro Fast generates dance videos quickly from images.
    video, dance, fast, seedance, bytedance, image-to-video

    Use cases:
    - Rapidly generate dance videos
    - Quick dance animation
    - Fast dance prototypes
    - Create dance previews
    - Efficient dance video generation
    """

    class SeeDanceV1ProFastAspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"
        AUTO = "auto"

    class SeeDanceV1ProFastDuration(Enum):
        """
        Duration of the video in seconds
        """
        VALUE_2 = "2"
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

    class SeeDanceV1ProFastResolution(Enum):
        """
        Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    aspect_ratio: SeeDanceV1ProFastAspectRatio = Field(
        default=SeeDanceV1ProFastAspectRatio.AUTO, description="The aspect ratio of the generated video"
    )
    duration: SeeDanceV1ProFastDuration = Field(
        default=SeeDanceV1ProFastDuration.VALUE_5, description="Duration of the video in seconds"
    )
    resolution: SeeDanceV1ProFastResolution = Field(
        default=SeeDanceV1ProFastResolution.VALUE_1080P, description="Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image used to generate video"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    camera_fixed: bool = Field(
        default=False, description="Whether to fix the camera position"
    )
    seed: int = Field(
        default=-1, description="Random seed to control video generation. Use -1 for random."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedance/v1/pro/fast/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class SeeDanceV1LiteReferenceToVideo(FALNode):
    """
    SeeDance v1 Lite generates lightweight dance videos using reference images.
    video, dance, lite, reference, seedance, image-to-video

    Use cases:
    - Generate efficient dance videos
    - Create reference-based animations
    - Produce lightweight dance content
    - Generate quick dance outputs
    - Create optimized dance videos
    """

    class SeeDanceV1LiteAspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"
        AUTO = "auto"

    class SeeDanceV1LiteDuration(Enum):
        """
        Duration of the video in seconds
        """
        VALUE_2 = "2"
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

    class SeeDanceV1LiteResolution(Enum):
        """
        Video resolution - 480p for faster generation, 720p for higher quality
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"


    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    aspect_ratio: SeeDanceV1LiteAspectRatio = Field(
        default=SeeDanceV1LiteAspectRatio.AUTO, description="The aspect ratio of the generated video"
    )
    duration: SeeDanceV1LiteDuration = Field(
        default=SeeDanceV1LiteDuration.VALUE_5, description="Duration of the video in seconds"
    )
    resolution: SeeDanceV1LiteResolution = Field(
        default=SeeDanceV1LiteResolution.VALUE_720P, description="Video resolution - 480p for faster generation, 720p for higher quality"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    camera_fixed: bool = Field(
        default=False, description="Whether to fix the camera position"
    )
    reference_images: list[str] = Field(
        default=[], description="Reference images to generate the video with."
    )
    seed: int = Field(
        default=-1, description="Random seed to control video generation. Use -1 for random."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_fixed": self.camera_fixed,
            "reference_image_urls": self.reference_images,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedance/v1/lite/reference-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "reference"]

class ByteDanceVideoStylize(FALNode):
    """
    ByteDance Video Stylize applies artistic styles to image-based video generation.
    video, style-transfer, artistic, bytedance, image-to-video

    Use cases:
    - Apply artistic styles to videos
    - Create stylized video content
    - Generate artistic animations
    - Produce style-transferred videos
    - Create visually unique content
    """

    style: str = Field(
        default="", description="The style for your character in the video. Please use a short description."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to make the stylized video from."
    )

    async def process(self, context: ProcessingContext) -> Any:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "style": self.style,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/video-stylize",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["image", "style"]

class OmniHumanV15(FALNode):
    """
    OmniHuman v1.5 generates realistic human videos from images.
    video, human, realistic, bytedance, image-to-video

    Use cases:
    - Generate realistic human videos
    - Create human motion animations
    - Produce lifelike character videos
    - Generate human performances
    - Create realistic human content
    """

    class OmniHumanV15Resolution(Enum):
        """
        The resolution of the generated video. Defaults to 1080p. 720p generation is faster and higher in quality. 1080p generation is limited to 30s audio and 720p generation is limited to 60s audio.
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default="", description="The text prompt used to guide the video generation."
    )
    resolution: OmniHumanV15Resolution = Field(
        default=OmniHumanV15Resolution.VALUE_1080P, description="The resolution of the generated video. Defaults to 1080p. 720p generation is faster and higher in quality. 1080p generation is limited to 30s audio and 720p generation is limited to 60s audio."
    )
    turbo_mode: bool = Field(
        default=False, description="Generate a video at a faster rate with a slight quality trade-off."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio file to generate the video. Audio must be under 30s long for 1080p generation and under 60s long for 720p generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image used to generate the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "turbo_mode": self.turbo_mode,
            "audio_url": self.audio,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/omnihuman/v1.5",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class CogVideoX5BImageToVideo(FALNode):
    """
    CogVideoX-5B generates high-quality videos from images with advanced motion.
    video, generation, cogvideo, image-to-video, img2vid

    Use cases:
    - Generate videos from images
    - Create dynamic image animations
    - Produce high-quality video content
    - Animate static images
    - Generate motion from photos
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    use_rife: bool = Field(
        default=True, description="Use RIFE for video interpolation"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL to the image to generate the video from."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. We currently support one lora."
    )
    video_size: str = Field(
        default="", description="The size of the generated video."
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
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "use_rife": self.use_rife,
            "image_url": f"data:image/png;base64,{image_base64}",
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "video_size": self.video_size,
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
            application="fal-ai/cogvideox-5b/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class StableVideoImageToVideo(FALNode):
    """
    Stable Video generates consistent video animations from images.
    video, generation, stable, consistent, image-to-video

    Use cases:
    - Generate stable video animations
    - Create consistent motion
    - Produce reliable video outputs
    - Animate images consistently
    - Generate predictable videos
    """

    motion_bucket_id: int = Field(
        default=127, description="The motion bucket id determines the motion of the generated video. The higher the number, the more motion there will be."
    )
    fps: int = Field(
        default=25, description="The frames per second of the generated video."
    )
    cond_aug: float = Field(
        default=0.02, description="The conditoning augmentation determines the amount of noise that will be added to the conditioning frame. The higher the number, the more noise there will be, and the less the video will look like the initial image. Increase it for more motion."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a starting point for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "motion_bucket_id": self.motion_bucket_id,
            "fps": self.fps,
            "cond_aug": self.cond_aug,
            "seed": self.seed,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LTXImageToVideo(FALNode):
    """
    LTX Video generates temporally consistent videos from images.
    video, generation, ltx, temporal, image-to-video

    Use cases:
    - Generate temporally consistent videos
    - Create smooth image animations
    - Produce coherent video sequences
    - Animate with temporal awareness
    - Generate fluid motion videos
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
    )
    seed: int = Field(
        default=-1, description="The seed to use for random number generation."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to take."
    )
    negative_prompt: str = Field(
        default="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly", description="The negative prompt to generate the video from."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate the video from."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class KlingVideoV1StandardImageToVideo(FALNode):
    """
    Kling Video v1 Standard generates videos from images with balanced quality.
    video, generation, kling, standard, image-to-video

    Use cases:
    - Generate standard quality videos
    - Create balanced video animations
    - Produce efficient video content
    - Generate videos for web use
    - Create moderate quality outputs
    """

    class KlingVideoV1StandardDuration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"


    prompt: str = Field(
        default="", description="The prompt for the video"
    )
    duration: KlingVideoV1StandardDuration = Field(
        default=KlingVideoV1StandardDuration.VALUE_5, description="The duration of the generated video in seconds"
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the video"
    )
    static_mask: ImageRef = Field(
        default=ImageRef(), description="URL of the image for Static Brush Application Area (Mask image created by users using the motion brush)"
    )
    dynamic_masks: list[DynamicMask] = Field(
        default=[], description="List of dynamic masks"
    )
    tail_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the end of the video"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        static_mask_base64 = await context.image_to_base64(self.static_mask)
        tail_image_base64 = await context.image_to_base64(self.tail_image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
            "static_mask_url": f"data:image/png;base64,{static_mask_base64}",
            "dynamic_masks": [item.model_dump(exclude={"type"}) for item in self.dynamic_masks],
            "tail_image_url": f"data:image/png;base64,{tail_image_base64}",
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV56Transition(FALNode):
    """
    Pixverse v5.6 Transition creates smooth video transitions between two images with professional effects.
    video, transition, pixverse, v5.6, effects

    Use cases:
    - Create smooth transitions between images
    - Generate professional video effects
    - Produce seamless image morphing
    - Create transition animations
    - Generate video connecting two scenes
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"

    class PixverseV56TransitionResolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class ThinkingType(Enum):
        """
        Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision
        """
        ENABLED = "enabled"
        DISABLED = "disabled"
        AUTO = "auto"

    class PixverseV56TransitionDuration(Enum):
        """
        The duration of the generated video in seconds. 1080p videos are limited to 5 or 8 seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"
        VALUE_10 = "10"


    first_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: PixverseV56TransitionResolution = Field(
        default=PixverseV56TransitionResolution.VALUE_720P, description="The resolution of the generated video"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    thinking_type: ThinkingType | None = Field(
        default=None, description="Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision"
    )
    prompt: str = Field(
        default="", description="The prompt for the transition"
    )
    duration: PixverseV56TransitionDuration = Field(
        default=PixverseV56TransitionDuration.VALUE_5, description="The duration of the generated video in seconds. 1080p videos are limited to 5 or 8 seconds"
    )
    generate_audio_switch: bool = Field(
        default=False, description="Enable audio generation (BGM, SFX, dialogue)"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the last frame"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        first_image_base64 = await context.image_to_base64(self.first_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "first_image_url": f"data:image/png;base64,{first_image_base64}",
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "prompt": self.prompt,
            "duration": self.duration.value,
            "generate_audio_switch": self.generate_audio_switch,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5.6/transition",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduQ2ReferenceToVideoPro(FALNode):
    """
    Vidu Q2 Reference-to-Video Pro generates professional quality videos using reference images for style and content.
    video, generation, vidu, q2, pro, reference

    Use cases:
    - Generate pro videos from references
    - Create style-consistent animations
    - Produce reference-guided videos
    - Generate videos matching examples
    - Create professional reference-based content
    """

    class ViduQ2ReferenceToVideoProResolution(Enum):
        """
        Output video resolution
        """
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class MovementAmplitude(Enum):
        """
        The movement amplitude of objects in the frame
        """
        AUTO = "auto"
        SMALL = "small"
        MEDIUM = "medium"
        LARGE = "large"


    prompt: str = Field(
        default="", description="Text prompt for video generation, max 2000 characters"
    )
    duration: int = Field(
        default=4, description="Duration of the video in seconds (0 for automatic duration)"
    )
    resolution: ViduQ2ReferenceToVideoProResolution = Field(
        default=ViduQ2ReferenceToVideoProResolution.VALUE_720P, description="Output video resolution"
    )
    aspect_ratio: str = Field(
        default="16:9", description="Aspect ratio of the output video (e.g., auto, 16:9, 9:16, 1:1, or any W:H)"
    )
    reference_videos: list[str] = Field(
        default=[], description="URLs of the reference videos for video editing or motion reference. Supports up to 2 videos."
    )
    bgm: bool = Field(
        default=False, description="Whether to add background music to the generated video"
    )
    reference_images: list[str] = Field(
        default=[], description="URLs of the reference images for subject appearance. If videos are provided, up to 4 images are allowed; otherwise up to 7 images."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    movement_amplitude: MovementAmplitude = Field(
        default=MovementAmplitude.AUTO, description="The movement amplitude of objects in the frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio,
            "reference_video_urls": self.reference_videos,
            "bgm": self.bgm,
            "reference_image_urls": self.reference_images,
            "seed": self.seed,
            "movement_amplitude": self.movement_amplitude.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/q2/reference-to-video/pro",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanV26ImageToVideoFlash(FALNode):
    """
    Wan v2.6 Flash generates videos from images with ultra-fast processing for rapid iteration.
    video, generation, wan, v2.6, flash, fast

    Use cases:
    - Generate videos at maximum speed
    - Create rapid video prototypes
    - Produce instant video previews
    - Generate quick video iterations
    - Create fast video animations
    """

    class WanV26FlashDuration(Enum):
        """
        Duration of the generated video in seconds. Choose between 5, 10 or 15 seconds.
        """
        VALUE_5 = "5"
        VALUE_10 = "10"
        VALUE_15 = "15"

    class WanV26FlashResolution(Enum):
        """
        Video resolution. Valid values: 720p, 1080p
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default="", description="The text prompt describing the desired video motion. Max 800 characters."
    )
    duration: WanV26FlashDuration = Field(
        default=WanV26FlashDuration.VALUE_5, description="Duration of the generated video in seconds. Choose between 5, 10 or 15 seconds."
    )
    resolution: WanV26FlashResolution = Field(
        default=WanV26FlashResolution.VALUE_1080P, description="Video resolution. Valid values: 720p, 1080p"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame. Must be publicly accessible or base64 data URI. Image dimensions must be between 240 and 7680."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="URL of the audio to use as the background music. Must be publicly accessible. Limit handling: If the audio duration exceeds the duration value (5, 10, or 15 seconds), the audio is truncated to the first N seconds, and the rest is discarded. If the audio is shorter than the video, the remaining part of the video will be silent. For example, if the audio is 3 seconds long and the video duration is 5 seconds, the first 3 seconds of the output video will have sound, and the last 2 seconds will be silent. - Format: WAV, MP3. - Duration: 3 to 30 s. - File size: Up to 15 MB."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    multi_shots: bool = Field(
        default=False, description="When true, enables intelligent multi-shot segmentation. Only active when enable_prompt_expansion is True. Set to false for single-shot generation."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to describe content to avoid. Max 500 characters."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt rewriting using LLM."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "image_url": f"data:image/png;base64,{image_base64}",
            "audio_url": self.audio,
            "seed": self.seed,
            "multi_shots": self.multi_shots,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="wan/v2.6/image-to-video/flash",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanV26ImageToVideo(FALNode):
    """
    Wan v2.6 generates high-quality videos from images with balanced quality and performance.
    video, generation, wan, v2.6, image-to-video

    Use cases:
    - Generate quality videos from images
    - Create balanced video animations
    - Produce reliable video content
    - Generate consistent videos
    - Create professional animations
    """

    class WanV26Duration(Enum):
        """
        Duration of the generated video in seconds. Choose between 5, 10 or 15 seconds.
        """
        VALUE_5 = "5"
        VALUE_10 = "10"
        VALUE_15 = "15"

    class WanV26Resolution(Enum):
        """
        Video resolution. Valid values: 720p, 1080p
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default="", description="The text prompt describing the desired video motion. Max 800 characters."
    )
    duration: WanV26Duration = Field(
        default=WanV26Duration.VALUE_5, description="Duration of the generated video in seconds. Choose between 5, 10 or 15 seconds."
    )
    resolution: WanV26Resolution = Field(
        default=WanV26Resolution.VALUE_1080P, description="Video resolution. Valid values: 720p, 1080p"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame. Must be publicly accessible or base64 data URI. Image dimensions must be between 240 and 7680."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="URL of the audio to use as the background music. Must be publicly accessible. Limit handling: If the audio duration exceeds the duration value (5, 10, or 15 seconds), the audio is truncated to the first N seconds, and the rest is discarded. If the audio is shorter than the video, the remaining part of the video will be silent. For example, if the audio is 3 seconds long and the video duration is 5 seconds, the first 3 seconds of the output video will have sound, and the last 2 seconds will be silent. - Format: WAV, MP3. - Duration: 3 to 30 s. - File size: Up to 15 MB."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    multi_shots: bool = Field(
        default=False, description="When true, enables intelligent multi-shot segmentation. Only active when enable_prompt_expansion is True. Set to false for single-shot generation."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to describe content to avoid. Max 500 characters."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt rewriting using LLM."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "image_url": f"data:image/png;base64,{image_base64}",
            "audio_url": self.audio,
            "seed": self.seed,
            "multi_shots": self.multi_shots,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="wan/v2.6/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Ltx219BImageToVideo(FALNode):
    """
    LTX-2 19B generates high-quality videos from images using the powerful 19-billion parameter model.
    video, generation, ltx-2, 19b, large-model

    Use cases:
    - Generate high-quality videos with large model
    - Create detailed video animations
    - Produce superior video content
    - Generate videos with powerful AI
    - Create premium video animations
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

    class InterpolationDirection(Enum):
        """
        The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image.
        """
        FORWARD = "forward"
        BACKWARD = "backward"


    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    prompt: str = Field(
        default="", description="The prompt used for the generation."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
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
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
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
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate the video from."
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
    interpolation_direction: InterpolationDirection = Field(
        default=InterpolationDirection.FORWARD, description="The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "image_strength": self.image_strength,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "interpolation_direction": self.interpolation_direction.value,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Ltx219BImageToVideoLora(FALNode):
    """
    LTX-2 19B with LoRA enables custom-trained 19B models for specialized video generation.
    video, generation, ltx-2, 19b, lora, custom

    Use cases:
    - Generate videos with custom 19B model
    - Create specialized video content
    - Produce domain-specific animations
    - Generate with fine-tuned large model
    - Create customized video animations
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

    class InterpolationDirection(Enum):
        """
        The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image.
        """
        FORWARD = "forward"
        BACKWARD = "backward"


    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    prompt: str = Field(
        default="", description="The prompt used for the generation."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
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
        default="auto", description="The size of the generated video."
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
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
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
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate the video from."
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
    interpolation_direction: InterpolationDirection = Field(
        default=InterpolationDirection.FORWARD, description="The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "fps": self.fps,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "image_strength": self.image_strength,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "interpolation_direction": self.interpolation_direction.value,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/image-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Ltx219BDistilledImageToVideo(FALNode):
    """
    LTX-2 19B Distilled generates videos efficiently using knowledge distillation from the 19B model.
    video, generation, ltx-2, 19b, distilled, efficient

    Use cases:
    - Generate videos efficiently with distilled model
    - Create fast quality video animations
    - Produce optimized video content
    - Generate videos with good performance
    - Create balanced quality-speed videos
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

    class InterpolationDirection(Enum):
        """
        The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image.
        """
        FORWARD = "forward"
        BACKWARD = "backward"


    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    prompt: str = Field(
        default="", description="The prompt used for the generation."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
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
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate the video from."
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
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    interpolation_direction: InterpolationDirection = Field(
        default=InterpolationDirection.FORWARD, description="The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "image_strength": self.image_strength,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "interpolation_direction": self.interpolation_direction.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Ltx219BDistilledImageToVideoLora(FALNode):
    """
    LTX-2 19B Distilled with LoRA combines efficient generation with custom-trained models.
    video, generation, ltx-2, 19b, distilled, lora

    Use cases:
    - Generate videos with custom distilled model
    - Create efficient specialized content
    - Produce fast domain-specific videos
    - Generate with optimized custom model
    - Create quick customized animations
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

    class InterpolationDirection(Enum):
        """
        The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image.
        """
        FORWARD = "forward"
        BACKWARD = "backward"


    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    prompt: str = Field(
        default="", description="The prompt used for the generation."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The acceleration level to use."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
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
        default="auto", description="The size of the generated video."
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
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate the video from."
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
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    interpolation_direction: InterpolationDirection = Field(
        default=InterpolationDirection.FORWARD, description="The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "use_multiscale": self.use_multiscale,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "fps": self.fps,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "image_strength": self.image_strength,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "end_image_strength": self.end_image_strength,
            "interpolation_direction": self.interpolation_direction.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/image-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanMove(FALNode):
    """
    Wan Move generates videos with natural motion and movement from static images.
    video, generation, wan, motion, animation

    Use cases:
    - Add natural motion to images
    - Create animated movements
    - Produce dynamic video content
    - Generate moving scenes from stills
    - Create motion animations
    """

    prompt: str = Field(
        default="", description="Text prompt to guide the video generation."
    )
    trajectories: list[list[str]] = Field(
        default=[], description="A list of trajectories. Each trajectory list means the movement of one object."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_inference_steps: int = Field(
        default=40, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    negative_prompt: str = Field(
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", description="Negative prompt to guide the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "trajectories": self.trajectories,
            "image_url": f"data:image/png;base64,{image_base64}",
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-move",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Kandinsky5ProImageToVideo(FALNode):
    """
    Kandinsky5 Pro generates professional quality videos from images with artistic style and control.
    video, generation, kandinsky, pro, artistic

    Use cases:
    - Generate artistic videos from images
    - Create stylized video animations
    - Produce creative video content
    - Generate videos with artistic flair
    - Create professional artistic videos
    """

    class Kandinsky5ProResolution(Enum):
        """
        Video resolution: 512p or 1024p.
        """
        VALUE_512P = "512P"
        VALUE_1024P = "1024P"

    class Acceleration(Enum):
        """
        Acceleration level for faster generation.
        """
        NONE = "none"
        REGULAR = "regular"

    class Kandinsky5ProDuration(Enum):
        """
        Video duration.
        """
        VALUE_5S = "5s"


    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    resolution: Kandinsky5ProResolution = Field(
        default=Kandinsky5ProResolution.VALUE_512P, description="Video resolution: 512p or 1024p."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for faster generation."
    )
    duration: Kandinsky5ProDuration = Field(
        default=Kandinsky5ProDuration.VALUE_5S, description="Video duration."
    )
    num_inference_steps: int = Field(
        default=28
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "duration": self.duration.value,
            "num_inference_steps": self.num_inference_steps,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kandinsky5-pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LiveAvatar(FALNode):
    """
    Live Avatar creates animated talking avatars from portrait images with realistic lip-sync and expressions.
    video, avatar, talking-head, animation, portrait

    Use cases:
    - Create talking avatar videos
    - Animate portrait images
    - Generate lip-synced avatars
    - Produce speaking character videos
    - Create animated presenters
    """

    class Acceleration(Enum):
        """
        Acceleration level for faster video decoding
        """
        NONE = "none"
        LIGHT = "light"
        REGULAR = "regular"
        HIGH = "high"


    frames_per_clip: int = Field(
        default=48, description="Number of frames per clip. Must be a multiple of 4. Higher values = smoother but slower generation."
    )
    prompt: str = Field(
        default="", description="A text prompt describing the scene and character. Helps guide the video generation style and context."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Acceleration level for faster video decoding"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the reference image for avatar generation. The character in this image will be animated."
    )
    num_clips: int = Field(
        default=10, description="Number of video clips to generate. Each clip is approximately 3 seconds. Set higher for longer videos."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the driving audio file (WAV or MP3). The avatar will be animated to match this audio."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )
    guidance_scale: float = Field(
        default=0, description="Classifier-free guidance scale. Higher values follow the prompt more closely."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker for content moderation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "frames_per_clip": self.frames_per_clip,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "num_clips": self.num_clips,
            "audio_url": self.audio,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/live-avatar",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class HunyuanVideoV15ImageToVideo(FALNode):
    """
    Hunyuan Video v1.5 generates high-quality videos from images with advanced AI capabilities.
    video, generation, hunyuan, v1.5, advanced

    Use cases:
    - Generate advanced quality videos
    - Create sophisticated animations
    - Produce high-fidelity video content
    - Generate videos with AI excellence
    - Create cutting-edge video animations
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the video.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class HunyuanVideoV15Resolution(Enum):
        """
        The resolution of the video.
        """
        VALUE_480P = "480p"


    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video."
    )
    resolution: HunyuanVideoV15Resolution = Field(
        default=HunyuanVideoV15Resolution.VALUE_480P, description="The resolution of the video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the reference image for image-to-video generation."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Enable prompt expansion to enhance the input prompt."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to guide what not to generate."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class KlingVideoO1StandardImageToVideo(FALNode):
    """
    Kling Video O1 Standard generates videos with optimized standard quality from images.
    video, generation, kling, o1, standard

    Use cases:
    - Generate standard O1 quality videos
    - Create optimized video animations
    - Produce efficient video content
    - Generate balanced quality videos
    - Create standard tier animations
    """

    class KlingVideoO1StandardDuration(Enum):
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


    prompt: str = Field(
        default="", description="Use @Image1 to reference the start frame, @Image2 to reference the end frame."
    )
    duration: KlingVideoO1StandardDuration = Field(
        default=KlingVideoO1StandardDuration.VALUE_5, description="Video duration in seconds."
    )
    start_image: ImageRef = Field(
        default=ImageRef(), description="Image to use as the first frame of the video. Max file size: 10.0MB, Min width: 300px, Min height: 300px, Min aspect ratio: 0.40, Max aspect ratio: 2.50, Timeout: 20.0s"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="Image to use as the last frame of the video. Max file size: 10.0MB, Min width: 300px, Min height: 300px, Min aspect ratio: 0.40, Max aspect ratio: 2.50, Timeout: 20.0s"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_base64 = await context.image_to_base64(self.start_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoO1StandardReferenceToVideo(FALNode):
    """
    Kling Video O1 Standard generates videos using reference images for style consistency.
    video, generation, kling, o1, standard, reference

    Use cases:
    - Generate videos from reference images
    - Create style-consistent animations
    - Produce reference-guided content
    - Generate videos matching examples
    - Create standardized reference videos
    """

    class KlingVideoO1StandardReferenceToVideoDuration(Enum):
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

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video frame.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"


    prompt: str = Field(
        default="", description="Take @Element1, @Element2 to reference elements and @Image1, @Image2 to reference images in order."
    )
    duration: KlingVideoO1StandardReferenceToVideoDuration = Field(
        default=KlingVideoO1StandardReferenceToVideoDuration.VALUE_5, description="Video duration in seconds."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame."
    )
    elements: list[OmniVideoElementInput] = Field(
        default=[], description="Elements (characters/objects) to include in the video. Reference in prompt as @Element1, @Element2, etc. Maximum 7 total (elements + reference images + start image)."
    )
    images: list[ImageRef] = Field(
        default=[], description="Additional reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 7 total (elements + reference images + start image)."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        images_data_urls = []
        for image in self.images or []:
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/standard/reference-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoV26ProImageToVideo(FALNode):
    """
    Kling Video v2.6 Pro generates professional quality videos with latest model improvements.
    video, generation, kling, v2.6, pro

    Use cases:
    - Generate professional v2.6 videos
    - Create latest quality animations
    - Produce premium video content
    - Generate advanced videos
    - Create pro-tier animations
    """

    class KlingVideoV26ProDuration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"


    prompt: str = Field(
        default=""
    )
    duration: KlingVideoV26ProDuration = Field(
        default=KlingVideoV26ProDuration.VALUE_5, description="The duration of the generated video in seconds"
    )
    voice_ids: list[str] = Field(
        default=[], description="Optional Voice IDs for video generation. Reference voices in your prompt with <<<voice_1>>> and <<<voice_2>>> (maximum 2 voices per task). Get voice IDs from the kling video create-voice endpoint: https://fal.ai/models/fal-ai/kling-video/create-voice"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate native audio for the video. Supports Chinese and English voice output. Other languages are automatically translated to English. For English speech, use lowercase letters; for acronyms or proper nouns, use uppercase."
    )
    start_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the video"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the end of the video"
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_base64 = await context.image_to_base64(self.start_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "voice_ids": self.voice_ids,
            "generate_audio": self.generate_audio,
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2.6/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoAiAvatarV2Standard(FALNode):
    """
    Kling Video AI Avatar v2 Standard creates animated talking avatars with standard quality.
    video, avatar, kling, v2, standard, talking-head

    Use cases:
    - Create standard quality talking avatars
    - Animate portraits with speech
    - Generate avatar presentations
    - Produce speaking character videos
    - Create AI-driven avatars
    """

    prompt: str = Field(
        default=".", description="The prompt to use for the video generation."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as your avatar"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "audio_url": self.audio,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/ai-avatar/v2/standard",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoAiAvatarV2Pro(FALNode):
    """
    Kling Video AI Avatar v2 Pro creates professional quality animated talking avatars with enhanced realism.
    video, avatar, kling, v2, pro, talking-head

    Use cases:
    - Create professional talking avatars
    - Animate portraits with high quality
    - Generate realistic avatar videos
    - Produce premium speaking characters
    - Create pro-grade AI avatars
    """

    prompt: str = Field(
        default=".", description="The prompt to use for the video generation."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as your avatar"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "audio_url": self.audio,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/ai-avatar/v2/pro",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class CreatifyAurora(FALNode):
    """
    Creatify Aurora generates creative and visually stunning videos from images with unique effects.
    video, generation, creatify, aurora, creative, effects

    Use cases:
    - Generate creative visual effects videos
    - Create stunning video animations
    - Produce artistic video content
    - Generate unique video effects
    - Create visually impressive videos
    """

    class CreatifyAuroraResolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"


    prompt: str = Field(
        default="", description="A text prompt to guide the video generation process."
    )
    resolution: CreatifyAuroraResolution = Field(
        default=CreatifyAuroraResolution.VALUE_720P, description="The resolution of the generated video."
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio file to be used for video generation."
    )
    audio_guidance_scale: float = Field(
        default=2, description="Guidance scale to be used for audio adherence."
    )
    guidance_scale: float = Field(
        default=1, description="Guidance scale to be used for text prompt adherence."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image file to be used for video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "audio_url": self.audio,
            "audio_guidance_scale": self.audio_guidance_scale,
            "guidance_scale": self.guidance_scale,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/creatify/aurora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV55Effects(FALNode):
    """
    Pixverse
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"
        VALUE_10 = "10"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class ThinkingType(Enum):
        """
        Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision
        """
        ENABLED = "enabled"
        DISABLED = "disabled"
        AUTO = "auto"

    class Effect(Enum):
        """
        The effect to apply to the video
        """
        KISS_ME_AI = "Kiss Me AI"
        KISS = "Kiss"
        MUSCLE_SURGE = "Muscle Surge"
        WARMTH_OF_JESUS = "Warmth of Jesus"
        ANYTHING_ROBOT = "Anything, Robot"
        THE_TIGER_TOUCH = "The Tiger Touch"
        HUG = "Hug"
        HOLY_WINGS = "Holy Wings"
        MICROWAVE = "Microwave"
        ZOMBIE_MODE = "Zombie Mode"
        SQUID_GAME = "Squid Game"
        BABY_FACE = "Baby Face"
        BLACK_MYTH_WUKONG = "Black Myth: Wukong"
        LONG_HAIR_MAGIC = "Long Hair Magic"
        LEGGY_RUN = "Leggy Run"
        FIN_TASTIC_MERMAID = "Fin-tastic Mermaid"
        PUNCH_FACE = "Punch Face"
        CREEPY_DEVIL_SMILE = "Creepy Devil Smile"
        THUNDER_GOD = "Thunder God"
        EYE_ZOOM_CHALLENGE = "Eye Zoom Challenge"
        WHOS_ARRESTED = "Who's Arrested?"
        BABY_ARRIVED = "Baby Arrived"
        WEREWOLF_RAGE = "Werewolf Rage"
        BALD_SWIPE = "Bald Swipe"
        BOOM_DROP = "BOOM DROP"
        HUGE_CUTIE = "Huge Cutie"
        LIQUID_METAL = "Liquid Metal"
        SHARKSNAP = "Sharksnap!"
        DUST_ME_AWAY = "Dust Me Away"
        FIGURINE_FACTOR_3D = "3D Figurine Factor"
        BIKINI_UP = "Bikini Up"
        MY_GIRLFRIENDS = "My Girlfriends"
        MY_BOYFRIENDS = "My Boyfriends"
        SUBJECT_3_FEVER = "Subject 3 Fever"
        EARTH_ZOOM = "Earth Zoom"
        POLE_DANCE = "Pole Dance"
        VROOM_DANCE = "Vroom Dance"
        GHOSTFACE_TERROR = "GhostFace Terror"
        DRAGON_EVOKER = "Dragon Evoker"
        SKELETAL_BAE = "Skeletal Bae"
        SUMMONING_SUCCUBUS = "Summoning succubus"
        HALLOWEEN_VOODOO_DOLL = "Halloween Voodoo Doll"
        NAKED_EYE_AD_3D = "3D Naked-Eye AD"
        PACKAGE_EXPLOSION = "Package Explosion"
        DISHES_SERVED = "Dishes Served"
        OCEAN_AD = "Ocean ad"
        SUPERMARKET_AD = "Supermarket AD"
        TREE_DOLL = "Tree doll"
        COME_FEEL_MY_ABS = "Come Feel My Abs"
        THE_BICEP_FLEX = "The Bicep Flex"
        LONDON_ELITE_VIBE = "London Elite Vibe"
        FLORA_NYMPH_GOWN = "Flora Nymph Gown"
        CHRISTMAS_COSTUME = "Christmas Costume"
        ITS_SNOWY = "It's Snowy"
        REINDEER_CRUISER = "Reindeer Cruiser"
        SNOW_GLOBE_MAKER = "Snow Globe Maker"
        PET_CHRISTMAS_OUTFIT = "Pet Christmas Outfit"
        ADOPT_A_POLAR_PAL = "Adopt a Polar Pal"
        CAT_CHRISTMAS_BOX = "Cat Christmas Box"
        STARLIGHT_GIFT_BOX = "Starlight Gift Box"
        XMAS_POSTER = "Xmas Poster"
        PET_CHRISTMAS_TREE = "Pet Christmas Tree"
        CITY_SANTA_HAT = "City Santa Hat"
        STOCKING_SWEETIE = "Stocking Sweetie"
        CHRISTMAS_NIGHT = "Christmas Night"
        XMAS_FRONT_PAGE_KARMA = "Xmas Front Page Karma"
        GRINCHS_XMAS_HIJACK = "Grinch's Xmas Hijack"
        GIANT_PRODUCT = "Giant Product"
        TRUCK_FASHION_SHOOT = "Truck Fashion Shoot"
        BEACH_AD = "Beach AD"
        SHOAL_SURROUND = "Shoal Surround"
        MECHANICAL_ASSEMBLY = "Mechanical Assembly"
        LIGHTING_AD = "Lighting AD"
        BILLBOARD_AD = "Billboard AD"
        PRODUCT_CLOSE_UP = "Product close-up"
        PARACHUTE_DELIVERY = "Parachute Delivery"
        DREAMLIKE_CLOUD = "Dreamlike Cloud"
        MACARON_MACHINE = "Macaron Machine"
        POSTER_AD = "Poster AD"
        TRUCK_AD = "Truck AD"
        GRAFFITI_AD = "Graffiti AD"
        FIGURINE_FACTORY_3D = "3D Figurine Factory"
        THE_EXCLUSIVE_FIRST_CLASS = "The Exclusive First Class"
        ART_ZOOM_CHALLENGE = "Art Zoom Challenge"
        I_QUIT = "I Quit"
        HITCHCOCK_DOLLY_ZOOM = "Hitchcock Dolly Zoom"
        SMELL_THE_LENS = "Smell the Lens"
        I_BELIEVE_I_CAN_FLY = "I believe I can fly"
        STRIKOUT_DANCE = "Strikout Dance"
        PIXEL_WORLD = "Pixel World"
        MINT_IN_BOX = "Mint in Box"
        HANDS_UP_HAND = "Hands up, Hand"
        FLORA_NYMPH_GO = "Flora Nymph Go"
        SOMBER_EMBRACE = "Somber Embrace"
        BEAM_ME_UP = "Beam me up"
        SUIT_SWAGGER = "Suit Swagger"


    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    thinking_type: ThinkingType | None = Field(
        default=None, description="Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision"
    )
    effect: Effect = Field(
        default="", description="The effect to apply to the video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the image to use as the first frame. If not provided, generates from text"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "negative_prompt": self.negative_prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "effect": self.effect.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5.5/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV55Transition(FALNode):
    """
    Pixverse
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class ThinkingType(Enum):
        """
        Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision
        """
        ENABLED = "enabled"
        DISABLED = "disabled"
        AUTO = "auto"

    class Duration(Enum):
        """
        The duration of the generated video in seconds. Longer durations cost more. 1080p videos are limited to 5 or 8 seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"
        VALUE_10 = "10"


    first_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    thinking_type: ThinkingType | None = Field(
        default=None, description="Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision"
    )
    prompt: str = Field(
        default="", description="The prompt for the transition"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds. Longer durations cost more. 1080p videos are limited to 5 or 8 seconds"
    )
    generate_audio_switch: bool = Field(
        default=False, description="Enable audio generation (BGM, SFX, dialogue)"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the last frame"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        first_image_base64 = await context.image_to_base64(self.first_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "first_image_url": f"data:image/png;base64,{first_image_base64}",
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "prompt": self.prompt,
            "duration": self.duration.value,
            "generate_audio_switch": self.generate_audio_switch,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5.5/transition",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV55ImageToVideo(FALNode):
    """
    Pixverse
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Duration(Enum):
        """
        The duration of the generated video in seconds. Longer durations cost more. 1080p videos are limited to 5 or 8 seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"
        VALUE_10 = "10"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class ThinkingType(Enum):
        """
        Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision
        """
        ENABLED = "enabled"
        DISABLED = "disabled"
        AUTO = "auto"


    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds. Longer durations cost more. 1080p videos are limited to 5 or 8 seconds"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    thinking_type: ThinkingType | None = Field(
        default=None, description="Prompt optimization mode: 'enabled' to optimize, 'disabled' to turn off, 'auto' for model decision"
    )
    generate_multi_clip_switch: bool = Field(
        default=False, description="Enable multi-clip generation with dynamic camera changes"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    generate_audio_switch: bool = Field(
        default=False, description="Enable audio generation (BGM, SFX, dialogue)"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "generate_multi_clip_switch": self.generate_multi_clip_switch,
            "image_url": f"data:image/png;base64,{image_base64}",
            "generate_audio_switch": self.generate_audio_switch,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5.5/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoO1ImageToVideo(FALNode):
    """
    Kling O1 First Frame Last Frame to Video [Pro]
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
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


    prompt: str = Field(
        default="", description="Use @Image1 to reference the start frame, @Image2 to reference the end frame."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Video duration in seconds."
    )
    start_image: ImageRef = Field(
        default=ImageRef(), description="Image to use as the first frame of the video. Max file size: 10.0MB, Min width: 300px, Min height: 300px, Min aspect ratio: 0.40, Max aspect ratio: 2.50, Timeout: 20.0s"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="Image to use as the last frame of the video. Max file size: 10.0MB, Min width: 300px, Min height: 300px, Min aspect ratio: 0.40, Max aspect ratio: 2.50, Timeout: 20.0s"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_base64 = await context.image_to_base64(self.start_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoO1ReferenceToVideo(FALNode):
    """
    Kling O1 Reference Image to Video [Pro]
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
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

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video frame.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"


    prompt: str = Field(
        default="", description="Take @Element1, @Element2 to reference elements and @Image1, @Image2 to reference images in order."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Video duration in seconds."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame."
    )
    elements: list[OmniVideoElementInput] = Field(
        default=[], description="Elements (characters/objects) to include in the video. Reference in prompt as @Element1, @Element2, etc. Maximum 7 total (elements + reference images + start image)."
    )
    images: list[ImageRef] = Field(
        default=[], description="Additional reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 7 total (elements + reference images + start image)."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        images_data_urls = []
        for image in self.images or []:
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "image_urls": images_data_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/o1/reference-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BytedanceLynx(FALNode):
    """
    Lynx
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        Resolution of the generated video (480p, 580p, or 720p)
        """
        VALUE_480P = "480p"
        VALUE_580P = "580p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video (16:9, 9:16, or 1:1)
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"


    prompt: str = Field(
        default="", description="Text prompt to guide video generation"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p, 580p, or 720p)"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9, 9:16, or 1:1)"
    )
    num_inference_steps: int = Field(
        default=50, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    guidance_scale_2: float = Field(
        default=2, description="Image guidance scale. Controls how closely the generated video follows the reference image. Higher values increase adherence to the reference image but may decrease quality."
    )
    strength: float = Field(
        default=1, description="Reference image scale. Controls the influence of the reference image on the generated video."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 30."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the subject image to be used for video generation"
    )
    guidance_scale: float = Field(
        default=5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    num_frames: int = Field(
        default=81, description="Number of frames in the generated video. Must be between 9 to 100."
    )
    negative_prompt: str = Field(
        default="Bright tones, overexposed, blurred background, static, subtitles, style, works, paintings, images, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", description="Negative prompt to guide what should not appear in the generated video"
    )
    ip_scale: float = Field(
        default=1, description="Identity preservation scale. Controls how closely the generated video preserves the subject's identity from the reference image."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale_2": self.guidance_scale_2,
            "strength": self.strength,
            "frames_per_second": self.frames_per_second,
            "image_url": f"data:image/png;base64,{image_base64}",
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_frames": self.num_frames,
            "negative_prompt": self.negative_prompt,
            "ip_scale": self.ip_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bytedance/lynx",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseSwap(FALNode):
    """
    Pixverse
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Mode(Enum):
        """
        The swap mode to use
        """
        PERSON = "person"
        OBJECT = "object"
        BACKGROUND = "background"

    class Resolution(Enum):
        """
        The output resolution (1080p not supported)
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"


    original_sound_switch: bool = Field(
        default=True, description="Whether to keep the original audio"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the external video to swap"
    )
    keyframe_id: int = Field(
        default=1, description="The keyframe ID (from 1 to the last frame position)"
    )
    mode: Mode = Field(
        default=Mode.PERSON, description="The swap mode to use"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The output resolution (1080p not supported)"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the target image for swapping"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "original_sound_switch": self.original_sound_switch,
            "video_url": self.video,
            "keyframe_id": self.keyframe_id,
            "mode": self.mode.value,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/swap",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PikaV22Pikaframes(FALNode):
    """
    Pika
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default="", description="Default prompt for all transitions. Individual transition prompts override this."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    transitions: list[KeyframeTransition] = Field(
        default=[], description="Configuration for each transition. Length must be len(image_urls) - 1. Total duration of all transitions must not exceed 25 seconds. If not provided, uses default 5-second transitions with the global prompt."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator"
    )
    images: list[ImageRef] = Field(
        default=[], description="URLs of keyframe images (2-5 images) to create transitions between"
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the model"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        images_data_urls = []
        for image in self.images or []:
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "transitions": [item.model_dump(exclude={"type"}) for item in self.transitions],
            "seed": self.seed,
            "image_urls": images_data_urls,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pika/v2.2/pikaframes",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LongcatVideoImageToVideo720P(FALNode):
    """
    LongCat Video
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Acceleration(Enum):
        """
        The acceleration level to use for the video generation.
        """
        NONE = "none"
        REGULAR = "regular"

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


    prompt: str = Field(
        default="First-person view from the cockpit of a Formula 1 car. The driver's gloved hands firmly grip the intricate, carbon-fiber steering wheel adorned with numerous colorful buttons and a vibrant digital display showing race data. Beyond the windshield, a sun-drenched racetrack stretches ahead, lined with cheering spectators in the grandstands. Several rival cars are visible in the distance, creating a dynamic sense of competition. The sky above is a clear, brilliant blue, reflecting the exhilarating atmosphere of a high-speed race. high resolution 4k", description="The prompt to guide the video generation."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for the video generation."
    )
    fps: int = Field(
        default=30, description="The frame rate of the generated video."
    )
    num_refine_inference_steps: int = Field(
        default=40, description="The number of inference steps to use for refinement."
    )
    guidance_scale: float = Field(
        default=4, description="The guidance scale to use for the video generation."
    )
    num_frames: int = Field(
        default=162, description="The number of frames to generate."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    negative_prompt: str = Field(
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", description="The negative prompt to use for the video generation."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate a video from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use for the video generation."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "fps": self.fps,
            "num_refine_inference_steps": self.num_refine_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-video/image-to-video/720p",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LongcatVideoImageToVideo480P(FALNode):
    """
    LongCat Video
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Acceleration(Enum):
        """
        The acceleration level to use for the video generation.
        """
        NONE = "none"
        REGULAR = "regular"

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


    prompt: str = Field(
        default="First-person view from the cockpit of a Formula 1 car. The driver's gloved hands firmly grip the intricate, carbon-fiber steering wheel adorned with numerous colorful buttons and a vibrant digital display showing race data. Beyond the windshield, a sun-drenched racetrack stretches ahead, lined with cheering spectators in the grandstands. Several rival cars are visible in the distance, creating a dynamic sense of competition. The sky above is a clear, brilliant blue, reflecting the exhilarating atmosphere of a high-speed race. high resolution 4k", description="The prompt to guide the video generation."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for the video generation."
    )
    fps: int = Field(
        default=15, description="The frame rate of the generated video."
    )
    guidance_scale: float = Field(
        default=4, description="The guidance scale to use for the video generation."
    )
    num_frames: int = Field(
        default=162, description="The number of frames to generate."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    negative_prompt: str = Field(
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards", description="The negative prompt to use for the video generation."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate a video from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use for the video generation."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "fps": self.fps,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-video/image-to-video/480p",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LongcatVideoDistilledImageToVideo720P(FALNode):
    """
    LongCat Video Distilled
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

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


    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    prompt: str = Field(
        default="First-person view from the cockpit of a Formula 1 car. The driver's gloved hands firmly grip the intricate, carbon-fiber steering wheel adorned with numerous colorful buttons and a vibrant digital display showing race data. Beyond the windshield, a sun-drenched racetrack stretches ahead, lined with cheering spectators in the grandstands. Several rival cars are visible in the distance, creating a dynamic sense of competition. The sky above is a clear, brilliant blue, reflecting the exhilarating atmosphere of a high-speed race. high resolution 4k", description="The prompt to guide the video generation."
    )
    fps: int = Field(
        default=30, description="The frame rate of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    num_refine_inference_steps: int = Field(
        default=12, description="The number of inference steps to use for refinement."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate a video from."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    num_inference_steps: int = Field(
        default=12, description="The number of inference steps to use."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    num_frames: int = Field(
        default=162, description="The number of frames to generate."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "prompt": self.prompt,
            "fps": self.fps,
            "sync_mode": self.sync_mode,
            "num_refine_inference_steps": self.num_refine_inference_steps,
            "image_url": f"data:image/png;base64,{image_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "num_frames": self.num_frames,
            "video_quality": self.video_quality.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-video/distilled/image-to-video/720p",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LongcatVideoDistilledImageToVideo480P(FALNode):
    """
    LongCat Video Distilled
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

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


    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
    )
    prompt: str = Field(
        default="First-person view from the cockpit of a Formula 1 car. The driver's gloved hands firmly grip the intricate, carbon-fiber steering wheel adorned with numerous colorful buttons and a vibrant digital display showing race data. Beyond the windshield, a sun-drenched racetrack stretches ahead, lined with cheering spectators in the grandstands. Several rival cars are visible in the distance, creating a dynamic sense of competition. The sky above is a clear, brilliant blue, reflecting the exhilarating atmosphere of a high-speed race. high resolution 4k", description="The prompt to guide the video generation."
    )
    fps: int = Field(
        default=15, description="The frame rate of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate a video from."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    num_inference_steps: int = Field(
        default=12, description="The number of inference steps to use."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    num_frames: int = Field(
        default=162, description="The number of frames to generate."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "prompt": self.prompt,
            "fps": self.fps,
            "sync_mode": self.sync_mode,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "num_frames": self.num_frames,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-video/distilled/image-to-video/480p",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class MinimaxHailuo23FastStandardImageToVideo(FALNode):
    """
    MiniMax Hailuo 2.3 Fast [Standard] (Image to Video)
    video, animation, image-to-video, img2vid, fast, professional

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the video in seconds.
        """
        VALUE_6 = "6"
        VALUE_10 = "10"


    prompt: str = Field(
        default="", description="Text prompt for video generation"
    )
    duration: Duration = Field(
        default=Duration.VALUE_6, description="The duration of the video in seconds."
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "prompt_optimizer": self.prompt_optimizer,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class MinimaxHailuo23StandardImageToVideo(FALNode):
    """
    MiniMax Hailuo 2.3 [Standard] (Image to Video)
    video, animation, image-to-video, img2vid, professional

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the video in seconds.
        """
        VALUE_6 = "6"
        VALUE_10 = "10"


    prompt: str = Field(
        default="", description="Text prompt for video generation"
    )
    duration: Duration = Field(
        default=Duration.VALUE_6, description="The duration of the video in seconds."
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "prompt_optimizer": self.prompt_optimizer,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-2.3/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class MinimaxHailuo23FastProImageToVideo(FALNode):
    """
    MiniMax Hailuo 2.3 Fast [Pro] (Image to Video)
    video, animation, image-to-video, img2vid, fast, professional

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    prompt: str = Field(
        default="", description="Text prompt for video generation"
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-2.3-fast/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoV25TurboStandardImageToVideo(FALNode):
    """
    Kling Video
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"


    prompt: str = Field(
        default=""
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "cfg_scale": self.cfg_scale,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2.5-turbo/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Veo31FastFirstLastFrameToVideo(FALNode):
    """
    Veo 3.1 Fast
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video.
        """
        VALUE_4S = "4s"
        VALUE_6S = "6s"
        VALUE_8S = "8s"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"
        VALUE_4K = "4k"


    prompt: str = Field(
        default="", description="The text prompt describing the video you want to generate"
    )
    duration: Duration = Field(
        default=Duration.VALUE_8S, description="The duration of the generated video."
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
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL of the first frame of the video"
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL of the last frame of the video"
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
            "resolution": self.resolution.value,
            "first_frame_url": self.first_frame,
            "seed": self.seed,
            "last_frame_url": self.last_frame,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3.1/fast/first-last-frame-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Veo31FirstLastFrameToVideo(FALNode):
    """
    Veo 3.1
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video.
        """
        VALUE_4S = "4s"
        VALUE_6S = "6s"
        VALUE_8S = "8s"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"
        VALUE_4K = "4k"


    prompt: str = Field(
        default="", description="The text prompt describing the video you want to generate"
    )
    duration: Duration = Field(
        default=Duration.VALUE_8S, description="The duration of the generated video."
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
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    first_frame: VideoRef = Field(
        default=VideoRef(), description="URL of the first frame of the video"
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    last_frame: VideoRef = Field(
        default=VideoRef(), description="URL of the last frame of the video"
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
            "resolution": self.resolution.value,
            "first_frame_url": self.first_frame,
            "seed": self.seed,
            "last_frame_url": self.last_frame,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3.1/first-last-frame-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Veo31ReferenceToVideo(FALNode):
    """
    Veo 3.1
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video.
        """
        VALUE_8S = "8s"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"
        VALUE_4K = "4k"


    prompt: str = Field(
        default="", description="The text prompt describing the video you want to generate"
    )
    duration: Duration = Field(
        default=Duration.VALUE_8S, description="The duration of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    auto_fix: bool = Field(
        default=False, description="Whether to automatically attempt to fix prompts that fail content policy or other validation checks by rewriting them."
    )
    images: list[ImageRef] = Field(
        default=[], description="URLs of the reference images to use for consistent subject appearance"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        images_data_urls = []
        for image in self.images or []:
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "resolution": self.resolution.value,
            "auto_fix": self.auto_fix,
            "image_urls": images_data_urls,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3.1/reference-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Veo31FastImageToVideo(FALNode):
    """
    Veo 3.1 Fast
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video.
        """
        VALUE_4S = "4s"
        VALUE_6S = "6s"
        VALUE_8S = "8s"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video. Only 16:9 and 9:16 are supported.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"
        VALUE_4K = "4k"


    prompt: str = Field(
        default="", description="The text prompt describing the video you want to generate"
    )
    duration: Duration = Field(
        default=Duration.VALUE_8S, description="The duration of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the generated video. Only 16:9 and 9:16 are supported."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    auto_fix: bool = Field(
        default=False, description="Whether to automatically attempt to fix prompts that fail content policy or other validation checks by rewriting them."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image to animate. Should be 720p or higher resolution in 16:9 or 9:16 aspect ratio. If the image is not in 16:9 or 9:16 aspect ratio, it will be cropped to fit."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3.1/fast/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Veo31ImageToVideo(FALNode):
    """
    Veo 3.1
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video.
        """
        VALUE_4S = "4s"
        VALUE_6S = "6s"
        VALUE_8S = "8s"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video. Only 16:9 and 9:16 are supported.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"
        VALUE_4K = "4k"


    prompt: str = Field(
        default="", description="The text prompt describing the video you want to generate"
    )
    duration: Duration = Field(
        default=Duration.VALUE_8S, description="The duration of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the generated video. Only 16:9 and 9:16 are supported."
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    auto_fix: bool = Field(
        default=False, description="Whether to automatically attempt to fix prompts that fail content policy or other validation checks by rewriting them."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image to animate. Should be 720p or higher resolution in 16:9 or 9:16 aspect ratio. If the image is not in 16:9 or 9:16 aspect ratio, it will be cropped to fit."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3.1/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class OviImageToVideo(FALNode):
    """
    Ovi
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps."
    )
    audio_negative_prompt: str = Field(
        default="robotic, muffled, echo, distorted", description="Negative prompt for audio generation."
    )
    negative_prompt: str = Field(
        default="jitter, bad hands, blur, distortion", description="Negative prompt for video generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The image URL to guide video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "audio_negative_prompt": self.audio_negative_prompt,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ovi/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class VeedFabric10Fast(FALNode):
    """
    Fabric 1.0 Fast
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        Resolution
        """
        VALUE_720P = "720p"
        VALUE_480P = "480p"


    resolution: Resolution = Field(
        default="", description="Resolution"
    )
    audio: AudioRef = Field(
        default=AudioRef()
    )
    image: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "resolution": self.resolution.value,
            "audio_url": self.audio,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/fabric-1.0/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class VeedFabric10(FALNode):
    """
    Fabric 1.0
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        Resolution
        """
        VALUE_720P = "720p"
        VALUE_480P = "480p"


    resolution: Resolution = Field(
        default="", description="Resolution"
    )
    audio: AudioRef = Field(
        default=AudioRef()
    )
    image: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "resolution": self.resolution.value,
            "audio_url": self.audio,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/fabric-1.0",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoV1StandardAiAvatar(FALNode):
    """
    Kling AI Avatar
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    prompt: str = Field(
        default=".", description="The prompt to use for the video generation."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as your avatar"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "audio_url": self.audio,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1/standard/ai-avatar",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoV1ProAiAvatar(FALNode):
    """
    Kling AI Avatar Pro
    video, animation, image-to-video, img2vid, professional

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    prompt: str = Field(
        default=".", description="The prompt to use for the video generation."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as your avatar"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "audio_url": self.audio,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1/pro/ai-avatar",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class DecartLucy14BImageToVideo(FALNode):
    """
    Decart Lucy 14b
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video.
        """
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"

    class Resolution(Enum):
        """
        Resolution of the generated video
        """
        VALUE_720P = "720p"


    sync_mode: bool = Field(
        default=True, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video."
    )
    prompt: str = Field(
        default="", description="Text description of the desired video content"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "sync_mode": self.sync_mode,
            "aspect_ratio": self.aspect_ratio.value,
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="decart/lucy-14b/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanAti(FALNode):
    """
    Wan Ati
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        Resolution of the generated video (480p, 580p, 720p).
        """
        VALUE_480P = "480p"
        VALUE_580P = "580p"
        VALUE_720P = "720p"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the generated video (480p, 580p, 720p)."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image."
    )
    track: list[list[str]] = Field(
        default=[], description="Motion tracks to guide video generation. Each track is a sequence of points defining a motion trajectory. Multiple tracks can control different elements or objects in the video. Expected format: array of tracks, where each track is an array of points with 'x' and 'y' coordinates (up to 121 points per track). Points will be automatically padded to 121 if fewer are provided. Coordinates should be within the image dimensions."
    )
    guidance_scale: float = Field(
        default=5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_inference_steps: int = Field(
        default=40, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "track": self.track,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-ati",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class DecartLucy5bImageToVideo(FALNode):
    """
    Lucy-5B is a model that can create 5-second I2V videos in under 5 seconds, achieving >1x RTF end-to-end
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video.
        """
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"

    class Resolution(Enum):
        """
        Resolution of the generated video
        """
        VALUE_720P = "720p"


    prompt: str = Field(
        default="", description="Text description of the desired video content"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video."
    )
    sync_mode: bool = Field(
        default=True, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/decart/lucy-5b/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV5Transition(FALNode):
    """
    Create seamless transition between images using PixVerse v5
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"


    first_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    prompt: str = Field(
        default="", description="The prompt for the transition"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the last frame"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        first_image_base64 = await context.image_to_base64(self.first_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "first_image_url": f"data:image/png;base64,{first_image_base64}",
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "prompt": self.prompt,
            "duration": self.duration.value,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5/transition",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV5Effects(FALNode):
    """
    Generate high quality video clips with different effects using PixVerse v5
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Effect(Enum):
        """
        The effect to apply to the video
        """
        KISS_ME_AI = "Kiss Me AI"
        KISS = "Kiss"
        MUSCLE_SURGE = "Muscle Surge"
        WARMTH_OF_JESUS = "Warmth of Jesus"
        ANYTHING_ROBOT = "Anything, Robot"
        THE_TIGER_TOUCH = "The Tiger Touch"
        HUG = "Hug"
        HOLY_WINGS = "Holy Wings"
        MICROWAVE = "Microwave"
        ZOMBIE_MODE = "Zombie Mode"
        SQUID_GAME = "Squid Game"
        BABY_FACE = "Baby Face"
        BLACK_MYTH_WUKONG = "Black Myth: Wukong"
        LONG_HAIR_MAGIC = "Long Hair Magic"
        LEGGY_RUN = "Leggy Run"
        FIN_TASTIC_MERMAID = "Fin-tastic Mermaid"
        PUNCH_FACE = "Punch Face"
        CREEPY_DEVIL_SMILE = "Creepy Devil Smile"
        THUNDER_GOD = "Thunder God"
        EYE_ZOOM_CHALLENGE = "Eye Zoom Challenge"
        WHOS_ARRESTED = "Who's Arrested?"
        BABY_ARRIVED = "Baby Arrived"
        WEREWOLF_RAGE = "Werewolf Rage"
        BALD_SWIPE = "Bald Swipe"
        BOOM_DROP = "BOOM DROP"
        HUGE_CUTIE = "Huge Cutie"
        LIQUID_METAL = "Liquid Metal"
        SHARKSNAP = "Sharksnap!"
        DUST_ME_AWAY = "Dust Me Away"
        FIGURINE_FACTOR_3D = "3D Figurine Factor"
        BIKINI_UP = "Bikini Up"
        MY_GIRLFRIENDS = "My Girlfriends"
        MY_BOYFRIENDS = "My Boyfriends"
        SUBJECT_3_FEVER = "Subject 3 Fever"
        EARTH_ZOOM = "Earth Zoom"
        POLE_DANCE = "Pole Dance"
        VROOM_DANCE = "Vroom Dance"
        GHOSTFACE_TERROR = "GhostFace Terror"
        DRAGON_EVOKER = "Dragon Evoker"
        SKELETAL_BAE = "Skeletal Bae"
        SUMMONING_SUCCUBUS = "Summoning succubus"
        HALLOWEEN_VOODOO_DOLL = "Halloween Voodoo Doll"
        NAKED_EYE_AD_3D = "3D Naked-Eye AD"
        PACKAGE_EXPLOSION = "Package Explosion"
        DISHES_SERVED = "Dishes Served"
        OCEAN_AD = "Ocean ad"
        SUPERMARKET_AD = "Supermarket AD"
        TREE_DOLL = "Tree doll"
        COME_FEEL_MY_ABS = "Come Feel My Abs"
        THE_BICEP_FLEX = "The Bicep Flex"
        LONDON_ELITE_VIBE = "London Elite Vibe"
        FLORA_NYMPH_GOWN = "Flora Nymph Gown"
        CHRISTMAS_COSTUME = "Christmas Costume"
        ITS_SNOWY = "It's Snowy"
        REINDEER_CRUISER = "Reindeer Cruiser"
        SNOW_GLOBE_MAKER = "Snow Globe Maker"
        PET_CHRISTMAS_OUTFIT = "Pet Christmas Outfit"
        ADOPT_A_POLAR_PAL = "Adopt a Polar Pal"
        CAT_CHRISTMAS_BOX = "Cat Christmas Box"
        STARLIGHT_GIFT_BOX = "Starlight Gift Box"
        XMAS_POSTER = "Xmas Poster"
        PET_CHRISTMAS_TREE = "Pet Christmas Tree"
        CITY_SANTA_HAT = "City Santa Hat"
        STOCKING_SWEETIE = "Stocking Sweetie"
        CHRISTMAS_NIGHT = "Christmas Night"
        XMAS_FRONT_PAGE_KARMA = "Xmas Front Page Karma"
        GRINCHS_XMAS_HIJACK = "Grinch's Xmas Hijack"
        GIANT_PRODUCT = "Giant Product"
        TRUCK_FASHION_SHOOT = "Truck Fashion Shoot"
        BEACH_AD = "Beach AD"
        SHOAL_SURROUND = "Shoal Surround"
        MECHANICAL_ASSEMBLY = "Mechanical Assembly"
        LIGHTING_AD = "Lighting AD"
        BILLBOARD_AD = "Billboard AD"
        PRODUCT_CLOSE_UP = "Product close-up"
        PARACHUTE_DELIVERY = "Parachute Delivery"
        DREAMLIKE_CLOUD = "Dreamlike Cloud"
        MACARON_MACHINE = "Macaron Machine"
        POSTER_AD = "Poster AD"
        TRUCK_AD = "Truck AD"
        GRAFFITI_AD = "Graffiti AD"
        FIGURINE_FACTORY_3D = "3D Figurine Factory"
        THE_EXCLUSIVE_FIRST_CLASS = "The Exclusive First Class"
        ART_ZOOM_CHALLENGE = "Art Zoom Challenge"
        I_QUIT = "I Quit"
        HITCHCOCK_DOLLY_ZOOM = "Hitchcock Dolly Zoom"
        SMELL_THE_LENS = "Smell the Lens"
        I_BELIEVE_I_CAN_FLY = "I believe I can fly"
        STRIKOUT_DANCE = "Strikout Dance"
        PIXEL_WORLD = "Pixel World"
        MINT_IN_BOX = "Mint in Box"
        HANDS_UP_HAND = "Hands up, Hand"
        FLORA_NYMPH_GO = "Flora Nymph Go"
        SOMBER_EMBRACE = "Somber Embrace"
        BEAM_ME_UP = "Beam me up"
        SUIT_SWAGGER = "Suit Swagger"


    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    effect: Effect = Field(
        default="", description="The effect to apply to the video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the image to use as the first frame. If not provided, generates from text"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "negative_prompt": self.negative_prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "effect": self.effect.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV5ImageToVideo(FALNode):
    """
    Generate high quality video clips from text and image prompts using PixVerse v5
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Duration(Enum):
        """
        The duration of the generated video in seconds. 8s videos cost double. 1080p videos are limited to 5 seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"


    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds. 8s videos cost double. 1080p videos are limited to 5 seconds"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "style": self.style.value if self.style else None,
            "image_url": f"data:image/png;base64,{image_base64}",
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class MoonvalleyMareyI2v(FALNode):
    """
    Generate a video starting from an image as the first frame with Marey, a generative video model trained exclusively on fully licensed data.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video.
        """
        VALUE_5S = "5s"
        VALUE_10S = "10s"

    class Dimensions(Enum):
        """
        The dimensions of the generated video in width x height format.
        """
        VALUE_1920X1080 = "1920x1080"
        VALUE_1080X1920 = "1080x1920"
        VALUE_1152X1152 = "1152x1152"
        VALUE_1536X1152 = "1536x1152"
        VALUE_1152X1536 = "1152x1536"


    prompt: str = Field(
        default="", description="The prompt to generate a video from"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5S, description="The duration of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the first frame of the video."
    )
    dimensions: Dimensions = Field(
        default=Dimensions.VALUE_1920X1080, description="The dimensions of the generated video in width x height format."
    )
    guidance_scale: str = Field(
        default="", description="Controls how strongly the generation is guided by the prompt (0-20). Higher values follow the prompt more closely."
    )
    seed: str = Field(
        default=-1, description="Seed for random number generation. Use -1 for random seed each run."
    )
    negative_prompt: str = Field(
        default="<synthetic> <scene cut> low-poly, flat shader, bad rigging, stiff animation, uncanny eyes, low-quality textures, looping glitch, cheap effect, overbloom, bloom spam, default lighting, game asset, stiff face, ugly specular, AI artifacts", description="Negative prompt used to guide the model away from undesirable features."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "dimensions": self.dimensions.value,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="moonvalley/marey/i2v",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanV22A14bImageToVideoLora(FALNode):
    """
    Wan-2.2 image-to-video is a video model that generates high-quality videos with high visual quality and motion diversity from text prompts and images. This endpoint supports LoRAs made for Wan 2.2
    video, animation, image-to-video, img2vid, lora

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Acceleration(Enum):
        """
        Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'.
        """
        NONE = "none"
        REGULAR = "regular"

    class VideoWriteMode(Enum):
        """
        The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

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

    class InterpolatorModel(Enum):
        """
        The model to use for frame interpolation. If None, no interpolation is applied.
        """
        NONE = "none"
        FILM = "film"
        RIFE = "rife"


    shift: float = Field(
        default=5, description="Shift value for the video. Must be between 1.0 and 10.0."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    num_interpolated_frames: int = Field(
        default=1, description="Number of frames to interpolate between each pair of generated frames. Must be between 0 and 4."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'."
    )
    reverse_video: bool = Field(
        default=False, description="If true, the video will be reversed."
    )
    loras: list[LoRAWeight] = Field(
        default=[], description="LoRA weights to be used in the inference."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 4 to 60. When using interpolation and `adjust_fps_for_interpolation` is set to true (default true,) the final FPS will be multiplied by the number of interpolated frames plus one. For example, if the generated frames per second is 16 and the number of interpolated frames is 1, the final frames per second will be 32. If `adjust_fps_for_interpolation` is set to false, this value will be used as-is."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 17 to 161 (inclusive)."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the end image."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p, 580p, or 720p)."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    guidance_scale_2: float = Field(
        default=4, description="Guidance scale for the second stage of the model. This is used to control the adherence to the prompt in the second stage of the model."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the output video. Higher quality means better visual quality but larger file size."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    adjust_fps_for_interpolation: bool = Field(
        default=True, description="If true, the number of frames per second will be multiplied by the number of interpolated frames plus one. For example, if the generated frames per second is 16 and the number of interpolated frames is 1, the final frames per second will be 32. If false, the passed frames per second will be used as-is."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: InterpolatorModel = Field(
        default=InterpolatorModel.FILM, description="The model to use for frame interpolation. If None, no interpolation is applied."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )
    num_inference_steps: int = Field(
        default=27, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "shift": self.shift,
            "prompt": self.prompt,
            "num_interpolated_frames": self.num_interpolated_frames,
            "acceleration": self.acceleration.value,
            "reverse_video": self.reverse_video,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "frames_per_second": self.frames_per_second,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "video_write_mode": self.video_write_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "guidance_scale_2": self.guidance_scale_2,
            "video_quality": self.video_quality.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "adjust_fps_for_interpolation": self.adjust_fps_for_interpolation,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-a14b/image-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class MinimaxHailuo02FastImageToVideo(FALNode):
    """
    Create blazing fast and economical videos with MiniMax Hailuo-02 Image To Video API at 512p resolution
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the video in seconds. 10 seconds videos are not supported for 1080p resolution.
        """
        VALUE_6 = "6"
        VALUE_10 = "10"


    prompt: str = Field(
        default=""
    )
    duration: Duration = Field(
        default=Duration.VALUE_6, description="The duration of the video in seconds. 10 seconds videos are not supported for 1080p resolution."
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    image: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "prompt_optimizer": self.prompt_optimizer,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-02-fast/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Veo3ImageToVideo(FALNode):
    """
    Veo 3 is the latest state-of-the art video generation model from Google DeepMind
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Duration(Enum):
        """
        The duration of the generated video.
        """
        VALUE_4S = "4s"
        VALUE_6S = "6s"
        VALUE_8S = "8s"


    prompt: str = Field(
        default="", description="The text prompt describing how the image should be animated"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
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
    duration: Duration = Field(
        default=Duration.VALUE_8S, description="The duration of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image to animate. Should be 720p or higher resolution in 16:9 or 9:16 aspect ratio. If the image is not in 16:9 or 9:16 aspect ratio, it will be cropped to fit."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "duration": self.duration.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanV22A14bImageToVideoTurbo(FALNode):
    """
    Wan-2.2 Turbo image-to-video is a video model that generates high-quality videos with high visual quality and motion diversity from text prompts. 
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
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

    class Acceleration(Enum):
        """
        Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'.
        """
        NONE = "none"
        REGULAR = "regular"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class VideoQuality(Enum):
        """
        The quality of the output video. Higher quality means better visual quality but larger file size.
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        MAXIMUM = "maximum"


    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p, 580p, or 720p)."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
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
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the end image."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "prompt": self.prompt,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-a14b/image-to-video/turbo",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanV225bImageToVideo(FALNode):
    """
    Wan 2.2's 5B model produces up to 5 seconds of video 720p at 24FPS with fluid motion and powerful prompt understanding
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class VideoWriteMode(Enum):
        """
        The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class Resolution(Enum):
        """
        Resolution of the generated video (580p or 720p).
        """
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

    class InterpolatorModel(Enum):
        """
        The model to use for frame interpolation. If None, no interpolation is applied.
        """
        NONE = "none"
        FILM = "film"
        RIFE = "rife"


    shift: float = Field(
        default=5, description="Shift value for the video. Must be between 1.0 and 10.0."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    num_interpolated_frames: int = Field(
        default=0, description="Number of frames to interpolate between each pair of generated frames. Must be between 0 and 4."
    )
    frames_per_second: int = Field(
        default=24, description="Frames per second of the generated video. Must be between 4 to 60. When using interpolation and `adjust_fps_for_interpolation` is set to true (default true,) the final FPS will be multiplied by the number of interpolated frames plus one. For example, if the generated frames per second is 16 and the number of interpolated frames is 1, the final frames per second will be 32. If `adjust_fps_for_interpolation` is set to false, this value will be used as-is."
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
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (580p or 720p)."
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
    adjust_fps_for_interpolation: bool = Field(
        default=True, description="If true, the number of frames per second will be multiplied by the number of interpolated frames plus one. For example, if the generated frames per second is 16 and the number of interpolated frames is 1, the final frames per second will be 32. If false, the passed frames per second will be used as-is."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: InterpolatorModel = Field(
        default=InterpolatorModel.FILM, description="The model to use for frame interpolation. If None, no interpolation is applied."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )
    num_inference_steps: int = Field(
        default=40, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "shift": self.shift,
            "prompt": self.prompt,
            "num_interpolated_frames": self.num_interpolated_frames,
            "frames_per_second": self.frames_per_second,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "image_url": f"data:image/png;base64,{image_base64}",
            "video_quality": self.video_quality.value,
            "adjust_fps_for_interpolation": self.adjust_fps_for_interpolation,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-5b/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanV22A14bImageToVideo(FALNode):
    """
    fal-ai/wan/v2.2-A14B/image-to-video
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Acceleration(Enum):
        """
        Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'.
        """
        NONE = "none"
        REGULAR = "regular"

    class VideoWriteMode(Enum):
        """
        The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

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

    class InterpolatorModel(Enum):
        """
        The model to use for frame interpolation. If None, no interpolation is applied.
        """
        NONE = "none"
        FILM = "film"
        RIFE = "rife"


    shift: float = Field(
        default=5, description="Shift value for the video. Must be between 1.0 and 10.0."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    num_interpolated_frames: int = Field(
        default=1, description="Number of frames to interpolate between each pair of generated frames. Must be between 0 and 4."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 4 to 60. When using interpolation and `adjust_fps_for_interpolation` is set to true (default true,) the final FPS will be multiplied by the number of interpolated frames plus one. For example, if the generated frames per second is 16 and the number of interpolated frames is 1, the final frames per second will be 32. If `adjust_fps_for_interpolation` is set to false, this value will be used as-is."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 17 to 161 (inclusive)."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the end image."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p, 580p, or 720p)."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    guidance_scale_2: float = Field(
        default=3.5, description="Guidance scale for the second stage of the model. This is used to control the adherence to the prompt in the second stage of the model."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the output video. Higher quality means better visual quality but larger file size."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    adjust_fps_for_interpolation: bool = Field(
        default=True, description="If true, the number of frames per second will be multiplied by the number of interpolated frames plus one. For example, if the generated frames per second is 16 and the number of interpolated frames is 1, the final frames per second will be 32. If false, the passed frames per second will be used as-is."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    interpolator_model: InterpolatorModel = Field(
        default=InterpolatorModel.FILM, description="The model to use for frame interpolation. If None, no interpolation is applied."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )
    num_inference_steps: int = Field(
        default=27, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "shift": self.shift,
            "prompt": self.prompt,
            "num_interpolated_frames": self.num_interpolated_frames,
            "acceleration": self.acceleration.value,
            "frames_per_second": self.frames_per_second,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "video_write_mode": self.video_write_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "guidance_scale_2": self.guidance_scale_2,
            "video_quality": self.video_quality.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "adjust_fps_for_interpolation": self.adjust_fps_for_interpolation,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-a14b/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BytedanceOmnihuman(FALNode):
    """
    OmniHuman generates video using an image of a human figure paired with an audio file. It produces vivid, high-quality videos where the character’s emotions and movements maintain a strong correlation with the audio.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    audio: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio file to generate the video. Audio must be under 30s long."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image used to generate the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "audio_url": self.audio,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/omnihuman",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Ltxv13b098DistilledImageToVideo(FALNode):
    """
    Generate long videos from prompts and images using LTX Video-0.9.8 13B Distilled and custom LoRA
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        Resolution of the generated video.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        The aspect ratio of the video.
        """
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"
        AUTO = "auto"


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
    enable_detail_pass: bool = Field(
        default=False, description="Whether to use a detail pass. If True, the model will perform a second pass to refine the video and enhance details. This incurs a 2.0x cost multiplier on the base price."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the video."
    )
    tone_map_compression_ratio: float = Field(
        default=0, description="The compression ratio for tone mapping. This is used to compress the dynamic range of the video to improve visual quality. A value of 0.0 means no compression, while a value of 1.0 means maximum compression."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Image URL for Image-to-Video task"
    )
    constant_rate_factor: int = Field(
        default=29, description="The constant rate factor (CRF) to compress input media with. Compressed input media more closely matches the model's training data, which can improve motion quality."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "second_pass_skip_initial_steps": self.second_pass_skip_initial_steps,
            "first_pass_num_inference_steps": self.first_pass_num_inference_steps,
            "frame_rate": self.frame_rate,
            "reverse_video": self.reverse_video,
            "prompt": self.prompt,
            "expand_prompt": self.expand_prompt,
            "temporal_adain_factor": self.temporal_adain_factor,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "enable_detail_pass": self.enable_detail_pass,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "tone_map_compression_ratio": self.tone_map_compression_ratio,
            "image_url": f"data:image/png;base64,{image_base64}",
            "constant_rate_factor": self.constant_rate_factor,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltxv-13b-098-distilled/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Veo3FastImageToVideo(FALNode):
    """
    Now with a 50% price drop. Generate videos from your image prompts using Veo 3 fast.
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Duration(Enum):
        """
        The duration of the generated video.
        """
        VALUE_4S = "4s"
        VALUE_6S = "6s"
        VALUE_8S = "8s"


    prompt: str = Field(
        default="", description="The text prompt describing how the image should be animated"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
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
    duration: Duration = Field(
        default=Duration.VALUE_8S, description="The duration of the generated video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image to animate. Should be 720p or higher resolution in 16:9 or 9:16 aspect ratio. If the image is not in 16:9 or 9:16 aspect ratio, it will be cropped to fit."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "duration": self.duration.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3/fast/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduQ1ReferenceToVideo(FALNode):
    """
    Generate video clips from your multiple image references using Vidu Q1
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the output video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class MovementAmplitude(Enum):
        """
        The movement amplitude of objects in the frame
        """
        AUTO = "auto"
        SMALL = "small"
        MEDIUM = "medium"
        LARGE = "large"


    prompt: str = Field(
        default="", description="Text prompt for video generation, max 1500 characters"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the output video"
    )
    bgm: bool = Field(
        default=False, description="Whether to add background music to the generated video"
    )
    reference_images: list[str] = Field(
        default=[], description="URLs of the reference images to use for consistent subject appearance. Q1 model supports up to 7 reference images."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )
    movement_amplitude: MovementAmplitude = Field(
        default=MovementAmplitude.AUTO, description="The movement amplitude of objects in the frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "bgm": self.bgm,
            "reference_image_urls": self.reference_images,
            "seed": self.seed,
            "movement_amplitude": self.movement_amplitude.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/q1/reference-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class MinimaxHailuo02ProImageToVideo(FALNode):
    """
    MiniMax Hailuo-02 Image To Video API (Pro, 1080p): Advanced image-to-video generation model with 1080p resolution
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    prompt: str = Field(
        default=""
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the image to use as the last frame of the video"
    )
    image: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_base64 = await context.image_to_base64(self.end_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-02/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class BytedanceSeedanceV1LiteImageToVideo(FALNode):
    """
    Seedance 1.0 Lite
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"
        AUTO = "auto"

    class Duration(Enum):
        """
        Duration of the video in seconds
        """
        VALUE_2 = "2"
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

    class Resolution(Enum):
        """
        Video resolution - 480p for faster generation, 720p for higher quality
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the generated video"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the video in seconds"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Video resolution - 480p for faster generation, 720p for higher quality"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image used to generate video"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    camera_fixed: bool = Field(
        default=False, description="Whether to fix the camera position"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image the video ends with. Defaults to None."
    )
    seed: int = Field(
        default=-1, description="Random seed to control video generation. Use -1 for random."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "camera_fixed": self.camera_fixed,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedance/v1/lite/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class HunyuanAvatar(FALNode):
    """
    HunyuanAvatar is a High-Fidelity Audio-Driven Human Animation model for Multiple Characters .
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    text: str = Field(
        default="A cat is singing.", description="Text prompt describing the scene."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the reference image."
    )
    turbo_mode: bool = Field(
        default=True, description="If true, the video will be generated faster with no noticeable degradation in the visual quality."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    num_frames: int = Field(
        default=129, description="Number of video frames to generate at 25 FPS. If greater than the input audio length, it will capped to the length of the input audio."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "text": self.text,
            "image_url": f"data:image/png;base64,{image_base64}",
            "turbo_mode": self.turbo_mode,
            "audio_url": self.audio,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "num_frames": self.num_frames,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-avatar",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoV21ProImageToVideo(FALNode):
    """
    Kling 2.1 Pro is an advanced endpoint for the Kling 2.1 model, offering professional-grade videos with enhanced visual fidelity, precise camera movements, and dynamic motion control, perfect for cinematic storytelling.  
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"


    prompt: str = Field(
        default=""
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    tail_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the end of the video"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        tail_image_base64 = await context.image_to_base64(self.tail_image)
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "tail_image_url": f"data:image/png;base64,{tail_image_base64}",
            "cfg_scale": self.cfg_scale,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2.1/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class HunyuanPortrait(FALNode):
    """
    HunyuanPortrait is a diffusion-based framework for generating lifelike, temporally consistent portrait animations.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the driving video."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation. If None, a random seed will be used."
    )
    use_arcface: bool = Field(
        default=True, description="Whether to use ArcFace for face recognition."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the source image."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "video_url": self.video,
            "seed": self.seed,
            "use_arcface": self.use_arcface,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-portrait",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoV16StandardElements(FALNode):
    """
    Generate video clips from your multiple image references using Kling 1.6 (standard)
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video frame
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"


    prompt: str = Field(
        default=""
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    input_images: list[str] = Field(
        default=[], description="List of image URLs to use for video generation. Supports up to 4 images."
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "input_image_urls": self.input_images,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.6/standard/elements",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoV16ProElements(FALNode):
    """
    Generate video clips from your multiple image references using Kling 1.6 (pro)
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video frame
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"


    prompt: str = Field(
        default=""
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    input_images: list[str] = Field(
        default=[], description="List of image URLs to use for video generation. Supports up to 4 images."
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "input_image_urls": self.input_images,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.6/pro/elements",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LtxVideo13bDistilledImageToVideo(FALNode):
    """
    Generate videos from prompts and images using LTX Video-0.9.7 13B Distilled and custom LoRA
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        Resolution of the generated video (480p or 720p).
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        The aspect ratio of the video.
        """
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"
        AUTO = "auto"


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
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p or 720p)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Image URL for Image-to-Video task"
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
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "second_pass_skip_initial_steps": self.second_pass_skip_initial_steps,
            "first_pass_num_inference_steps": self.first_pass_num_inference_steps,
            "frame_rate": self.frame_rate,
            "reverse_video": self.reverse_video,
            "prompt": self.prompt,
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "constant_rate_factor": self.constant_rate_factor,
            "first_pass_skip_final_steps": self.first_pass_skip_final_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-13b-distilled/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LtxVideo13bDevImageToVideo(FALNode):
    """
    Generate videos from prompts and images using LTX Video-0.9.7 13B and custom LoRA
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        Resolution of the generated video (480p or 720p).
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        The aspect ratio of the video.
        """
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"
        AUTO = "auto"


    second_pass_skip_initial_steps: int = Field(
        default=17, description="The number of inference steps to skip in the initial steps of the second pass. By skipping some steps at the beginning, the second pass can focus on smaller details instead of larger changes."
    )
    first_pass_num_inference_steps: int = Field(
        default=30, description="Number of inference steps during the first pass."
    )
    frame_rate: int = Field(
        default=30, description="The frame rate of the video."
    )
    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video."
    )
    expand_prompt: bool = Field(
        default=False, description="Whether to expand the prompt using a language model."
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
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted", description="Negative prompt for generation"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p or 720p)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Image URL for Image-to-Video task"
    )
    constant_rate_factor: int = Field(
        default=35, description="The constant rate factor (CRF) to compress input media with. Compressed input media more closely matches the model's training data, which can improve motion quality."
    )
    first_pass_skip_final_steps: int = Field(
        default=3, description="Number of inference steps to skip in the final steps of the first pass. By skipping some steps at the end, the first pass can focus on larger changes instead of smaller details."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "second_pass_skip_initial_steps": self.second_pass_skip_initial_steps,
            "first_pass_num_inference_steps": self.first_pass_num_inference_steps,
            "frame_rate": self.frame_rate,
            "prompt": self.prompt,
            "reverse_video": self.reverse_video,
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "constant_rate_factor": self.constant_rate_factor,
            "first_pass_skip_final_steps": self.first_pass_skip_final_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-13b-dev/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LtxVideoLoraImageToVideo(FALNode):
    """
    Generate videos from prompts and images using LTX Video-0.9.7 and custom LoRA
    video, animation, image-to-video, img2vid, lora

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the video.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        The aspect ratio of the video.
        """
        RATIO_16_9 = "16:9"
        RATIO_1_1 = "1:1"
        RATIO_9_16 = "9:16"
        AUTO = "auto"


    number_of_steps: int = Field(
        default=30, description="The number of inference steps to use."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the video."
    )
    reverse_video: bool = Field(
        default=False, description="Whether to reverse the video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="The aspect ratio of the video."
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
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as input."
    )
    loras: list[LoRAWeight] = Field(
        default=[], description="The LoRA weights to use for generation."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    seed: int = Field(
        default=-1, description="The seed to use for generation."
    )
    negative_prompt: str = Field(
        default="blurry, low quality, low resolution, inconsistent motion, jittery, distorted", description="The negative prompt to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "number_of_steps": self.number_of_steps,
            "resolution": self.resolution.value,
            "reverse_video": self.reverse_video,
            "aspect_ratio": self.aspect_ratio.value,
            "frame_rate": self.frame_rate,
            "expand_prompt": self.expand_prompt,
            "number_of_frames": self.number_of_frames,
            "image_url": f"data:image/png;base64,{image_base64}",
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "prompt": self.prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-lora/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV45Transition(FALNode):
    """
    Create seamless transition between images using PixVerse v4.5
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"


    first_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    prompt: str = Field(
        default="", description="The prompt for the transition"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the last frame"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        first_image_base64 = await context.image_to_base64(self.first_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "first_image_url": f"data:image/png;base64,{first_image_base64}",
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "prompt": self.prompt,
            "duration": self.duration.value,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4.5/transition",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV45ImageToVideoFast(FALNode):
    """
    Generate fast high quality video clips from text and image prompts using PixVerse v4.5
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class CameraMovement(Enum):
        """
        The type of camera movement to apply to the video
        """
        HORIZONTAL_LEFT = "horizontal_left"
        HORIZONTAL_RIGHT = "horizontal_right"
        VERTICAL_UP = "vertical_up"
        VERTICAL_DOWN = "vertical_down"
        ZOOM_IN = "zoom_in"
        ZOOM_OUT = "zoom_out"
        CRANE_UP = "crane_up"
        QUICKLY_ZOOM_IN = "quickly_zoom_in"
        QUICKLY_ZOOM_OUT = "quickly_zoom_out"
        SMOOTH_ZOOM_IN = "smooth_zoom_in"
        CAMERA_ROTATION = "camera_rotation"
        ROBO_ARM = "robo_arm"
        SUPER_DOLLY_OUT = "super_dolly_out"
        WHIP_PAN = "whip_pan"
        HITCHCOCK = "hitchcock"
        LEFT_FOLLOW = "left_follow"
        RIGHT_FOLLOW = "right_follow"
        PAN_LEFT = "pan_left"
        PAN_RIGHT = "pan_right"
        FIX_BG = "fix_bg"


    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    camera_movement: CameraMovement | None = Field(
        default=None, description="The type of camera movement to apply to the video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "camera_movement": self.camera_movement.value if self.camera_movement else None,
            "image_url": f"data:image/png;base64,{image_base64}",
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4.5/image-to-video/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV45Effects(FALNode):
    """
    Generate high quality video clips with different effects using PixVerse v4.5
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Effect(Enum):
        """
        The effect to apply to the video
        """
        KISS_ME_AI = "Kiss Me AI"
        KISS = "Kiss"
        MUSCLE_SURGE = "Muscle Surge"
        WARMTH_OF_JESUS = "Warmth of Jesus"
        ANYTHING_ROBOT = "Anything, Robot"
        THE_TIGER_TOUCH = "The Tiger Touch"
        HUG = "Hug"
        HOLY_WINGS = "Holy Wings"
        MICROWAVE = "Microwave"
        ZOMBIE_MODE = "Zombie Mode"
        SQUID_GAME = "Squid Game"
        BABY_FACE = "Baby Face"
        BLACK_MYTH_WUKONG = "Black Myth: Wukong"
        LONG_HAIR_MAGIC = "Long Hair Magic"
        LEGGY_RUN = "Leggy Run"
        FIN_TASTIC_MERMAID = "Fin-tastic Mermaid"
        PUNCH_FACE = "Punch Face"
        CREEPY_DEVIL_SMILE = "Creepy Devil Smile"
        THUNDER_GOD = "Thunder God"
        EYE_ZOOM_CHALLENGE = "Eye Zoom Challenge"
        WHOS_ARRESTED = "Who's Arrested?"
        BABY_ARRIVED = "Baby Arrived"
        WEREWOLF_RAGE = "Werewolf Rage"
        BALD_SWIPE = "Bald Swipe"
        BOOM_DROP = "BOOM DROP"
        HUGE_CUTIE = "Huge Cutie"
        LIQUID_METAL = "Liquid Metal"
        SHARKSNAP = "Sharksnap!"
        DUST_ME_AWAY = "Dust Me Away"
        FIGURINE_FACTOR_3D = "3D Figurine Factor"
        BIKINI_UP = "Bikini Up"
        MY_GIRLFRIENDS = "My Girlfriends"
        MY_BOYFRIENDS = "My Boyfriends"
        SUBJECT_3_FEVER = "Subject 3 Fever"
        EARTH_ZOOM = "Earth Zoom"
        POLE_DANCE = "Pole Dance"
        VROOM_DANCE = "Vroom Dance"
        GHOSTFACE_TERROR = "GhostFace Terror"
        DRAGON_EVOKER = "Dragon Evoker"
        SKELETAL_BAE = "Skeletal Bae"
        SUMMONING_SUCCUBUS = "Summoning succubus"
        HALLOWEEN_VOODOO_DOLL = "Halloween Voodoo Doll"
        NAKED_EYE_AD_3D = "3D Naked-Eye AD"
        PACKAGE_EXPLOSION = "Package Explosion"
        DISHES_SERVED = "Dishes Served"
        OCEAN_AD = "Ocean ad"
        SUPERMARKET_AD = "Supermarket AD"
        TREE_DOLL = "Tree doll"
        COME_FEEL_MY_ABS = "Come Feel My Abs"
        THE_BICEP_FLEX = "The Bicep Flex"
        LONDON_ELITE_VIBE = "London Elite Vibe"
        FLORA_NYMPH_GOWN = "Flora Nymph Gown"
        CHRISTMAS_COSTUME = "Christmas Costume"
        ITS_SNOWY = "It's Snowy"
        REINDEER_CRUISER = "Reindeer Cruiser"
        SNOW_GLOBE_MAKER = "Snow Globe Maker"
        PET_CHRISTMAS_OUTFIT = "Pet Christmas Outfit"
        ADOPT_A_POLAR_PAL = "Adopt a Polar Pal"
        CAT_CHRISTMAS_BOX = "Cat Christmas Box"
        STARLIGHT_GIFT_BOX = "Starlight Gift Box"
        XMAS_POSTER = "Xmas Poster"
        PET_CHRISTMAS_TREE = "Pet Christmas Tree"
        CITY_SANTA_HAT = "City Santa Hat"
        STOCKING_SWEETIE = "Stocking Sweetie"
        CHRISTMAS_NIGHT = "Christmas Night"
        XMAS_FRONT_PAGE_KARMA = "Xmas Front Page Karma"
        GRINCHS_XMAS_HIJACK = "Grinch's Xmas Hijack"
        GIANT_PRODUCT = "Giant Product"
        TRUCK_FASHION_SHOOT = "Truck Fashion Shoot"
        BEACH_AD = "Beach AD"
        SHOAL_SURROUND = "Shoal Surround"
        MECHANICAL_ASSEMBLY = "Mechanical Assembly"
        LIGHTING_AD = "Lighting AD"
        BILLBOARD_AD = "Billboard AD"
        PRODUCT_CLOSE_UP = "Product close-up"
        PARACHUTE_DELIVERY = "Parachute Delivery"
        DREAMLIKE_CLOUD = "Dreamlike Cloud"
        MACARON_MACHINE = "Macaron Machine"
        POSTER_AD = "Poster AD"
        TRUCK_AD = "Truck AD"
        GRAFFITI_AD = "Graffiti AD"
        FIGURINE_FACTORY_3D = "3D Figurine Factory"
        THE_EXCLUSIVE_FIRST_CLASS = "The Exclusive First Class"
        ART_ZOOM_CHALLENGE = "Art Zoom Challenge"
        I_QUIT = "I Quit"
        HITCHCOCK_DOLLY_ZOOM = "Hitchcock Dolly Zoom"
        SMELL_THE_LENS = "Smell the Lens"
        I_BELIEVE_I_CAN_FLY = "I believe I can fly"
        STRIKOUT_DANCE = "Strikout Dance"
        PIXEL_WORLD = "Pixel World"
        MINT_IN_BOX = "Mint in Box"
        HANDS_UP_HAND = "Hands up, Hand"
        FLORA_NYMPH_GO = "Flora Nymph Go"
        SOMBER_EMBRACE = "Somber Embrace"
        BEAM_ME_UP = "Beam me up"
        SUIT_SWAGGER = "Suit Swagger"


    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    effect: Effect = Field(
        default="", description="The effect to apply to the video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the image to use as the first frame. If not provided, generates from text"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "negative_prompt": self.negative_prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "effect": self.effect.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4.5/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class HunyuanCustom(FALNode):
    """
    HunyuanCustom revolutionizes video generation with unmatched identity consistency across multiple input types. Its innovative fusion modules and alignment networks outperform competitors, maintaining subject integrity while responding flexibly to text, image, audio, and video conditions.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the video to generate.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the video to generate. 720p generations cost 1.5x more than 480p generations.
        """
        VALUE_512P = "512p"
        VALUE_720P = "720p"


    prompt: str = Field(
        default="", description="Text prompt for video generation (max 500 characters)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video to generate."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_512P, description="The resolution of the video to generate. 720p generations cost 1.5x more than 480p generations."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_frames: int = Field(
        default=129, description="The number of frames to generate."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image input."
    )
    fps: int = Field(
        default=25, description="The frames per second of the generated video."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed to use for generating the video."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to run. Lower gets faster results, higher gets better results."
    )
    negative_prompt: str = Field(
        default="Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.", description="Negative prompt for video generation."
    )
    cfg_scale: float = Field(
        default=7.5, description="Classifier-Free Guidance scale for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_base64}",
            "fps": self.fps,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-custom",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FramepackF1(FALNode):
    """
    Framepack is an efficient Image-to-video model that autoregressively generates videos.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the video to generate.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the video to generate. 720p generations cost 1.5x more than 480p generations.
        """
        VALUE_720P = "720p"
        VALUE_480P = "480p"


    prompt: str = Field(
        default="", description="Text prompt for video generation (max 500 characters)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video to generate."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the video to generate. 720p generations cost 1.5x more than 480p generations."
    )
    num_frames: int = Field(
        default=180, description="The number of frames to generate."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image input."
    )
    guidance_scale: float = Field(
        default=10, description="Guidance scale for the generation."
    )
    seed: str = Field(
        default="", description="The seed to use for generating the video."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    cfg_scale: float = Field(
        default=1, description="Classifier-Free Guidance scale for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_base64}",
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/framepack/f1",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduQ1StartEndToVideo(FALNode):
    """
    Vidu Q1 Start-End to Video generates smooth transition 1080p videos between specified start and end images.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class MovementAmplitude(Enum):
        """
        The movement amplitude of objects in the frame
        """
        AUTO = "auto"
        SMALL = "small"
        MEDIUM = "medium"
        LARGE = "large"


    prompt: str = Field(
        default="", description="Text prompt for video generation, max 1500 characters"
    )
    start_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the last frame"
    )
    seed: int = Field(
        default=-1, description="Seed for the random number generator"
    )
    movement_amplitude: MovementAmplitude = Field(
        default=MovementAmplitude.AUTO, description="The movement amplitude of objects in the frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_base64 = await context.image_to_base64(self.start_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "seed": self.seed,
            "movement_amplitude": self.movement_amplitude.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/q1/start-end-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduQ1ImageToVideo(FALNode):
    """
    Vidu Q1 Image to Video generates high-quality 1080p videos with exceptional visual quality and motion diversity from a single image
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class MovementAmplitude(Enum):
        """
        The movement amplitude of objects in the frame
        """
        AUTO = "auto"
        SMALL = "small"
        MEDIUM = "medium"
        LARGE = "large"


    prompt: str = Field(
        default="", description="Text prompt for video generation, max 1500 characters"
    )
    seed: int = Field(
        default=-1, description="Seed for the random number generator"
    )
    movement_amplitude: MovementAmplitude = Field(
        default=MovementAmplitude.AUTO, description="The movement amplitude of objects in the frame"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "movement_amplitude": self.movement_amplitude.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/q1/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV4Effects(FALNode):
    """
    Generate high quality video clips with different effects using PixVerse v4
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Effect(Enum):
        """
        The effect to apply to the video
        """
        KISS_ME_AI = "Kiss Me AI"
        KISS = "Kiss"
        MUSCLE_SURGE = "Muscle Surge"
        WARMTH_OF_JESUS = "Warmth of Jesus"
        ANYTHING_ROBOT = "Anything, Robot"
        THE_TIGER_TOUCH = "The Tiger Touch"
        HUG = "Hug"
        HOLY_WINGS = "Holy Wings"
        MICROWAVE = "Microwave"
        ZOMBIE_MODE = "Zombie Mode"
        SQUID_GAME = "Squid Game"
        BABY_FACE = "Baby Face"
        BLACK_MYTH_WUKONG = "Black Myth: Wukong"
        LONG_HAIR_MAGIC = "Long Hair Magic"
        LEGGY_RUN = "Leggy Run"
        FIN_TASTIC_MERMAID = "Fin-tastic Mermaid"
        PUNCH_FACE = "Punch Face"
        CREEPY_DEVIL_SMILE = "Creepy Devil Smile"
        THUNDER_GOD = "Thunder God"
        EYE_ZOOM_CHALLENGE = "Eye Zoom Challenge"
        WHOS_ARRESTED = "Who's Arrested?"
        BABY_ARRIVED = "Baby Arrived"
        WEREWOLF_RAGE = "Werewolf Rage"
        BALD_SWIPE = "Bald Swipe"
        BOOM_DROP = "BOOM DROP"
        HUGE_CUTIE = "Huge Cutie"
        LIQUID_METAL = "Liquid Metal"
        SHARKSNAP = "Sharksnap!"
        DUST_ME_AWAY = "Dust Me Away"
        FIGURINE_FACTOR_3D = "3D Figurine Factor"
        BIKINI_UP = "Bikini Up"
        MY_GIRLFRIENDS = "My Girlfriends"
        MY_BOYFRIENDS = "My Boyfriends"
        SUBJECT_3_FEVER = "Subject 3 Fever"
        EARTH_ZOOM = "Earth Zoom"
        POLE_DANCE = "Pole Dance"
        VROOM_DANCE = "Vroom Dance"
        GHOSTFACE_TERROR = "GhostFace Terror"
        DRAGON_EVOKER = "Dragon Evoker"
        SKELETAL_BAE = "Skeletal Bae"
        SUMMONING_SUCCUBUS = "Summoning succubus"
        HALLOWEEN_VOODOO_DOLL = "Halloween Voodoo Doll"
        NAKED_EYE_AD_3D = "3D Naked-Eye AD"
        PACKAGE_EXPLOSION = "Package Explosion"
        DISHES_SERVED = "Dishes Served"
        OCEAN_AD = "Ocean ad"
        SUPERMARKET_AD = "Supermarket AD"
        TREE_DOLL = "Tree doll"
        COME_FEEL_MY_ABS = "Come Feel My Abs"
        THE_BICEP_FLEX = "The Bicep Flex"
        LONDON_ELITE_VIBE = "London Elite Vibe"
        FLORA_NYMPH_GOWN = "Flora Nymph Gown"
        CHRISTMAS_COSTUME = "Christmas Costume"
        ITS_SNOWY = "It's Snowy"
        REINDEER_CRUISER = "Reindeer Cruiser"
        SNOW_GLOBE_MAKER = "Snow Globe Maker"
        PET_CHRISTMAS_OUTFIT = "Pet Christmas Outfit"
        ADOPT_A_POLAR_PAL = "Adopt a Polar Pal"
        CAT_CHRISTMAS_BOX = "Cat Christmas Box"
        STARLIGHT_GIFT_BOX = "Starlight Gift Box"
        XMAS_POSTER = "Xmas Poster"
        PET_CHRISTMAS_TREE = "Pet Christmas Tree"
        CITY_SANTA_HAT = "City Santa Hat"
        STOCKING_SWEETIE = "Stocking Sweetie"
        CHRISTMAS_NIGHT = "Christmas Night"
        XMAS_FRONT_PAGE_KARMA = "Xmas Front Page Karma"
        GRINCHS_XMAS_HIJACK = "Grinch's Xmas Hijack"
        GIANT_PRODUCT = "Giant Product"
        TRUCK_FASHION_SHOOT = "Truck Fashion Shoot"
        BEACH_AD = "Beach AD"
        SHOAL_SURROUND = "Shoal Surround"
        MECHANICAL_ASSEMBLY = "Mechanical Assembly"
        LIGHTING_AD = "Lighting AD"
        BILLBOARD_AD = "Billboard AD"
        PRODUCT_CLOSE_UP = "Product close-up"
        PARACHUTE_DELIVERY = "Parachute Delivery"
        DREAMLIKE_CLOUD = "Dreamlike Cloud"
        MACARON_MACHINE = "Macaron Machine"
        POSTER_AD = "Poster AD"
        TRUCK_AD = "Truck AD"
        GRAFFITI_AD = "Graffiti AD"
        FIGURINE_FACTORY_3D = "3D Figurine Factory"
        THE_EXCLUSIVE_FIRST_CLASS = "The Exclusive First Class"
        ART_ZOOM_CHALLENGE = "Art Zoom Challenge"
        I_QUIT = "I Quit"
        HITCHCOCK_DOLLY_ZOOM = "Hitchcock Dolly Zoom"
        SMELL_THE_LENS = "Smell the Lens"
        I_BELIEVE_I_CAN_FLY = "I believe I can fly"
        STRIKOUT_DANCE = "Strikout Dance"
        PIXEL_WORLD = "Pixel World"
        MINT_IN_BOX = "Mint in Box"
        HANDS_UP_HAND = "Hands up, Hand"
        FLORA_NYMPH_GO = "Flora Nymph Go"
        SOMBER_EMBRACE = "Somber Embrace"
        BEAM_ME_UP = "Beam me up"
        SUIT_SWAGGER = "Suit Swagger"


    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    effect: Effect = Field(
        default="", description="The effect to apply to the video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the image to use as the first frame. If not provided, generates from text"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "negative_prompt": self.negative_prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "effect": self.effect.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FramepackFlf2v(FALNode):
    """
    Framepack is an efficient Image-to-video model that autoregressively generates videos.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the video to generate.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the video to generate. 720p generations cost 1.5x more than 480p generations.
        """
        VALUE_720P = "720p"
        VALUE_480P = "480p"


    prompt: str = Field(
        default="", description="Text prompt for video generation (max 500 characters)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video to generate."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the video to generate. 720p generations cost 1.5x more than 480p generations."
    )
    num_frames: int = Field(
        default=240, description="The number of frames to generate."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image input."
    )
    strength: float = Field(
        default=0.8, description="Determines the influence of the final frame on the generated video. Higher values result in the output being more heavily influenced by the last frame."
    )
    guidance_scale: float = Field(
        default=10, description="Guidance scale for the generation."
    )
    seed: str = Field(
        default="", description="The seed to use for generating the video."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the end image input."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    cfg_scale: float = Field(
        default=1, description="Classifier-Free Guidance scale for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "image_url": f"data:image/png;base64,{image_base64}",
            "strength": self.strength,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/framepack/flf2v",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanFlf2v(FALNode):
    """
    Wan-2.1 flf2v generates dynamic videos by intelligently bridging a given first frame to a desired end frame through smooth, coherent motion sequences.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Acceleration(Enum):
        """
        Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'.
        """
        NONE = "none"
        REGULAR = "regular"

    class Resolution(Enum):
        """
        Resolution of the generated video (480p or 720p). 480p is 0.5 billing units, and 720p is 1 billing unit.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 24."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    start_image: ImageRef = Field(
        default=ImageRef(), description="URL of the starting image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the ending image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    negative_prompt: str = Field(
        default="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 100 (inclusive). If the number of frames is greater than 81, the video will be generated with 1.25x more billing units."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p or 720p). 480p is 0.5 billing units, and 720p is 1 billing unit."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.AUTO, description="Aspect ratio of the generated video. If 'auto', the aspect ratio will be determined automatically based on the input image."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    guide_scale: float = Field(
        default=5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_base64 = await context.image_to_base64(self.start_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "shift": self.shift,
            "acceleration": self.acceleration.value,
            "frames_per_second": self.frames_per_second,
            "enable_safety_checker": self.enable_safety_checker,
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "guide_scale": self.guide_scale,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-flf2v",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Framepack(FALNode):
    """
    Framepack is an efficient Image-to-video model that autoregressively generates videos.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the video to generate.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the video to generate. 720p generations cost 1.5x more than 480p generations.
        """
        VALUE_720P = "720p"
        VALUE_480P = "480p"


    prompt: str = Field(
        default="", description="Text prompt for video generation (max 500 characters)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video to generate."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the video to generate. 720p generations cost 1.5x more than 480p generations."
    )
    num_frames: int = Field(
        default=180, description="The number of frames to generate."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image input."
    )
    guidance_scale: float = Field(
        default=10, description="Guidance scale for the generation."
    )
    seed: str = Field(
        default="", description="The seed to use for generating the video."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    cfg_scale: float = Field(
        default=1, description="Classifier-Free Guidance scale for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_base64}",
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/framepack",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV4ImageToVideoFast(FALNode):
    """
    Generate fast high quality video clips from text and image prompts using PixVerse v4
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class CameraMovement(Enum):
        """
        The type of camera movement to apply to the video
        """
        HORIZONTAL_LEFT = "horizontal_left"
        HORIZONTAL_RIGHT = "horizontal_right"
        VERTICAL_UP = "vertical_up"
        VERTICAL_DOWN = "vertical_down"
        ZOOM_IN = "zoom_in"
        ZOOM_OUT = "zoom_out"
        CRANE_UP = "crane_up"
        QUICKLY_ZOOM_IN = "quickly_zoom_in"
        QUICKLY_ZOOM_OUT = "quickly_zoom_out"
        SMOOTH_ZOOM_IN = "smooth_zoom_in"
        CAMERA_ROTATION = "camera_rotation"
        ROBO_ARM = "robo_arm"
        SUPER_DOLLY_OUT = "super_dolly_out"
        WHIP_PAN = "whip_pan"
        HITCHCOCK = "hitchcock"
        LEFT_FOLLOW = "left_follow"
        RIGHT_FOLLOW = "right_follow"
        PAN_LEFT = "pan_left"
        PAN_RIGHT = "pan_right"
        FIX_BG = "fix_bg"


    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    camera_movement: CameraMovement | None = Field(
        default=None, description="The type of camera movement to apply to the video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "camera_movement": self.camera_movement.value if self.camera_movement else None,
            "image_url": f"data:image/png;base64,{image_base64}",
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4/image-to-video/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV4ImageToVideo(FALNode):
    """
    Generate high quality video clips from text and image prompts using PixVerse v4
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Duration(Enum):
        """
        The duration of the generated video in seconds. 8s videos cost double. 1080p videos are limited to 5 seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class CameraMovement(Enum):
        """
        The type of camera movement to apply to the video
        """
        HORIZONTAL_LEFT = "horizontal_left"
        HORIZONTAL_RIGHT = "horizontal_right"
        VERTICAL_UP = "vertical_up"
        VERTICAL_DOWN = "vertical_down"
        ZOOM_IN = "zoom_in"
        ZOOM_OUT = "zoom_out"
        CRANE_UP = "crane_up"
        QUICKLY_ZOOM_IN = "quickly_zoom_in"
        QUICKLY_ZOOM_OUT = "quickly_zoom_out"
        SMOOTH_ZOOM_IN = "smooth_zoom_in"
        CAMERA_ROTATION = "camera_rotation"
        ROBO_ARM = "robo_arm"
        SUPER_DOLLY_OUT = "super_dolly_out"
        WHIP_PAN = "whip_pan"
        HITCHCOCK = "hitchcock"
        LEFT_FOLLOW = "left_follow"
        RIGHT_FOLLOW = "right_follow"
        PAN_LEFT = "pan_left"
        PAN_RIGHT = "pan_right"
        FIX_BG = "fix_bg"


    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds. 8s videos cost double. 1080p videos are limited to 5 seconds"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    camera_movement: CameraMovement | None = Field(
        default=None, description="The type of camera movement to apply to the video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "style": self.style.value if self.style else None,
            "camera_movement": self.camera_movement.value if self.camera_movement else None,
            "image_url": f"data:image/png;base64,{image_base64}",
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV35Effects(FALNode):
    """
    Generate high quality video clips with different effects using PixVerse v3.5
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Effect(Enum):
        """
        The effect to apply to the video
        """
        KISS_ME_AI = "Kiss Me AI"
        KISS = "Kiss"
        MUSCLE_SURGE = "Muscle Surge"
        WARMTH_OF_JESUS = "Warmth of Jesus"
        ANYTHING_ROBOT = "Anything, Robot"
        THE_TIGER_TOUCH = "The Tiger Touch"
        HUG = "Hug"
        HOLY_WINGS = "Holy Wings"
        MICROWAVE = "Microwave"
        ZOMBIE_MODE = "Zombie Mode"
        SQUID_GAME = "Squid Game"
        BABY_FACE = "Baby Face"
        BLACK_MYTH_WUKONG = "Black Myth: Wukong"
        LONG_HAIR_MAGIC = "Long Hair Magic"
        LEGGY_RUN = "Leggy Run"
        FIN_TASTIC_MERMAID = "Fin-tastic Mermaid"
        PUNCH_FACE = "Punch Face"
        CREEPY_DEVIL_SMILE = "Creepy Devil Smile"
        THUNDER_GOD = "Thunder God"
        EYE_ZOOM_CHALLENGE = "Eye Zoom Challenge"
        WHOS_ARRESTED = "Who's Arrested?"
        BABY_ARRIVED = "Baby Arrived"
        WEREWOLF_RAGE = "Werewolf Rage"
        BALD_SWIPE = "Bald Swipe"
        BOOM_DROP = "BOOM DROP"
        HUGE_CUTIE = "Huge Cutie"
        LIQUID_METAL = "Liquid Metal"
        SHARKSNAP = "Sharksnap!"
        DUST_ME_AWAY = "Dust Me Away"
        FIGURINE_FACTOR_3D = "3D Figurine Factor"
        BIKINI_UP = "Bikini Up"
        MY_GIRLFRIENDS = "My Girlfriends"
        MY_BOYFRIENDS = "My Boyfriends"
        SUBJECT_3_FEVER = "Subject 3 Fever"
        EARTH_ZOOM = "Earth Zoom"
        POLE_DANCE = "Pole Dance"
        VROOM_DANCE = "Vroom Dance"
        GHOSTFACE_TERROR = "GhostFace Terror"
        DRAGON_EVOKER = "Dragon Evoker"
        SKELETAL_BAE = "Skeletal Bae"
        SUMMONING_SUCCUBUS = "Summoning succubus"
        HALLOWEEN_VOODOO_DOLL = "Halloween Voodoo Doll"
        NAKED_EYE_AD_3D = "3D Naked-Eye AD"
        PACKAGE_EXPLOSION = "Package Explosion"
        DISHES_SERVED = "Dishes Served"
        OCEAN_AD = "Ocean ad"
        SUPERMARKET_AD = "Supermarket AD"
        TREE_DOLL = "Tree doll"
        COME_FEEL_MY_ABS = "Come Feel My Abs"
        THE_BICEP_FLEX = "The Bicep Flex"
        LONDON_ELITE_VIBE = "London Elite Vibe"
        FLORA_NYMPH_GOWN = "Flora Nymph Gown"
        CHRISTMAS_COSTUME = "Christmas Costume"
        ITS_SNOWY = "It's Snowy"
        REINDEER_CRUISER = "Reindeer Cruiser"
        SNOW_GLOBE_MAKER = "Snow Globe Maker"
        PET_CHRISTMAS_OUTFIT = "Pet Christmas Outfit"
        ADOPT_A_POLAR_PAL = "Adopt a Polar Pal"
        CAT_CHRISTMAS_BOX = "Cat Christmas Box"
        STARLIGHT_GIFT_BOX = "Starlight Gift Box"
        XMAS_POSTER = "Xmas Poster"
        PET_CHRISTMAS_TREE = "Pet Christmas Tree"
        CITY_SANTA_HAT = "City Santa Hat"
        STOCKING_SWEETIE = "Stocking Sweetie"
        CHRISTMAS_NIGHT = "Christmas Night"
        XMAS_FRONT_PAGE_KARMA = "Xmas Front Page Karma"
        GRINCHS_XMAS_HIJACK = "Grinch's Xmas Hijack"
        GIANT_PRODUCT = "Giant Product"
        TRUCK_FASHION_SHOOT = "Truck Fashion Shoot"
        BEACH_AD = "Beach AD"
        SHOAL_SURROUND = "Shoal Surround"
        MECHANICAL_ASSEMBLY = "Mechanical Assembly"
        LIGHTING_AD = "Lighting AD"
        BILLBOARD_AD = "Billboard AD"
        PRODUCT_CLOSE_UP = "Product close-up"
        PARACHUTE_DELIVERY = "Parachute Delivery"
        DREAMLIKE_CLOUD = "Dreamlike Cloud"
        MACARON_MACHINE = "Macaron Machine"
        POSTER_AD = "Poster AD"
        TRUCK_AD = "Truck AD"
        GRAFFITI_AD = "Graffiti AD"
        FIGURINE_FACTORY_3D = "3D Figurine Factory"
        THE_EXCLUSIVE_FIRST_CLASS = "The Exclusive First Class"
        ART_ZOOM_CHALLENGE = "Art Zoom Challenge"
        I_QUIT = "I Quit"
        HITCHCOCK_DOLLY_ZOOM = "Hitchcock Dolly Zoom"
        SMELL_THE_LENS = "Smell the Lens"
        I_BELIEVE_I_CAN_FLY = "I believe I can fly"
        STRIKOUT_DANCE = "Strikout Dance"
        PIXEL_WORLD = "Pixel World"
        MINT_IN_BOX = "Mint in Box"
        HANDS_UP_HAND = "Hands up, Hand"
        FLORA_NYMPH_GO = "Flora Nymph Go"
        SOMBER_EMBRACE = "Somber Embrace"
        BEAM_ME_UP = "Beam me up"
        SUIT_SWAGGER = "Suit Swagger"


    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video."
    )
    effect: Effect = Field(
        default="", description="The effect to apply to the video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the image to use as the first frame. If not provided, generates from text"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "negative_prompt": self.negative_prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "effect": self.effect.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v3.5/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV35Transition(FALNode):
    """
    Create seamless transition between images using PixVerse v3.5
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"


    first_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    prompt: str = Field(
        default="", description="The prompt for the transition"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the last frame"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        first_image_base64 = await context.image_to_base64(self.first_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "first_image_url": f"data:image/png;base64,{first_image_base64}",
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "prompt": self.prompt,
            "duration": self.duration.value,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v3.5/transition",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LumaDreamMachineRay2FlashImageToVideo(FALNode):
    """
    Ray2 Flash is a fast video generative model capable of creating realistic visuals with natural, coherent motion.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"
        RATIO_21_9 = "21:9"
        RATIO_9_21 = "9:21"

    class Resolution(Enum):
        """
        The resolution of the generated video (720p costs 2x more, 1080p costs 4x more)
        """
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Duration(Enum):
        """
        The duration of the generated video
        """
        VALUE_5S = "5s"
        VALUE_9S = "9s"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_540P, description="The resolution of the generated video (720p costs 2x more, 1080p costs 4x more)"
    )
    loop: bool = Field(
        default=False, description="Whether the video should loop (end of video is blended with the beginning)"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5S, description="The duration of the generated video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Initial image to start the video from. Can be used together with end_image_url."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="Final image to end the video with. Can be used together with image_url."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "loop": self.loop,
            "duration": self.duration.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2-flash/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PikaV15Pikaffects(FALNode):
    """
    Pika Effects are AI-powered video effects designed to modify objects, characters, and environments in a fun, engaging, and visually compelling manner.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Pikaffect(Enum):
        """
        The Pikaffect to apply
        """
        CAKE_IFY = "Cake-ify"
        CRUMBLE = "Crumble"
        CRUSH = "Crush"
        DECAPITATE = "Decapitate"
        DEFLATE = "Deflate"
        DISSOLVE = "Dissolve"
        EXPLODE = "Explode"
        EYE_POP = "Eye-pop"
        INFLATE = "Inflate"
        LEVITATE = "Levitate"
        MELT = "Melt"
        PEEL = "Peel"
        POKE = "Poke"
        SQUISH = "Squish"
        TA_DA = "Ta-da"
        TEAR = "Tear"


    pikaffect: Pikaffect = Field(
        default="", description="The Pikaffect to apply"
    )
    prompt: str = Field(
        default="", description="Text prompt to guide the effect"
    )
    seed: str = Field(
        default="", description="The seed for the random number generator"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to guide the model"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "pikaffect": self.pikaffect.value,
            "prompt": self.prompt,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pika/v1.5/pikaffects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PikaV21ImageToVideo(FALNode):
    """
    Turn photos into mind-blowing, dynamic videos. Your images can can come to life with sharp details, impressive character control and cinematic camera moves.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    duration: int = Field(
        default=5, description="The duration of the generated video in seconds"
    )
    seed: str = Field(
        default="", description="The seed for the random number generator"
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the model"
    )
    image: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class PikaV2TurboImageToVideo(FALNode):
    """
    Turbo is the model to use when you feel the need for speed. Turn your image to stunning video up to 3x faster – all with high quality outputs. 
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    duration: int = Field(
        default=5, description="The duration of the generated video in seconds"
    )
    seed: str = Field(
        default="", description="The seed for the random number generator"
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the model"
    )
    image: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pika/v2/turbo/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduImageToVideo(FALNode):
    """
    Vidu Image to Video generates high-quality videos with exceptional visual quality and motion diversity from a single image
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class MovementAmplitude(Enum):
        """
        The movement amplitude of objects in the frame
        """
        AUTO = "auto"
        SMALL = "small"
        MEDIUM = "medium"
        LARGE = "large"


    prompt: str = Field(
        default="", description="Text prompt for video generation, max 1500 characters"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )
    movement_amplitude: MovementAmplitude = Field(
        default=MovementAmplitude.AUTO, description="The movement amplitude of objects in the frame"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "movement_amplitude": self.movement_amplitude.value,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduReferenceToVideo(FALNode):
    """
    Vidu Reference to Video creates videos by using a reference images and combining them with a prompt.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the output video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class MovementAmplitude(Enum):
        """
        The movement amplitude of objects in the frame
        """
        AUTO = "auto"
        SMALL = "small"
        MEDIUM = "medium"
        LARGE = "large"


    prompt: str = Field(
        default="", description="Text prompt for video generation, max 1500 characters"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the output video"
    )
    reference_images: list[str] = Field(
        default=[], description="URLs of the reference images to use for consistent subject appearance"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )
    movement_amplitude: MovementAmplitude = Field(
        default=MovementAmplitude.AUTO, description="The movement amplitude of objects in the frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "reference_image_urls": self.reference_images,
            "seed": self.seed,
            "movement_amplitude": self.movement_amplitude.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/reference-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduStartEndToVideo(FALNode):
    """
    Vidu Start-End to Video generates smooth transition videos between specified start and end images.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class MovementAmplitude(Enum):
        """
        The movement amplitude of objects in the frame
        """
        AUTO = "auto"
        SMALL = "small"
        MEDIUM = "medium"
        LARGE = "large"


    prompt: str = Field(
        default="", description="Text prompt for video generation, max 1500 characters"
    )
    start_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the last frame"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )
    movement_amplitude: MovementAmplitude = Field(
        default=MovementAmplitude.AUTO, description="The movement amplitude of objects in the frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_base64 = await context.image_to_base64(self.start_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "seed": self.seed,
            "movement_amplitude": self.movement_amplitude.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/start-end-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class ViduTemplateToVideo(FALNode):
    """
    Vidu Template to Video lets you create different effects by applying motion templates to your images.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the output video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Template(Enum):
        """
        AI video template to use. Pricing varies by template: Standard templates (hug, kiss, love_pose, etc.) cost 4 credits ($0.20), Premium templates (lunar_newyear, dynasty_dress, dreamy_wedding, etc.) cost 6 credits ($0.30), and Advanced templates (live_photo) cost 10 credits ($0.50).
        """
        DREAMY_WEDDING = "dreamy_wedding"
        ROMANTIC_LIFT = "romantic_lift"
        SWEET_PROPOSAL = "sweet_proposal"
        COUPLE_ARRIVAL = "couple_arrival"
        CUPID_ARROW = "cupid_arrow"
        PET_LOVERS = "pet_lovers"
        LUNAR_NEWYEAR = "lunar_newyear"
        HUG = "hug"
        KISS = "kiss"
        DYNASTY_DRESS = "dynasty_dress"
        WISH_SENDER = "wish_sender"
        LOVE_POSE = "love_pose"
        HAIR_SWAP = "hair_swap"
        YOUTH_REWIND = "youth_rewind"
        MORPHLAB = "morphlab"
        LIVE_PHOTO = "live_photo"
        EMOTIONLAB = "emotionlab"
        LIVE_MEMORY = "live_memory"
        INTERACTION = "interaction"
        CHRISTMAS = "christmas"
        PET_FINGER = "pet_finger"
        EAT_MUSHROOMS = "eat_mushrooms"
        BEAST_CHASE_LIBRARY = "beast_chase_library"
        BEAST_CHASE_SUPERMARKET = "beast_chase_supermarket"
        PETAL_SCATTERED = "petal_scattered"
        EMOJI_FIGURE = "emoji_figure"
        HAIR_COLOR_CHANGE = "hair_color_change"
        MULTIPLE_PEOPLE_KISSING = "multiple_people_kissing"
        BEAST_CHASE_AMAZON = "beast_chase_amazon"
        BEAST_CHASE_MOUNTAIN = "beast_chase_mountain"
        BALLOONMAN_EXPLODES_PRO = "balloonman_explodes_pro"
        GET_THINNER = "get_thinner"
        JUMP2POOL = "jump2pool"
        BODYSHAKE = "bodyshake"
        JIGGLE_UP = "jiggle_up"
        SHAKE_IT_DANCE = "shake_it_dance"
        SUBJECT_3 = "subject_3"
        PUBG_WINNER_HIT = "pubg_winner_hit"
        SHAKE_IT_DOWN = "shake_it_down"
        BLUEPRINT_SUPREME = "blueprint_supreme"
        HIP_TWIST = "hip_twist"
        MOTOR_DANCE = "motor_dance"
        RAT_DANCE = "rat_dance"
        KWOK_DANCE = "kwok_dance"
        LEG_SWEEP_DANCE = "leg_sweep_dance"
        HEESEUNG_MARCH = "heeseung_march"
        SHAKE_TO_MAX = "shake_to_max"
        DAME_UN_GRRR = "dame_un_grrr"
        I_KNOW = "i_know"
        LIT_BOUNCE = "lit_bounce"
        WAVE_DANCE = "wave_dance"
        CHILL_DANCE = "chill_dance"
        HIP_FLICKING = "hip_flicking"
        SAKURA_SEASON = "sakura_season"
        ZONGZI_WRAP = "zongzi_wrap"
        ZONGZI_DROP = "zongzi_drop"
        DRAGONBOAT_SHOT = "dragonboat_shot"
        RAIN_KISS = "rain_kiss"
        CHILD_MEMORY = "child_memory"
        COUPLE_DROP = "couple_drop"
        COUPLE_WALK = "couple_walk"
        FLOWER_RECEIVE = "flower_receive"
        LOVE_DROP = "love_drop"
        CHEEK_KISS = "cheek_kiss"
        CARRY_ME = "carry_me"
        BLOW_KISS = "blow_kiss"
        LOVE_FALL = "love_fall"
        FRENCH_KISS_8S = "french_kiss_8s"
        WORKDAY_FEELS = "workday_feels"
        LOVE_STORY = "love_story"
        BLOOM_MAGIC = "bloom_magic"
        GHIBLI = "ghibli"
        MINECRAFT = "minecraft"
        BOX_ME = "box_me"
        CLAW_ME = "claw_me"
        CLAYSHOT = "clayshot"
        MANGA_MEME = "manga_meme"
        QUAD_MEME = "quad_meme"
        PIXEL_ME = "pixel_me"
        CLAYSHOT_DUO = "clayshot_duo"
        IRASUTOYA = "irasutoya"
        AMERICAN_COMIC = "american_comic"
        SIMPSONS_COMIC = "simpsons_comic"
        YAYOI_KUSAMA_STYLE = "yayoi_kusama_style"
        POP_ART = "pop_art"
        JOJO_STYLE = "jojo_style"
        SLICE_THERAPY = "slice_therapy"
        BALLOON_FLYAWAY = "balloon_flyaway"
        FLYING = "flying"
        PAPERMAN = "paperman"
        PINCH = "pinch"
        BLOOM_DOOROBEAR = "bloom_doorobear"
        GENDER_SWAP = "gender_swap"
        NAP_ME = "nap_me"
        SEXY_ME = "sexy_me"
        SPIN360 = "spin360"
        SMOOTH_SHIFT = "smooth_shift"
        PAPER_FALL = "paper_fall"
        JUMP_TO_CLOUD = "jump_to_cloud"
        PILOT = "pilot"
        SWEET_DREAMS = "sweet_dreams"
        SOUL_DEPART = "soul_depart"
        PUNCH_HIT = "punch_hit"
        WATERMELON_HIT = "watermelon_hit"
        SPLIT_STANCE_PET = "split_stance_pet"
        MAKE_FACE = "make_face"
        BREAK_GLASS = "break_glass"
        SPLIT_STANCE_HUMAN = "split_stance_human"
        COVERED_LIQUID_METAL = "covered_liquid_metal"
        FLUFFY_PLUNGE = "fluffy_plunge"
        PET_BELLY_DANCE = "pet_belly_dance"
        WATER_FLOAT = "water_float"
        RELAX_CUT = "relax_cut"
        HEAD_TO_BALLOON = "head_to_balloon"
        CLONING = "cloning"
        ACROSS_THE_UNIVERSE_JUNGLE = "across_the_universe_jungle"
        CLOTHES_SPINNING_REMNANT = "clothes_spinning_remnant"
        ACROSS_THE_UNIVERSE_JURASSIC = "across_the_universe_jurassic"
        ACROSS_THE_UNIVERSE_MOON = "across_the_universe_moon"
        FISHEYE_PET = "fisheye_pet"
        HITCHCOCK_ZOOM = "hitchcock_zoom"
        CUTE_BANGS = "cute_bangs"
        EARTH_ZOOM_OUT = "earth_zoom_out"
        FISHEYE_HUMAN = "fisheye_human"
        DRIVE_YACHT = "drive_yacht"
        VIRTUAL_SINGER = "virtual_singer"
        EARTH_ZOOM_IN = "earth_zoom_in"
        ALIENS_COMING = "aliens_coming"
        DRIVE_FERRARI = "drive_ferrari"
        BJD_STYLE = "bjd_style"
        VIRTUAL_FITTING = "virtual_fitting"
        ORBIT = "orbit"
        ZOOM_IN = "zoom_in"
        AI_OUTFIT = "ai_outfit"
        SPIN180 = "spin180"
        ORBIT_DOLLY = "orbit_dolly"
        ORBIT_DOLLY_FAST = "orbit_dolly_fast"
        AUTO_SPIN = "auto_spin"
        WALK_FORWARD = "walk_forward"
        OUTFIT_SHOW = "outfit_show"
        ZOOM_IN_FAST = "zoom_in_fast"
        ZOOM_OUT_IMAGE = "zoom_out_image"
        ZOOM_OUT_STARTEND = "zoom_out_startend"
        MUSCLING = "muscling"
        CAPTAIN_AMERICA = "captain_america"
        HULK = "hulk"
        CAP_WALK = "cap_walk"
        HULK_DIVE = "hulk_dive"
        EXOTIC_PRINCESS = "exotic_princess"
        BEAST_COMPANION = "beast_companion"
        CARTOON_DOLL = "cartoon_doll"
        GOLDEN_EPOCH = "golden_epoch"
        OSCAR_GALA = "oscar_gala"
        FASHION_STRIDE = "fashion_stride"
        STAR_CARPET = "star_carpet"
        FLAME_CARPET = "flame_carpet"
        FROST_CARPET = "frost_carpet"
        MECHA_X = "mecha_x"
        STYLE_ME = "style_me"
        TAP_ME = "tap_me"
        SABER_WARRIOR = "saber_warrior"
        PET2HUMAN = "pet2human"
        GRADUATION = "graduation"
        FISHERMEN = "fishermen"
        HAPPY_BIRTHDAY = "happy_birthday"
        FAIRY_ME = "fairy_me"
        LADUDU_ME = "ladudu_me"
        LADUDU_ME_RANDOM = "ladudu_me_random"
        SQUID_GAME = "squid_game"
        SUPERMAN = "superman"
        GROW_WINGS = "grow_wings"
        CLEVAGE = "clevage"
        FLY_WITH_DORAEMON = "fly_with_doraemon"
        CREATICE_PRODUCT_DOWN = "creatice_product_down"
        POLE_DANCE = "pole_dance"
        HUG_FROM_BEHIND = "hug_from_behind"
        CREATICE_PRODUCT_UP_CYBERCITY = "creatice_product_up_cybercity"
        CREATICE_PRODUCT_UP_BLUECIRCUIT = "creatice_product_up_bluecircuit"
        CREATICE_PRODUCT_UP = "creatice_product_up"
        RUN_FAST = "run_fast"
        BACKGROUND_EXPLOSION = "background_explosion"


    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the output video"
    )
    template: Template = Field(
        default=Template.HUG, description="AI video template to use. Pricing varies by template: Standard templates (hug, kiss, love_pose, etc.) cost 4 credits ($0.20), Premium templates (lunar_newyear, dynasty_dress, dreamy_wedding, etc.) cost 6 credits ($0.30), and Advanced templates (live_photo) cost 10 credits ($0.50)."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )
    input_images: list[str] = Field(
        default=[], description="URLs of the images to use with the template. Number of images required varies by template: 'dynasty_dress' and 'shop_frame' accept 1-2 images, 'wish_sender' requires exactly 3 images, all other templates accept only 1 image."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "aspect_ratio": self.aspect_ratio.value,
            "template": self.template.value,
            "seed": self.seed,
            "input_image_urls": self.input_images,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/template-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class WanI2vLora(FALNode):
    """
    Add custom LoRAs to Wan-2.1 is a image-to-video model that generates high-quality videos with high visual quality and motion diversity from images
    video, animation, image-to-video, img2vid, lora

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        Aspect ratio of the output video.
        """
        AUTO = "auto"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class Resolution(Enum):
        """
        Resolution of the generated video (480p or 720p). 480p is 0.5 billing units, and 720p is 1 billing unit.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    shift: float = Field(
        default=5, description="Shift parameter for video generation."
    )
    reverse_video: bool = Field(
        default=False, description="If true, the video will be reversed."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="LoRA weights to be used in the inference."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 24."
    )
    turbo_mode: bool = Field(
        default=True, description="If true, the video will be generated faster with no noticeable degradation in the visual quality."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 100 (inclusive). If the number of frames is greater than 81, the video will be generated with 1.25x more billing units."
    )
    negative_prompt: str = Field(
        default="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the output video."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p or 720p). 480p is 0.5 billing units, and 720p is 1 billing unit."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    guide_scale: float = Field(
        default=5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "shift": self.shift,
            "reverse_video": self.reverse_video,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "frames_per_second": self.frames_per_second,
            "turbo_mode": self.turbo_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "guide_scale": self.guide_scale,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-i2v-lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class HunyuanVideoImageToVideo(FALNode):
    """
    Image to Video for the high-quality Hunyuan Video I2V model.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

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
        VALUE_720P = "720p"

    class NumFrames(Enum):
        """
        The number of frames to generate.
        """
        VALUE_129 = "129"


    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video to generate."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the video to generate."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image input."
    )
    seed: int = Field(
        default=-1, description="The seed to use for generating the video."
    )
    num_frames: NumFrames = Field(
        default=129, description="The number of frames to generate."
    )
    i2v_stability: bool = Field(
        default=False, description="Turning on I2V Stability reduces hallucination but also reduces motion."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "seed": self.seed,
            "num_frames": self.num_frames.value,
            "i2v_stability": self.i2v_stability,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class MinimaxVideo01DirectorImageToVideo(FALNode):
    """
    Generate video clips more accurately with respect to initial image, natural language descriptions, and using camera movement instructions for shot control.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    prompt: str = Field(
        default="", description="Text prompt for video generation. Camera movement instructions can be added using square brackets (e.g. [Pan left] or [Zoom in]). You can use up to 3 combined movements per prompt. Supported movements: Truck left/right, Pan left/right, Push in/Pull out, Pedestal up/down, Tilt up/down, Zoom in/out, Shake, Tracking shot, Static shot. For example: [Truck left, Pan right, Zoom in]. For a more detailed guide, refer https://sixth-switch-2ac.notion.site/T2V-01-Director-Model-Tutorial-with-camera-movement-1886c20a98eb80f395b8e05291ad8645"
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/video-01-director/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class SkyreelsI2v(FALNode):
    """
    SkyReels V1 is the first and most advanced open-source human-centric video foundation model. By fine-tuning HunyuanVideo on O(10M) high-quality film and television clips
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        Aspect ratio of the output video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"


    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the output video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image input."
    )
    guidance_scale: float = Field(
        default=6, description="Guidance scale for generation (between 1.0 and 20.0)"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation. If not provided, a random seed will be used."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of denoising steps (between 1 and 50). Higher values give better quality but take longer."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to guide generation away from certain attributes."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/skyreels-i2v",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LumaDreamMachineRay2ImageToVideo(FALNode):
    """
    Ray2 is a large-scale video generative model capable of creating realistic visuals with natural, coherent motion.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"
        RATIO_21_9 = "21:9"
        RATIO_9_21 = "9:21"

    class Resolution(Enum):
        """
        The resolution of the generated video (720p costs 2x more, 1080p costs 4x more)
        """
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Duration(Enum):
        """
        The duration of the generated video
        """
        VALUE_5S = "5s"
        VALUE_9S = "9s"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_540P, description="The resolution of the generated video (720p costs 2x more, 1080p costs 4x more)"
    )
    loop: bool = Field(
        default=False, description="Whether the video should loop (end of video is blended with the beginning)"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5S, description="The duration of the generated video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Initial image to start the video from. Can be used together with end_image_url."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="Final image to end the video with. Can be used together with image_url."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "loop": self.loop,
            "duration": self.duration.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class HunyuanVideoImg2vidLora(FALNode):
    """
    Image to Video for the Hunyuan Video model using a custom trained LoRA.
    video, animation, image-to-video, img2vid, lora

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    seed: int = Field(
        default=-1, description="The seed to use for generating the video."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL to the image to generate the video from. The image must be 960x544 or it will get cropped and resized to that size."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video-img2vid-lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV35ImageToVideoFast(FALNode):
    """
    Generate high quality video clips from text and image prompts quickly using PixVerse v3.5 Fast
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"


    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v3.5/image-to-video/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class PixverseV35ImageToVideo(FALNode):
    """
    Generate high quality video clips from text and image prompts using PixVerse v3.5
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_360P = "360p"
        VALUE_540P = "540p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Duration(Enum):
        """
        The duration of the generated video in seconds. 8s videos cost double. 1080p videos are limited to 5 seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"

    class Style(Enum):
        """
        The style of the generated video
        """
        ANIME = "anime"
        ANIMATION_3D = "3d_animation"
        CLAY = "clay"
        COMIC = "comic"
        CYBERPUNK = "cyberpunk"


    prompt: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds. 8s videos cost double. 1080p videos are limited to 5 seconds"
    )
    style: Style | None = Field(
        default=None, description="The style of the generated video"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "style": self.style.value if self.style else None,
            "image_url": f"data:image/png;base64,{image_base64}",
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v3.5/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class MinimaxVideo01SubjectReference(FALNode):
    """
    Generate video clips maintaining consistent, realistic facial features and identity across dynamic video content
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    prompt: str = Field(
        default=""
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    subject_reference_image: ImageRef = Field(
        default=ImageRef(), description="URL of the subject reference image to use for consistent subject appearance"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        subject_reference_image_base64 = await context.image_to_base64(self.subject_reference_image)
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
            "subject_reference_image_url": f"data:image/png;base64,{subject_reference_image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/video-01-subject-reference",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoV16StandardImageToVideo(FALNode):
    """
    Generate video clips from your images using Kling 1.6 (std)
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"


    prompt: str = Field(
        default=""
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    image: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "cfg_scale": self.cfg_scale,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.6/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class SadtalkerReference(FALNode):
    """
    Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class FaceEnhancer(Enum):
        """
        The type of face enhancer to use
        """
        GFPGAN = "gfpgan"

    class FaceModelResolution(Enum):
        """
        The resolution of the face model
        """
        VALUE_256 = "256"
        VALUE_512 = "512"

    class Preprocess(Enum):
        """
        The type of preprocessing to use
        """
        CROP = "crop"
        EXTCROP = "extcrop"
        RESIZE = "resize"
        FULL = "full"
        EXTFULL = "extfull"


    pose_style: int = Field(
        default=0, description="The style of the pose"
    )
    source_image: ImageRef = Field(
        default=ImageRef(), description="URL of the source image"
    )
    reference_pose_video: VideoRef = Field(
        default=VideoRef(), description="URL of the reference video"
    )
    driven_audio: AudioRef = Field(
        default=AudioRef(), description="URL of the driven audio"
    )
    face_enhancer: FaceEnhancer | None = Field(
        default=None, description="The type of face enhancer to use"
    )
    expression_scale: float = Field(
        default=1, description="The scale of the expression"
    )
    face_model_resolution: FaceModelResolution = Field(
        default=FaceModelResolution.VALUE_256, description="The resolution of the face model"
    )
    still_mode: bool = Field(
        default=False, description="Whether to use still mode. Fewer head motion, works with preprocess `full`."
    )
    preprocess: Preprocess = Field(
        default=Preprocess.CROP, description="The type of preprocessing to use"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        source_image_base64 = await context.image_to_base64(self.source_image)
        arguments = {
            "pose_style": self.pose_style,
            "source_image_url": f"data:image/png;base64,{source_image_base64}",
            "reference_pose_video_url": self.reference_pose_video,
            "driven_audio_url": self.driven_audio,
            "face_enhancer": self.face_enhancer.value if self.face_enhancer else None,
            "expression_scale": self.expression_scale,
            "face_model_resolution": self.face_model_resolution.value,
            "still_mode": self.still_mode,
            "preprocess": self.preprocess.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sadtalker/reference",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class MinimaxVideo01LiveImageToVideo(FALNode):
    """
    Generate video clips from your images using MiniMax Video model
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    prompt: str = Field(
        default=""
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class KlingVideoV15ProImageToVideo(FALNode):
    """
    Generate video clips from your images using Kling 1.5 (pro)
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video frame
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"


    prompt: str = Field(
        default=""
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    image: ImageRef = Field(
        default=ImageRef()
    )
    static_mask: ImageRef = Field(
        default=ImageRef(), description="URL of the image for Static Brush Application Area (Mask image created by users using the motion brush)"
    )
    dynamic_masks: list[DynamicMask] = Field(
        default=[], description="List of dynamic masks"
    )
    tail_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the end of the video"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        static_mask_base64 = await context.image_to_base64(self.static_mask)
        tail_image_base64 = await context.image_to_base64(self.tail_image)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "static_mask_url": f"data:image/png;base64,{static_mask_base64}",
            "dynamic_masks": [item.model_dump(exclude={"type"}) for item in self.dynamic_masks],
            "tail_image_url": f"data:image/png;base64,{tail_image_base64}",
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.5/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class LivePortrait(FALNode):
    """
    Transfer expression from a video to a portrait.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    smile: float = Field(
        default=0, description="Amount to smile"
    )
    video: VideoRef = Field(
        default=VideoRef(), description="URL of the video to drive the lip syncing."
    )
    flag_stitching: bool = Field(
        default=True, description="Whether to enable stitching. Recommended to set to True."
    )
    eyebrow: float = Field(
        default=0, description="Amount to raise or lower eyebrows"
    )
    wink: float = Field(
        default=0, description="Amount to wink"
    )
    rotate_pitch: float = Field(
        default=0, description="Amount to rotate the face in pitch"
    )
    blink: float = Field(
        default=0, description="Amount to blink the eyes"
    )
    scale: float = Field(
        default=2.3, description="Scaling factor for the face crop."
    )
    eee: float = Field(
        default=0, description="Amount to shape mouth in 'eee' position"
    )
    flag_pasteback: bool = Field(
        default=True, description="Whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space."
    )
    pupil_y: float = Field(
        default=0, description="Amount to move pupils vertically"
    )
    rotate_yaw: float = Field(
        default=0, description="Amount to rotate the face in yaw"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be animated"
    )
    woo: float = Field(
        default=0, description="Amount to shape mouth in 'woo' position"
    )
    aaa: float = Field(
        default=0, description="Amount to open mouth in 'aaa' shape"
    )
    flag_do_rot: bool = Field(
        default=True, description="Whether to conduct the rotation when flag_do_crop is True."
    )
    flag_relative: bool = Field(
        default=True, description="Whether to use relative motion."
    )
    flag_eye_retargeting: bool = Field(
        default=False, description="Whether to enable eye retargeting."
    )
    flag_lip_zero: bool = Field(
        default=True, description="Whether to set the lip to closed state before animation. Only takes effect when flag_eye_retargeting and flag_lip_retargeting are False."
    )
    batch_size: int = Field(
        default=32, description="Batch size for the model. The larger the batch size, the faster the model will run, but the more memory it will consume."
    )
    rotate_roll: float = Field(
        default=0, description="Amount to rotate the face in roll"
    )
    dsize: int = Field(
        default=512, description="Size of the output image."
    )
    vy_ratio: float = Field(
        default=-0.125, description="Vertical offset ratio for face crop. Positive values move up, negative values move down."
    )
    pupil_x: float = Field(
        default=0, description="Amount to move pupils horizontally"
    )
    enable_safety_checker: bool = Field(
        default=False, description="Whether to enable the safety checker. If enabled, the model will check if the input image contains a face before processing it. The safety checker will process the input image"
    )
    vx_ratio: float = Field(
        default=0, description="Horizontal offset ratio for face crop."
    )
    flag_lip_retargeting: bool = Field(
        default=False, description="Whether to enable lip retargeting."
    )
    flag_do_crop: bool = Field(
        default=True, description="Whether to crop the source portrait to the face-cropping space."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "smile": self.smile,
            "video_url": self.video,
            "flag_stitching": self.flag_stitching,
            "eyebrow": self.eyebrow,
            "wink": self.wink,
            "rotate_pitch": self.rotate_pitch,
            "blink": self.blink,
            "scale": self.scale,
            "eee": self.eee,
            "flag_pasteback": self.flag_pasteback,
            "pupil_y": self.pupil_y,
            "rotate_yaw": self.rotate_yaw,
            "image_url": f"data:image/png;base64,{image_base64}",
            "woo": self.woo,
            "aaa": self.aaa,
            "flag_do_rot": self.flag_do_rot,
            "flag_relative": self.flag_relative,
            "flag_eye_retargeting": self.flag_eye_retargeting,
            "flag_lip_zero": self.flag_lip_zero,
            "batch_size": self.batch_size,
            "rotate_roll": self.rotate_roll,
            "dsize": self.dsize,
            "vy_ratio": self.vy_ratio,
            "pupil_x": self.pupil_x,
            "enable_safety_checker": self.enable_safety_checker,
            "vx_ratio": self.vx_ratio,
            "flag_lip_retargeting": self.flag_lip_retargeting,
            "flag_do_crop": self.flag_do_crop,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/live-portrait",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Musetalk(FALNode):
    """
    MuseTalk is a real-time high quality audio-driven lip-syncing model. Use MuseTalk to animate a face with your own audio.
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    source_video: VideoRef = Field(
        default=VideoRef(), description="URL of the source video"
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="URL of the audio"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "source_video_url": self.source_video,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/musetalk",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class Sadtalker(FALNode):
    """
    Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation
    video, animation, image-to-video, img2vid

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    class FaceEnhancer(Enum):
        """
        The type of face enhancer to use
        """
        GFPGAN = "gfpgan"

    class FaceModelResolution(Enum):
        """
        The resolution of the face model
        """
        VALUE_256 = "256"
        VALUE_512 = "512"

    class Preprocess(Enum):
        """
        The type of preprocessing to use
        """
        CROP = "crop"
        EXTCROP = "extcrop"
        RESIZE = "resize"
        FULL = "full"
        EXTFULL = "extfull"


    pose_style: int = Field(
        default=0, description="The style of the pose"
    )
    source_image: ImageRef = Field(
        default=ImageRef(), description="URL of the source image"
    )
    driven_audio: AudioRef = Field(
        default=AudioRef(), description="URL of the driven audio"
    )
    face_enhancer: FaceEnhancer | None = Field(
        default=None, description="The type of face enhancer to use"
    )
    expression_scale: float = Field(
        default=1, description="The scale of the expression"
    )
    face_model_resolution: FaceModelResolution = Field(
        default=FaceModelResolution.VALUE_256, description="The resolution of the face model"
    )
    still_mode: bool = Field(
        default=False, description="Whether to use still mode. Fewer head motion, works with preprocess `full`."
    )
    preprocess: Preprocess = Field(
        default=Preprocess.CROP, description="The type of preprocessing to use"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        source_image_base64 = await context.image_to_base64(self.source_image)
        arguments = {
            "pose_style": self.pose_style,
            "source_image_url": f"data:image/png;base64,{source_image_base64}",
            "driven_audio_url": self.driven_audio,
            "face_enhancer": self.face_enhancer.value if self.face_enhancer else None,
            "expression_scale": self.expression_scale,
            "face_model_resolution": self.face_model_resolution.value,
            "still_mode": self.still_mode,
            "preprocess": self.preprocess.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sadtalker",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class FastSvdLcm(FALNode):
    """
    Generate short video clips from your images using SVD v1.1 at Lightning Speed
    video, animation, image-to-video, img2vid, fast

    Use cases:
    - Animate static images
    - Create engaging social media content
    - Product demonstrations
    - Marketing and promotional videos
    - Visual storytelling
    """

    motion_bucket_id: int = Field(
        default=127, description="The motion bucket id determines the motion of the generated video. The higher the number, the more motion there will be."
    )
    fps: int = Field(
        default=10, description="The FPS of the generated video. The higher the number, the faster the video will play. Total video length is 25 frames."
    )
    steps: int = Field(
        default=4, description="The number of steps to run the model for. The higher the number the better the quality and longer it will take to generate."
    )
    cond_aug: float = Field(
        default=0.02, description="The conditoning augmentation determines the amount of noise that will be added to the conditioning frame. The higher the number, the more noise there will be, and the less the video will look like the initial image. Increase it for more motion."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a starting point for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "motion_bucket_id": self.motion_bucket_id,
            "fps": self.fps,
            "steps": self.steps,
            "cond_aug": self.cond_aug,
            "seed": self.seed,
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-svd-lcm",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image", "prompt"]

class KlingVideoV3StandardImageToVideo(FALNode):
    """
    Kling Video V3 Standard generates videos from images with balanced quality and speed using the latest V3 model.
    video, generation, kling, v3, standard, image-to-video

    Use cases:
    - Animate static images into short video clips
    - Create engaging social media content from photos
    - Generate product demonstration videos
    - Produce marketing and promotional videos
    - Transform images into cinematic animations
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
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

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video frame
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class ShotType(Enum):
        """
        The type of multi-shot video generation. Required when multi_prompt is provided.
        """
        CUSTOMIZE = "customize"


    prompt: str = Field(
        default="", description="Text prompt for video generation. Either prompt or multi_prompt must be provided, but not both."
    )
    voice_ids: list[str] = Field(
        default=[], description="Optional Voice IDs for video generation. Reference voices in your prompt with <<<voice_1>>> and <<<voice_2>>> (maximum 2 voices per task). Get voice IDs from the kling video create-voice endpoint: https://fal.ai/models/fal-ai/kling-video/create-voice"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate native audio for the video. Supports Chinese and English voice output. Other languages are automatically translated to English. For English speech, use lowercase letters; for acronyms or proper nouns, use uppercase."
    )
    multi_prompt: list[KlingV3MultiPromptElement] = Field(
        default=[], description="List of prompts for multi-shot video generation. If provided, divides the video into multiple shots."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    start_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the video"
    )
    shot_type: ShotType = Field(
        default=ShotType.CUSTOMIZE, description="The type of multi-shot video generation. Required when multi_prompt is provided."
    )
    elements: list[KlingV3ComboElementInput] = Field(
        default=[], description="Elements (characters/objects) to include in the video. Each example can either be an image set (frontal + reference images) or a video. Reference in prompt as @Element1, @Element2, etc."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the end of the video"
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_base64 = await context.image_to_base64(self.start_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "voice_ids": self.voice_ids,
            "duration": self.duration.value,
            "generate_audio": self.generate_audio,
            "multi_prompt": [item.model_dump(exclude={"type"}) for item in self.multi_prompt],
            "aspect_ratio": self.aspect_ratio.value,
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "shot_type": self.shot_type.value,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v3/standard/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["start_image_url", "prompt", "duration"]

class KlingVideoV3ProImageToVideo(FALNode):
    """
    Kling Video V3 Pro generates professional quality videos from images with enhanced visual fidelity using the latest V3 model.
    video, generation, kling, v3, pro, image-to-video

    Use cases:
    - Create professional-grade video animations from images
    - Generate cinematic video content with precise motion
    - Produce high-fidelity product showcase videos
    - Animate images with enhanced visual quality
    - Create premium video content for advertising
    """

    class Duration(Enum):
        """
        The duration of the generated video in seconds
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

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video frame
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class ShotType(Enum):
        """
        The type of multi-shot video generation. Required when multi_prompt is provided.
        """
        CUSTOMIZE = "customize"


    prompt: str = Field(
        default="", description="Text prompt for video generation. Either prompt or multi_prompt must be provided, but not both."
    )
    voice_ids: list[str] = Field(
        default=[], description="Optional Voice IDs for video generation. Reference voices in your prompt with <<<voice_1>>> and <<<voice_2>>> (maximum 2 voices per task). Get voice IDs from the kling video create-voice endpoint: https://fal.ai/models/fal-ai/kling-video/create-voice"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate native audio for the video. Supports Chinese and English voice output. Other languages are automatically translated to English. For English speech, use lowercase letters; for acronyms or proper nouns, use uppercase."
    )
    multi_prompt: list[KlingV3MultiPromptElement] = Field(
        default=[], description="List of prompts for multi-shot video generation. If provided, divides the video into multiple shots."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    start_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the video"
    )
    shot_type: ShotType = Field(
        default=ShotType.CUSTOMIZE, description="The type of multi-shot video generation. Required when multi_prompt is provided."
    )
    elements: list[KlingV3ComboElementInput] = Field(
        default=[], description="Elements (characters/objects) to include in the video. Each example can either be an image set (frontal + reference images) or a video. Reference in prompt as @Element1, @Element2, etc."
    )
    end_image: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the end of the video"
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_base64 = await context.image_to_base64(self.start_image)
        end_image_base64 = await context.image_to_base64(self.end_image)
        arguments = {
            "prompt": self.prompt,
            "voice_ids": self.voice_ids,
            "duration": self.duration.value,
            "generate_audio": self.generate_audio,
            "multi_prompt": [item.model_dump(exclude={"type"}) for item in self.multi_prompt],
            "aspect_ratio": self.aspect_ratio.value,
            "start_image_url": f"data:image/png;base64,{start_image_base64}",
            "shot_type": self.shot_type.value,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
            "end_image_url": f"data:image/png;base64,{end_image_base64}",
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v3/pro/image-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["start_image_url", "prompt", "duration"]