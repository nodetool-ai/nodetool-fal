from enum import Enum
from pydantic import Field
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class HunyuanVideo(FALNode):
    """
    Hunyuan Video is Tencent's advanced text-to-video model for high-quality video generation.
    video, generation, hunyuan, text-to-video, txt2vid

    Use cases:
    - Generate cinematic videos from text descriptions
    - Create marketing videos from product descriptions
    - Produce educational video content
    - Generate creative video concepts
    - Create animated scenes from stories
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
        VALUE_480P = "480p"
        VALUE_580P = "580p"
        VALUE_720P = "720p"

    class NumFrames(Enum):
        """
        The number of frames to generate.
        """
        VALUE_129 = "129"
        VALUE_85 = "85"


    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video to generate."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the video to generate."
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
            application="fal-ai/hunyuan-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class CogVideoX5B(FALNode):
    """
    CogVideoX-5B is a powerful open-source text-to-video generation model with 5 billion parameters.
    video, generation, cogvideo, text-to-video, txt2vid

    Use cases:
    - Generate detailed videos from text prompts
    - Create animated storytelling content
    - Produce concept videos for pitches
    - Generate video storyboards
    - Create educational demonstrations
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
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
            "use_rife": self.use_rife,
            "loras": self.loras,
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
            application="fal-ai/cogvideox-5b",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class AnimateDiffTextToVideo(FALNode):
    """
    AnimateDiff generates smooth animations from text prompts using diffusion models.
    video, generation, animatediff, animation, text-to-video, txt2vid

    Use cases:
    - Animate ideas from text descriptions
    - Create animated content quickly
    - Generate motion graphics from prompts
    - Produce animated concept art
    - Create video loops and sequences
    """

    prompt: str = Field(
        default="", description="The prompt to use for generating the video. Be as descriptive as possible for best results."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    fps: int = Field(
        default=8, description="Number of frames per second to extract from the video."
    )
    video_size: str = Field(
        default="square", description="The size of the video to generate."
    )
    guidance_scale: float = Field(
        default=7.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_frames: int = Field(
        default=16, description="The number of frames to generate for the video."
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
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "fps": self.fps,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "motions": self.motions,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-animatediff/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class AnimateDiffTurboTextToVideo(FALNode):
    """
    AnimateDiff Turbo generates animations at lightning speed with reduced steps.
    video, generation, animatediff, turbo, fast, text-to-video, txt2vid

    Use cases:
    - Rapidly prototype video animations
    - Create quick video previews
    - Generate animations with minimal latency
    - Iterate on video concepts quickly
    - Produce real-time animation effects
    """

    prompt: str = Field(
        default="", description="The prompt to use for generating the video. Be as descriptive as possible for best results."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    fps: int = Field(
        default=8, description="Number of frames per second to extract from the video."
    )
    video_size: str = Field(
        default="square", description="The size of the video to generate."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_frames: int = Field(
        default=16, description="The number of frames to generate for the video."
    )
    num_inference_steps: int = Field(
        default=4, description="The number of inference steps to perform. 4-12 is recommended for turbo mode."
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
            "seed": self.seed,
            "fps": self.fps,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "motions": self.motions,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-animatediff/turbo/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class AnimateDiffSparseCtrlLCM(FALNode):
    """
    AnimateDiff SparseCtrl LCM animates drawings with latent consistency models for fast generation.
    video, generation, animatediff, sparsectrl, lcm, animation, text-to-video

    Use cases:
    - Animate hand-drawn sketches
    - Bring drawings to life
    - Create animated illustrations
    - Generate animations from concept art
    - Produce animation from sparse frames
    """

    class ControlnetType(Enum):
        """
        The type of controlnet to use for generating the video. The controlnet determines how the video will be animated.
        """
        SCRIBBLE = "scribble"
        RGB = "rgb"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    controlnet_type: ControlnetType = Field(
        default=ControlnetType.SCRIBBLE, description="The type of controlnet to use for generating the video. The controlnet determines how the video will be animated."
    )
    keyframe_2_index: int = Field(
        default=0, description="The frame index of the third keyframe to use for the generation."
    )
    keyframe_0_index: int = Field(
        default=0, description="The frame index of the first keyframe to use for the generation."
    )
    keyframe_1_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the second keyframe to use for the generation."
    )
    keyframe_1_index: int = Field(
        default=0, description="The frame index of the second keyframe to use for the generation."
    )
    guidance_scale: int = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=4, description="Increasing the amount of steps tells Stable Diffusion that it should take more steps to generate your final result which can increase the amount of detail in your image."
    )
    keyframe_2_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the third keyframe to use for the generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to specify what you don't want."
    )
    keyframe_0_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the first keyframe to use for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        keyframe_1_image_url_base64 = await context.image_to_base64(self.keyframe_1_image_url)
        keyframe_2_image_url_base64 = await context.image_to_base64(self.keyframe_2_image_url)
        keyframe_0_image_url_base64 = await context.image_to_base64(self.keyframe_0_image_url)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "controlnet_type": self.controlnet_type.value,
            "keyframe_2_index": self.keyframe_2_index,
            "keyframe_0_index": self.keyframe_0_index,
            "keyframe_1_image_url": f"data:image/png;base64,{keyframe_1_image_url_base64}",
            "keyframe_1_index": self.keyframe_1_index,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "keyframe_2_image_url": f"data:image/png;base64,{keyframe_2_image_url_base64}",
            "negative_prompt": self.negative_prompt,
            "keyframe_0_image_url": f"data:image/png;base64,{keyframe_0_image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/animatediff-sparsectrl-lcm",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class VeedAvatarsTextToVideo(FALNode):
    """
    VEED Avatars generates talking avatar videos from text using realistic AI-powered characters.
    video, generation, avatar, talking-head, veed, text-to-video

    Use cases:
    - Create talking avatar presentations
    - Generate spokesperson videos
    - Produce educational talking head videos
    - Create personalized video messages
    - Generate multilingual avatar content
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


    text: str = Field(
        default=""
    )
    avatar_id: AvatarId = Field(
        default="", description="The avatar to use for the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "text": self.text,
            "avatar_id": self.avatar_id.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/avatars/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ArgilAvatarsTextToVideo(FALNode):
    """
    Argil Avatars creates realistic talking avatar videos from text descriptions.
    video, generation, avatar, talking-head, argil, text-to-video

    Use cases:
    - Generate avatar spokesperson videos
    - Create virtual presenter content
    - Produce automated video announcements
    - Generate character-based narratives
    - Create social media avatar videos
    """

    class Voice(Enum):
        RACHEL = "Rachel"
        CLYDE = "Clyde"
        ROGER = "Roger"
        SARAH = "Sarah"
        LAURA = "Laura"
        THOMAS = "Thomas"
        CHARLIE = "Charlie"
        GEORGE = "George"
        CALLUM = "Callum"
        RIVER = "River"
        HARRY = "Harry"
        LIAM = "Liam"
        ALICE = "Alice"
        MATILDA = "Matilda"
        WILL = "Will"
        JESSICA = "Jessica"
        LILLY = "Lilly"
        BILL = "Bill"
        OXLEY = "Oxley"
        LUNA = "Luna"

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


    text: str = Field(
        default=""
    )
    voice: Voice = Field(
        default=""
    )
    remove_background: bool = Field(
        default=False, description="Enabling the remove background feature will result in a 50% increase in the price."
    )
    avatar: Avatar = Field(
        default=""
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "text": self.text,
            "voice": self.voice.value,
            "remove_background": self.remove_background,
            "avatar": self.avatar.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="argil/avatars/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class SeeDanceV15ProTextToVideo(FALNode):
    """
    SeeDance v1.5 Pro from ByteDance generates high-quality dance videos from text prompts.
    video, generation, dance, seedance, bytedance, text-to-video

    Use cases:
    - Generate dance choreography videos
    - Create dance performance visualizations
    - Produce music video concepts
    - Generate dance training content
    - Create dance animation prototypes
    """

    class Resolution(Enum):
        """
        Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class Duration(Enum):
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


    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the video in seconds"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
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
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "generate_audio": self.generate_audio,
            "aspect_ratio": self.aspect_ratio.value,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedance/v1.5/pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class SeeDanceV1ProFastTextToVideo(FALNode):
    """
    SeeDance v1 Pro Fast generates dance videos quickly from text with reduced generation time.
    video, generation, dance, seedance, fast, bytedance, text-to-video

    Use cases:
    - Rapidly prototype dance videos
    - Create quick dance previews
    - Generate dance concepts efficiently
    - Iterate on choreography ideas
    - Produce dance storyboards
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
        Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the video in seconds"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1080P, description="Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality"
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
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedance/v1/pro/fast/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class VeedFabric10Text(FALNode):
    """
    VEED Fabric 1.0 generates video content from text using advanced video synthesis.
    video, generation, fabric, veed, text-to-video, txt2vid

    Use cases:
    - Generate marketing videos from text
    - Create explainer video content
    - Produce video ads from copy
    - Generate social media videos
    - Create branded video content
    """

    class Resolution(Enum):
        """
        Resolution
        """
        VALUE_720P = "720p"
        VALUE_480P = "480p"


    text: str = Field(
        default=""
    )
    resolution: Resolution = Field(
        default="", description="Resolution"
    )
    voice_description: str = Field(
        default="", description="Optional additional voice description. The primary voice description is auto-generated from the image. You can use simple descriptors like 'British accent' or 'Confident' or provide a detailed description like 'Confident male voice, mid-20s, with notes of...'"
    )
    image_url: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "text": self.text,
            "resolution": self.resolution.value,
            "voice_description": self.voice_description,
            "image_url": f"data:image/png;base64,{image_url_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="veed/fabric-1.0/text",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LTXVideo(FALNode):
    """
    LTX Video generates high-quality videos from text prompts with advanced temporal consistency.
    video, generation, ltx, text-to-video, txt2vid

    Use cases:
    - Generate temporally consistent videos
    - Create smooth video sequences
    - Produce high-quality video content
    - Generate professional video clips
    - Create cinematic video scenes
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV1StandardTextToVideo(FALNode):
    """
    Kling Video v1 Standard generates videos from text with balanced quality and speed.
    video, generation, kling, text-to-video, txt2vid

    Use cases:
    - Generate standard quality videos
    - Create video content efficiently
    - Produce videos for web use
    - Generate video previews
    - Create video concepts
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video frame
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

    class CameraControl(Enum):
        """
        Camera control parameters
        """
        DOWN_BACK = "down_back"
        FORWARD_UP = "forward_up"
        RIGHT_TURN_FORWARD = "right_turn_forward"
        LEFT_TURN_FORWARD = "left_turn_forward"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    advanced_camera_control: str = Field(
        default="", description="Advanced Camera control parameters"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    camera_control: CameraControl | None = Field(
        default=None, description="Camera control parameters"
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "advanced_camera_control": self.advanced_camera_control,
            "duration": self.duration.value,
            "camera_control": self.camera_control.value if self.camera_control else None,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1/standard/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class MochiV1(FALNode):
    """
    Mochi v1 generates creative videos from text with unique artistic style.
    video, generation, mochi, artistic, text-to-video, txt2vid

    Use cases:
    - Generate artistic video content
    - Create stylized animations
    - Produce creative video art
    - Generate experimental videos
    - Create unique visual content
    """

    prompt: str = Field(
        default="", description="The prompt to generate a video from."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="The seed to use for generating the video."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt for the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/mochi-v1",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class StableVideo(FALNode):
    """
    Stable Video generates consistent and stable video sequences from text prompts.
    video, generation, stable, text-to-video, txt2vid

    Use cases:
    - Generate stable video sequences
    - Create consistent video content
    - Produce reliable video outputs
    - Generate predictable video scenes
    - Create controlled video generation
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a starting point for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "motion_bucket_id": self.motion_bucket_id,
            "fps": self.fps,
            "cond_aug": self.cond_aug,
            "seed": self.seed,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
        return ["prompt"]

class T2VTurbo(FALNode):
    """
    T2V Turbo generates videos from text at high speed with optimized performance.
    video, generation, turbo, fast, text-to-video, txt2vid

    Use cases:
    - Generate videos with minimal latency
    - Create rapid video prototypes
    - Produce quick video previews
    - Generate real-time video content
    - Create efficient video workflows
    """

    prompt: str = Field(
        default="", description="The prompt to generate images from"
    )
    guidance_scale: float = Field(
        default=7.5, description="The guidance scale"
    )
    seed: str = Field(
        default="", description="The seed to use for the random number generator"
    )
    export_fps: int = Field(
        default=8, description="The FPS of the exported video"
    )
    num_frames: int = Field(
        default=16, description="The number of frames to generate"
    )
    num_inference_steps: int = Field(
        default=4, description="The number of steps to sample"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "export_fps": self.export_fps,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/t2v-turbo",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LumaDreamMachineTextToVideo(FALNode):
    """
    Luma Dream Machine generates creative videos from text with dreamlike aesthetics.
    video, generation, luma, dream-machine, text-to-video, txt2vid

    Use cases:
    - Generate dreamlike video content
    - Create surreal video sequences
    - Produce artistic video interpretations
    - Generate creative video concepts
    - Create imaginative video art
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "loop": self.loop,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LumaPhoton(FALNode):
    """
    Luma Photon generates photorealistic videos from text with high visual fidelity.
    video, generation, luma, photon, photorealistic, text-to-video

    Use cases:
    - Generate photorealistic video content
    - Create realistic video simulations
    - Produce lifelike video scenes
    - Generate high-fidelity video outputs
    - Create realistic visual content
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"
        RATIO_21_9 = "21:9"
        RATIO_9_21 = "9:21"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated video"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-photon",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PixverseV5_6TextToVideo(FALNode):
    """
    Pixverse
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
        The duration of the generated video in seconds. 1080p videos are limited to 5 or 8 seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"
        VALUE_10 = "10"


    prompt: str = Field(
        default=""
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
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds. 1080p videos are limited to 5 or 8 seconds"
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
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "duration": self.duration.value,
            "generate_audio_switch": self.generate_audio_switch,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5.6/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Ltx219BDistilledTextToVideoLora(FALNode):
    """
    LTX-2 19B Distilled
    video, generation, text-to-video, txt2vid, lora

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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


    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
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
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the generation."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_size: str = Field(
        default="landscape_4_3", description="The size of the generated video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "use_multiscale": self.use_multiscale,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "fps": self.fps,
            "loras": self.loras,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "num_frames": self.num_frames,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/text-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Ltx219BDistilledTextToVideo(FALNode):
    """
    LTX-2 19B Distilled
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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


    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
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
        default="landscape_4_3", description="The size of the generated video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    num_frames: int = Field(
        default=121, description="The number of frames to generate."
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

    async def process(self, context: ProcessingContext) -> VideoRef:
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
            "num_frames": self.num_frames,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/distilled/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Ltx219BTextToVideoLora(FALNode):
    """
    LTX-2 19B
    video, generation, text-to-video, txt2vid, lora

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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


    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
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
    loras: list[str] = Field(
        default=[], description="The LoRAs to use for the generation."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    video_size: str = Field(
        default="landscape_4_3", description="The size of the generated video."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
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
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "use_multiscale": self.use_multiscale,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "fps": self.fps,
            "loras": self.loras,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/text-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Ltx219BTextToVideo(FALNode):
    """
    LTX-2 19B
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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


    use_multiscale: bool = Field(
        default=True, description="Whether to use multi-scale generation. If True, the model will generate the video at a smaller scale first, then use the smaller video to guide the generation of a video at or above your requested size. This results in better coherence and details."
    )
    prompt: str = Field(
        default="", description="The prompt to generate the video from."
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
        default="landscape_4_3", description="The size of the generated video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    camera_lora_scale: float = Field(
        default=1, description="The scale of the camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
    )
    guidance_scale: float = Field(
        default=3, description="The guidance scale to use."
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
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
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
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-2-19b/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Kandinsky5ProTextToVideo(FALNode):
    """
    Kandinsky5 Pro
    video, generation, text-to-video, txt2vid, professional

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Resolution(Enum):
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

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video. One of (3:2, 1:1, 2:3).
        """
        RATIO_3_2 = "3:2"
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"

    class Duration(Enum):
        """
        The length of the video to generate (5s or 10s)
        """
        VALUE_5S = "5s"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_512P, description="Video resolution: 512p or 1024p."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for faster generation."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_3_2, description="Aspect ratio of the generated video. One of (3:2, 1:1, 2:3)."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5S, description="The length of the video to generate (5s or 10s)"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "num_inference_steps": self.num_inference_steps,
            "duration": self.duration.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kandinsky5-pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV2_6TextToVideo(FALNode):
    """
    Wan v2.6 Text to Video
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Duration(Enum):
        """
        Duration of the generated video in seconds. Choose between 5, 10, or 15 seconds.
        """
        VALUE_5 = "5"
        VALUE_10 = "10"
        VALUE_15 = "15"

    class Resolution(Enum):
        """
        Video resolution tier. Wan 2.6 T2V only supports 720p and 1080p (no 480p).
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video. Wan 2.6 supports additional ratios.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"


    prompt: str = Field(
        default="", description="The text prompt for video generation. Supports Chinese and English, max 800 characters. For multi-shot videos, use format: 'Overall description. First shot [0-3s] content. Second shot [3-5s] content.'"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the generated video in seconds. Choose between 5, 10, or 15 seconds."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1080P, description="Video resolution tier. Wan 2.6 T2V only supports 720p and 1080p (no 480p)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video. Wan 2.6 supports additional ratios."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt rewriting using LLM. Improves results for short prompts but increases processing time."
    )
    audio_url: VideoRef = Field(
        default=VideoRef(), description="URL of the audio to use as the background music. Must be publicly accessible. Limit handling: If the audio duration exceeds the duration value (5, 10, or 15 seconds), the audio is truncated to the first N seconds, and the rest is discarded. If the audio is shorter than the video, the remaining part of the video will be silent. For example, if the audio is 3 seconds long and the video duration is 5 seconds, the first 3 seconds of the output video will have sound, and the last 2 seconds will be silent. - Format: WAV, MP3. - Duration: 3 to 30 s. - File size: Up to 15 MB."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    multi_shots: bool = Field(
        default=True, description="When true, enables intelligent multi-shot segmentation for coherent narrative videos. Only active when enable_prompt_expansion is True. Set to false for single-shot generation."
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
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "audio_url": self.audio_url,
            "seed": self.seed,
            "multi_shots": self.multi_shots,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="wan/v2.6/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV2_6ProTextToVideo(FALNode):
    """
    Kling Video v2.6 Text to Video
    video, generation, text-to-video, txt2vid, professional

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video frame
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class Duration(Enum):
        """
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate native audio for the video. Supports Chinese and English voice output. Other languages are automatically translated to English. For English speech, use lowercase letters; for acronyms or proper nouns, use uppercase."
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "generate_audio": self.generate_audio,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2.6/pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PixverseV5_5TextToVideo(FALNode):
    """
    Pixverse
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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


    prompt: str = Field(
        default=""
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
    generate_multi_clip_switch: bool = Field(
        default=False, description="Enable multi-clip generation with dynamic camera changes"
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
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "generate_multi_clip_switch": self.generate_multi_clip_switch,
            "duration": self.duration.value,
            "generate_audio_switch": self.generate_audio_switch,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5.5/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class HunyuanVideoV1_5TextToVideo(FALNode):
    """
    Hunyuan Video V1.5
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the video.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"

    class Resolution(Enum):
        """
        The resolution of the video.
        """
        VALUE_480P = "480p"


    prompt: str = Field(
        default="", description="The prompt to generate the video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the video."
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
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
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
            application="fal-ai/hunyuan-video-v1.5/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class InfinityStarTextToVideo(FALNode):
    """
    Infinity Star
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated output
        """
        RATIO_16_9 = "16:9"
        RATIO_1_1 = "1:1"
        RATIO_9_16 = "9:16"


    prompt: str = Field(
        default="", description="Text prompt for generating the video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated output"
    )
    enhance_prompt: bool = Field(
        default=True, description="Whether to use an LLM to enhance the prompt."
    )
    use_apg: bool = Field(
        default=True, description="Whether to use APG"
    )
    guidance_scale: float = Field(
        default=7.5, description="Guidance scale for generation"
    )
    num_inference_steps: int = Field(
        default=50, description="Number of inference steps"
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. Leave empty for random generation."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to guide what to avoid in generation"
    )
    tau_video: float = Field(
        default=0.4, description="Tau value for video scale"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "enhance_prompt": self.enhance_prompt,
            "use_apg": self.use_apg,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "tau_video": self.tau_video,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/infinity-star/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class SanaVideo(FALNode):
    """
    Sana Video
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Resolution(Enum):
        """
        The resolution of the output video
        """
        VALUE_480P = "480p"


    prompt: str = Field(
        default="", description="The text prompt describing the video to generate"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the output video"
    )
    fps: int = Field(
        default=16, description="Frames per second for the output video"
    )
    motion_score: int = Field(
        default=30, description="Motion intensity score (higher = more motion)"
    )
    guidance_scale: float = Field(
        default=6, description="Guidance scale for generation (higher = more prompt adherence)"
    )
    num_inference_steps: int = Field(
        default=28, description="Number of denoising steps"
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation. If not provided, a random seed will be used."
    )
    negative_prompt: str = Field(
        default="A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience.", description="The negative prompt describing what to avoid in the generation"
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "fps": self.fps,
            "motion_score": self.motion_score,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sana-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LongcatVideoTextToVideo720P(FALNode):
    """
    LongCat Video
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
        X264__MP4 = "X264 (.mp4)"
        VP9__WEBM = "VP9 (.webm)"
        PRORES4444__MOV = "PRORES4444 (.mov)"
        GIF__GIF = "GIF (.gif)"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class VideoQuality(Enum):
        """
        The quality of the generated video.
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        MAXIMUM = "maximum"


    prompt: str = Field(
        default="", description="The prompt to guide the video generation."
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
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
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
            "aspect_ratio": self.aspect_ratio.value,
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
            application="fal-ai/longcat-video/text-to-video/720p",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LongcatVideoTextToVideo480P(FALNode):
    """
    LongCat Video
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
        X264__MP4 = "X264 (.mp4)"
        VP9__WEBM = "VP9 (.webm)"
        PRORES4444__MOV = "PRORES4444 (.mov)"
        GIF__GIF = "GIF (.gif)"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class VideoQuality(Enum):
        """
        The quality of the generated video.
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        MAXIMUM = "maximum"


    prompt: str = Field(
        default="", description="The prompt to guide the video generation."
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
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
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
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to use for the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
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
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-video/text-to-video/480p",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LongcatVideoDistilledTextToVideo720P(FALNode):
    """
    LongCat Video Distilled
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class VideoWriteMode(Enum):
        """
        The write mode of the generated video.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

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


    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
    )
    prompt: str = Field(
        default="", description="The prompt to guide the video generation."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
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
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    num_frames: int = Field(
        default=162, description="The number of frames to generate."
    )
    num_inference_steps: int = Field(
        default=12, description="The number of inference steps to use."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "prompt": self.prompt,
            "video_output_type": self.video_output_type.value,
            "fps": self.fps,
            "sync_mode": self.sync_mode,
            "num_refine_inference_steps": self.num_refine_inference_steps,
            "video_quality": self.video_quality.value,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-video/distilled/text-to-video/720p",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LongcatVideoDistilledTextToVideo480P(FALNode):
    """
    LongCat Video Distilled
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class VideoWriteMode(Enum):
        """
        The write mode of the generated video.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

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


    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
    )
    prompt: str = Field(
        default="", description="The prompt to guide the video generation."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    fps: int = Field(
        default=15, description="The frame rate of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    num_frames: int = Field(
        default=162, description="The number of frames to generate."
    )
    num_inference_steps: int = Field(
        default=12, description="The number of inference steps to use."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "prompt": self.prompt,
            "video_output_type": self.video_output_type.value,
            "fps": self.fps,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-video/distilled/text-to-video/480p",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class MinimaxHailuo2_3StandardTextToVideo(FALNode):
    """
    MiniMax Hailuo 2.3 [Standard] (Text to Video)
    video, generation, text-to-video, txt2vid, professional

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Duration(Enum):
        """
        The duration of the video in seconds.
        """
        VALUE_6 = "6"
        VALUE_10 = "10"


    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    duration: Duration = Field(
        default=Duration.VALUE_6, description="The duration of the video in seconds."
    )
    prompt: str = Field(
        default=""
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt_optimizer": self.prompt_optimizer,
            "duration": self.duration.value,
            "prompt": self.prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-2.3/standard/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class MinimaxHailuo2_3ProTextToVideo(FALNode):
    """
    MiniMax Hailuo 2.3 [Pro] (Text to Video)
    video, generation, text-to-video, txt2vid, professional

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    prompt: str = Field(
        default="", description="Text prompt for video generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt_optimizer": self.prompt_optimizer,
            "prompt": self.prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-2.3/pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KreaWan14BTextToVideo(FALNode):
    """
    Krea Wan 14b- Text to Video
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    prompt: str = Field(
        default="", description="Prompt for the video-to-video generation."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )
    num_frames: int = Field(
        default=78, description="Number of frames to generate. Must be a multiple of 12 plus 6, for example 6, 18, 30, 42, etc."
    )
    seed: str = Field(
        default="", description="Seed for the video-to-video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_frames": self.num_frames,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/krea-wan-14b/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanAlpha(FALNode):
    """
    Wan Alpha
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Sampler(Enum):
        """
        The sampler to use.
        """
        UNIPC = "unipc"
        DPMPP = "dpmPP"
        EULER = "euler"

    class VideoWriteMode(Enum):
        """
        The write mode of the generated video.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class Resolution(Enum):
        """
        The resolution of the generated video.
        """
        VALUE_240P = "240p"
        VALUE_360P = "360p"
        VALUE_480P = "480p"
        VALUE_580P = "580p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video.
        """
        RATIO_16_9 = "16:9"
        RATIO_1_1 = "1:1"
        RATIO_9_16 = "9:16"

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


    prompt: str = Field(
        default="", description="The prompt to guide the video generation."
    )
    shift: float = Field(
        default=10.5, description="The shift of the generated video."
    )
    mask_clamp_upper: float = Field(
        default=0.75, description="The upper bound of the mask clamping."
    )
    fps: int = Field(
        default=16, description="The frame rate of the generated video."
    )
    mask_clamp_lower: float = Field(
        default=0.1, description="The lower bound of the mask clamping."
    )
    num_frames: int = Field(
        default=81, description="The number of frames to generate."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable safety checker."
    )
    mask_binarization_threshold: float = Field(
        default=0.8, description="The threshold for mask binarization. When binarize_mask is True, this threshold will be used to binarize the mask. This will also be used for transparency when the output type is `.webm`."
    )
    sampler: Sampler = Field(
        default=Sampler.EULER, description="The sampler to use."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="The resolution of the generated video."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.VP9__WEBM, description="The output type of the generated video."
    )
    binarize_mask: bool = Field(
        default=False, description="Whether to binarize the mask."
    )
    video_quality: VideoQuality = Field(
        default=VideoQuality.HIGH, description="The quality of the generated video."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to use."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "shift": self.shift,
            "mask_clamp_upper": self.mask_clamp_upper,
            "fps": self.fps,
            "mask_clamp_lower": self.mask_clamp_lower,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "mask_binarization_threshold": self.mask_binarization_threshold,
            "sampler": self.sampler.value,
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "video_output_type": self.video_output_type.value,
            "binarize_mask": self.binarize_mask,
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-alpha",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Kandinsky5TextToVideoDistill(FALNode):
    """
    Kandinsky5
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Duration(Enum):
        """
        The length of the video to generate (5s or 10s)
        """
        VALUE_5S = "5s"
        VALUE_10S = "10s"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video. One of (3:2, 1:1, 2:3).
        """
        RATIO_3_2 = "3:2"
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"

    class Resolution(Enum):
        """
        Resolution of the generated video in W:H format. Will be calculated based on the aspect ratio(768x512, 512x512, 512x768).
        """
        VALUE_768X512 = "768x512"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5S, description="The length of the video to generate (5s or 10s)"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_3_2, description="Aspect ratio of the generated video. One of (3:2, 1:1, 2:3)."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_768X512, description="Resolution of the generated video in W:H format. Will be calculated based on the aspect ratio(768x512, 512x512, 512x768)."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kandinsky5/text-to-video/distill",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Kandinsky5TextToVideo(FALNode):
    """
    Kandinsky5
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Resolution(Enum):
        """
        Resolution of the generated video in W:H format. Will be calculated based on the aspect ratio(768x512, 512x512, 512x768).
        """
        VALUE_768X512 = "768x512"

    class Duration(Enum):
        """
        The length of the video to generate (5s or 10s)
        """
        VALUE_5S = "5s"
        VALUE_10S = "10s"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video. One of (3:2, 1:1, 2:3).
        """
        RATIO_3_2 = "3:2"
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_768X512, description="Resolution of the generated video in W:H format. Will be calculated based on the aspect ratio(768x512, 512x512, 512x768)."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5S, description="The length of the video to generate (5s or 10s)"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_3_2, description="Aspect ratio of the generated video. One of (3:2, 1:1, 2:3)."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kandinsky5/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Veo3_1Fast(FALNode):
    """
    Veo 3.1 Fast
    video, generation, text-to-video, txt2vid, fast

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
        Aspect ratio of the generated video
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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    auto_fix: bool = Field(
        default=True, description="Whether to automatically attempt to fix prompts that fail content policy or other validation checks by rewriting them."
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
            "resolution": self.resolution.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3.1/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Veo3_1(FALNode):
    """
    Veo 3.1
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
        Aspect ratio of the generated video
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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video."
    )
    auto_fix: bool = Field(
        default=True, description="Whether to automatically attempt to fix prompts that fail content policy or other validation checks by rewriting them."
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
            "resolution": self.resolution.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo3.1",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Ovi(FALNode):
    """
    Ovi Text to Video
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Resolution(Enum):
        """
        Resolution of the generated video in W:H format. One of (512x992, 992x512, 960x512, 512x960, 720x720, or 448x1120).
        """
        VALUE_512X992 = "512x992"
        VALUE_992X512 = "992x512"
        VALUE_960X512 = "960x512"
        VALUE_512X960 = "512x960"
        VALUE_720X720 = "720x720"
        VALUE_448X1120 = "448x1120"
        VALUE_1120X448 = "1120x448"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_992X512, description="Resolution of the generated video in W:H format. One of (512x992, 992x512, 960x512, 512x960, 720x720, or 448x1120)."
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
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "num_inference_steps": self.num_inference_steps,
            "audio_negative_prompt": self.audio_negative_prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ovi",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Wan25PreviewTextToVideo(FALNode):
    """
    Wan 2.5 Text to Video
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Duration(Enum):
        """
        Duration of the generated video in seconds. Choose between 5 or 10 seconds.
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class Resolution(Enum):
        """
        Video resolution tier
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default="", description="The text prompt for video generation. Supports Chinese and English, max 800 characters."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the generated video in seconds. Choose between 5 or 10 seconds."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1080P, description="Video resolution tier"
    )
    audio_url: VideoRef = Field(
        default=VideoRef(), description="URL of the audio to use as the background music. Must be publicly accessible. Limit handling: If the audio duration exceeds the duration value (5 or 10 seconds), the audio is truncated to the first 5 or 10 seconds, and the rest is discarded. If the audio is shorter than the video, the remaining part of the video will be silent. For example, if the audio is 3 seconds long and the video duration is 5 seconds, the first 3 seconds of the output video will have sound, and the last 2 seconds will be silent. - Format: WAV, MP3. - Duration: 3 to 30 s. - File size: Up to 15 MB."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt rewriting using LLM. Improves results for short prompts but increases processing time."
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
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "audio_url": self.audio_url,
            "seed": self.seed,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-25-preview/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PixverseV5TextToVideo(FALNode):
    """
    Pixverse
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
        The duration of the generated video in seconds. 8s videos cost double. 1080p videos are limited to 5 seconds
        """
        VALUE_5 = "5"
        VALUE_8 = "8"


    prompt: str = Field(
        default=""
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
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds. 8s videos cost double. 1080p videos are limited to 5 seconds"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "duration": self.duration.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v5/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class InfinitalkSingleText(FALNode):
    """
    Infinitalk
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Resolution(Enum):
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
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for generation."
    )
    text_input: str = Field(
        default="", description="The text input to guide video generation."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    voice: Voice = Field(
        default="", description="The voice to use for speech generation"
    )
    num_frames: int = Field(
        default=145, description="Number of frames to generate. Must be between 41 to 721."
    )
    seed: int = Field(
        default=42, description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "text_input": self.text_input,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "voice": self.voice.value,
            "num_frames": self.num_frames,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/infinitalk/single-text",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class MoonvalleyMareyT2V(FALNode):
    """
    Marey Realism V1.5
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
        VALUE_1152X1152 = "1152x1152"
        VALUE_1536X1152 = "1536x1152"
        VALUE_1152X1536 = "1152x1536"


    prompt: str = Field(
        default="", description="The prompt to generate a video from"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5S, description="The duration of the generated video."
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
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "dimensions": self.dimensions.value,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="moonvalley/marey/t2v",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]