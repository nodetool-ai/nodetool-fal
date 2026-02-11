from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.nodes.fal.types import KlingV3MultiPromptElement, LoRAInput, LoRAWeight, LoraWeight
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
        arguments = {
            "prompt": self.prompt,
            "use_rife": self.use_rife,
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
    num_frames: int = Field(
        default=16, description="The number of frames to generate for the video."
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
        arguments = {
            "prompt": self.prompt,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
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
    fps: int = Field(
        default=8, description="Number of frames per second to extract from the video."
    )
    video_size: str = Field(
        default="square", description="The size of the video to generate."
    )
    guidance_scale: float = Field(
        default=2, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform. 4-12 is recommended for turbo mode."
    )
    negative_prompt: str = Field(
        default="(bad quality, worst quality:1.2), ugly faces, bad anime", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    num_frames: int = Field(
        default=16, description="The number of frames to generate for the video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "fps": self.fps,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
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
    keyframe_1_image: ImageRef = Field(
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
    keyframe_2_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the third keyframe to use for the generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to specify what you don't want."
    )
    keyframe_0_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the first keyframe to use for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        keyframe_1_image_base64 = await context.image_to_base64(self.keyframe_1_image)
        keyframe_2_image_base64 = await context.image_to_base64(self.keyframe_2_image)
        keyframe_0_image_base64 = await context.image_to_base64(self.keyframe_0_image)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "controlnet_type": self.controlnet_type.value,
            "keyframe_2_index": self.keyframe_2_index,
            "keyframe_0_index": self.keyframe_0_index,
            "keyframe_1_image_url": f"data:image/png;base64,{keyframe_1_image_base64}",
            "keyframe_1_index": self.keyframe_1_index,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "keyframe_2_image_url": f"data:image/png;base64,{keyframe_2_image_base64}",
            "negative_prompt": self.negative_prompt,
            "keyframe_0_image_url": f"data:image/png;base64,{keyframe_0_image_base64}",
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
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality"
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
            "generate_audio": self.generate_audio,
            "resolution": self.resolution.value,
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


    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1080P, description="Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the video in seconds"
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
            "resolution": self.resolution.value,
            "duration": self.duration.value,
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
    image: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "text": self.text,
            "resolution": self.resolution.value,
            "voice_description": self.voice_description,
            "image_url": f"data:image/png;base64,{image_base64}",
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
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    advanced_camera_control: str = Field(
        default="", description="Advanced Camera control parameters"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    camera_control: CameraControl | None = Field(
        default=None, description="Camera control parameters"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "advanced_camera_control": self.advanced_camera_control,
            "aspect_ratio": self.aspect_ratio.value,
            "cfg_scale": self.cfg_scale,
            "negative_prompt": self.negative_prompt,
            "camera_control": self.camera_control.value if self.camera_control else None,
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

class PixverseV56TextToVideo(FALNode):
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
    loras: list[LoRAInput] = Field(
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
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "use_multiscale": self.use_multiscale,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "fps": self.fps,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "num_frames": self.num_frames,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
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
    video_size: str = Field(
        default="landscape_4_3", description="The size of the generated video."
    )
    camera_lora: CameraLora = Field(
        default=CameraLora.NONE, description="The camera LoRA to use. This allows you to control the camera movement of the generated video more accurately than just prompting the model to move the camera."
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
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "use_multiscale": self.use_multiscale,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "fps": self.fps,
            "video_size": self.video_size,
            "camera_lora": self.camera_lora.value,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "num_frames": self.num_frames,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
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
    loras: list[LoRAInput] = Field(
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
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
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
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
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
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
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
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "guidance_scale": self.guidance_scale,
            "camera_lora_scale": self.camera_lora_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_frames": self.num_frames,
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
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

class WanV26TextToVideo(FALNode):
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
    audio: VideoRef = Field(
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
            "audio_url": self.audio,
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

class KlingVideoV26ProTextToVideo(FALNode):
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
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
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

class PixverseV55TextToVideo(FALNode):
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

class HunyuanVideoV15TextToVideo(FALNode):
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
        X264_MP4 = "X264 (.mp4)"
        VP9_WEBM = "VP9 (.webm)"
        PRORES4444_MOV = "PRORES4444 (.mov)"
        GIF_GIF = "GIF (.gif)"

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
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
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
        X264_MP4 = "X264 (.mp4)"
        VP9_WEBM = "VP9 (.webm)"
        PRORES4444_MOV = "PRORES4444 (.mov)"
        GIF_GIF = "GIF (.gif)"

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
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
    )
    prompt: str = Field(
        default="", description="The prompt to guide the video generation."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video."
    )
    prompt: str = Field(
        default="", description="The prompt to guide the video generation."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264_MP4, description="The output type of the generated video."
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

class MinimaxHailuo23StandardTextToVideo(FALNode):
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


    prompt: str = Field(
        default=""
    )
    duration: Duration = Field(
        default=Duration.VALUE_6, description="The duration of the video in seconds."
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "prompt_optimizer": self.prompt_optimizer,
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

class MinimaxHailuo23ProTextToVideo(FALNode):
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

    prompt: str = Field(
        default="", description="Text prompt for video generation"
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
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
        DPM_PLUS_PLUS = "dpm++"
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
        default=VideoOutputType.VP9_WEBM, description="The output type of the generated video."
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

class Veo31Fast(FALNode):
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

class Veo31(FALNode):
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

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class Duration(Enum):
        """
        Duration of the generated video in seconds. Choose between 5 or 10 seconds.
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the generated video in seconds. Choose between 5 or 10 seconds."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1080P, description="Video resolution tier"
    )
    audio: VideoRef = Field(
        default=VideoRef(), description="URL of the audio to use as the background music. Must be publicly accessible. Limit handling: If the audio duration exceeds the duration value (5 or 10 seconds), the audio is truncated to the first 5 or 10 seconds, and the rest is discarded. If the audio is shorter than the video, the remaining part of the video will be silent. For example, if the audio is 3 seconds long and the video duration is 5 seconds, the first 3 seconds of the output video will have sound, and the last 2 seconds will be silent. - Format: WAV, MP3. - Duration: 3 to 30 s. - File size: Up to 15 MB."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to describe content to avoid. Max 500 characters."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt rewriting using LLM. Improves results for short prompts but increases processing time."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "audio_url": self.audio,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
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
    image: ImageRef = Field(
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
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "text_input": self.text_input,
            "image_url": f"data:image/png;base64,{image_base64}",
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

class WanV22A14bTextToVideoLora(FALNode):
    """
    Wan-2.2 text-to-video is a video model that generates high-quality videos with high visual quality and motion diversity from text prompts. This endpoint supports LoRAs made for Wan 2.2.
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
        Aspect ratio of the generated video (16:9 or 9:16).
        """
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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
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
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "video_write_mode": self.video_write_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "guidance_scale_2": self.guidance_scale_2,
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
            application="fal-ai/wan/v2.2-a14b/text-to-video/lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV225bTextToVideoDistill(FALNode):
    """
    Wan 2.2's 5B distill model produces up to 5 seconds of video 720p at 24FPS with fluid motion and powerful prompt understanding
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
        The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video (16:9 or 9:16).
        """
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
        default=1, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 17 to 161 (inclusive)."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (580p or 720p)."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
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
        arguments = {
            "shift": self.shift,
            "prompt": self.prompt,
            "num_interpolated_frames": self.num_interpolated_frames,
            "frames_per_second": self.frames_per_second,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "enable_safety_checker": self.enable_safety_checker,
            "video_write_mode": self.video_write_mode.value,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
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
            application="fal-ai/wan/v2.2-5b/text-to-video/distill",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV225bTextToVideoFastWan(FALNode):
    """
    Wan 2.2's 5B FastVideo model produces up to 5 seconds of video 720p at 24FPS with fluid motion and powerful prompt understanding
    video, generation, text-to-video, txt2vid, fast

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
        Aspect ratio of the generated video (16:9 or 9:16).
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class Resolution(Enum):
        """
        Resolution of the generated video (580p or 720p).
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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (580p or 720p)."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
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
            "video_quality": self.video_quality.value,
            "adjust_fps_for_interpolation": self.adjust_fps_for_interpolation,
            "seed": self.seed,
            "interpolator_model": self.interpolator_model.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-5b/text-to-video/fast-wan",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV22A14bTextToVideoTurbo(FALNode):
    """
    Wan-2.2 turbo text-to-video is a video model that generates high-quality videos with high visual quality and motion diversity from text prompts. 
    video, generation, text-to-video, txt2vid, fast

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
        Aspect ratio of the generated video (16:9 or 9:16).
        """
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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
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
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "prompt": self.prompt,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "video_quality": self.video_quality.value,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-a14b/text-to-video/turbo",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV225bTextToVideo(FALNode):
    """
    Wan 2.2's 5B model produces up to 5 seconds of video 720p at 24FPS with fluid motion and powerful prompt understanding
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
        The write mode of the output video. Faster write mode means faster results but larger file size, balanced write mode is a good compromise between speed and quality, and small write mode is the slowest but produces the smallest file size.
        """
        FAST = "fast"
        BALANCED = "balanced"
        SMALL = "small"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video (16:9 or 9:16).
        """
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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (580p or 720p)."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
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
            application="fal-ai/wan/v2.2-5b/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV22A14bTextToVideo(FALNode):
    """
    Wan-2.2 text-to-video is a video model that generates high-quality videos with high visual quality and motion diversity from text prompts. 
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
        Aspect ratio of the generated video (16:9 or 9:16).
        """
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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
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
        arguments = {
            "shift": self.shift,
            "prompt": self.prompt,
            "num_interpolated_frames": self.num_interpolated_frames,
            "acceleration": self.acceleration.value,
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
            application="fal-ai/wan/v2.2-a14b/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Ltxv13b098Distilled(FALNode):
    """
    Generate long videos from prompts using LTX Video-0.9.8 13B Distilled and custom LoRA
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
        Resolution of the generated video.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video.
        """
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"


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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video."
    )
    tone_map_compression_ratio: float = Field(
        default=0, description="The compression ratio for tone mapping. This is used to compress the dynamic range of the video to improve visual quality. A value of 0.0 means no compression, while a value of 1.0 means maximum compression."
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
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "enable_detail_pass": self.enable_detail_pass,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "tone_map_compression_ratio": self.tone_map_compression_ratio,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltxv-13b-098-distilled",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class MinimaxHailuo02ProTextToVideo(FALNode):
    """
    MiniMax Hailuo-02 Text To Video API (Pro, 1080p): Advanced video generation model with 1080p resolution
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    prompt: str = Field(
        default=""
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/hailuo-02/pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BytedanceSeedanceV1ProTextToVideo(FALNode):
    """
    Seedance 1.0 Pro, a high quality video generation model developed by Bytedance.
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
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"

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


    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1080P, description="Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the video in seconds"
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
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedance/v1/pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BytedanceSeedanceV1LiteTextToVideo(FALNode):
    """
    Seedance 1.0 Lite
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
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_1_1 = "1:1"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"
        RATIO_9_21 = "9:21"

    class Resolution(Enum):
        """
        Video resolution - 480p for faster generation, 720p for higher quality
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"

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


    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Video resolution - 480p for faster generation, 720p for higher quality"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Duration of the video in seconds"
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
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_fixed": self.camera_fixed,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedance/v1/lite/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV21MasterTextToVideo(FALNode):
    """
    Kling 2.1 Master: The premium endpoint for Kling 2.1, designed for top-tier text-to-video generation with unparalleled motion fluidity, cinematic visuals, and exceptional prompt precision.
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
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v2.1/master/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LtxVideo13bDev(FALNode):
    """
    Generate videos from prompts using LTX Video-0.9.7 13B and custom LoRA
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
        Resolution of the generated video (480p or 720p).
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video (16:9, 1:1 or 9:16).
        """
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"


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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9, 1:1 or 9:16)."
    )
    first_pass_skip_final_steps: int = Field(
        default=3, description="Number of inference steps to skip in the final steps of the first pass. By skipping some steps at the end, the first pass can focus on larger changes instead of smaller details."
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
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
            "first_pass_skip_final_steps": self.first_pass_skip_final_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-13b-dev",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LtxVideo13bDistilled(FALNode):
    """
    Generate videos from prompts using LTX Video-0.9.7 13B Distilled and custom LoRA
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
        Resolution of the generated video (480p or 720p).
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video (16:9, 1:1 or 9:16).
        """
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"


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
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9, 1:1 or 9:16)."
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
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "second_pass_num_inference_steps": self.second_pass_num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "first_pass_skip_final_steps": self.first_pass_skip_final_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-13b-distilled",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PixverseV45TextToVideoFast(FALNode):
    """
    Generate high quality and fast video clips from text and image prompts using PixVerse v4.5 fast
    video, generation, text-to-video, txt2vid, fast

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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class PixverseV45TextToVideo(FALNode):
    """
    Generate high quality video clips from text and image prompts using PixVerse v4.5
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
            application="fal-ai/pixverse/v4.5/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ViduQ1TextToVideo(FALNode):
    """
    Vidu Q1 Text to Video generates high-quality 1080p videos with exceptional visual quality and motion diversity
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
        The aspect ratio of the output video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class Style(Enum):
        """
        The style of output video
        """
        GENERAL = "general"
        ANIME = "anime"

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
    style: Style = Field(
        default=Style.GENERAL, description="The style of output video"
    )
    seed: int = Field(
        default=-1, description="Seed for the random number generator"
    )
    movement_amplitude: MovementAmplitude = Field(
        default=MovementAmplitude.AUTO, description="The movement amplitude of objects in the frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "style": self.style.value,
            "seed": self.seed,
            "movement_amplitude": self.movement_amplitude.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/q1/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PixverseV4TextToVideo(FALNode):
    """
    Generate high quality video clips from text and image prompts using PixVerse v4
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
            application="fal-ai/pixverse/v4/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PixverseV4TextToVideoFast(FALNode):
    """
    Generate high quality and fast video clips from text and image prompts using PixVerse v4 fast
    video, generation, text-to-video, txt2vid, fast

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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v4/text-to-video/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoLipsyncAudioToVideo(FALNode):
    """
    Kling LipSync is an audio-to-video model that generates realistic lip movements from audio input.
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the lip sync for. Supports .mp4/.mov, ≤100MB, 2–10s, 720p/1080p only, width/height 720–1920px."
    )
    audio: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio to generate the lip sync for. Minimum duration is 2s and maximum duration is 60s. Maximum file size is 5MB."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "video_url": self.video,
            "audio_url": self.audio,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/lipsync/audio-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoLipsyncTextToVideo(FALNode):
    """
    Kling LipSync is a text-to-video model that generates realistic lip movements from text input.
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class VoiceId(Enum):
        """
        Voice ID to use for speech synthesis
        """
        GENSHIN_VINDI2 = "genshin_vindi2"
        ZHINEN_XUESHENG = "zhinen_xuesheng"
        AOT = "AOT"
        AI_SHATANG = "ai_shatang"
        GENSHIN_KLEE2 = "genshin_klee2"
        GENSHIN_KIRARA = "genshin_kirara"
        AI_KAIYA = "ai_kaiya"
        OVERSEA_MALE1 = "oversea_male1"
        AI_CHENJIAHAO_712 = "ai_chenjiahao_712"
        GIRLFRIEND_4_SPEECH02 = "girlfriend_4_speech02"
        CHAT1_FEMALE_NEW_3 = "chat1_female_new-3"
        CHAT_0407_5_1 = "chat_0407_5-1"
        CARTOON_BOY_07 = "cartoon-boy-07"
        UK_BOY1 = "uk_boy1"
        CARTOON_GIRL_01 = "cartoon-girl-01"
        PEPPAPIG_PLATFORM = "PeppaPig_platform"
        AI_HUANGZHONG_712 = "ai_huangzhong_712"
        AI_HUANGYAOSHI_712 = "ai_huangyaoshi_712"
        AI_LAOGUOWANG_712 = "ai_laoguowang_712"
        CHENGSHU_JIEJIE = "chengshu_jiejie"
        YOU_PINGJING = "you_pingjing"
        CALM_STORY1 = "calm_story1"
        UK_MAN2 = "uk_man2"
        LAOPOPO_SPEECH02 = "laopopo_speech02"
        HEAINAINAI_SPEECH02 = "heainainai_speech02"
        READER_EN_M_V1 = "reader_en_m-v1"
        COMMERCIAL_LADY_EN_F_V1 = "commercial_lady_en_f-v1"
        TIYUXI_XUEDI = "tiyuxi_xuedi"
        TIEXIN_NANYOU = "tiexin_nanyou"
        GIRLFRIEND_1_SPEECH02 = "girlfriend_1_speech02"
        GIRLFRIEND_2_SPEECH02 = "girlfriend_2_speech02"
        ZHUXI_SPEECH02 = "zhuxi_speech02"
        UK_OLDMAN3 = "uk_oldman3"
        DONGBEILAOTIE_SPEECH02 = "dongbeilaotie_speech02"
        CHONGQINGXIAOHUO_SPEECH02 = "chongqingxiaohuo_speech02"
        CHUANMEIZI_SPEECH02 = "chuanmeizi_speech02"
        CHAOSHANDASHU_SPEECH02 = "chaoshandashu_speech02"
        AI_TAIWAN_MAN2_SPEECH02 = "ai_taiwan_man2_speech02"
        XIANZHANGGUI_SPEECH02 = "xianzhanggui_speech02"
        TIANJINJIEJIE_SPEECH02 = "tianjinjiejie_speech02"
        DIYINNANSANG_DB_CN_M_04_V2 = "diyinnansang_DB_CN_M_04-v2"
        YIZHIPIANNAN_V1 = "yizhipiannan-v1"
        GUANXIAOFANG_V2 = "guanxiaofang-v2"
        TIANMEIXUEMEI_V1 = "tianmeixuemei-v1"
        DAOPIANYANSANG_V1 = "daopianyansang-v1"
        MENGWA_V1 = "mengwa-v1"

    class VoiceLanguage(Enum):
        """
        The voice language corresponding to the Voice ID
        """
        ZH = "zh"
        EN = "en"


    text: str = Field(
        default="", description="Text content for lip-sync video generation. Max 120 characters."
    )
    video: VideoRef = Field(
        default=VideoRef(), description="The URL of the video to generate the lip sync for. Supports .mp4/.mov, ≤100MB, 2-60s, 720p/1080p only, width/height 720–1920px. If validation fails, an error is returned."
    )
    voice_id: VoiceId = Field(
        default="", description="Voice ID to use for speech synthesis"
    )
    voice_language: VoiceLanguage = Field(
        default=VoiceLanguage.EN, description="The voice language corresponding to the Voice ID"
    )
    voice_speed: float = Field(
        default=1, description="Speech rate for Text to Video generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "text": self.text,
            "video_url": self.video,
            "voice_id": self.voice_id.value,
            "voice_language": self.voice_language.value,
            "voice_speed": self.voice_speed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/lipsync/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanT2vLora(FALNode):
    """
    Add custom LoRAs to Wan-2.1 is a text-to-video model that generates high-quality videos with high visual quality and motion diversity from images
    video, generation, text-to-video, txt2vid, lora

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class Resolution(Enum):
        """
        Resolution of the generated video (480p,580p, or 720p).
        """
        VALUE_480P = "480p"
        VALUE_580P = "580p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video (16:9 or 9:16).
        """
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the generated video (480p,580p, or 720p)."
    )
    reverse_video: bool = Field(
        default=False, description="If true, the video will be reversed."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
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
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 100 (inclusive)."
    )
    negative_prompt: str = Field(
        default="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "reverse_video": self.reverse_video,
            "seed": self.seed,
            "aspect_ratio": self.aspect_ratio.value,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "frames_per_second": self.frames_per_second,
            "turbo_mode": self.turbo_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "num_frames": self.num_frames,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-t2v-lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LumaDreamMachineRay2Flash(FALNode):
    """
    Ray2 Flash is a fast video generative model capable of creating realistic visuals with natural, coherent motion.
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
        The duration of the generated video (9s costs 2x more)
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
        default=Duration.VALUE_5S, description="The duration of the generated video (9s costs 2x more)"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "loop": self.loop,
            "duration": self.duration.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2-flash",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PikaV21TextToVideo(FALNode):
    """
    Start with a simple text input to create dynamic generations that defy expectations. Anything you dream can come to life with sharp details, impressive character control and cinematic camera moves.
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
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_4_5 = "4:5"
        RATIO_5_4 = "5:4"
        RATIO_3_2 = "3:2"
        RATIO_2_3 = "2:3"

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": self.duration,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class PikaV2TurboTextToVideo(FALNode):
    """
    Pika v2 Turbo creates videos from a text prompt with high quality output.
    video, generation, text-to-video, txt2vid, fast

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
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_4_5 = "4:5"
        RATIO_5_4 = "5:4"
        RATIO_3_2 = "3:2"
        RATIO_2_3 = "2:3"

    class Resolution(Enum):
        """
        The resolution of the generated video
        """
        VALUE_720P = "720p"
        VALUE_1080P = "1080p"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "duration": self.duration,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pika/v2/turbo/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanProTextToVideo(FALNode):
    """
    Wan-2.1 Pro is a premium text-to-video model that generates high-quality 1080p videos at 30fps with up to 6 seconds duration, delivering exceptional visual quality and motion diversity from text prompts
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker"
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. If None, a random seed is chosen."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV16ProEffects(FALNode):
    """
    Generate video clips from your prompts using Kling 1.6 (pro)
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
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

    class EffectScene(Enum):
        """
        The effect scene to use for the video generation
        """
        HUG = "hug"
        KISS = "kiss"
        HEART_GESTURE = "heart_gesture"
        SQUISH = "squish"
        EXPANSION = "expansion"
        FUZZYFUZZY = "fuzzyfuzzy"
        BLOOMBLOOM = "bloombloom"
        DIZZYDIZZY = "dizzydizzy"
        JELLY_PRESS = "jelly_press"
        JELLY_SLICE = "jelly_slice"
        JELLY_SQUISH = "jelly_squish"
        JELLY_JIGGLE = "jelly_jiggle"
        PIXELPIXEL = "pixelpixel"
        YEARBOOK = "yearbook"
        INSTANT_FILM = "instant_film"
        ANIME_FIGURE = "anime_figure"
        ROCKETROCKET = "rocketrocket"
        FLY_FLY = "fly_fly"
        DISAPPEAR = "disappear"
        LIGHTNING_POWER = "lightning_power"
        BULLET_TIME = "bullet_time"
        BULLET_TIME_360 = "bullet_time_360"
        MEDIA_INTERVIEW = "media_interview"
        DAY_TO_NIGHT = "day_to_night"
        LETS_RIDE = "let's_ride"
        JUMPDROP = "jumpdrop"
        SWISH_SWISH = "swish_swish"
        RUNNING_MAN = "running_man"
        JAZZ_JAZZ = "jazz_jazz"
        SWING_SWING = "swing_swing"
        SKATESKATE = "skateskate"
        BUILDING_SWEATER = "building_sweater"
        PURE_WHITE_WINGS = "pure_white_wings"
        BLACK_WINGS = "black_wings"
        GOLDEN_WING = "golden_wing"
        PINK_PINK_WINGS = "pink_pink_wings"
        RAMPAGE_APE = "rampage_ape"
        A_LIST_LOOK = "a_list_look"
        COUNTDOWN_TELEPORT = "countdown_teleport"
        FIREWORK_2026 = "firework_2026"
        INSTANT_CHRISTMAS = "instant_christmas"
        BIRTHDAY_STAR = "birthday_star"
        FIREWORK = "firework"
        CELEBRATION = "celebration"
        TIGER_HUG_PRO = "tiger_hug_pro"
        PET_LION_PRO = "pet_lion_pro"
        GUARDIAN_SPIRIT = "guardian_spirit"
        SQUEEZE_SCREAM = "squeeze_scream"
        INNER_VOICE = "inner_voice"
        MEMORY_ALIVE = "memory_alive"
        GUESS_WHAT = "guess_what"
        EAGLE_SNATCH = "eagle_snatch"
        HUG_FROM_PAST = "hug_from_past"
        INSTANT_KID = "instant_kid"
        DOLLAR_RAIN = "dollar_rain"
        CRY_CRY = "cry_cry"
        BUILDING_COLLAPSE = "building_collapse"
        MUSHROOM = "mushroom"
        JESUS_HUG = "jesus_hug"
        SHARK_ALERT = "shark_alert"
        LIE_FLAT = "lie_flat"
        POLAR_BEAR_HUG = "polar_bear_hug"
        BROWN_BEAR_HUG = "brown_bear_hug"
        OFFICE_ESCAPE_PLOW = "office_escape_plow"
        WATERMELON_BOMB = "watermelon_bomb"
        BOSS_COMING = "boss_coming"
        WIG_OUT = "wig_out"
        CAR_EXPLOSION = "car_explosion"
        TIGER_HUG = "tiger_hug"
        SIBLINGS = "siblings"
        CONSTRUCTION_WORKER = "construction_worker"
        SNATCHED = "snatched"
        FELT_FELT = "felt_felt"
        PLUSHCUT = "plushcut"
        DRUNK_DANCE = "drunk_dance"
        DRUNK_DANCE_PET = "drunk_dance_pet"
        DAOMA_DANCE = "daoma_dance"
        BOUNCY_DANCE = "bouncy_dance"
        SMOOTH_SAILING_DANCE = "smooth_sailing_dance"
        NEW_YEAR_GREETING = "new_year_greeting"
        LION_DANCE = "lion_dance"
        PROSPERITY = "prosperity"
        GREAT_SUCCESS = "great_success"
        GOLDEN_HORSE_FORTUNE = "golden_horse_fortune"
        RED_PACKET_BOX = "red_packet_box"
        LUCKY_HORSE_YEAR = "lucky_horse_year"
        LUCKY_RED_PACKET = "lucky_red_packet"
        LUCKY_MONEY_COME = "lucky_money_come"
        LION_DANCE_PET = "lion_dance_pet"
        DUMPLING_MAKING_PET = "dumpling_making_pet"
        FISH_MAKING_PET = "fish_making_pet"
        PET_RED_PACKET = "pet_red_packet"
        LANTERN_GLOW = "lantern_glow"
        EXPRESSION_CHALLENGE = "expression_challenge"
        OVERDRIVE = "overdrive"
        HEART_GESTURE_DANCE = "heart_gesture_dance"
        POPING = "poping"
        MARTIAL_ARTS = "martial_arts"
        RUNNING = "running"
        NEZHA = "nezha"
        MOTORCYCLE_DANCE = "motorcycle_dance"
        SUBJECT_3_DANCE = "subject_3_dance"
        GHOST_STEP_DANCE = "ghost_step_dance"
        PHANTOM_JEWEL = "phantom_jewel"
        ZOOM_OUT = "zoom_out"
        CHEERS_2026 = "cheers_2026"
        KISS_PRO = "kiss_pro"
        FIGHT_PRO = "fight_pro"
        HUG_PRO = "hug_pro"
        HEART_GESTURE_PRO = "heart_gesture_pro"
        DOLLAR_RAIN_PRO = "dollar_rain_pro"
        PET_BEE_PRO = "pet_bee_pro"
        SANTA_RANDOM_SURPRISE = "santa_random_surprise"
        MAGIC_MATCH_TREE = "magic_match_tree"
        HAPPY_BIRTHDAY = "happy_birthday"
        THUMBS_UP_PRO = "thumbs_up_pro"
        SURPRISE_BOUQUET = "surprise_bouquet"
        BOUQUET_DROP = "bouquet_drop"
        CARTOON_1_PRO_3D = "3d_cartoon_1_pro"
        GLAMOUR_PHOTO_SHOOT = "glamour_photo_shoot"
        BOX_OF_JOY = "box_of_joy"
        FIRST_TOAST_OF_THE_YEAR = "first_toast_of_the_year"
        MY_SANTA_PIC = "my_santa_pic"
        SANTA_GIFT = "santa_gift"
        STEAMPUNK_CHRISTMAS = "steampunk_christmas"
        SNOWGLOBE = "snowglobe"
        CHRISTMAS_PHOTO_SHOOT = "christmas_photo_shoot"
        ORNAMENT_CRASH = "ornament_crash"
        SANTA_EXPRESS = "santa_express"
        PARTICLE_SANTA_SURROUND = "particle_santa_surround"
        CORONATION_OF_FROST = "coronation_of_frost"
        SPARK_IN_THE_SNOW = "spark_in_the_snow"
        SCARLET_AND_SNOW = "scarlet_and_snow"
        COZY_TOON_WRAP = "cozy_toon_wrap"
        BULLET_TIME_LITE = "bullet_time_lite"
        MAGIC_CLOAK = "magic_cloak"
        BALLOON_PARADE = "balloon_parade"
        JUMPING_GINGER_JOY = "jumping_ginger_joy"
        C4D_CARTOON_PRO = "c4d_cartoon_pro"
        VENOMOUS_SPIDER = "venomous_spider"
        THRONE_OF_KING = "throne_of_king"
        LUMINOUS_ELF = "luminous_elf"
        WOODLAND_ELF = "woodland_elf"
        JAPANESE_ANIME_1 = "japanese_anime_1"
        AMERICAN_COMICS = "american_comics"
        SNOWBOARDING = "snowboarding"
        WITCH_TRANSFORM = "witch_transform"
        VAMPIRE_TRANSFORM = "vampire_transform"
        PUMPKIN_HEAD_TRANSFORM = "pumpkin_head_transform"
        DEMON_TRANSFORM = "demon_transform"
        MUMMY_TRANSFORM = "mummy_transform"
        ZOMBIE_TRANSFORM = "zombie_transform"
        CUTE_PUMPKIN_TRANSFORM = "cute_pumpkin_transform"
        CUTE_GHOST_TRANSFORM = "cute_ghost_transform"
        KNOCK_KNOCK_HALLOWEEN = "knock_knock_halloween"
        HALLOWEEN_ESCAPE = "halloween_escape"
        BASEBALL = "baseball"
        TRAMPOLINE = "trampoline"
        TRAMPOLINE_NIGHT = "trampoline_night"
        PUCKER_UP = "pucker_up"
        FEED_MOONCAKE = "feed_mooncake"
        FLYER = "flyer"
        DISHWASHER = "dishwasher"
        PET_CHINESE_OPERA = "pet_chinese_opera"
        MAGIC_FIREBALL = "magic_fireball"
        GALLERY_RING = "gallery_ring"
        PET_MOTO_RIDER = "pet_moto_rider"
        MUSCLE_PET = "muscle_pet"
        PET_DELIVERY = "pet_delivery"
        MYTHIC_STYLE = "mythic_style"
        STEAMPUNK = "steampunk"
        CARTOON_2_3D = "3d_cartoon_2"
        PET_CHEF = "pet_chef"
        SANTA_GIFTS = "santa_gifts"
        SANTA_HUG = "santa_hug"
        GIRLFRIEND = "girlfriend"
        BOYFRIEND = "boyfriend"
        HEART_GESTURE_1 = "heart_gesture_1"
        PET_WIZARD = "pet_wizard"
        SMOKE_SMOKE = "smoke_smoke"
        GUN_SHOT = "gun_shot"
        DOUBLE_GUN = "double_gun"
        PET_WARRIOR = "pet_warrior"
        LONG_HAIR = "long_hair"
        PET_DANCE = "pet_dance"
        WOOL_CURLY = "wool_curly"
        PET_BEE = "pet_bee"
        MARRY_ME = "marry_me"
        PIGGY_MORPH = "piggy_morph"
        SKI_SKI = "ski_ski"
        MAGIC_BROOM = "magic_broom"
        SPLASHSPLASH = "splashsplash"
        SURFSURF = "surfsurf"
        FAIRY_WING = "fairy_wing"
        ANGEL_WING = "angel_wing"
        DARK_WING = "dark_wing"
        EMOJI = "emoji"


    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    effect_scene: EffectScene = Field(
        default="", description="The effect scene to use for the video generation"
    )
    input_images: list[str] = Field(
        default=[], description="URL of images to be used for hug, kiss or heart_gesture video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "duration": self.duration.value,
            "effect_scene": self.effect_scene.value,
            "input_image_urls": self.input_images,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.6/pro/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV16StandardEffects(FALNode):
    """
    Generate video clips from your prompts using Kling 1.6 (std)
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
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

    class EffectScene(Enum):
        """
        The effect scene to use for the video generation
        """
        HUG = "hug"
        KISS = "kiss"
        HEART_GESTURE = "heart_gesture"
        SQUISH = "squish"
        EXPANSION = "expansion"
        FUZZYFUZZY = "fuzzyfuzzy"
        BLOOMBLOOM = "bloombloom"
        DIZZYDIZZY = "dizzydizzy"
        JELLY_PRESS = "jelly_press"
        JELLY_SLICE = "jelly_slice"
        JELLY_SQUISH = "jelly_squish"
        JELLY_JIGGLE = "jelly_jiggle"
        PIXELPIXEL = "pixelpixel"
        YEARBOOK = "yearbook"
        INSTANT_FILM = "instant_film"
        ANIME_FIGURE = "anime_figure"
        ROCKETROCKET = "rocketrocket"
        FLY_FLY = "fly_fly"
        DISAPPEAR = "disappear"
        LIGHTNING_POWER = "lightning_power"
        BULLET_TIME = "bullet_time"
        BULLET_TIME_360 = "bullet_time_360"
        MEDIA_INTERVIEW = "media_interview"
        DAY_TO_NIGHT = "day_to_night"
        LETS_RIDE = "let's_ride"
        JUMPDROP = "jumpdrop"
        SWISH_SWISH = "swish_swish"
        RUNNING_MAN = "running_man"
        JAZZ_JAZZ = "jazz_jazz"
        SWING_SWING = "swing_swing"
        SKATESKATE = "skateskate"
        BUILDING_SWEATER = "building_sweater"
        PURE_WHITE_WINGS = "pure_white_wings"
        BLACK_WINGS = "black_wings"
        GOLDEN_WING = "golden_wing"
        PINK_PINK_WINGS = "pink_pink_wings"
        RAMPAGE_APE = "rampage_ape"
        A_LIST_LOOK = "a_list_look"
        COUNTDOWN_TELEPORT = "countdown_teleport"
        FIREWORK_2026 = "firework_2026"
        INSTANT_CHRISTMAS = "instant_christmas"
        BIRTHDAY_STAR = "birthday_star"
        FIREWORK = "firework"
        CELEBRATION = "celebration"
        TIGER_HUG_PRO = "tiger_hug_pro"
        PET_LION_PRO = "pet_lion_pro"
        GUARDIAN_SPIRIT = "guardian_spirit"
        SQUEEZE_SCREAM = "squeeze_scream"
        INNER_VOICE = "inner_voice"
        MEMORY_ALIVE = "memory_alive"
        GUESS_WHAT = "guess_what"
        EAGLE_SNATCH = "eagle_snatch"
        HUG_FROM_PAST = "hug_from_past"
        INSTANT_KID = "instant_kid"
        DOLLAR_RAIN = "dollar_rain"
        CRY_CRY = "cry_cry"
        BUILDING_COLLAPSE = "building_collapse"
        MUSHROOM = "mushroom"
        JESUS_HUG = "jesus_hug"
        SHARK_ALERT = "shark_alert"
        LIE_FLAT = "lie_flat"
        POLAR_BEAR_HUG = "polar_bear_hug"
        BROWN_BEAR_HUG = "brown_bear_hug"
        OFFICE_ESCAPE_PLOW = "office_escape_plow"
        WATERMELON_BOMB = "watermelon_bomb"
        BOSS_COMING = "boss_coming"
        WIG_OUT = "wig_out"
        CAR_EXPLOSION = "car_explosion"
        TIGER_HUG = "tiger_hug"
        SIBLINGS = "siblings"
        CONSTRUCTION_WORKER = "construction_worker"
        SNATCHED = "snatched"
        FELT_FELT = "felt_felt"
        PLUSHCUT = "plushcut"
        DRUNK_DANCE = "drunk_dance"
        DRUNK_DANCE_PET = "drunk_dance_pet"
        DAOMA_DANCE = "daoma_dance"
        BOUNCY_DANCE = "bouncy_dance"
        SMOOTH_SAILING_DANCE = "smooth_sailing_dance"
        NEW_YEAR_GREETING = "new_year_greeting"
        LION_DANCE = "lion_dance"
        PROSPERITY = "prosperity"
        GREAT_SUCCESS = "great_success"
        GOLDEN_HORSE_FORTUNE = "golden_horse_fortune"
        RED_PACKET_BOX = "red_packet_box"
        LUCKY_HORSE_YEAR = "lucky_horse_year"
        LUCKY_RED_PACKET = "lucky_red_packet"
        LUCKY_MONEY_COME = "lucky_money_come"
        LION_DANCE_PET = "lion_dance_pet"
        DUMPLING_MAKING_PET = "dumpling_making_pet"
        FISH_MAKING_PET = "fish_making_pet"
        PET_RED_PACKET = "pet_red_packet"
        LANTERN_GLOW = "lantern_glow"
        EXPRESSION_CHALLENGE = "expression_challenge"
        OVERDRIVE = "overdrive"
        HEART_GESTURE_DANCE = "heart_gesture_dance"
        POPING = "poping"
        MARTIAL_ARTS = "martial_arts"
        RUNNING = "running"
        NEZHA = "nezha"
        MOTORCYCLE_DANCE = "motorcycle_dance"
        SUBJECT_3_DANCE = "subject_3_dance"
        GHOST_STEP_DANCE = "ghost_step_dance"
        PHANTOM_JEWEL = "phantom_jewel"
        ZOOM_OUT = "zoom_out"
        CHEERS_2026 = "cheers_2026"
        KISS_PRO = "kiss_pro"
        FIGHT_PRO = "fight_pro"
        HUG_PRO = "hug_pro"
        HEART_GESTURE_PRO = "heart_gesture_pro"
        DOLLAR_RAIN_PRO = "dollar_rain_pro"
        PET_BEE_PRO = "pet_bee_pro"
        SANTA_RANDOM_SURPRISE = "santa_random_surprise"
        MAGIC_MATCH_TREE = "magic_match_tree"
        HAPPY_BIRTHDAY = "happy_birthday"
        THUMBS_UP_PRO = "thumbs_up_pro"
        SURPRISE_BOUQUET = "surprise_bouquet"
        BOUQUET_DROP = "bouquet_drop"
        CARTOON_1_PRO_3D = "3d_cartoon_1_pro"
        GLAMOUR_PHOTO_SHOOT = "glamour_photo_shoot"
        BOX_OF_JOY = "box_of_joy"
        FIRST_TOAST_OF_THE_YEAR = "first_toast_of_the_year"
        MY_SANTA_PIC = "my_santa_pic"
        SANTA_GIFT = "santa_gift"
        STEAMPUNK_CHRISTMAS = "steampunk_christmas"
        SNOWGLOBE = "snowglobe"
        CHRISTMAS_PHOTO_SHOOT = "christmas_photo_shoot"
        ORNAMENT_CRASH = "ornament_crash"
        SANTA_EXPRESS = "santa_express"
        PARTICLE_SANTA_SURROUND = "particle_santa_surround"
        CORONATION_OF_FROST = "coronation_of_frost"
        SPARK_IN_THE_SNOW = "spark_in_the_snow"
        SCARLET_AND_SNOW = "scarlet_and_snow"
        COZY_TOON_WRAP = "cozy_toon_wrap"
        BULLET_TIME_LITE = "bullet_time_lite"
        MAGIC_CLOAK = "magic_cloak"
        BALLOON_PARADE = "balloon_parade"
        JUMPING_GINGER_JOY = "jumping_ginger_joy"
        C4D_CARTOON_PRO = "c4d_cartoon_pro"
        VENOMOUS_SPIDER = "venomous_spider"
        THRONE_OF_KING = "throne_of_king"
        LUMINOUS_ELF = "luminous_elf"
        WOODLAND_ELF = "woodland_elf"
        JAPANESE_ANIME_1 = "japanese_anime_1"
        AMERICAN_COMICS = "american_comics"
        SNOWBOARDING = "snowboarding"
        WITCH_TRANSFORM = "witch_transform"
        VAMPIRE_TRANSFORM = "vampire_transform"
        PUMPKIN_HEAD_TRANSFORM = "pumpkin_head_transform"
        DEMON_TRANSFORM = "demon_transform"
        MUMMY_TRANSFORM = "mummy_transform"
        ZOMBIE_TRANSFORM = "zombie_transform"
        CUTE_PUMPKIN_TRANSFORM = "cute_pumpkin_transform"
        CUTE_GHOST_TRANSFORM = "cute_ghost_transform"
        KNOCK_KNOCK_HALLOWEEN = "knock_knock_halloween"
        HALLOWEEN_ESCAPE = "halloween_escape"
        BASEBALL = "baseball"
        TRAMPOLINE = "trampoline"
        TRAMPOLINE_NIGHT = "trampoline_night"
        PUCKER_UP = "pucker_up"
        FEED_MOONCAKE = "feed_mooncake"
        FLYER = "flyer"
        DISHWASHER = "dishwasher"
        PET_CHINESE_OPERA = "pet_chinese_opera"
        MAGIC_FIREBALL = "magic_fireball"
        GALLERY_RING = "gallery_ring"
        PET_MOTO_RIDER = "pet_moto_rider"
        MUSCLE_PET = "muscle_pet"
        PET_DELIVERY = "pet_delivery"
        MYTHIC_STYLE = "mythic_style"
        STEAMPUNK = "steampunk"
        CARTOON_2_3D = "3d_cartoon_2"
        PET_CHEF = "pet_chef"
        SANTA_GIFTS = "santa_gifts"
        SANTA_HUG = "santa_hug"
        GIRLFRIEND = "girlfriend"
        BOYFRIEND = "boyfriend"
        HEART_GESTURE_1 = "heart_gesture_1"
        PET_WIZARD = "pet_wizard"
        SMOKE_SMOKE = "smoke_smoke"
        GUN_SHOT = "gun_shot"
        DOUBLE_GUN = "double_gun"
        PET_WARRIOR = "pet_warrior"
        LONG_HAIR = "long_hair"
        PET_DANCE = "pet_dance"
        WOOL_CURLY = "wool_curly"
        PET_BEE = "pet_bee"
        MARRY_ME = "marry_me"
        PIGGY_MORPH = "piggy_morph"
        SKI_SKI = "ski_ski"
        MAGIC_BROOM = "magic_broom"
        SPLASHSPLASH = "splashsplash"
        SURFSURF = "surfsurf"
        FAIRY_WING = "fairy_wing"
        ANGEL_WING = "angel_wing"
        DARK_WING = "dark_wing"
        EMOJI = "emoji"


    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    effect_scene: EffectScene = Field(
        default="", description="The effect scene to use for the video generation"
    )
    input_images: list[str] = Field(
        default=[], description="URL of images to be used for hug, kiss or heart_gesture video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "duration": self.duration.value,
            "effect_scene": self.effect_scene.value,
            "input_image_urls": self.input_images,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.6/standard/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV15ProEffects(FALNode):
    """
    Generate video clips from your prompts using Kling 1.5 (pro)
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
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

    class EffectScene(Enum):
        """
        The effect scene to use for the video generation
        """
        HUG = "hug"
        KISS = "kiss"
        HEART_GESTURE = "heart_gesture"
        SQUISH = "squish"
        EXPANSION = "expansion"
        FUZZYFUZZY = "fuzzyfuzzy"
        BLOOMBLOOM = "bloombloom"
        DIZZYDIZZY = "dizzydizzy"
        JELLY_PRESS = "jelly_press"
        JELLY_SLICE = "jelly_slice"
        JELLY_SQUISH = "jelly_squish"
        JELLY_JIGGLE = "jelly_jiggle"
        PIXELPIXEL = "pixelpixel"
        YEARBOOK = "yearbook"
        INSTANT_FILM = "instant_film"
        ANIME_FIGURE = "anime_figure"
        ROCKETROCKET = "rocketrocket"
        FLY_FLY = "fly_fly"
        DISAPPEAR = "disappear"
        LIGHTNING_POWER = "lightning_power"
        BULLET_TIME = "bullet_time"
        BULLET_TIME_360 = "bullet_time_360"
        MEDIA_INTERVIEW = "media_interview"
        DAY_TO_NIGHT = "day_to_night"
        LETS_RIDE = "let's_ride"
        JUMPDROP = "jumpdrop"
        SWISH_SWISH = "swish_swish"
        RUNNING_MAN = "running_man"
        JAZZ_JAZZ = "jazz_jazz"
        SWING_SWING = "swing_swing"
        SKATESKATE = "skateskate"
        BUILDING_SWEATER = "building_sweater"
        PURE_WHITE_WINGS = "pure_white_wings"
        BLACK_WINGS = "black_wings"
        GOLDEN_WING = "golden_wing"
        PINK_PINK_WINGS = "pink_pink_wings"
        RAMPAGE_APE = "rampage_ape"
        A_LIST_LOOK = "a_list_look"
        COUNTDOWN_TELEPORT = "countdown_teleport"
        FIREWORK_2026 = "firework_2026"
        INSTANT_CHRISTMAS = "instant_christmas"
        BIRTHDAY_STAR = "birthday_star"
        FIREWORK = "firework"
        CELEBRATION = "celebration"
        TIGER_HUG_PRO = "tiger_hug_pro"
        PET_LION_PRO = "pet_lion_pro"
        GUARDIAN_SPIRIT = "guardian_spirit"
        SQUEEZE_SCREAM = "squeeze_scream"
        INNER_VOICE = "inner_voice"
        MEMORY_ALIVE = "memory_alive"
        GUESS_WHAT = "guess_what"
        EAGLE_SNATCH = "eagle_snatch"
        HUG_FROM_PAST = "hug_from_past"
        INSTANT_KID = "instant_kid"
        DOLLAR_RAIN = "dollar_rain"
        CRY_CRY = "cry_cry"
        BUILDING_COLLAPSE = "building_collapse"
        MUSHROOM = "mushroom"
        JESUS_HUG = "jesus_hug"
        SHARK_ALERT = "shark_alert"
        LIE_FLAT = "lie_flat"
        POLAR_BEAR_HUG = "polar_bear_hug"
        BROWN_BEAR_HUG = "brown_bear_hug"
        OFFICE_ESCAPE_PLOW = "office_escape_plow"
        WATERMELON_BOMB = "watermelon_bomb"
        BOSS_COMING = "boss_coming"
        WIG_OUT = "wig_out"
        CAR_EXPLOSION = "car_explosion"
        TIGER_HUG = "tiger_hug"
        SIBLINGS = "siblings"
        CONSTRUCTION_WORKER = "construction_worker"
        SNATCHED = "snatched"
        FELT_FELT = "felt_felt"
        PLUSHCUT = "plushcut"
        DRUNK_DANCE = "drunk_dance"
        DRUNK_DANCE_PET = "drunk_dance_pet"
        DAOMA_DANCE = "daoma_dance"
        BOUNCY_DANCE = "bouncy_dance"
        SMOOTH_SAILING_DANCE = "smooth_sailing_dance"
        NEW_YEAR_GREETING = "new_year_greeting"
        LION_DANCE = "lion_dance"
        PROSPERITY = "prosperity"
        GREAT_SUCCESS = "great_success"
        GOLDEN_HORSE_FORTUNE = "golden_horse_fortune"
        RED_PACKET_BOX = "red_packet_box"
        LUCKY_HORSE_YEAR = "lucky_horse_year"
        LUCKY_RED_PACKET = "lucky_red_packet"
        LUCKY_MONEY_COME = "lucky_money_come"
        LION_DANCE_PET = "lion_dance_pet"
        DUMPLING_MAKING_PET = "dumpling_making_pet"
        FISH_MAKING_PET = "fish_making_pet"
        PET_RED_PACKET = "pet_red_packet"
        LANTERN_GLOW = "lantern_glow"
        EXPRESSION_CHALLENGE = "expression_challenge"
        OVERDRIVE = "overdrive"
        HEART_GESTURE_DANCE = "heart_gesture_dance"
        POPING = "poping"
        MARTIAL_ARTS = "martial_arts"
        RUNNING = "running"
        NEZHA = "nezha"
        MOTORCYCLE_DANCE = "motorcycle_dance"
        SUBJECT_3_DANCE = "subject_3_dance"
        GHOST_STEP_DANCE = "ghost_step_dance"
        PHANTOM_JEWEL = "phantom_jewel"
        ZOOM_OUT = "zoom_out"
        CHEERS_2026 = "cheers_2026"
        KISS_PRO = "kiss_pro"
        FIGHT_PRO = "fight_pro"
        HUG_PRO = "hug_pro"
        HEART_GESTURE_PRO = "heart_gesture_pro"
        DOLLAR_RAIN_PRO = "dollar_rain_pro"
        PET_BEE_PRO = "pet_bee_pro"
        SANTA_RANDOM_SURPRISE = "santa_random_surprise"
        MAGIC_MATCH_TREE = "magic_match_tree"
        HAPPY_BIRTHDAY = "happy_birthday"
        THUMBS_UP_PRO = "thumbs_up_pro"
        SURPRISE_BOUQUET = "surprise_bouquet"
        BOUQUET_DROP = "bouquet_drop"
        CARTOON_1_PRO_3D = "3d_cartoon_1_pro"
        GLAMOUR_PHOTO_SHOOT = "glamour_photo_shoot"
        BOX_OF_JOY = "box_of_joy"
        FIRST_TOAST_OF_THE_YEAR = "first_toast_of_the_year"
        MY_SANTA_PIC = "my_santa_pic"
        SANTA_GIFT = "santa_gift"
        STEAMPUNK_CHRISTMAS = "steampunk_christmas"
        SNOWGLOBE = "snowglobe"
        CHRISTMAS_PHOTO_SHOOT = "christmas_photo_shoot"
        ORNAMENT_CRASH = "ornament_crash"
        SANTA_EXPRESS = "santa_express"
        PARTICLE_SANTA_SURROUND = "particle_santa_surround"
        CORONATION_OF_FROST = "coronation_of_frost"
        SPARK_IN_THE_SNOW = "spark_in_the_snow"
        SCARLET_AND_SNOW = "scarlet_and_snow"
        COZY_TOON_WRAP = "cozy_toon_wrap"
        BULLET_TIME_LITE = "bullet_time_lite"
        MAGIC_CLOAK = "magic_cloak"
        BALLOON_PARADE = "balloon_parade"
        JUMPING_GINGER_JOY = "jumping_ginger_joy"
        C4D_CARTOON_PRO = "c4d_cartoon_pro"
        VENOMOUS_SPIDER = "venomous_spider"
        THRONE_OF_KING = "throne_of_king"
        LUMINOUS_ELF = "luminous_elf"
        WOODLAND_ELF = "woodland_elf"
        JAPANESE_ANIME_1 = "japanese_anime_1"
        AMERICAN_COMICS = "american_comics"
        SNOWBOARDING = "snowboarding"
        WITCH_TRANSFORM = "witch_transform"
        VAMPIRE_TRANSFORM = "vampire_transform"
        PUMPKIN_HEAD_TRANSFORM = "pumpkin_head_transform"
        DEMON_TRANSFORM = "demon_transform"
        MUMMY_TRANSFORM = "mummy_transform"
        ZOMBIE_TRANSFORM = "zombie_transform"
        CUTE_PUMPKIN_TRANSFORM = "cute_pumpkin_transform"
        CUTE_GHOST_TRANSFORM = "cute_ghost_transform"
        KNOCK_KNOCK_HALLOWEEN = "knock_knock_halloween"
        HALLOWEEN_ESCAPE = "halloween_escape"
        BASEBALL = "baseball"
        TRAMPOLINE = "trampoline"
        TRAMPOLINE_NIGHT = "trampoline_night"
        PUCKER_UP = "pucker_up"
        FEED_MOONCAKE = "feed_mooncake"
        FLYER = "flyer"
        DISHWASHER = "dishwasher"
        PET_CHINESE_OPERA = "pet_chinese_opera"
        MAGIC_FIREBALL = "magic_fireball"
        GALLERY_RING = "gallery_ring"
        PET_MOTO_RIDER = "pet_moto_rider"
        MUSCLE_PET = "muscle_pet"
        PET_DELIVERY = "pet_delivery"
        MYTHIC_STYLE = "mythic_style"
        STEAMPUNK = "steampunk"
        CARTOON_2_3D = "3d_cartoon_2"
        PET_CHEF = "pet_chef"
        SANTA_GIFTS = "santa_gifts"
        SANTA_HUG = "santa_hug"
        GIRLFRIEND = "girlfriend"
        BOYFRIEND = "boyfriend"
        HEART_GESTURE_1 = "heart_gesture_1"
        PET_WIZARD = "pet_wizard"
        SMOKE_SMOKE = "smoke_smoke"
        GUN_SHOT = "gun_shot"
        DOUBLE_GUN = "double_gun"
        PET_WARRIOR = "pet_warrior"
        LONG_HAIR = "long_hair"
        PET_DANCE = "pet_dance"
        WOOL_CURLY = "wool_curly"
        PET_BEE = "pet_bee"
        MARRY_ME = "marry_me"
        PIGGY_MORPH = "piggy_morph"
        SKI_SKI = "ski_ski"
        MAGIC_BROOM = "magic_broom"
        SPLASHSPLASH = "splashsplash"
        SURFSURF = "surfsurf"
        FAIRY_WING = "fairy_wing"
        ANGEL_WING = "angel_wing"
        DARK_WING = "dark_wing"
        EMOJI = "emoji"


    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    effect_scene: EffectScene = Field(
        default="", description="The effect scene to use for the video generation"
    )
    input_images: list[str] = Field(
        default=[], description="URL of images to be used for hug, kiss or heart_gesture video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "duration": self.duration.value,
            "effect_scene": self.effect_scene.value,
            "input_image_urls": self.input_images,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.5/pro/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV1StandardEffects(FALNode):
    """
    Generate video clips from your prompts using Kling 1.0
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
        The duration of the generated video in seconds
        """
        VALUE_5 = "5"
        VALUE_10 = "10"

    class EffectScene(Enum):
        """
        The effect scene to use for the video generation
        """
        HUG = "hug"
        KISS = "kiss"
        HEART_GESTURE = "heart_gesture"
        SQUISH = "squish"
        EXPANSION = "expansion"
        FUZZYFUZZY = "fuzzyfuzzy"
        BLOOMBLOOM = "bloombloom"
        DIZZYDIZZY = "dizzydizzy"
        JELLY_PRESS = "jelly_press"
        JELLY_SLICE = "jelly_slice"
        JELLY_SQUISH = "jelly_squish"
        JELLY_JIGGLE = "jelly_jiggle"
        PIXELPIXEL = "pixelpixel"
        YEARBOOK = "yearbook"
        INSTANT_FILM = "instant_film"
        ANIME_FIGURE = "anime_figure"
        ROCKETROCKET = "rocketrocket"
        FLY_FLY = "fly_fly"
        DISAPPEAR = "disappear"
        LIGHTNING_POWER = "lightning_power"
        BULLET_TIME = "bullet_time"
        BULLET_TIME_360 = "bullet_time_360"
        MEDIA_INTERVIEW = "media_interview"
        DAY_TO_NIGHT = "day_to_night"
        LETS_RIDE = "let's_ride"
        JUMPDROP = "jumpdrop"
        SWISH_SWISH = "swish_swish"
        RUNNING_MAN = "running_man"
        JAZZ_JAZZ = "jazz_jazz"
        SWING_SWING = "swing_swing"
        SKATESKATE = "skateskate"
        BUILDING_SWEATER = "building_sweater"
        PURE_WHITE_WINGS = "pure_white_wings"
        BLACK_WINGS = "black_wings"
        GOLDEN_WING = "golden_wing"
        PINK_PINK_WINGS = "pink_pink_wings"
        RAMPAGE_APE = "rampage_ape"
        A_LIST_LOOK = "a_list_look"
        COUNTDOWN_TELEPORT = "countdown_teleport"
        FIREWORK_2026 = "firework_2026"
        INSTANT_CHRISTMAS = "instant_christmas"
        BIRTHDAY_STAR = "birthday_star"
        FIREWORK = "firework"
        CELEBRATION = "celebration"
        TIGER_HUG_PRO = "tiger_hug_pro"
        PET_LION_PRO = "pet_lion_pro"
        GUARDIAN_SPIRIT = "guardian_spirit"
        SQUEEZE_SCREAM = "squeeze_scream"
        INNER_VOICE = "inner_voice"
        MEMORY_ALIVE = "memory_alive"
        GUESS_WHAT = "guess_what"
        EAGLE_SNATCH = "eagle_snatch"
        HUG_FROM_PAST = "hug_from_past"
        INSTANT_KID = "instant_kid"
        DOLLAR_RAIN = "dollar_rain"
        CRY_CRY = "cry_cry"
        BUILDING_COLLAPSE = "building_collapse"
        MUSHROOM = "mushroom"
        JESUS_HUG = "jesus_hug"
        SHARK_ALERT = "shark_alert"
        LIE_FLAT = "lie_flat"
        POLAR_BEAR_HUG = "polar_bear_hug"
        BROWN_BEAR_HUG = "brown_bear_hug"
        OFFICE_ESCAPE_PLOW = "office_escape_plow"
        WATERMELON_BOMB = "watermelon_bomb"
        BOSS_COMING = "boss_coming"
        WIG_OUT = "wig_out"
        CAR_EXPLOSION = "car_explosion"
        TIGER_HUG = "tiger_hug"
        SIBLINGS = "siblings"
        CONSTRUCTION_WORKER = "construction_worker"
        SNATCHED = "snatched"
        FELT_FELT = "felt_felt"
        PLUSHCUT = "plushcut"
        DRUNK_DANCE = "drunk_dance"
        DRUNK_DANCE_PET = "drunk_dance_pet"
        DAOMA_DANCE = "daoma_dance"
        BOUNCY_DANCE = "bouncy_dance"
        SMOOTH_SAILING_DANCE = "smooth_sailing_dance"
        NEW_YEAR_GREETING = "new_year_greeting"
        LION_DANCE = "lion_dance"
        PROSPERITY = "prosperity"
        GREAT_SUCCESS = "great_success"
        GOLDEN_HORSE_FORTUNE = "golden_horse_fortune"
        RED_PACKET_BOX = "red_packet_box"
        LUCKY_HORSE_YEAR = "lucky_horse_year"
        LUCKY_RED_PACKET = "lucky_red_packet"
        LUCKY_MONEY_COME = "lucky_money_come"
        LION_DANCE_PET = "lion_dance_pet"
        DUMPLING_MAKING_PET = "dumpling_making_pet"
        FISH_MAKING_PET = "fish_making_pet"
        PET_RED_PACKET = "pet_red_packet"
        LANTERN_GLOW = "lantern_glow"
        EXPRESSION_CHALLENGE = "expression_challenge"
        OVERDRIVE = "overdrive"
        HEART_GESTURE_DANCE = "heart_gesture_dance"
        POPING = "poping"
        MARTIAL_ARTS = "martial_arts"
        RUNNING = "running"
        NEZHA = "nezha"
        MOTORCYCLE_DANCE = "motorcycle_dance"
        SUBJECT_3_DANCE = "subject_3_dance"
        GHOST_STEP_DANCE = "ghost_step_dance"
        PHANTOM_JEWEL = "phantom_jewel"
        ZOOM_OUT = "zoom_out"
        CHEERS_2026 = "cheers_2026"
        KISS_PRO = "kiss_pro"
        FIGHT_PRO = "fight_pro"
        HUG_PRO = "hug_pro"
        HEART_GESTURE_PRO = "heart_gesture_pro"
        DOLLAR_RAIN_PRO = "dollar_rain_pro"
        PET_BEE_PRO = "pet_bee_pro"
        SANTA_RANDOM_SURPRISE = "santa_random_surprise"
        MAGIC_MATCH_TREE = "magic_match_tree"
        HAPPY_BIRTHDAY = "happy_birthday"
        THUMBS_UP_PRO = "thumbs_up_pro"
        SURPRISE_BOUQUET = "surprise_bouquet"
        BOUQUET_DROP = "bouquet_drop"
        CARTOON_1_PRO_3D = "3d_cartoon_1_pro"
        GLAMOUR_PHOTO_SHOOT = "glamour_photo_shoot"
        BOX_OF_JOY = "box_of_joy"
        FIRST_TOAST_OF_THE_YEAR = "first_toast_of_the_year"
        MY_SANTA_PIC = "my_santa_pic"
        SANTA_GIFT = "santa_gift"
        STEAMPUNK_CHRISTMAS = "steampunk_christmas"
        SNOWGLOBE = "snowglobe"
        CHRISTMAS_PHOTO_SHOOT = "christmas_photo_shoot"
        ORNAMENT_CRASH = "ornament_crash"
        SANTA_EXPRESS = "santa_express"
        PARTICLE_SANTA_SURROUND = "particle_santa_surround"
        CORONATION_OF_FROST = "coronation_of_frost"
        SPARK_IN_THE_SNOW = "spark_in_the_snow"
        SCARLET_AND_SNOW = "scarlet_and_snow"
        COZY_TOON_WRAP = "cozy_toon_wrap"
        BULLET_TIME_LITE = "bullet_time_lite"
        MAGIC_CLOAK = "magic_cloak"
        BALLOON_PARADE = "balloon_parade"
        JUMPING_GINGER_JOY = "jumping_ginger_joy"
        C4D_CARTOON_PRO = "c4d_cartoon_pro"
        VENOMOUS_SPIDER = "venomous_spider"
        THRONE_OF_KING = "throne_of_king"
        LUMINOUS_ELF = "luminous_elf"
        WOODLAND_ELF = "woodland_elf"
        JAPANESE_ANIME_1 = "japanese_anime_1"
        AMERICAN_COMICS = "american_comics"
        SNOWBOARDING = "snowboarding"
        WITCH_TRANSFORM = "witch_transform"
        VAMPIRE_TRANSFORM = "vampire_transform"
        PUMPKIN_HEAD_TRANSFORM = "pumpkin_head_transform"
        DEMON_TRANSFORM = "demon_transform"
        MUMMY_TRANSFORM = "mummy_transform"
        ZOMBIE_TRANSFORM = "zombie_transform"
        CUTE_PUMPKIN_TRANSFORM = "cute_pumpkin_transform"
        CUTE_GHOST_TRANSFORM = "cute_ghost_transform"
        KNOCK_KNOCK_HALLOWEEN = "knock_knock_halloween"
        HALLOWEEN_ESCAPE = "halloween_escape"
        BASEBALL = "baseball"
        TRAMPOLINE = "trampoline"
        TRAMPOLINE_NIGHT = "trampoline_night"
        PUCKER_UP = "pucker_up"
        FEED_MOONCAKE = "feed_mooncake"
        FLYER = "flyer"
        DISHWASHER = "dishwasher"
        PET_CHINESE_OPERA = "pet_chinese_opera"
        MAGIC_FIREBALL = "magic_fireball"
        GALLERY_RING = "gallery_ring"
        PET_MOTO_RIDER = "pet_moto_rider"
        MUSCLE_PET = "muscle_pet"
        PET_DELIVERY = "pet_delivery"
        MYTHIC_STYLE = "mythic_style"
        STEAMPUNK = "steampunk"
        CARTOON_2_3D = "3d_cartoon_2"
        PET_CHEF = "pet_chef"
        SANTA_GIFTS = "santa_gifts"
        SANTA_HUG = "santa_hug"
        GIRLFRIEND = "girlfriend"
        BOYFRIEND = "boyfriend"
        HEART_GESTURE_1 = "heart_gesture_1"
        PET_WIZARD = "pet_wizard"
        SMOKE_SMOKE = "smoke_smoke"
        GUN_SHOT = "gun_shot"
        DOUBLE_GUN = "double_gun"
        PET_WARRIOR = "pet_warrior"
        LONG_HAIR = "long_hair"
        PET_DANCE = "pet_dance"
        WOOL_CURLY = "wool_curly"
        PET_BEE = "pet_bee"
        MARRY_ME = "marry_me"
        PIGGY_MORPH = "piggy_morph"
        SKI_SKI = "ski_ski"
        MAGIC_BROOM = "magic_broom"
        SPLASHSPLASH = "splashsplash"
        SURFSURF = "surfsurf"
        FAIRY_WING = "fairy_wing"
        ANGEL_WING = "angel_wing"
        DARK_WING = "dark_wing"
        EMOJI = "emoji"


    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    effect_scene: EffectScene = Field(
        default="", description="The effect scene to use for the video generation"
    )
    input_images: list[str] = Field(
        default=[], description="URL of images to be used for hug, kiss or heart_gesture video."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "duration": self.duration.value,
            "effect_scene": self.effect_scene.value,
            "input_image_urls": self.input_images,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1/standard/effects",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LtxVideoV095(FALNode):
    """
    Generate videos from prompts using LTX Video-0.9.5
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
        Resolution of the generated video (480p or 720p).
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated video (16:9 or 9:16).
        """
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"


    prompt: str = Field(
        default="", description="Text prompt to guide generation"
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p or 720p)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "expand_prompt": self.expand_prompt,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ltx-video-v095",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV16ProTextToVideo(FALNode):
    """
    Generate video clips from your prompts using Kling 1.6 (pro)
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
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.6/pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanT2v(FALNode):
    """
    Wan-2.1 is a text-to-video model that generates high-quality videos with high visual quality and motion diversity from text prompts
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
        Aspect ratio of the generated video (16:9 or 9:16).
        """
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"

    class Resolution(Enum):
        """
        Resolution of the generated video (480p, 580p, or 720p).
        """
        VALUE_480P = "480p"
        VALUE_580P = "580p"
        VALUE_720P = "720p"


    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video (16:9 or 9:16)."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="Resolution of the generated video (480p, 580p, or 720p)."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    turbo_mode: bool = Field(
        default=False, description="If true, the video will be generated faster with no noticeable degradation in the visual quality."
    )
    frames_per_second: int = Field(
        default=16, description="Frames per second of the generated video. Must be between 5 to 24."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    num_frames: int = Field(
        default=81, description="Number of frames to generate. Must be between 81 to 100 (inclusive)."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    negative_prompt: str = Field(
        default="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "seed": self.seed,
            "turbo_mode": self.turbo_mode,
            "frames_per_second": self.frames_per_second,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-t2v",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Veo2(FALNode):
    """
    Veo 2 creates videos with realistic motion and high quality output. Explore different styles and find your own with extensive camera controls.
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
        The duration of the generated video in seconds
        """
        VALUE_5S = "5s"
        VALUE_6S = "6s"
        VALUE_7S = "7s"
        VALUE_8S = "8s"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"


    prompt: str = Field(
        default="", description="The text prompt describing the video you want to generate"
    )
    duration: Duration = Field(
        default=Duration.VALUE_5S, description="The duration of the generated video in seconds"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    seed: int = Field(
        default=-1, description="A seed to use for the video generation"
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation"
    )
    enhance_prompt: bool = Field(
        default=True, description="Whether to enhance the video generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/veo2",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class MinimaxVideo01Director(FALNode):
    """
    Generate video clips more accurately with respect to natural language descriptions and using camera movement instructions for shot control.
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    prompt: str = Field(
        default="", description="Text prompt for video generation. Camera movement instructions can be added using square brackets (e.g. [Pan left] or [Zoom in]). You can use up to 3 combined movements per prompt. Supported movements: Truck left/right, Pan left/right, Push in/Pull out, Pedestal up/down, Tilt up/down, Zoom in/out, Shake, Tracking shot, Static shot. For example: [Truck left, Pan right, Zoom in]. For a more detailed guide, refer https://sixth-switch-2ac.notion.site/T2V-01-Director-Model-Tutorial-with-camera-movement-1886c20a98eb80f395b8e05291ad8645"
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/video-01-director",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PixverseV35TextToVideo(FALNode):
    """
    Generate high quality video clips from text prompts using PixVerse v3.5
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
            application="fal-ai/pixverse/v3.5/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PixverseV35TextToVideoFast(FALNode):
    """
    Generate high quality video clips quickly from text prompts using PixVerse v3.5 Fast
    video, generation, text-to-video, txt2vid, fast

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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
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

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixverse/v3.5/text-to-video/fast",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LumaDreamMachineRay2(FALNode):
    """
    Ray2 is a large-scale video generative model capable of creating realistic visuals with natural, coherent motion.
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
        The duration of the generated video (9s costs 2x more)
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
        default=Duration.VALUE_5S, description="The duration of the generated video (9s costs 2x more)"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "loop": self.loop,
            "duration": self.duration.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-dream-machine/ray-2",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class HunyuanVideoLora(FALNode):
    """
    Hunyuan Video is an Open video generation model with high visual quality, motion diversity, text-video alignment, and generation stability
    video, generation, text-to-video, txt2vid, lora

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
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
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
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
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "num_frames": self.num_frames.value,
            "pro_mode": self.pro_mode,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-video-lora",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Transpixar(FALNode):
    """
    Transform text into stunning videos with TransPixar - an AI model that generates both RGB footage and alpha channels, enabling seamless compositing and creative video effects.
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    guidance_scale: float = Field(
        default=7, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related video to show you."
    )
    num_inference_steps: int = Field(
        default=24, description="The number of inference steps to perform."
    )
    export_fps: int = Field(
        default=8, description="The target FPS of the video"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate video from"
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same video every time."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
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
            application="fal-ai/transpixar",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV16StandardTextToVideo(FALNode):
    """
    Generate video clips from your prompts using Kling 1.6 (std)
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
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.6/standard/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class MinimaxVideo01Live(FALNode):
    """
    Generate video clips from your prompts using MiniMax model
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    prompt: str = Field(
        default=""
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/video-01-live",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV15ProTextToVideo(FALNode):
    """
    Generate video clips from your prompts using Kling 1.5 (pro)
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
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-video/v1.5/pro/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FastSvdTextToVideo(FALNode):
    """
    Generate short video clips from your prompts using SVD v1.1
    video, generation, text-to-video, txt2vid, fast

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    class DeepCache(Enum):
        """
        Enabling [DeepCache](https://github.com/horseee/DeepCache) will make the execution
        faster, but might sometimes degrade overall quality. The higher the setting, the
        faster the execution will be, but the more quality might be lost.
        """
        NONE = "none"
        MINIMUM = "minimum"
        MEDIUM = "medium"
        HIGH = "high"


    prompt: str = Field(
        default="", description="The prompt to use as a starting point for the generation."
    )
    cond_aug: float = Field(
        default=0.02, description="The conditoning augmentation determines the amount of noise that will be added to the conditioning frame. The higher the number, the more noise there will be, and the less the video will look like the initial image. Increase it for more motion."
    )
    deep_cache: DeepCache = Field(
        default=DeepCache.NONE, description="Enabling [DeepCache](https://github.com/horseee/DeepCache) will make the execution faster, but might sometimes degrade overall quality. The higher the setting, the faster the execution will be, but the more quality might be lost."
    )
    fps: int = Field(
        default=10, description="The FPS of the generated video. The higher the number, the faster the video will play. Total video length is 25 frames."
    )
    motion_bucket_id: int = Field(
        default=127, description="The motion bucket id determines the motion of the generated video. The higher the number, the more motion there will be."
    )
    video_size: str = Field(
        default="landscape_16_9", description="The size of the generated video."
    )
    steps: int = Field(
        default=20, description="The number of steps to run the model for. The higher the number the better the quality and longer it will take to generate."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    negative_prompt: str = Field(
        default="unrealistic, saturated, high contrast, big nose, painting, drawing, sketch, cartoon, anime, manga, render, CG, 3d, watermark, signature, label", description="The negative prompt to use as a starting point for the generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "cond_aug": self.cond_aug,
            "deep_cache": self.deep_cache.value,
            "fps": self.fps,
            "motion_bucket_id": self.motion_bucket_id,
            "video_size": self.video_size,
            "steps": self.steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-svd/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FastSvdLcmTextToVideo(FALNode):
    """
    Generate short video clips from your images using SVD v1.1 at Lightning Speed
    video, generation, text-to-video, txt2vid, fast

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    prompt: str = Field(
        default="", description="The prompt to use as a starting point for the generation."
    )
    cond_aug: float = Field(
        default=0.02, description="The conditoning augmentation determines the amount of noise that will be added to the conditioning frame. The higher the number, the more noise there will be, and the less the video will look like the initial image. Increase it for more motion."
    )
    fps: int = Field(
        default=10, description="The FPS of the generated video. The higher the number, the faster the video will play. Total video length is 25 frames."
    )
    motion_bucket_id: int = Field(
        default=127, description="The motion bucket id determines the motion of the generated video. The higher the number, the more motion there will be."
    )
    video_size: str = Field(
        default="landscape_16_9", description="The size of the generated video."
    )
    steps: int = Field(
        default=4, description="The number of steps to run the model for. The higher the number the better the quality and longer it will take to generate."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "cond_aug": self.cond_aug,
            "fps": self.fps,
            "motion_bucket_id": self.motion_bucket_id,
            "video_size": self.video_size,
            "steps": self.steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-svd-lcm/text-to-video",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class MinimaxVideo01(FALNode):
    """
    Generate video clips from your prompts using MiniMax model
    video, generation, text-to-video, txt2vid

    Use cases:
    - AI-generated video content
    - Marketing and advertising videos
    - Educational content creation
    - Social media video posts
    - Automated video production
    """

    prompt: str = Field(
        default=""
    )
    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "prompt_optimizer": self.prompt_optimizer,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/video-01",
            arguments=arguments,
        )
        assert "video" in res
        return VideoRef(uri=res["video"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingVideoV3StandardTextToVideo(FALNode):
    """
    Kling Video V3 Standard generates videos from text prompts with balanced quality and speed using the latest V3 model.
    video, generation, kling, v3, standard, text-to-video, txt2vid

    Use cases:
    - Generate cinematic videos from text descriptions
    - Create marketing videos from product descriptions
    - Produce educational video content from scripts
    - Generate social media video content
    - Create animated scenes from text prompts
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
        The type of multi-shot video generation
        """
        CUSTOMIZE = "customize"
        INTELLIGENT = "intelligent"


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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    multi_prompt: list[KlingV3MultiPromptElement] = Field(
        default=[], description="List of prompts for multi-shot video generation. If provided, overrides the single prompt and divides the video into multiple shots with specified prompts and durations."
    )
    shot_type: ShotType = Field(
        default=ShotType.CUSTOMIZE, description="The type of multi-shot video generation"
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
            "voice_ids": self.voice_ids,
            "duration": self.duration.value,
            "generate_audio": self.generate_audio,
            "aspect_ratio": self.aspect_ratio.value,
            "multi_prompt": [item.model_dump(exclude={"type"}) for item in self.multi_prompt],
            "shot_type": self.shot_type.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class KlingVideoO3StandardTextToVideo(FALNode):
    """
    Kling Video O3 Standard generates videos from text prompts with balanced quality and speed.
    video, generation, kling, o3, standard, text-to-video, txt2vid

    Use cases:
    - Generate cinematic videos from text descriptions
    - Create marketing videos from product descriptions
    - Produce educational video content from scripts
    - Generate social media video content
    - Create animated scenes from text prompts
    """

    class Duration(Enum):
        """
        Video duration in seconds (3-15s).
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
        Aspect ratio of the generated video.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class ShotType(Enum):
        """
        The type of multi-shot video generation.
        """
        CUSTOMIZE = "customize"


    prompt: str = Field(
        default="", description="Text prompt for video generation. Required unless multi_prompt is provided."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Video duration in seconds (3-15s)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video."
    )
    multi_prompt: list[KlingV3MultiPromptElement] = Field(
        default=[], description="List of prompts for multi-shot video generation."
    )
    voice_ids: list[str] = Field(
        default=[], description="Optional Voice IDs for video generation. Reference voices in your prompt with <<<voice_1>>> and <<<voice_2>>> (maximum 2 voices per task). Get voice IDs from the kling video create-voice endpoint: https://fal.ai/models/fal-ai/kling-video/create-voice"
    )
    generate_audio: bool = Field(
        default=False, description="Whether to generate native audio for the video."
    )
    shot_type: ShotType = Field(
        default=ShotType.CUSTOMIZE, description="The type of multi-shot video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "multi_prompt": [item.model_dump(exclude={"type"}) for item in self.multi_prompt],
            "voice_ids": self.voice_ids,
            "generate_audio": self.generate_audio,
            "shot_type": self.shot_type.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class KlingVideoO3ProTextToVideo(FALNode):
    """
    Kling Video O3 Pro generates professional quality videos from text prompts with enhanced fidelity.
    video, generation, kling, o3, pro, text-to-video, txt2vid

    Use cases:
    - Create professional-grade videos from detailed prompts
    - Generate cinematic video content with precise motion
    - Produce high-fidelity advertising videos
    - Create premium animated content from scripts
    - Generate top-tier video for film and media
    """

    class Duration(Enum):
        """
        Video duration in seconds (3-15s).
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
        Aspect ratio of the generated video.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"

    class ShotType(Enum):
        """
        The type of multi-shot video generation.
        """
        CUSTOMIZE = "customize"


    prompt: str = Field(
        default="", description="Text prompt for video generation. Required unless multi_prompt is provided."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Video duration in seconds (3-15s)."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of the generated video."
    )
    multi_prompt: list[KlingV3MultiPromptElement] = Field(
        default=[], description="List of prompts for multi-shot video generation."
    )
    voice_ids: list[str] = Field(
        default=[], description="Optional Voice IDs for video generation. Reference voices in your prompt with <<<voice_1>>> and <<<voice_2>>> (maximum 2 voices per task). Get voice IDs from the kling video create-voice endpoint: https://fal.ai/models/fal-ai/kling-video/create-voice"
    )
    generate_audio: bool = Field(
        default=False, description="Whether to generate native audio for the video."
    )
    shot_type: ShotType = Field(
        default=ShotType.CUSTOMIZE, description="The type of multi-shot video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "multi_prompt": [item.model_dump(exclude={"type"}) for item in self.multi_prompt],
            "voice_ids": self.voice_ids,
            "generate_audio": self.generate_audio,
            "shot_type": self.shot_type.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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

class KlingVideoV3ProTextToVideo(FALNode):
    """
    Kling Video V3 Pro generates professional quality videos from text prompts with enhanced visual fidelity using the latest V3 model.
    video, generation, kling, v3, pro, text-to-video, txt2vid

    Use cases:
    - Create professional-grade videos from detailed prompts
    - Generate cinematic video content with precise motion
    - Produce high-fidelity advertising videos
    - Create premium animated content from scripts
    - Generate top-tier video for film and media
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
        The type of multi-shot video generation
        """
        CUSTOMIZE = "customize"
        INTELLIGENT = "intelligent"


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
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame"
    )
    multi_prompt: list[KlingV3MultiPromptElement] = Field(
        default=[], description="List of prompts for multi-shot video generation. If provided, overrides the single prompt and divides the video into multiple shots with specified prompts and durations."
    )
    shot_type: ShotType = Field(
        default=ShotType.CUSTOMIZE, description="The type of multi-shot video generation"
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
            "voice_ids": self.voice_ids,
            "duration": self.duration.value,
            "generate_audio": self.generate_audio,
            "aspect_ratio": self.aspect_ratio.value,
            "multi_prompt": [item.model_dump(exclude={"type"}) for item in self.multi_prompt],
            "shot_type": self.shot_type.value,
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

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
