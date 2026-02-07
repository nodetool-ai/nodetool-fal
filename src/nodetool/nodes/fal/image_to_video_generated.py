from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef, VideoRef, AudioRef
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


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


class AIAvatarSingleTextResolution(Enum):
    """
    Resolution of the video to generate. Must be either 480p or 720p.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"


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


class AIAvatarMultiResolution(Enum):
    """
    Resolution of the video to generate. Must be either 480p or 720p.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"


class SeeDanceV15ProResolution(Enum):
    """
    Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"
    VALUE_1080P = "1080p"


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


class SeeDanceV1ProFastResolution(Enum):
    """
    Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"
    VALUE_1080P = "1080p"


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


class SeeDanceV1LiteResolution(Enum):
    """
    Video resolution - 480p for faster generation, 720p for higher quality
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"


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


class OmniHumanV15Resolution(Enum):
    """
    The resolution of the generated video. Defaults to 1080p. 720p generation is faster and higher in quality. 1080p generation is limited to 30s audio and 720p generation is limited to 60s audio.
    """
    VALUE_720P = "720p"
    VALUE_1080P = "1080p"


class KlingVideoV1StandardDuration(Enum):
    """
    The duration of the generated video in seconds
    """
    VALUE_5 = "5"
    VALUE_10 = "10"


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


class InterpolationDirection(Enum):
    """
    The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image.
    """
    FORWARD = "forward"
    BACKWARD = "backward"


class Kandinsky5ProResolution(Enum):
    """
    Video resolution: 512p or 1024p.
    """
    VALUE_512P = "512P"
    VALUE_1024P = "1024P"


class Kandinsky5ProDuration(Enum):
    """
    Video duration.
    """
    VALUE_5S = "5s"


class HunyuanVideoV15Resolution(Enum):
    """
    The resolution of the video.
    """
    VALUE_480P = "480p"


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


class KlingVideoV26ProDuration(Enum):
    """
    The duration of the generated video in seconds
    """
    VALUE_5 = "5"
    VALUE_10 = "10"


class CreatifyAuroraResolution(Enum):
    """
    The resolution of the generated video.
    """
    VALUE_480P = "480p"
    VALUE_720P = "720p"


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


class Effect(Enum):
    """
    The effect to apply to the video
    """
    KISS_ME_AI = "Kiss Me AI"
    KISS = "Kiss"
    MUSCLE_SURGE = "Muscle Surge"
    WARMTH_OF_JESUS = "Warmth of Jesus"
    ANYTHING__ROBOT = "Anything, Robot"
    THE_TIGER_TOUCH = "The Tiger Touch"
    HUG = "Hug"
    HOLY_WINGS = "Holy Wings"
    MICROWAVE = "Microwave"
    ZOMBIE_MODE = "Zombie Mode"
    SQUID_GAME = "Squid Game"
    BABY_FACE = "Baby Face"
    RATIO_BLACK_MYTH__WUKONG = "Black Myth: Wukong"
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
    HANDS_UP__HAND = "Hands up, Hand"
    FLORA_NYMPH_GO = "Flora Nymph Go"
    SOMBER_EMBRACE = "Somber Embrace"
    BEAM_ME_UP = "Beam me up"
    SUIT_SWAGGER = "Suit Swagger"


class Mode(Enum):
    """
    The swap mode to use
    """
    PERSON = "person"
    OBJECT = "object"
    BACKGROUND = "background"






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

    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    loop: bool = Field(
        default=False, description="Whether the video should loop (end of video is blended with the beginning)"
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="An image to blend the end of the video with"
    )
    image_url: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "loop": self.loop,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    frames: list[str] = Field(
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
            "frames": self.frames,
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

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: AIAvatarResolution = Field(
        default=AIAvatarResolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for generation."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "audio_url": self.audio_url,
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
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "text_input": self.text_input,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "second_text_input": self.second_text_input,
            "acceleration": self.acceleration.value,
            "resolution": self.resolution.value,
            "first_text_input": self.first_text_input,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: AIAvatarMultiResolution = Field(
        default=AIAvatarMultiResolution.VALUE_480P, description="Resolution of the video to generate. Must be either 480p or 720p."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for generation."
    )
    first_audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the Person 1 audio file."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the input image. If the input image does not match the chosen aspect ratio, it is resized and center cropped."
    )
    second_audio_url: AudioRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "first_audio_url": self.first_audio_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "second_audio_url": self.second_audio_url,
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

    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    resolution: SeeDanceV15ProResolution = Field(
        default=SeeDanceV15ProResolution.VALUE_720P, description="Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality"
    )
    duration: SeeDanceV15ProDuration = Field(
        default=SeeDanceV15ProDuration.VALUE_5, description="Duration of the video in seconds"
    )
    generate_audio: bool = Field(
        default=True, description="Whether to generate audio for the video"
    )
    aspect_ratio: SeeDanceV15ProAspectRatio = Field(
        default=SeeDanceV15ProAspectRatio.RATIO_16_9, description="The aspect ratio of the generated video"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image used to generate video"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="Random seed to control video generation. Use -1 for random."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image the video ends with. Defaults to None."
    )
    camera_fixed: bool = Field(
        default=False, description="Whether to fix the camera position"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "generate_audio": self.generate_audio,
            "aspect_ratio": self.aspect_ratio.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "camera_fixed": self.camera_fixed,
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

    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    resolution: SeeDanceV1ProFastResolution = Field(
        default=SeeDanceV1ProFastResolution.VALUE_1080P, description="Video resolution - 480p for faster generation, 720p for balance, 1080p for higher quality"
    )
    duration: SeeDanceV1ProFastDuration = Field(
        default=SeeDanceV1ProFastDuration.VALUE_5, description="Duration of the video in seconds"
    )
    aspect_ratio: SeeDanceV1ProFastAspectRatio = Field(
        default=SeeDanceV1ProFastAspectRatio.AUTO, description="The aspect ratio of the generated video"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image used to generate video"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="Random seed to control video generation. Use -1 for random."
    )
    camera_fixed: bool = Field(
        default=False, description="Whether to fix the camera position"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "camera_fixed": self.camera_fixed,
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

    prompt: str = Field(
        default="", description="The text prompt used to generate the video"
    )
    resolution: SeeDanceV1LiteResolution = Field(
        default=SeeDanceV1LiteResolution.VALUE_720P, description="Video resolution - 480p for faster generation, 720p for higher quality"
    )
    duration: SeeDanceV1LiteDuration = Field(
        default=SeeDanceV1LiteDuration.VALUE_5, description="Duration of the video in seconds"
    )
    aspect_ratio: SeeDanceV1LiteAspectRatio = Field(
        default=SeeDanceV1LiteAspectRatio.AUTO, description="The aspect ratio of the generated video"
    )
    reference_image_urls: list[str] = Field(
        default=[], description="Reference images to generate the video with."
    )
    seed: int = Field(
        default=-1, description="Random seed to control video generation. Use -1 for random."
    )
    camera_fixed: bool = Field(
        default=False, description="Whether to fix the camera position"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "reference_image_urls": self.reference_image_urls,
            "seed": self.seed,
            "camera_fixed": self.camera_fixed,
            "enable_safety_checker": self.enable_safety_checker,
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to make the stylized video from."
    )

    async def process(self, context: ProcessingContext) -> Any:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "style": self.style,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    turbo_mode: bool = Field(
        default=False, description="Generate a video at a faster rate with a slight quality trade-off."
    )
    resolution: OmniHumanV15Resolution = Field(
        default=OmniHumanV15Resolution.VALUE_1080P, description="The resolution of the generated video. Defaults to 1080p. 720p generation is faster and higher in quality. 1080p generation is limited to 30s audio and 720p generation is limited to 60s audio."
    )
    prompt: str = Field(
        default="", description="The text prompt used to guide the video generation."
    )
    audio_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio file to generate the video. Audio must be under 30s long for 1080p generation and under 60s long for 720p generation."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image used to generate the video"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "turbo_mode": self.turbo_mode,
            "resolution": self.resolution.value,
            "prompt": self.prompt,
            "audio_url": self.audio_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL to the image to generate the video from."
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "use_rife": self.use_rife,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to generate the video from."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    prompt: str = Field(
        default="", description="The prompt for the video"
    )
    duration: KlingVideoV1StandardDuration = Field(
        default=KlingVideoV1StandardDuration.VALUE_5, description="The duration of the generated video in seconds"
    )
    tail_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the end of the video"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the video"
    )
    static_mask_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image for Static Brush Application Area (Mask image created by users using the motion brush)"
    )
    dynamic_masks: list[str] = Field(
        default=[], description="List of dynamic masks"
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        tail_image_url_base64 = await context.image_to_base64(self.tail_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        static_mask_url_base64 = await context.image_to_base64(self.static_mask_url)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "tail_image_url": f"data:image/png;base64,{tail_image_url_base64}",
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "static_mask_url": f"data:image/png;base64,{static_mask_url_base64}",
            "dynamic_masks": self.dynamic_masks,
            "negative_prompt": self.negative_prompt,
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

    first_image_url: ImageRef = Field(
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
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the last frame"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        first_image_url_base64 = await context.image_to_base64(self.first_image_url)
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "first_image_url": f"data:image/png;base64,{first_image_url_base64}",
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "prompt": self.prompt,
            "duration": self.duration.value,
            "generate_audio_switch": self.generate_audio_switch,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
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

    prompt: str = Field(
        default="", description="Text prompt for video generation, max 2000 characters"
    )
    resolution: ViduQ2ReferenceToVideoProResolution = Field(
        default=ViduQ2ReferenceToVideoProResolution.VALUE_720P, description="Output video resolution"
    )
    aspect_ratio: str = Field(
        default="16:9", description="Aspect ratio of the output video (e.g., auto, 16:9, 9:16, 1:1, or any W:H)"
    )
    duration: int = Field(
        default=4, description="Duration of the video in seconds (0 for automatic duration)"
    )
    reference_video_urls: list[str] = Field(
        default=[], description="URLs of the reference videos for video editing or motion reference. Supports up to 2 videos."
    )
    bgm: bool = Field(
        default=False, description="Whether to add background music to the generated video"
    )
    reference_image_urls: list[str] = Field(
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
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio,
            "duration": self.duration,
            "reference_video_urls": self.reference_video_urls,
            "bgm": self.bgm,
            "reference_image_urls": self.reference_image_urls,
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame. Must be publicly accessible or base64 data URI. Image dimensions must be between 240 and 7680."
    )
    audio_url: VideoRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "audio_url": self.audio_url,
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame. Must be publicly accessible or base64 data URI. Image dimensions must be between 240 and 7680."
    )
    audio_url: VideoRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "enable_safety_checker": self.enable_safety_checker,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "audio_url": self.audio_url,
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

    prompt: str = Field(
        default="", description="The prompt used for the generation."
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
        default="auto", description="The size of the generated video."
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
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
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
    image_url: ImageRef = Field(
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
    seed: str = Field(
        default="", description="The seed for the random number generator."
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
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
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
            "image_strength": self.image_strength,
            "negative_prompt": self.negative_prompt,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    prompt: str = Field(
        default="", description="The prompt used for the generation."
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
        default="auto", description="The size of the generated video."
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
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
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
    image_url: ImageRef = Field(
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
    seed: str = Field(
        default="", description="The seed for the random number generator."
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
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
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
            "image_strength": self.image_strength,
            "negative_prompt": self.negative_prompt,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "enable_safety_checker": self.enable_safety_checker,
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "video_quality": self.video_quality.value,
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

    prompt: str = Field(
        default="", description="The prompt used for the generation."
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
    image_strength: float = Field(
        default=1, description="The strength of the image to use for the video generation."
    )
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
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
    image_url: ImageRef = Field(
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
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    interpolation_direction: InterpolationDirection = Field(
        default=InterpolationDirection.FORWARD, description="The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "use_multiscale": self.use_multiscale,
            "acceleration": self.acceleration.value,
            "generate_audio": self.generate_audio,
            "fps": self.fps,
            "camera_lora": self.camera_lora.value,
            "video_size": self.video_size,
            "enable_safety_checker": self.enable_safety_checker,
            "camera_lora_scale": self.camera_lora_scale,
            "image_strength": self.image_strength,
            "negative_prompt": self.negative_prompt,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "video_quality": self.video_quality.value,
            "sync_mode": self.sync_mode,
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

    prompt: str = Field(
        default="", description="The prompt used for the generation."
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
        default="auto", description="The size of the generated video."
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
    negative_prompt: str = Field(
        default="blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio,incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.", description="The negative prompt to generate the video from."
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as the end of the video."
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
    image_url: ImageRef = Field(
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
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )
    end_image_strength: float = Field(
        default=1, description="The strength of the end image to use for the video generation."
    )
    interpolation_direction: InterpolationDirection = Field(
        default=InterpolationDirection.FORWARD, description="The direction to interpolate the image sequence in. 'Forward' goes from the start image to the end image, 'Backward' goes from the end image to the start image."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        image_url_base64 = await context.image_to_base64(self.image_url)
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
            "image_strength": self.image_strength,
            "negative_prompt": self.negative_prompt,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "num_frames": self.num_frames,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "trajectories": self.trajectories,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a reference for the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "acceleration": self.acceleration.value,
            "duration": self.duration.value,
            "num_inference_steps": self.num_inference_steps,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    frames_per_clip: int = Field(
        default=48, description="Number of frames per clip. Must be a multiple of 4. Higher values = smoother but slower generation."
    )
    prompt: str = Field(
        default="", description="A text prompt describing the scene and character. Helps guide the video generation style and context."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Acceleration level for faster video decoding"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the reference image for avatar generation. The character in this image will be animated."
    )
    num_clips: int = Field(
        default=10, description="Number of video clips to generate. Each clip is approximately 3 seconds. Set higher for longer videos."
    )
    audio_url: AudioRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "frames_per_clip": self.frames_per_clip,
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "num_clips": self.num_clips,
            "audio_url": self.audio_url,
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

    prompt: str = Field(
        default="", description="The prompt to generate the video from."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the video."
    )
    resolution: HunyuanVideoV15Resolution = Field(
        default=HunyuanVideoV15Resolution.VALUE_480P, description="The resolution of the video."
    )
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    prompt: str = Field(
        default="", description="Use @Image1 to reference the start frame, @Image2 to reference the end frame."
    )
    duration: KlingVideoO1StandardDuration = Field(
        default=KlingVideoO1StandardDuration.VALUE_5, description="Video duration in seconds."
    )
    start_image_url: ImageRef = Field(
        default=ImageRef(), description="Image to use as the first frame of the video. Max file size: 10.0MB, Min width: 300px, Min height: 300px, Min aspect ratio: 0.40, Max aspect ratio: 2.50, Timeout: 20.0s"
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="Image to use as the last frame of the video. Max file size: 10.0MB, Min width: 300px, Min height: 300px, Min aspect ratio: 0.40, Max aspect ratio: 2.50, Timeout: 20.0s"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_url_base64 = await context.image_to_base64(self.start_image_url)
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "start_image_url": f"data:image/png;base64,{start_image_url_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
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

    prompt: str = Field(
        default="", description="Take @Element1, @Element2 to reference elements and @Image1, @Image2 to reference images in order."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame."
    )
    duration: KlingVideoO1StandardReferenceToVideoDuration = Field(
        default=KlingVideoO1StandardReferenceToVideoDuration.VALUE_5, description="Video duration in seconds."
    )
    elements: list[str] = Field(
        default=[], description="Elements (characters/objects) to include in the video. Reference in prompt as @Element1, @Element2, etc. Maximum 7 total (elements + reference images + start image)."
    )
    image_urls: list[str] = Field(
        default=[], description="Additional reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 7 total (elements + reference images + start image)."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "elements": self.elements,
            "image_urls": self.image_urls,
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
    start_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the video"
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the end of the video"
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_url_base64 = await context.image_to_base64(self.start_image_url)
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "voice_ids": self.voice_ids,
            "generate_audio": self.generate_audio,
            "start_image_url": f"data:image/png;base64,{start_image_url_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
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
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as your avatar"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "audio_url": self.audio_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as your avatar"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "audio_url": self.audio_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    prompt: str = Field(
        default="", description="A text prompt to guide the video generation process."
    )
    resolution: CreatifyAuroraResolution = Field(
        default=CreatifyAuroraResolution.VALUE_720P, description="The resolution of the generated video."
    )
    guidance_scale: float = Field(
        default=1, description="Guidance scale to be used for text prompt adherence."
    )
    audio_guidance_scale: float = Field(
        default=2, description="Guidance scale to be used for audio adherence."
    )
    audio_url: VideoRef = Field(
        default=VideoRef(), description="The URL of the audio file to be used for video generation."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image file to be used for video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "guidance_scale": self.guidance_scale,
            "audio_guidance_scale": self.audio_guidance_scale,
            "audio_url": self.audio_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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





class PixverseV5_5Effects(FALNode):
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="Optional URL of the image to use as the first frame. If not provided, generates from text"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "negative_prompt": self.negative_prompt,
            "duration": self.duration.value,
            "resolution": self.resolution.value,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "effect": self.effect.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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






class PixverseV5_5Transition(FALNode):
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

    first_image_url: ImageRef = Field(
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
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the last frame"
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt to be used for the generation"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        first_image_url_base64 = await context.image_to_base64(self.first_image_url)
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "first_image_url": f"data:image/png;base64,{first_image_url_base64}",
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "prompt": self.prompt,
            "duration": self.duration.value,
            "generate_audio_switch": self.generate_audio_switch,
            "seed": self.seed,
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
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





class PixverseV5_5ImageToVideo(FALNode):
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
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "duration": self.duration.value,
            "style": self.style.value if self.style else None,
            "thinking_type": self.thinking_type.value if self.thinking_type else None,
            "generate_multi_clip_switch": self.generate_multi_clip_switch,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    prompt: str = Field(
        default="", description="Use @Image1 to reference the start frame, @Image2 to reference the end frame."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Video duration in seconds."
    )
    start_image_url: ImageRef = Field(
        default=ImageRef(), description="Image to use as the first frame of the video. Max file size: 10.0MB, Min width: 300px, Min height: 300px, Min aspect ratio: 0.40, Max aspect ratio: 2.50, Timeout: 20.0s"
    )
    end_image_url: ImageRef = Field(
        default=ImageRef(), description="Image to use as the last frame of the video. Max file size: 10.0MB, Min width: 300px, Min height: 300px, Min aspect ratio: 0.40, Max aspect ratio: 2.50, Timeout: 20.0s"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        start_image_url_base64 = await context.image_to_base64(self.start_image_url)
        end_image_url_base64 = await context.image_to_base64(self.end_image_url)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "start_image_url": f"data:image/png;base64,{start_image_url_base64}",
            "end_image_url": f"data:image/png;base64,{end_image_url_base64}",
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

    prompt: str = Field(
        default="", description="Take @Element1, @Element2 to reference elements and @Image1, @Image2 to reference images in order."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the generated video frame."
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="Video duration in seconds."
    )
    elements: list[str] = Field(
        default=[], description="Elements (characters/objects) to include in the video. Reference in prompt as @Element1, @Element2, etc. Maximum 7 total (elements + reference images + start image)."
    )
    image_urls: list[str] = Field(
        default=[], description="Additional reference images for style/appearance. Reference in prompt as @Image1, @Image2, etc. Maximum 7 total (elements + reference images + start image)."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "duration": self.duration.value,
            "elements": self.elements,
            "image_urls": self.image_urls,
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
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale_2": self.guidance_scale_2,
            "strength": self.strength,
            "frames_per_second": self.frames_per_second,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    original_sound_switch: bool = Field(
        default=True, description="Whether to keep the original audio"
    )
    video_url: VideoRef = Field(
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the target image for swapping"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "original_sound_switch": self.original_sound_switch,
            "video_url": self.video_url,
            "keyframe_id": self.keyframe_id,
            "mode": self.mode.value,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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


class PikaV2_2Pikaframes(FALNode):
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

    prompt: str = Field(
        default="", description="Default prompt for all transitions. Individual transition prompts override this."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the generated video"
    )
    transitions: list[str] = Field(
        default=[], description="Configuration for each transition. Length must be len(image_urls) - 1. Total duration of all transitions must not exceed 25 seconds. If not provided, uses default 5-second transitions with the global prompt."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator"
    )
    image_urls: list[str] = Field(
        default=[], description="URLs of keyframe images (2-5 images) to create transitions between"
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the model"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "transitions": self.transitions,
            "seed": self.seed,
            "image_urls": self.image_urls,
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
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
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
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
    )
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
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
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
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
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "prompt": self.prompt,
            "fps": self.fps,
            "sync_mode": self.sync_mode,
            "num_refine_inference_steps": self.num_refine_inference_steps,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    video_write_mode: VideoWriteMode = Field(
        default=VideoWriteMode.BALANCED, description="The write mode of the generated video."
    )
    video_output_type: VideoOutputType = Field(
        default=VideoOutputType.X264__MP4, description="The output type of the generated video."
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
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "video_write_mode": self.video_write_mode.value,
            "video_output_type": self.video_output_type.value,
            "prompt": self.prompt,
            "fps": self.fps,
            "sync_mode": self.sync_mode,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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


class MinimaxHailuo2_3FastStandardImageToVideo(FALNode):
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

    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    duration: Duration = Field(
        default=Duration.VALUE_6, description="The duration of the video in seconds."
    )
    prompt: str = Field(
        default="", description="Text prompt for video generation"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt_optimizer": self.prompt_optimizer,
            "duration": self.duration.value,
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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


class MinimaxHailuo2_3StandardImageToVideo(FALNode):
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

    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    duration: Duration = Field(
        default=Duration.VALUE_6, description="The duration of the video in seconds."
    )
    prompt: str = Field(
        default="", description="Text prompt for video generation"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt_optimizer": self.prompt_optimizer,
            "duration": self.duration.value,
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

class MinimaxHailuo2_3FastProImageToVideo(FALNode):
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

    prompt_optimizer: bool = Field(
        default=True, description="Whether to use the model's prompt optimizer"
    )
    prompt: str = Field(
        default="", description="Text prompt for video generation"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt_optimizer": self.prompt_optimizer,
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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


class KlingVideoV2_5TurboStandardImageToVideo(FALNode):
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

    prompt: str = Field(
        default=""
    )
    duration: Duration = Field(
        default=Duration.VALUE_5, description="The duration of the generated video in seconds"
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to be used for the video"
    )
    negative_prompt: str = Field(
        default="blur, distort, and low quality"
    )
    cfg_scale: float = Field(
        default=0.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "negative_prompt": self.negative_prompt,
            "cfg_scale": self.cfg_scale,
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




class Veo3_1FastFirstLastFrameToVideo(FALNode):
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
    first_frame_url: VideoRef = Field(
        default=VideoRef(), description="URL of the first frame of the video"
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    last_frame_url: VideoRef = Field(
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
            "first_frame_url": self.first_frame_url,
            "seed": self.seed,
            "last_frame_url": self.last_frame_url,
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




class Veo3_1FirstLastFrameToVideo(FALNode):
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
    first_frame_url: VideoRef = Field(
        default=VideoRef(), description="URL of the first frame of the video"
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    last_frame_url: VideoRef = Field(
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
            "first_frame_url": self.first_frame_url,
            "seed": self.seed,
            "last_frame_url": self.last_frame_url,
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




class Veo3_1ReferenceToVideo(FALNode):
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
    image_urls: list[str] = Field(
        default=[], description="URLs of the reference images to use for consistent subject appearance"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "resolution": self.resolution.value,
            "auto_fix": self.auto_fix,
            "image_urls": self.image_urls,
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




class Veo3_1FastImageToVideo(FALNode):
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the input image to animate. Should be 720p or higher resolution in 16:9 or 9:16 aspect ratio. If the image is not in 16:9 or 9:16 aspect ratio, it will be cropped to fit."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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




class Veo3_1ImageToVideo(FALNode):
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the input image to animate. Should be 720p or higher resolution in 16:9 or 9:16 aspect ratio. If the image is not in 16:9 or 9:16 aspect ratio, it will be cropped to fit."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to guide the video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "duration": self.duration.value,
            "aspect_ratio": self.aspect_ratio.value,
            "generate_audio": self.generate_audio,
            "auto_fix": self.auto_fix,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to guide video generation."
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "audio_negative_prompt": self.audio_negative_prompt,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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


class VeedFabric1_0Fast(FALNode):
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

    resolution: Resolution = Field(
        default="", description="Resolution"
    )
    audio_url: AudioRef = Field(
        default=AudioRef()
    )
    image_url: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "resolution": self.resolution.value,
            "audio_url": self.audio_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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


class VeedFabric1_0(FALNode):
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

    resolution: Resolution = Field(
        default="", description="Resolution"
    )
    audio_url: AudioRef = Field(
        default=AudioRef()
    )
    image_url: ImageRef = Field(
        default=ImageRef()
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "resolution": self.resolution.value,
            "audio_url": self.audio_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as your avatar"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "audio_url": self.audio_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    audio_url: AudioRef = Field(
        default=AudioRef(), description="The URL of the audio file."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as your avatar"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "audio_url": self.audio_url,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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
    image_url: ImageRef = Field(
        default=ImageRef(), description="URL of the image to use as the first frame"
    )

    async def process(self, context: ProcessingContext) -> VideoRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "sync_mode": self.sync_mode,
            "aspect_ratio": self.aspect_ratio.value,
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_480P, description="Resolution of the generated video (480p, 580p, 720p)."
    )
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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