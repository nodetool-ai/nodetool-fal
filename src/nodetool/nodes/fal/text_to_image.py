from enum import Enum
from pydantic import Field
from nodetool.metadata.types import ColorRef, ImageRef, LoraWeight
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext
from typing import Any, Optional


class ImageSizePreset(str, Enum):
    SQUARE_HD = "square_hd"
    SQUARE = "square"
    PORTRAIT_4_3 = "portrait_4_3"
    PORTRAIT_16_9 = "portrait_16_9"
    LANDSCAPE_4_3 = "landscape_4_3"
    LANDSCAPE_16_9 = "landscape_16_9"


class StylePreset(str, Enum):
    ANY = "any"
    REALISTIC_IMAGE = "realistic_image"
    DIGITAL_ILLUSTRATION = "digital_illustration"
    VECTOR_ILLUSTRATION = "vector_illustration"
    PIXEL_ART = "pixel_art"
    FLAT_ILLUSTRATION = "flat_illustration"
    ISOMETRIC_ILLUSTRATION = "isometric_illustration"
    WATERCOLOR = "watercolor"
    LINE_ART = "line_art"
    PENCIL_DRAWING = "pencil_drawing"
    OIL_PAINTING = "oil_painting"
    ANIME = "anime"
    COMIC_BOOK = "comic_book"
    RETRO = "retro"
    STICKER = "sticker"
    _3D_RENDER = "3d_render"
    CINEMATIC = "cinematic"
    PHOTOGRAPHIC = "photographic"
    CLAY = "clay"
    CUTOUT = "cutout"
    ORIGAMI = "origami"
    PATTERN = "pattern"
    POP_ART = "pop_art"
    RENAISSANCE = "renaissance"
    STUDIO_GHIBLI = "studio_ghibli"
    STORYBOOK = "storybook"


# Define enums for aspect ratio and style
class AspectRatio(str, Enum):
    RATIO_10_16 = "10:16"
    RATIO_16_10 = "16:10"
    RATIO_9_16 = "9:16"
    RATIO_16_9 = "16:9"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_1_1 = "1:1"
    RATIO_1_3 = "1:3"
    RATIO_3_1 = "3:1"
    RATIO_3_2 = "3:2"
    RATIO_2_3 = "2:3"
    RATIO_4_5 = "4:5"
    RATIO_5_4 = "5:4"


class IdeogramStyle(str, Enum):
    AUTO = "auto"
    GENERAL = "general"
    REALISTIC = "realistic"
    DESIGN = "design"
    RENDER_3D = "render_3D"
    ANIME = "anime"


class IdeogramV2(FALNode):
    """
    Ideogram V2 is a state-of-the-art image generation model optimized for commercial and creative use, featuring exceptional typography handling and realistic outputs.
    image, generation, ai, typography, realistic, text-to-image, txt2img

    Use cases:
    - Create commercial artwork and designs
    - Generate realistic product visualizations
    - Design marketing materials with text
    - Produce high-quality illustrations
    - Create brand assets and logos
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the generated image.",
    )
    expand_prompt: bool = Field(
        default=True,
        description="Whether to expand the prompt with MagicPrompt functionality.",
    )
    style: IdeogramStyle = Field(
        default=IdeogramStyle.AUTO, description="The style of the generated image."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to avoid in the generated image."
    )
    seed: int = Field(default=-1, description="Seed for the random number generator.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "expand_prompt": self.expand_prompt,
            "style": self.style.value,
            "negative_prompt": self.negative_prompt,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "style"]


class IdeogramV2Turbo(FALNode):
    """
    Accelerated image generation with Ideogram V2 Turbo. Create high-quality visuals, posters,
    and logos with enhanced speed while maintaining Ideogram's signature quality.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the generated image.",
    )
    expand_prompt: bool = Field(
        default=True,
        description="Whether to expand the prompt with MagicPrompt functionality.",
    )
    style: IdeogramStyle = Field(
        default=IdeogramStyle.AUTO, description="The style of the generated image."
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to avoid in the generated image."
    )
    seed: int = Field(default=-1, description="Seed for the random number generator.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "expand_prompt": self.expand_prompt,
            "style": self.style.value,
            "negative_prompt": self.negative_prompt,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2/turbo",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "style"]


class FluxV1Pro(FALNode):
    """
    FLUX1.1 [pro] is an enhanced version of FLUX.1 [pro], improved image generation capabilities, delivering superior composition, detail, and artistic fidelity compared to its predecessor.
    image, generation, composition, detail, artistic, text-to-image, txt2img

    Use cases:
    - Generate high-fidelity artwork
    - Create detailed illustrations
    - Design complex compositions
    - Produce artistic renderings
    - Generate professional visuals
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="Either a preset size or a custom {width, height} dictionary. Max dimension 14142",
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1,
        le=20,
        description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you.",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, le=50, description="The number of inference steps to perform."
    )
    seed: Optional[int] = Field(
        default=None,
        description="The same seed and the same prompt given to the same version of the model will output the same image every time.",
    )
    num_images: int = Field(
        default=1, ge=1, le=4, description="The number of images to generate (1-4)"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    safety_tolerance: str = Field(
        default="2",
        description="Safety tolerance level (1-6), 1 being strict, 6 being permissive",
    )
    output_format: str = Field(
        default="jpeg", description="Output format (jpeg or png)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "safety_tolerance": self.safety_tolerance,
            "output_format": self.output_format,
        }
        if self.seed is not None:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1.1",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale", "num_images"]


class FluxV1ProUltra(FALNode):
    """
    FLUX1.1 [ultra] is the latest and most advanced version of FLUX.1 [pro],
    featuring cutting-edge improvements in image generation, delivering unparalleled
    composition, detail, and artistic fidelity.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="Either a preset size or a custom {width, height} dictionary. Max dimension 14142",
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1,
        le=20,
        description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you.",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, le=50, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1,
        description="The same seed and the same prompt given to the same version of the model will output the same image every time.",
    )
    num_images: int = Field(
        default=1, ge=1, le=4, description="The number of images to generate (1-4)"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    safety_tolerance: str = Field(
        default="2",
        description="Safety tolerance level (1-6), 1 being strict, 6 being permissive",
    )
    output_format: str = Field(
        default="jpeg", description="Output format (jpeg or png)"
    )
    raw: bool = Field(
        default=False,
        description="Generate less processed, more natural-looking images",
    )
    aspect_ratio: str = Field(
        default="16:9", description="Aspect ratio of the generated image"
    )
    image_prompt_strength: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Strength of the image prompt"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "image_size": self.image_size.value,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "safety_tolerance": self.safety_tolerance,
            "output_format": self.output_format,
            "raw": self.raw,
            "aspect_ratio": self.aspect_ratio,
            "image_prompt_strength": self.image_prompt_strength,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1.1-ultra",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale", "aspect_ratio"]


class RecraftV3(FALNode):
    """
    Recraft V3 is a text-to-image model with the ability to generate long texts, vector art, images in brand style, and much more.
    image, text
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="Either a preset size or a custom {width, height} dictionary. Max dimension 14142",
    )
    style: StylePreset = Field(
        default=StylePreset.REALISTIC_IMAGE,
        description="The style of the generated images. Vector images cost 2X as much.",
    )
    colors: list[ColorRef] = Field(
        default=[], description="An array of preferable colors"
    )
    style_id: str = Field(
        default="", description="The ID of the custom style reference (optional)"
    )

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "style"]

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "style": self.style.value,
            "image_size": self.image_size.value,
            "colors": [color.value for color in self.colors],
            "output_format": "png",
        }

        if self.style_id:
            arguments["style_id"] = self.style_id

        res = await self.submit_request(
            context=context,
            application="fal-ai/recraft-v3",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])


class Switti(FALNode):
    """
    Switti is a scale-wise transformer for fast text-to-image generation that outperforms existing T2I AR models and competes with state-of-the-art T2I diffusion models while being faster than distilled diffusion models.
    image, generation, fast, transformer, efficient, text-to-image, txt2img

    Use cases:
    - Rapid image prototyping
    - Real-time image generation
    - Quick visual concept testing
    - Fast artistic visualization
    - Efficient batch image creation
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    sampling_top_k: int = Field(
        default=400, description="The number of top-k tokens to sample from"
    )
    sampling_top_p: float = Field(
        default=0.95, description="The top-p probability to sample from"
    )
    more_smooth: bool = Field(
        default=True, description="Smoothing with Gumbel softmax sampling"
    )
    more_diverse: bool = Field(default=False, description="More diverse sampling")
    smooth_start_si: int = Field(default=2, description="Smoothing starting scale")
    turn_off_cfg_start_si: int = Field(
        default=8, description="Disable CFG starting scale"
    )
    last_scale_temp: float = Field(
        default=0.1, description="Temperature after disabling CFG"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    guidance_scale: float = Field(
        default=6.0, description="How closely the model should stick to your prompt"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "sampling_top_k": self.sampling_top_k,
            "sampling_top_p": self.sampling_top_p,
            "more_smooth": self.more_smooth,
            "more_diverse": self.more_diverse,
            "smooth_start_si": self.smooth_start_si,
            "turn_off_cfg_start_si": self.turn_off_cfg_start_si,
            "last_scale_temp": self.last_scale_temp,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/switti/1024",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "guidance_scale", "negative_prompt"]


class AuraFlowV03(FALNode):
    """
    AuraFlow v0.3 is an open-source flow-based text-to-image generation model that achieves state-of-the-art results on GenEval.
    image, generation, flow-based, text-to-image, txt2img

    Use cases:
    - Generate high-quality images
    - Create artistic visualizations
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier free guidance scale"
    )
    num_inference_steps: int = Field(
        default=50, ge=1, description="The number of inference steps to take"
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to perform prompt expansion (recommended)"
    )
    seed: int = Field(default=-1, description="The seed to use for generating images")

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "expand_prompt": self.expand_prompt,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/aura-flow",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "guidance_scale", "num_inference_steps"]


class FluxDev(FALNode):
    """
    FLUX.1 [dev] is a 12 billion parameter flow transformer that generates high-quality images from text.
    It is suitable for personal and commercial use.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="Either a preset size or a custom {width, height} dictionary",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5,
        description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt",
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/dev",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class Flux2KleinAcceleration(str, Enum):
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"


class Flux2Klein9bBase(FALNode):
    """
    FLUX.2 [klein] 9B Base: fast text-to-image for real-time apps and high volume. Sub-second speed, high quality (not maximum). Supports up to 4 images.
    flux, text-to-image, klein, black-forest-labs, real-time, fast, high-volume

    Use cases:
    - Real-time and interactive image generation
    - High-volume batch or API workloads
    - Fast iteration and previews
    - Apps needing sub-second latency
    - High-quality output where speed matters more than max quality
    """

    prompt: str = Field(default="", description="The prompt to generate an image from.")
    negative_prompt: str | None = Field(
        default="",
        description="Negative prompt for classifier-free guidance. Describes what to avoid in the image.",
    )
    guidance_scale: float = Field(
        default=5.0,
        ge=0,
        le=20,
        description="Guidance scale for classifier-free guidance.",
    )
    seed: int = Field(
        default=-1,
        description="The seed to use for the generation. If not provided, a random seed will be used.",
    )
    num_inference_steps: int = Field(
        default=28,
        ge=4,
        le=50,
        description="The number of inference steps to perform.",
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the image to generate.",
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=4,
        description="The number of images to generate.",
    )
    acceleration: Flux2KleinAcceleration = Field(
        default=Flux2KleinAcceleration.REGULAR,
        description="The acceleration level to use for image generation.",
    )
    sync_mode: bool = Field(
        default=False,
        description="If True, the media will be returned as a data URI. Output is not stored when this is True.",
    )
    enable_safety_checker: bool = Field(
        default=True,
        description="If set to true, the safety checker will be enabled.",
    )
    output_format: str = Field(
        default="png",
        description="The format of the generated image.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments: dict[str, Any] = {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "image_size": self.image_size.value,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": self.output_format,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt.strip()
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/9b/base",
            arguments=arguments,
        )
        assert "images" in res
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale", "num_images"]


class FluxLora(FALNode):
    """
    FLUX.1 [dev] with LoRAs is a text-to-image model that supports LoRA adaptations, enabling rapid and high-quality image generation with pre-trained LoRA weights for personalization, specific styles, brand identities, and product-specific outputs.
    image, generation, lora, personalization, style-transfer, text-to-image, txt2img

    Use cases:
    - Create brand-specific visuals
    - Generate custom styled images
    - Adapt existing styles to new content
    - Produce personalized artwork
    - Design consistent visual identities
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="Either a preset size or a custom {width, height} dictionary",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5,
        description="The CFG scale to determine how closely the model follows the prompt",
    )
    loras: list[LoraWeight] = Field(
        default=[],
        description="List of LoRA weights to use for image generation",
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
            "loras": [{"path": lora.url, "scale": lora.scale} for lora in self.loras],
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-lora",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "loras"]


class FluxLoraInpainting(FALNode):
    """
    FLUX.1 [dev] Inpainting with LoRAs is a text-to-image model that supports inpainting and LoRA adaptations,
    enabling rapid and high-quality image inpainting using pre-trained LoRA weights for personalization,
    specific styles, brand identities, and product-specific outputs.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image: ImageRef = Field(
        default=ImageRef(), description="The input image to inpaint"
    )
    mask: ImageRef = Field(
        default=ImageRef(),
        description="The mask indicating areas to inpaint (white=inpaint, black=keep)",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5,
        description="The CFG scale to determine how closely the model follows the prompt",
    )
    strength: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="The strength to use for inpainting. 1.0 completely remakes the image while 0.0 preserves the original",
    )
    loras: list[LoraWeight] = Field(
        default=[],
        description="List of LoRA weights to use for image generation",
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        # Convert input image and mask to base64 data URIs
        image_b64 = await context.image_to_base64(self.image)
        mask_b64 = await context.image_to_base64(self.mask)

        arguments = {
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_b64}",
            "mask_url": f"data:image/png;base64,{mask_b64}",
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "strength": self.strength,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
            "loras": [{"path": lora.url, "scale": lora.scale} for lora in self.loras],
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-lora/inpainting",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "mask", "loras"]


class FluxSchnell(FALNode):
    """
    FLUX.1 [schnell] is a 12 billion parameter flow transformer that generates high-quality images
    from text in 1 to 4 steps, suitable for personal and commercial use.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="Either a preset size or a custom {width, height} dictionary",
    )
    num_inference_steps: int = Field(
        default=4, ge=1, description="The number of inference steps to perform"
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/schnell",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]


class FluxSubject(FALNode):
    """
    FLUX.1 Subject is a super fast endpoint for the FLUX.1 [schnell] model with subject input capabilities, enabling rapid and high-quality image generation for personalization, specific styles, brand identities, and product-specific outputs.
    image, generation, subject-driven, personalization, fast, text-to-image, txt2img

    Use cases:
    - Create variations of existing subjects
    - Generate personalized product images
    - Design brand-specific visuals
    - Produce custom character artwork
    - Create subject-based illustrations
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image: ImageRef = Field(default=ImageRef(), description="The image of the subject")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="Either a preset size or a custom {width, height} dictionary",
    )
    num_inference_steps: int = Field(
        default=8, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5,
        description="The CFG scale to determine how closely the model follows the prompt",
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        # Convert input image to base64 data URI if it's not already a URL
        image_base64 = await context.image_to_base64(self.image)
        image_url = f"data:image/png;base64,{image_base64}"

        arguments = {
            "prompt": self.prompt,
            "image_url": image_url,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-subject",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "image_size"]


class FluxV1ProNew(FALNode):
    """
    FLUX.1 [pro] new is an accelerated version of FLUX.1 [pro], maintaining professional-grade
    image quality while delivering significantly faster generation speeds.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="Either a preset size or a custom {width, height} dictionary",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1,
        le=20,
        description="The CFG scale to determine how closely the model follows the prompt",
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    safety_tolerance: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Safety tolerance level (1=strict, 6=permissive)",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_images": self.num_images,
            "safety_tolerance": str(self.safety_tolerance),
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/new",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class SanaV1(FALNode):
    """
    Sana can synthesize high-resolution, high-quality images with strong text-image alignment at a remarkably fast speed, with the ability to generate 4K images in less than a second.
    image, generation, high-resolution, fast, text-alignment, text-to-image, txt2img

    Use cases:
    - Generate 4K quality images
    - Create high-resolution artwork
    - Produce rapid visual prototypes
    - Design detailed illustrations
    - Generate precise text-aligned visuals
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=18, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=5.0, description="How closely the model should stick to your prompt"
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/sana",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class OmniGenV1(FALNode):
    """
    OmniGen is a unified image generation model that can generate a wide range of images from multi-modal prompts. It can be used for various tasks such as Image Editing, Personalized Image Generation, Virtual Try-On, Multi Person Generation and more!
    image, generation, multi-modal, editing, personalization, text-to-image, txt2img

    Use cases:
    - Edit and modify existing images
    - Create personalized visual content
    - Generate virtual try-on images
    - Create multi-person compositions
    - Combine multiple images creatively
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    input_image_1: ImageRef = Field(
        default=ImageRef(), description="The first input image to use for generation"
    )
    input_image_2: ImageRef = Field(
        default=ImageRef(), description="The second input image to use for generation"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=50, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.0, description="How closely the model should stick to your prompt"
    )
    img_guidance_scale: float = Field(
        default=1.6,
        description="How closely the model should stick to your input image",
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_urls = []
        if self.input_image_1.is_set():
            image_1_base64 = await context.image_to_base64(self.input_image_1)
            image_urls.append(f"data:image/png;base64,{image_1_base64}")
        if self.input_image_2.is_set():
            image_2_base64 = await context.image_to_base64(self.input_image_2)
            image_urls.append(f"data:image/png;base64,{image_2_base64}")

        arguments = {
            "prompt": self.prompt,
            "input_image_urls": image_urls,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "img_guidance_scale": self.img_guidance_scale,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/omnigen-v1",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "input_image_1", "image_size"]


class StableDiffusionV35Large(FALNode):
    """
    Stable Diffusion 3.5 Large is a Multimodal Diffusion Transformer (MMDiT) text-to-image model that features
    improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely the model should stick to your prompt"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "image_size": self.image_size.value,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-diffusion-v35-large",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "guidance_scale"]


class Recraft20B(FALNode):
    """
    Recraft 20B is a new and affordable text-to-image model that delivers state-of-the-art results.
     image, generation, efficient, text-to-image, txt2img

    Use cases:
    - Generate cost-effective visuals
    - Create high-quality images
    - Produce professional artwork
    - Design marketing materials
    - Generate commercial content
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="Either a preset size or a custom {width, height} dictionary",
    )
    style: StylePreset = Field(
        default=StylePreset.REALISTIC_IMAGE,
        description="The style of the generated images. Vector images cost 2X as much.",
    )
    colors: list[ColorRef] = Field(
        default=[], description="An array of preferable colors"
    )
    style_id: str = Field(
        default="", description="The ID of the custom style reference (optional)"
    )

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "style"]

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "style": self.style.value,
            "image_size": self.image_size.value,
            "colors": [color.value for color in self.colors],
            "output_format": "png",
        }

        if self.style_id:
            arguments["style_id"] = self.style_id

        res = await self.submit_request(
            context=context,
            application="fal-ai/recraft-20b",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])


class BriaV1(FALNode):
    """
    Bria's Text-to-Image model, trained exclusively on licensed data for safe and risk-free commercial use.
    Features exceptional image quality and commercial licensing safety.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to avoid certain elements in the generated image",
    )
    num_images: int = Field(
        default=4,
        ge=1,
        description="How many images to generate. When using guidance, value is set to 1",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the image. Ignored when guidance is used",
    )
    num_inference_steps: int = Field(
        default=30,
        ge=1,
        description="The number of iterations for refining the generated image",
    )
    guidance_scale: float = Field(
        default=5.0,
        description="How closely the model should stick to your prompt (CFG scale)",
    )
    prompt_enhancement: bool = Field(
        default=False,
        description="When true, enhances the prompt with more descriptive variations",
    )
    medium: str = Field(
        default="", description="Optional medium specification ('photography' or 'art')"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "prompt_enhancement": self.prompt_enhancement,
            "output_format": "png",
        }
        if self.medium:
            arguments["medium"] = self.medium
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/text-to-image/base",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "aspect_ratio"]


class BriaV1Fast(FALNode):
    """
    Bria's Text-to-Image model with perfect harmony of latency and quality.
    Trained exclusively on licensed data for safe and risk-free commercial use.
    Features faster inference times while maintaining high image quality.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to avoid certain elements in the generated image",
    )
    num_images: int = Field(
        default=4,
        ge=1,
        description="How many images to generate. When using guidance, value is set to 1",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the image. Ignored when guidance is used",
    )
    num_inference_steps: int = Field(
        default=8,
        ge=1,
        description="The number of iterations for refining the generated image",
    )
    guidance_scale: float = Field(
        default=5.0,
        description="How closely the model should stick to your prompt (CFG scale)",
    )
    prompt_enhancement: bool = Field(
        default=False,
        description="When true, enhances the prompt with more descriptive variations",
    )
    medium: str = Field(
        default="", description="Optional medium specification ('photography' or 'art')"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "prompt_enhancement": self.prompt_enhancement,
            "output_format": "png",
        }
        if self.medium:
            arguments["medium"] = self.medium
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/text-to-image/fast",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "aspect_ratio"]


class BriaV1HD(FALNode):
    """
    Bria's Text-to-Image model for HD images. Trained exclusively on licensed data for safe and risk-free commercial use. Features exceptional image quality and commercial licensing safety.
    image, generation, hd, text-to-image, txt2img

    Use cases:
    - Create commercial marketing materials
    - Generate licensed artwork
    - Produce high-definition visuals
    - Design professional content
    - Create legally safe visual assets
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="The negative prompt to avoid certain elements in the generated image",
    )
    num_images: int = Field(
        default=4,
        ge=1,
        description="How many images to generate. When using guidance, value is set to 1",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the image. Ignored when guidance is used",
    )
    num_inference_steps: int = Field(
        default=30,
        ge=1,
        description="The number of iterations for refining the generated image",
    )
    guidance_scale: float = Field(
        default=5.0,
        description="How closely the model should stick to your prompt (CFG scale)",
    )
    prompt_enhancement: bool = Field(
        default=False,
        description="When true, enhances the prompt with more descriptive variations",
    )
    medium: str = Field(
        default="", description="Optional medium specification ('photography' or 'art')"
    )
    seed: int = Field(default=-1, description="The seed to use for generating images")

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "prompt_enhancement": self.prompt_enhancement,
            "output_format": "png",
        }
        if self.medium:
            arguments["medium"] = self.medium
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/text-to-image/hd",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "aspect_ratio"]


class FluxGeneral(FALNode):
    """
    FLUX.1 [dev] with Controlnets and Loras is a versatile text-to-image model that supports multiple AI extensions including LoRA, ControlNet conditioning, and IP-Adapter integration, enabling comprehensive control over image generation through various guidance methods.
    image, generation, controlnet, lora, ip-adapter, text-to-image, txt2img

    Use cases:
    - Create controlled image generations
    - Apply multiple AI extensions
    - Generate guided visual content
    - Produce customized artwork
    - Design with precise control
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=28, ge=1, le=50, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1,
        le=20,
        description="How closely the model should stick to your prompt (CFG scale)",
    )
    real_cfg_scale: float = Field(
        default=3.5, description="Classical CFG scale as in SD1.5, SDXL, etc."
    )
    use_real_cfg: bool = Field(
        default=False,
        description="Uses classical CFG. Increases generation times and price when true",
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    reference_strength: float = Field(
        default=0.65,
        description="Strength of reference_only generation. Only used if a reference image is provided",
    )
    reference_end: float = Field(
        default=1.0,
        description="The percentage of total timesteps when reference guidance should end",
    )
    base_shift: float = Field(
        default=0.5, description="Base shift for the scheduled timesteps"
    )
    max_shift: float = Field(
        default=1.15, description="Max shift for the scheduled timesteps"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "real_cfg_scale": self.real_cfg_scale,
            "use_real_cfg": self.use_real_cfg,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "reference_strength": self.reference_strength,
            "reference_end": self.reference_end,
            "base_shift": self.base_shift,
            "max_shift": self.max_shift,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-general",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class StableDiffusionV3Medium(FALNode):
    """
    Stable Diffusion 3 Medium (Text to Image) is a Multimodal Diffusion Transformer (MMDiT) model
    that improves image quality, typography, prompt understanding, and efficiency.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate an image from"
    )
    prompt_expansion: bool = Field(
        default=False,
        description="If set to true, prompt will be upsampled with more details",
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=5.0,
        description="How closely the model should stick to your prompt (CFG scale)",
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "prompt_expansion": self.prompt_expansion,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-diffusion-v3-medium",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "guidance_scale"]


class FastSDXL(FALNode):
    """
    Fast SDXL is a high-performance text-to-image model that runs SDXL at exceptional speeds
    while maintaining high-quality output.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=25, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=7.5,
        description="How closely the model should stick to your prompt (CFG scale)",
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    expand_prompt: bool = Field(
        default=False,
        description="If true, the prompt will be expanded with additional prompts",
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The list of LoRA weights to use"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "expand_prompt": self.expand_prompt,
            "output_format": "png",
            "loras": [{"path": lora.url, "scale": lora.scale} for lora in self.loras],
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-sdxl",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "guidance_scale"]


class LoraModel(str, Enum):
    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    SD_1_5 = "runwayml/stable-diffusion-v1-5"
    SD_2_1 = "stabilityai/stable-diffusion-2-1"
    ANYTHING_V5 = "gsdf/Anything-V5.0"
    DREAMSHAPER_8 = "lykon/dreamshaper-8"
    DELIBERATE_V3 = "XpucT/Deliberate_v3"
    REALISTIC_VISION_5_1 = "SG161222/Realistic_Vision_V5.1_noVAE"


class FluxLoraTTI(FALNode):
    """
    FLUX.1 with LoRAs is a text-to-image model that supports LoRA adaptations,
    enabling high-quality image generation with customizable LoRA weights for
    personalization, specific styles, and brand identities.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    model_name: LoraModel = Field(
        default=LoraModel.SDXL_BASE, description="The base model to use for generation"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=30, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=7.5, description="How closely the model should stick to your prompt"
    )
    loras: list[LoraWeight] = Field(
        default=[],
        description="List of LoRA weights to use for image generation",
    )
    prompt_weighting: bool = Field(
        default=True,
        description="If true, prompt weighting syntax will be used and 77 token limit lifted",
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "model_name": self.model_name.value,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "prompt_weighting": self.prompt_weighting,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
            "loras": [{"path": lora.url, "scale": lora.scale} for lora in self.loras],
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/lora",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "model_name", "loras"]


class StableCascade(FALNode):
    """
    Stable Cascade is a state-of-the-art text-to-image model that generates images on a smaller & cheaper
    latent space while maintaining high quality output.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    first_stage_steps: int = Field(
        default=20, description="Number of steps to run the first stage for"
    )
    second_stage_steps: int = Field(
        default=10, description="Number of steps to run the second stage for"
    )
    guidance_scale: float = Field(
        default=4.0, description="How closely the model should stick to your prompt"
    )
    second_stage_guidance_scale: float = Field(
        default=4.0, description="Guidance scale for the second stage of generation"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "first_stage_steps": self.first_stage_steps,
            "second_stage_steps": self.second_stage_steps,
            "guidance_scale": self.guidance_scale,
            "second_stage_guidance_scale": self.second_stage_guidance_scale,
            "image_size": self.image_size.value,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-cascade",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "guidance_scale"]


class AspectRatioLuma(str, Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_21_9 = "21:9"
    RATIO_9_21 = "9:21"


class LumaPhoton(FALNode):
    """
    Luma Photon is a creative and personalizable text-to-image model that brings a step-function
    change in the cost of high-quality image generation, optimized for creatives.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    aspect_ratio: AspectRatioLuma = Field(
        default=AspectRatioLuma.RATIO_1_1,
        description="The aspect ratio of the generated image",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "output_format": "png",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-photon",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio"]


class LumaPhotonFlash(FALNode):
    """
    Luma Photon Flash is the most creative, personalizable, and intelligent visual model for creatives,
    bringing a step-function change in the cost of high-quality image generation with faster inference times.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    aspect_ratio: AspectRatioLuma = Field(
        default=AspectRatioLuma.RATIO_1_1,
        description="The aspect ratio of the generated image",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "output_format": "png",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/luma-photon/flash",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio"]


class ModelNameEnum(str, Enum):
    SDXL_TURBO = "stabilityai/sdxl-turbo"
    SD_TURBO = "stabilityai/sd-turbo"


class FastTurboDiffusion(FALNode):
    """
    Fast Turbo Diffusion runs SDXL at exceptional speeds while maintaining high-quality output.
    Supports both SDXL Turbo and SD Turbo models for ultra-fast image generation.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    model_name: ModelNameEnum = Field(
        default=ModelNameEnum.SDXL_TURBO, description="The name of the model to use"
    )
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=2, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=1.0, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    expand_prompt: bool = Field(
        default=False,
        description="If true, the prompt will be expanded with additional prompts",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "model_name": self.model_name.value,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "expand_prompt": self.expand_prompt,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-turbo-diffusion",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "model_name", "guidance_scale"]


class ModelNameFastLCM(str, Enum):
    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    SD_1_5 = "runwayml/stable-diffusion-v1-5"


class SafetyCheckerVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


class FastLCMDiffusion(FALNode):
    """
    Fast Latent Consistency Models (v1.5/XL) Text to Image runs SDXL at the speed of light,
    enabling rapid and high-quality image generation.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    model_name: ModelNameFastLCM = Field(
        default=ModelNameFastLCM.SDXL_BASE, description="The name of the model to use"
    )
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=6, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=1.5, description="How closely the model should stick to your prompt"
    )
    sync_mode: bool = Field(
        default=True,
        description="If true, wait for image generation and upload before returning",
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1,
        description="The version of the safety checker to use",
    )
    expand_prompt: bool = Field(
        default=False,
        description="If true, the prompt will be expanded with additional prompts",
    )
    guidance_rescale: float = Field(
        default=0.0, description="The rescale factor for the CFG"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "model_name": self.model_name.value,
            "negative_prompt": self.negative_prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "sync_mode": self.sync_mode,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "safety_checker_version": self.safety_checker_version.value,
            "expand_prompt": self.expand_prompt,
            "output_format": "png",
        }
        if self.guidance_rescale > 0:
            arguments["guidance_rescale"] = self.guidance_rescale
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-lcm-diffusion",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "model_name", "guidance_scale"]


class FastLightningSDXL(FALNode):
    """
    Stable Diffusion XL Lightning Text to Image runs SDXL at the speed of light, enabling
    ultra-fast high-quality image generation.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=4,
        ge=1,
        le=8,
        description="The number of inference steps to perform (1, 2, 4, or 8)",
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    expand_prompt: bool = Field(
        default=False,
        description="If true, the prompt will be expanded with additional prompts",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "expand_prompt": self.expand_prompt,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-lightning-sdxl",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]


class HyperSDXL(FALNode):
    """
    Hyper SDXL is a hyper-charged version of SDXL that delivers exceptional performance and creativity
    while maintaining high-quality output and ultra-fast generation speeds.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=1,
        ge=1,
        le=4,
        description="The number of inference steps to perform (1, 2, or 4)",
    )
    sync_mode: bool = Field(
        default=True,
        description="If true, wait for image generation and upload before returning",
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    expand_prompt: bool = Field(
        default=False,
        description="If true, the prompt will be expanded with additional prompts",
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "sync_mode": self.sync_mode,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "expand_prompt": self.expand_prompt,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/hyper-sdxl",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]


class PlaygroundV25(FALNode):
    """
    Playground v2.5 is a state-of-the-art open-source model that excels in aesthetic quality
    for text-to-image generation.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=30, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=7.5, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/playground-v25",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class ModelNameLCM(str, Enum):
    SDXL = "sdxl"
    SD_1_5 = "sdv1-5"


class LCMDiffusion(FALNode):
    """
    Latent Consistency Models (SDXL & SDv1.5) Text to Image produces high-quality images
    with minimal inference steps.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    model: ModelNameLCM = Field(
        default=ModelNameLCM.SD_1_5,
        description="The model to use for generating the image",
    )
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE, description="The size of the generated image"
    )
    num_inference_steps: int = Field(
        default=4, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=1.0, description="How closely the model should stick to your prompt"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "model": self.model.value,
            "negative_prompt": self.negative_prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checks": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/lcm",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "model", "guidance_scale"]


class PerformanceEnum(str, Enum):
    SPEED = "Speed"
    QUALITY = "Quality"
    EXTREME_SPEED = "Extreme Speed"
    LIGHTNING = "Lightning"


class ControlTypeEnum(str, Enum):
    IMAGE_PROMPT = "ImagePrompt"
    PYRA_CANNY = "PyraCanny"
    CPDS = "CPDS"
    FACE_SWAP = "FaceSwap"


class RefinerModelEnum(str, Enum):
    NONE = "None"
    REALISTIC_VISION = "realisticVisionV60B1_v51VAE.safetensors"


class Fooocus(FALNode):
    """
    Fooocus is a text-to-image model with default parameters and automated optimizations
    for quality improvements.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    styles: list[str] = Field(
        default=["Fooocus Enhance", "Fooocus V2", "Fooocus Sharp"],
        description="The styles to apply to the generated image",
    )
    performance: PerformanceEnum = Field(
        default=PerformanceEnum.EXTREME_SPEED,
        description="You can choose Speed or Quality",
    )
    guidance_scale: float = Field(
        default=4.0, description="How closely the model should stick to your prompt"
    )
    sharpness: float = Field(
        default=2.0, description="Higher value means image and texture are sharper"
    )
    aspect_ratio: str = Field(
        default="1024x1024",
        description="The size of the generated image (must be multiples of 8)",
    )
    loras: list[LoraWeight] = Field(
        default=[],
        description="Up to 5 LoRAs that will be merged for generation",
    )
    refiner_model: RefinerModelEnum = Field(
        default=RefinerModelEnum.NONE,
        description="Refiner model to use (SDXL or SD 1.5)",
    )
    refiner_switch: float = Field(
        default=0.8,
        description="Switch point for refiner (0.4 for SD1.5 realistic, 0.667 for SD1.5 anime, 0.8 for XL)",
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    control_image: ImageRef = Field(
        default=ImageRef(), description="Reference image for generation"
    )
    control_type: ControlTypeEnum = Field(
        default=ControlTypeEnum.PYRA_CANNY, description="The type of image control"
    )
    control_image_weight: float = Field(
        default=1.0, description="Strength of the control image influence"
    )
    control_image_stop_at: float = Field(
        default=1.0, description="When to stop applying control image influence"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If false, the safety checker will be disabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "styles": self.styles,
            "performance": self.performance.value,
            "guidance_scale": self.guidance_scale,
            "sharpness": self.sharpness,
            "aspect_ratio": self.aspect_ratio,
            "refiner_model": self.refiner_model.value,
            "refiner_switch": self.refiner_switch,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
            "loras": [{"path": lora.url, "scale": lora.scale} for lora in self.loras],
        }

        if self.seed != -1:
            arguments["seed"] = self.seed

        if self.control_image.is_set():
            control_image_base64 = await context.image_to_base64(self.control_image)
            arguments["control_image_url"] = (
                f"data:image/png;base64,{control_image_base64}"
            )
            arguments["control_type"] = self.control_type.value
            arguments["control_image_weight"] = self.control_image_weight
            arguments["control_image_stop_at"] = self.control_image_stop_at

        res = await self.submit_request(
            context=context,
            application="fal-ai/fooocus",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "styles"]


class IllusionDiffusion(FALNode):
    """
    Illusion Diffusion is a model that creates illusions conditioned on an input image.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    image: ImageRef = Field(
        default=ImageRef(),
        description="Input image URL for conditioning the generation",
    )
    guidance_scale: float = Field(
        default=7.5, description="How closely the model should stick to your prompt"
    )
    num_inference_steps: int = Field(
        default=40, ge=1, description="The number of inference steps to perform"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "image_size": self.image_size.value,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/illusion-diffusion",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "guidance_scale"]


class FastSDXLControlNetCanny(FALNode):
    """
    Fast SDXL ControlNet Canny is a model that generates images using ControlNet with SDXL.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    control_image: ImageRef = Field(
        default=ImageRef(), description="The control image to use for generation"
    )
    negative_prompt: str = Field(
        default="",
        description="Use it to address details that you don't want in the image",
    )
    guidance_scale: float = Field(
        default=7.5, description="How closely the model should stick to your prompt"
    )
    num_inference_steps: int = Field(
        default=25, ge=1, description="The number of inference steps to perform"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD, description="The size of the generated image"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image_base64 = await context.image_to_base64(self.control_image)
        arguments = {
            "prompt": self.prompt,
            "control_image_url": f"data:image/png;base64,{control_image_base64}",
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "image_size": self.image_size.value,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-sdxl-controlnet-canny",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "control_image", "guidance_scale"]


class FluxDevImageToImage(FALNode):
    """
    FLUX.1 [dev] Image-to-Image is a high-performance endpoint that enables rapid transformation
    of existing images, delivering high-quality style transfers and image modifications with
    the core FLUX capabilities.
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image: ImageRef = Field(
        default=ImageRef(), description="The input image to transform"
    )
    strength: float = Field(
        default=0.95,
        description="The strength of the initial image. Higher strength values are better for this model",
    )
    num_inference_steps: int = Field(
        default=40, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely the model should stick to your prompt"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        # Convert input image to base64 data URI
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "prompt": self.prompt,
            "image_url": f"data:image/png;base64,{image_base64}",
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/dev/image-to-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image", "strength"]


class DiffusionEdge(FALNode):
    """
    Diffusion Edge is a diffusion-based high-quality edge detection model that generates
    edge maps from input images.
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The input image to detect edges from"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        # Convert input image to base64 data URI
        image_base64 = await context.image_to_base64(self.image)

        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
            "output_format": "png",
        }

        res = await self.submit_request(
            context=context,
            application="fal-ai/diffusion-edge",
            arguments=arguments,
        )
        assert res["image"] is not None
        return ImageRef(uri=res["image"]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["image"]


class Imagen4Preview(FALNode):
    """
    Imagen 4 Preview is the next iteration of Google's Imagen series, offering
    high quality text-to-image generation with strong prompt adherence and
    improved realism.
    image, generation, google, text-to-image, txt2img

    Use cases:
    - Generate photorealistic artwork and designs
    - Create marketing and product visuals
    - Produce concept art or storyboards
    - Explore creative ideas with high fidelity
    - Rapid prototyping of imagery
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="",
        description="Elements to avoid in the generated image",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the generated image",
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=50, ge=1, description="The number of inference steps to perform"
    )
    guidance_scale: float = Field(
        default=5.0, description="How closely the model should follow the prompt"
    )
    num_images: int = Field(
        default=1, ge=1, description="The number of images to generate"
    )
    seed: int = Field(
        default=-1,
        description="The same seed and prompt will output the same image every time",
    )
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "image_size": self.image_size.value,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/imagen4/preview",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "guidance_scale"]


class IdeogramV3(FALNode):
    """
    Ideogram V3 is the latest generation text-to-image model with enhanced typography and photorealistic outputs.
    image, generation, typography, realistic, text-to-image, txt2img, ideogram

    Use cases:
    - Create professional marketing materials with text
    - Generate logos and brand assets
    - Design posters and advertisements
    - Produce photorealistic product images
    - Create typography-heavy artwork
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the generated image",
    )
    style: IdeogramStyle = Field(
        default=IdeogramStyle.AUTO, description="The style of the generated image"
    )
    expand_prompt: bool = Field(
        default=True,
        description="Whether to expand the prompt with MagicPrompt functionality",
    )
    negative_prompt: str = Field(
        default="", description="A negative prompt to avoid in the generated image"
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "style": self.style.value,
            "expand_prompt": self.expand_prompt,
            "output_format": "png",
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v3",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "style"]


class GPTImage1(FALNode):
    """
    OpenAI's GPT Image 1 model for generating images from text prompts with high quality and creative outputs.
    image, generation, openai, gpt, text-to-image, txt2img, creative

    Use cases:
    - Generate creative illustrations
    - Create concept art and designs
    - Produce marketing visuals
    - Design digital artwork
    - Create custom graphics
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    size: str = Field(
        default="1024x1024",
        description="The size of the generated image (e.g., 1024x1024, 1792x1024, 1024x1792)",
    )
    quality: str = Field(
        default="auto",
        description="The quality of the generated image (auto, high, medium, low)",
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "size": self.size,
            "quality": self.quality,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/gpt-image-1/text-to-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "size", "quality"]


class Gemini25FlashImage(FALNode):
    """
    Google's Gemini 2.5 Flash model for fast high-quality image generation from text.
    image, generation, google, gemini, text-to-image, txt2img, fast

    Use cases:
    - Generate images quickly
    - Create visual content at scale
    - Produce concept visualizations
    - Design marketing materials
    - Create educational illustrations
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the generated image",
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/gemini-25-flash-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio"]


class Flux2Turbo(FALNode):
    """
    FLUX 2 Turbo is a fast text-to-image model delivering high-quality results with reduced generation time.
    image, generation, flux, fast, text-to-image, txt2img, turbo

    Use cases:
    - Rapid image prototyping
    - High-volume image generation
    - Quick concept visualization
    - Fast design iterations
    - Real-time creative workflows
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=4, ge=1, le=12, description="The number of inference steps"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/turbo",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]


class Flux2Flash(FALNode):
    """
    FLUX 2 Flash is an ultra-fast text-to-image model optimized for speed while maintaining quality.
    image, generation, flux, ultra-fast, text-to-image, txt2img, flash

    Use cases:
    - Real-time image generation
    - Interactive creative tools
    - Rapid prototyping
    - High-throughput applications
    - Quick visual exploration
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=4, ge=1, le=8, description="The number of inference steps"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/flash",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]


class HunyuanImageV3(FALNode):
    """
    Hunyuan Image V3 is Tencent's advanced text-to-image model with exceptional detail and artistic quality.
    image, generation, hunyuan, tencent, text-to-image, txt2img, artistic

    Use cases:
    - Create detailed digital artwork
    - Generate photorealistic images
    - Produce high-quality illustrations
    - Design creative visuals
    - Create artistic compositions
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=30, ge=1, description="The number of inference steps"
    )
    guidance_scale: float = Field(
        default=5.0, description="How closely to follow the prompt"
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-image/v3/text-to-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class HunyuanImageSizePreset(str, Enum):
    AUTO = "auto"
    SQUARE_HD = "square_hd"
    SQUARE = "square"
    PORTRAIT_4_3 = "portrait_4_3"
    PORTRAIT_16_9 = "portrait_16_9"
    LANDSCAPE_4_3 = "landscape_4_3"
    LANDSCAPE_16_9 = "landscape_16_9"


class HunyuanImageV3Instruct(FALNode):
    """
    Hunyuan Image V3 Instruct with internal reasoning capabilities for advanced text-to-image generation.
    image, generation, hunyuan, tencent, instruct, reasoning, text-to-image, txt2img, advanced

    Use cases:
    - Generate highly detailed images with reasoning
    - Create complex compositions with multiple elements
    - Produce photorealistic images with fine control
    - Generate artistic images with advanced understanding
    - Create images with complex prompt interpretation
    """

    prompt: str = Field(
        default="", description="The text prompt to generate an image from"
    )
    image_size: HunyuanImageSizePreset = Field(
        default=HunyuanImageSizePreset.AUTO,
        description="The desired size of the generated image. If auto, size is determined by the model",
    )
    num_images: int = Field(
        default=1, ge=1, le=4, description="The number of images to generate"
    )
    guidance_scale: float = Field(
        default=3.5,
        ge=1.0,
        le=20.0,
        description="How closely to follow the prompt (higher = stricter adherence)",
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker to filter unsafe content"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_images": self.num_images,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
        }
        if self.seed != -1:
            arguments["seed"] = self.seed

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-image/v3/instruct/text-to-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class CogView4(FALNode):
    """
    CogView4 is a powerful text-to-image model with strong understanding and generation capabilities.
    image, generation, cogview, text-to-image, txt2img, ai

    Use cases:
    - Generate creative images from descriptions
    - Create concept art
    - Design visual content
    - Produce illustrations
    - Create artistic images
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=50, ge=1, description="The number of inference steps"
    )
    guidance_scale: float = Field(
        default=7.0, description="How closely to follow the prompt"
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/cogview4",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class Kolors(FALNode):
    """
    Kolors is an advanced text-to-image model with excellent color reproduction and artistic style.
    image, generation, kolors, text-to-image, txt2img, artistic, color

    Use cases:
    - Create vibrant colorful artwork
    - Generate stylized illustrations
    - Design visually striking content
    - Produce artistic images
    - Create color-rich visuals
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=25, ge=1, description="The number of inference steps"
    )
    guidance_scale: float = Field(
        default=5.0, description="How closely to follow the prompt"
    )
    seed: int = Field(default=-1, description="Seed for reproducible generation")
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": "png",
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/kolors",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class Seedream45(FALNode):
    """
    ByteDance Seedream V4.5 is a state-of-the-art text-to-image model with exceptional detail and artistic quality.
    image, generation, bytedance, seedream, text-to-image, txt2img, artistic

    Use cases:
    - Create high-quality digital art
    - Generate photorealistic images
    - Design marketing visuals
    - Produce detailed illustrations
    - Create professional graphics
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="The size of the generated image",
    )
    guidance_scale: float = Field(
        default=5.0, description="How closely to follow the prompt"
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedream/v4.5/text-to-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class Reve(FALNode):
    """
    Reve is a creative text-to-image model with unique artistic capabilities and style.
    image, generation, reve, text-to-image, txt2img, artistic, creative

    Use cases:
    - Create artistic illustrations
    - Generate unique visual content
    - Design creative artwork
    - Produce stylized images
    - Create imaginative visuals
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, ge=1, description="The number of inference steps"
    )
    guidance_scale: float = Field(
        default=3.5, description="How closely to follow the prompt"
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/reve/text-to-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]


class Imagen3(FALNode):
    """
    Google Imagen 3 is a state-of-the-art text-to-image model with exceptional quality and understanding.
    image, generation, google, imagen, text-to-image, txt2img, high-quality

    Use cases:
    - Generate photorealistic images
    - Create professional marketing content
    - Design visual assets
    - Produce high-quality illustrations
    - Create detailed artwork
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1,
        description="The aspect ratio of the generated image",
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/imagen3",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio"]


class QwenImageMax(FALNode):
    """
    Qwen Image Max is Alibaba's advanced text-to-image model with exceptional quality and detail.
    image, generation, qwen, alibaba, text-to-image, txt2img, high-quality

    Use cases:
    - Generate detailed images
    - Create professional visuals
    - Design marketing content
    - Produce high-quality artwork
    - Create commercial graphics
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="The size of the generated image",
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-max/text-to-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size"]


class ZImageTurbo(FALNode):
    """
    Z-Image Turbo is a fast text-to-image model optimized for quick generation with good quality.
    image, generation, z-image, text-to-image, txt2img, fast, turbo

    Use cases:
    - Rapid image generation
    - Quick prototyping
    - High-volume content creation
    - Fast design iterations
    - Real-time applications
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.SQUARE_HD,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=4, ge=1, description="The number of inference steps"
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/turbo",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]


class ZImageAcceleration(str, Enum):
    NONE = "none"
    REGULAR = "regular"
    HIGH = "high"


class OutputFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


class ZImageBase(FALNode):
    """
    Generate high-quality images using the Z-Image Base model. Provides detailed image generation with multiple acceleration and quality options.
    image, generation, text-to-image, z-image, detailed, quality

    Use cases:
    - Generate detailed images from text prompts
    - Create high-quality artwork
    - Produce professional illustrations
    - Generate concept art
    - Create visual content for projects
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    num_images: int = Field(default=1, description="The number of images to generate")
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image"
    )
    acceleration: ZImageAcceleration = Field(
        default=ZImageAcceleration.REGULAR,
        description="The acceleration level to use",
    )
    guidance_scale: float = Field(
        default=4.0, description="The guidance scale to use for image generation"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use for image generation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": self.output_format.value,
            "acceleration": self.acceleration.value,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/base",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]


class ZImageBaseLora(FALNode):
    """
    Generate high-quality images using the Z-Image Base model with LoRA support. Allows fine-tuned image generation with custom LoRA models.
    image, generation, text-to-image, z-image, lora, fine-tuning

    Use cases:
    - Generate images with custom LoRA fine-tuning
    - Create specialized style images
    - Produce character-consistent artwork
    - Generate images matching specific aesthetics
    - Create brand-aligned visual content
    """

    prompt: str = Field(default="", description="The prompt to generate an image from")
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3,
        description="The size of the generated image",
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform"
    )
    seed: int = Field(
        default=-1, description="The same seed will output the same image every time"
    )
    num_images: int = Field(default=1, description="The number of images to generate")
    enable_safety_checker: bool = Field(
        default=True, description="If true, the safety checker will be enabled"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image"
    )
    acceleration: ZImageAcceleration = Field(
        default=ZImageAcceleration.REGULAR,
        description="The acceleration level to use",
    )
    guidance_scale: float = Field(
        default=4.0, description="The guidance scale to use for image generation"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use for image generation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size.value,
            "num_inference_steps": self.num_inference_steps,
            "num_images": self.num_images,
            "enable_safety_checker": self.enable_safety_checker,
            "output_format": self.output_format.value,
            "acceleration": self.acceleration.value,
            "guidance_scale": self.guidance_scale,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/base/lora",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]


class Kling3ImageAspectRatio(Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_1_1 = "1:1"
    RATIO_3_2 = "3:2"
    RATIO_2_3 = "2:3"
    RATIO_21_9 = "21:9"


class Kling3ImageResolution(Enum):
    RES_1K = "1K"
    RES_2K = "2K"


class KlingImage3TextToImage(FALNode):
    """
    Generate high-quality images from text prompts using Kling Image 3.0.
    Supports sharp outputs up to 2K resolution with strong prompt adherence.
    image, generation, kling, v3, text-to-image, high-resolution

    Use cases:
    - Create high-resolution images from descriptions
    - Generate 2K quality artwork
    - Produce photorealistic images
    - Create detailed visual content
    - Generate images with strong prompt adherence
    """

    prompt: str = Field(
        default="",
        description="The text prompt describing the desired image (max 2500 characters)",
    )
    negative_prompt: str = Field(
        default="", description="What to avoid in the generated image"
    )
    aspect_ratio: Kling3ImageAspectRatio = Field(
        default=Kling3ImageAspectRatio.RATIO_16_9,
        description="The aspect ratio of the generated image",
    )
    resolution: Kling3ImageResolution = Field(
        default=Kling3ImageResolution.RES_1K,
        description="Image generation resolution. 1K: standard, 2K: high-res",
    )
    num_images: int = Field(
        default=1, ge=1, le=9, description="Number of images to generate (1-9)"
    )
    elements: list[ImageRef] = Field(
        default=[],
        description="Optional elements for face/character control. Reference as @Element1, @Element2 in prompt",
    )
    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments: dict[str, Any] = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "num_images": self.num_images,
        }
        if self.negative_prompt is not None and self.negative_prompt.strip():
            arguments["negative_prompt"] = self.negative_prompt

        # Add elements for face/character control
        if self.elements:
            element_list = []
            for elem in self.elements:
                if elem.uri:
                    elem_base64 = await context.image_to_base64(elem)
                    ref_data_url = f"data:image/png;base64,{elem_base64}"
                    element_list.append(
                        {
                            "frontal_image_url": ref_data_url,
                            "reference_image_urls": [ref_data_url],
                        }
                    )
            if element_list:
                arguments["elements"] = element_list

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-image/v3/text-to-image",
            arguments=arguments,
        )
        assert res["images"] is not None
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio"]
