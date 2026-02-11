from enum import Enum
from pydantic import Field
from typing import Any
from nodetool.metadata.types import ImageRef
from nodetool.nodes.fal.types import ControlNet, ElementInput, Embedding, GuidanceInput, IPAdapter, LoRAInput, LoRAWeight, LoraWeight, RGBColor
from nodetool.nodes.fal.fal_node import FALNode
from nodetool.workflows.processing_context import ProcessingContext


class ImageSizePreset(str, Enum):
    """Preset sizes for image generation"""
    SQUARE_HD = "square_hd"
    SQUARE = "square"
    PORTRAIT_4_3 = "portrait_4_3"
    PORTRAIT_16_9 = "portrait_16_9"
    LANDSCAPE_4_3 = "landscape_4_3"
    LANDSCAPE_16_9 = "landscape_16_9"


class FluxDev(FALNode):
    """
    FLUX.1 [dev] is a powerful open-weight text-to-image model with 12 billion parameters. Optimized for prompt following and visual quality.
    image, generation, flux, text-to-image, txt2img

    Use cases:
    - Generate high-quality images from text prompts
    - Create detailed illustrations with precise control
    - Produce professional artwork and designs
    - Generate multiple variations from one prompt
    - Create safe-for-work content with built-in safety checker
    """

    class Acceleration(Enum):
        """
        The speed of the generation. The higher the speed, the faster the generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="Size preset for the generated image"
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker to filter unsafe content"
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    guidance_scale: float = Field(
        default=3.5, description="How strictly to follow the prompt. Higher values are more literal"
    )
    num_inference_steps: int = Field(
        default=28, description="Number of denoising steps. More steps typically improve quality"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/dev",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]

class FluxSchnell(FALNode):
    """
    FLUX.1 [schnell] is a fast distilled version of FLUX.1 optimized for speed. Can generate high-quality images in 1-4 steps.
    image, generation, flux, fast, text-to-image, txt2img

    Use cases:
    - Generate images quickly for rapid iteration
    - Create concept art with minimal latency
    - Produce preview images before final generation
    - Generate multiple variations efficiently
    - Real-time image generation applications
    """

    class Acceleration(Enum):
        """
        The speed of the generation. The higher the speed, the faster the generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="Size preset for the generated image"
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker to filter unsafe content"
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=4, description="Number of denoising steps (1-4 recommended for schnell)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/schnell",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_inference_steps"]

class FluxV1Pro(FALNode):
    """
    FLUX.1 Pro is a state-of-the-art image generation model with superior prompt following and image quality.
    image, generation, flux, pro, text-to-image, txt2img

    Use cases:
    - Generate professional-grade images for commercial use
    - Create highly detailed artwork with complex prompts
    - Produce marketing materials and brand assets
    - Generate photorealistic images
    - Create custom visual content with precise control
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"
        VALUE_6 = "6"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="landscape_4_3", description="Size preset for the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="Output image format (jpeg or png)"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="Safety checker tolerance level (1-6). Higher is more permissive"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker to filter unsafe content"
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1.1",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]

class FluxV1ProUltra(FALNode):
    """
    FLUX.1 Pro Ultra delivers the highest quality image generation with enhanced detail and realism.
    image, generation, flux, pro, ultra, text-to-image, txt2img

    Use cases:
    - Generate ultra-high quality photorealistic images
    - Create professional photography-grade visuals
    - Produce detailed product renders
    - Generate premium marketing materials
    - Create artistic masterpieces with fine details
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"
        VALUE_6 = "6"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    aspect_ratio: str = Field(
        default="16:9", description="Aspect ratio for the generated image"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    raw: bool = Field(
        default=False, description="Generate less processed, more natural results"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The image URL to generate an image from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    image_prompt_strength: float = Field(
        default=0.1, description="Strength of image prompt influence (0-1)"
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio,
            "num_images": self.num_images,
            "raw": self.raw,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "image_prompt_strength": self.image_prompt_strength,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1.1-ultra",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "aspect_ratio"]

class FluxLora(FALNode):
    """
    FLUX with LoRA support enables fine-tuned image generation using custom LoRA models for specific styles or subjects.
    image, generation, flux, lora, fine-tuning, text-to-image, txt2img

    Use cases:
    - Generate images with custom artistic styles
    - Create consistent characters across images
    - Apply brand-specific visual styles
    - Generate images with specialized subjects
    - Combine multiple LoRA models for unique results
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. 'regular' balances speed and quality.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate. This is always set to 1 for streaming output."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    image_size: str = Field(
        default="landscape_4_3", description="Size preset for the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="List of LoRA models to apply with their weights"
    )
    guidance_scale: float = Field(
        default=3.5, description="How strictly to follow the prompt"
    )
    num_inference_steps: int = Field(
        default=28, description="Number of denoising steps"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker to filter unsafe content"
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "loras", "image_size"]

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

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image
        """
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

    class Style(Enum):
        """
        The style of the generated image
        """
        AUTO = "auto"
        GENERAL = "general"
        REALISTIC = "realistic"
        DESIGN = "design"
        RENDER_3D = "render_3D"
        ANIME = "anime"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image"
    )
    style: Style = Field(
        default=Style.AUTO, description="The style of the generated image"
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to expand the prompt with MagicPrompt functionality"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seed: str = Field(
        default="", description="Seed for reproducible results. Use -1 for random"
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
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "style"]

class IdeogramV2Turbo(FALNode):
    """
    Ideogram V2 Turbo offers faster image generation with the same exceptional quality and typography handling as V2.
    image, generation, ai, typography, realistic, fast, text-to-image, txt2img

    Use cases:
    - Rapidly generate commercial designs
    - Quick iteration on marketing materials
    - Fast prototyping of visual concepts
    - Real-time design exploration
    - Efficient batch generation of branded content
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image
        """
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

    class Style(Enum):
        """
        The style of the generated image
        """
        AUTO = "auto"
        GENERAL = "general"
        REALISTIC = "realistic"
        DESIGN = "design"
        RENDER_3D = "render_3D"
        ANIME = "anime"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image"
    )
    style: Style = Field(
        default=Style.AUTO, description="The style of the generated image"
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to expand the prompt with MagicPrompt functionality"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seed: str = Field(
        default="", description="Seed for reproducible results. Use -1 for random"
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
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2/turbo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "style"]

class RecraftV3(FALNode):
    """
    Recraft V3 is a powerful image generation model with exceptional control over style and colors, ideal for brand consistency and design work.
    image, generation, design, branding, style, text-to-image, txt2img

    Use cases:
    - Create brand-consistent visual assets
    - Generate designs with specific color palettes
    - Produce stylized illustrations and artwork
    - Design marketing materials with brand colors
    - Create cohesive visual content series
    """

    class RecraftV3Style(Enum):
        """
        The style of the generated images. Vector images cost 2X as much.
        """
        ANY = "any"
        REALISTIC_IMAGE = "realistic_image"
        DIGITAL_ILLUSTRATION = "digital_illustration"
        VECTOR_ILLUSTRATION = "vector_illustration"
        REALISTIC_IMAGE_B_AND_W = "realistic_image/b_and_w"
        REALISTIC_IMAGE_HARD_FLASH = "realistic_image/hard_flash"
        REALISTIC_IMAGE_HDR = "realistic_image/hdr"
        REALISTIC_IMAGE_NATURAL_LIGHT = "realistic_image/natural_light"
        REALISTIC_IMAGE_STUDIO_PORTRAIT = "realistic_image/studio_portrait"
        REALISTIC_IMAGE_ENTERPRISE = "realistic_image/enterprise"
        REALISTIC_IMAGE_MOTION_BLUR = "realistic_image/motion_blur"
        REALISTIC_IMAGE_EVENING_LIGHT = "realistic_image/evening_light"
        REALISTIC_IMAGE_FADED_NOSTALGIA = "realistic_image/faded_nostalgia"
        REALISTIC_IMAGE_FOREST_LIFE = "realistic_image/forest_life"
        REALISTIC_IMAGE_MYSTIC_NATURALISM = "realistic_image/mystic_naturalism"
        REALISTIC_IMAGE_NATURAL_TONES = "realistic_image/natural_tones"
        REALISTIC_IMAGE_ORGANIC_CALM = "realistic_image/organic_calm"
        REALISTIC_IMAGE_REAL_LIFE_GLOW = "realistic_image/real_life_glow"
        REALISTIC_IMAGE_RETRO_REALISM = "realistic_image/retro_realism"
        REALISTIC_IMAGE_RETRO_SNAPSHOT = "realistic_image/retro_snapshot"
        REALISTIC_IMAGE_URBAN_DRAMA = "realistic_image/urban_drama"
        REALISTIC_IMAGE_VILLAGE_REALISM = "realistic_image/village_realism"
        REALISTIC_IMAGE_WARM_FOLK = "realistic_image/warm_folk"
        DIGITAL_ILLUSTRATION_PIXEL_ART = "digital_illustration/pixel_art"
        DIGITAL_ILLUSTRATION_HAND_DRAWN = "digital_illustration/hand_drawn"
        DIGITAL_ILLUSTRATION_GRAIN = "digital_illustration/grain"
        DIGITAL_ILLUSTRATION_INFANTILE_SKETCH = "digital_illustration/infantile_sketch"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER = "digital_illustration/2d_art_poster"
        DIGITAL_ILLUSTRATION_HANDMADE_3D = "digital_illustration/handmade_3d"
        DIGITAL_ILLUSTRATION_HAND_DRAWN_OUTLINE = "digital_illustration/hand_drawn_outline"
        DIGITAL_ILLUSTRATION_ENGRAVING_COLOR = "digital_illustration/engraving_color"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER_2 = "digital_illustration/2d_art_poster_2"
        DIGITAL_ILLUSTRATION_ANTIQUARIAN = "digital_illustration/antiquarian"
        DIGITAL_ILLUSTRATION_BOLD_FANTASY = "digital_illustration/bold_fantasy"
        DIGITAL_ILLUSTRATION_CHILD_BOOK = "digital_illustration/child_book"
        DIGITAL_ILLUSTRATION_CHILD_BOOKS = "digital_illustration/child_books"
        DIGITAL_ILLUSTRATION_COVER = "digital_illustration/cover"
        DIGITAL_ILLUSTRATION_CROSSHATCH = "digital_illustration/crosshatch"
        DIGITAL_ILLUSTRATION_DIGITAL_ENGRAVING = "digital_illustration/digital_engraving"
        DIGITAL_ILLUSTRATION_EXPRESSIONISM = "digital_illustration/expressionism"
        DIGITAL_ILLUSTRATION_FREEHAND_DETAILS = "digital_illustration/freehand_details"
        DIGITAL_ILLUSTRATION_GRAIN_20 = "digital_illustration/grain_20"
        DIGITAL_ILLUSTRATION_GRAPHIC_INTENSITY = "digital_illustration/graphic_intensity"
        DIGITAL_ILLUSTRATION_HARD_COMICS = "digital_illustration/hard_comics"
        DIGITAL_ILLUSTRATION_LONG_SHADOW = "digital_illustration/long_shadow"
        DIGITAL_ILLUSTRATION_MODERN_FOLK = "digital_illustration/modern_folk"
        DIGITAL_ILLUSTRATION_MULTICOLOR = "digital_illustration/multicolor"
        DIGITAL_ILLUSTRATION_NEON_CALM = "digital_illustration/neon_calm"
        DIGITAL_ILLUSTRATION_NOIR = "digital_illustration/noir"
        DIGITAL_ILLUSTRATION_NOSTALGIC_PASTEL = "digital_illustration/nostalgic_pastel"
        DIGITAL_ILLUSTRATION_OUTLINE_DETAILS = "digital_illustration/outline_details"
        DIGITAL_ILLUSTRATION_PASTEL_GRADIENT = "digital_illustration/pastel_gradient"
        DIGITAL_ILLUSTRATION_PASTEL_SKETCH = "digital_illustration/pastel_sketch"
        DIGITAL_ILLUSTRATION_POP_ART = "digital_illustration/pop_art"
        DIGITAL_ILLUSTRATION_POP_RENAISSANCE = "digital_illustration/pop_renaissance"
        DIGITAL_ILLUSTRATION_STREET_ART = "digital_illustration/street_art"
        DIGITAL_ILLUSTRATION_TABLET_SKETCH = "digital_illustration/tablet_sketch"
        DIGITAL_ILLUSTRATION_URBAN_GLOW = "digital_illustration/urban_glow"
        DIGITAL_ILLUSTRATION_URBAN_SKETCHING = "digital_illustration/urban_sketching"
        DIGITAL_ILLUSTRATION_VANILLA_DREAMS = "digital_illustration/vanilla_dreams"
        DIGITAL_ILLUSTRATION_YOUNG_ADULT_BOOK = "digital_illustration/young_adult_book"
        DIGITAL_ILLUSTRATION_YOUNG_ADULT_BOOK_2 = "digital_illustration/young_adult_book_2"
        VECTOR_ILLUSTRATION_BOLD_STROKE = "vector_illustration/bold_stroke"
        VECTOR_ILLUSTRATION_CHEMISTRY = "vector_illustration/chemistry"
        VECTOR_ILLUSTRATION_COLORED_STENCIL = "vector_illustration/colored_stencil"
        VECTOR_ILLUSTRATION_CONTOUR_POP_ART = "vector_illustration/contour_pop_art"
        VECTOR_ILLUSTRATION_COSMICS = "vector_illustration/cosmics"
        VECTOR_ILLUSTRATION_CUTOUT = "vector_illustration/cutout"
        VECTOR_ILLUSTRATION_DEPRESSIVE = "vector_illustration/depressive"
        VECTOR_ILLUSTRATION_EDITORIAL = "vector_illustration/editorial"
        VECTOR_ILLUSTRATION_EMOTIONAL_FLAT = "vector_illustration/emotional_flat"
        VECTOR_ILLUSTRATION_INFOGRAPHICAL = "vector_illustration/infographical"
        VECTOR_ILLUSTRATION_MARKER_OUTLINE = "vector_illustration/marker_outline"
        VECTOR_ILLUSTRATION_MOSAIC = "vector_illustration/mosaic"
        VECTOR_ILLUSTRATION_NAIVECTOR = "vector_illustration/naivector"
        VECTOR_ILLUSTRATION_ROUNDISH_FLAT = "vector_illustration/roundish_flat"
        VECTOR_ILLUSTRATION_SEGMENTED_COLORS = "vector_illustration/segmented_colors"
        VECTOR_ILLUSTRATION_SHARP_CONTRAST = "vector_illustration/sharp_contrast"
        VECTOR_ILLUSTRATION_THIN = "vector_illustration/thin"
        VECTOR_ILLUSTRATION_VECTOR_PHOTO = "vector_illustration/vector_photo"
        VECTOR_ILLUSTRATION_VIVID_SHAPES = "vector_illustration/vivid_shapes"
        VECTOR_ILLUSTRATION_ENGRAVING = "vector_illustration/engraving"
        VECTOR_ILLUSTRATION_LINE_ART = "vector_illustration/line_art"
        VECTOR_ILLUSTRATION_LINE_CIRCUIT = "vector_illustration/line_circuit"
        VECTOR_ILLUSTRATION_LINOCUT = "vector_illustration/linocut"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    image_size: str = Field(
        default="square_hd", description="Size preset for the generated image"
    )
    style: RecraftV3Style = Field(
        default=RecraftV3Style.REALISTIC_IMAGE, description="Visual style preset for the generated image"
    )
    colors: list[RGBColor] = Field(
        default=[], description="Specific color palette to use in the generation"
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    style_id: str = Field(
        default="", description="Custom style ID for brand-specific styles"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "style": self.style.value,
            "colors": [item.model_dump(exclude={"type"}) for item in self.colors],
            "enable_safety_checker": self.enable_safety_checker,
            "style_id": self.style_id,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/recraft-v3",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "style", "colors"]

class StableDiffusionV35Large(FALNode):
    """
    Stable Diffusion 3.5 Large is a powerful open-weight model with excellent prompt adherence and diverse output capabilities.
    image, generation, stable-diffusion, open-source, text-to-image, txt2img

    Use cases:
    - Generate diverse artistic styles
    - Create high-quality illustrations
    - Produce photorealistic images
    - Generate concept art and designs
    - Create custom visual content
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. Defaults to landscape_4_3 if no controlnet has been passed, otherwise defaults to the size of the controlnet conditioning image."
    )
    controlnet: str = Field(
        default="", description="ControlNet for inference."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    ip_adapter: str = Field(
        default="", description="IP-Adapter to use during inference."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    negative_prompt: str = Field(
        default="", description="Elements to avoid in the generated image"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "controlnet": self.controlnet,
            "output_format": self.output_format.value,
            "ip_adapter": self.ip_adapter,
            "sync_mode": self.sync_mode,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-diffusion-v35-large",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "negative_prompt", "aspect_ratio"]

class FluxProNew(FALNode):
    """
    FLUX.1 Pro New is the latest version of the professional FLUX model with enhanced capabilities and improved output quality.
    image, generation, flux, professional, text-to-image, txt2img

    Use cases:
    - Generate professional-grade marketing visuals
    - Create high-quality product renders
    - Produce detailed architectural visualizations
    - Design premium brand assets
    - Generate photorealistic commercial imagery
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"
        VALUE_6 = "6"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="Size preset for the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/new",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size"]

class Flux2Turbo(FALNode):
    """
    FLUX.2 Turbo is a blazing-fast image generation model optimized for speed without sacrificing quality, ideal for real-time applications.
    image, generation, flux, fast, turbo, text-to-image, txt2img

    Use cases:
    - Real-time image generation for interactive apps
    - Rapid prototyping of visual concepts
    - Generate multiple variations instantly
    - Live visual effects and augmented reality
    - High-throughput batch image processing
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="Size preset for the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="If set to true, the prompt will be expanded for better results."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    guidance_scale: float = Field(
        default=2.5, description="Guidance Scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/turbo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "num_images"]

class Flux2Flash(FALNode):
    """
    FLUX.2 Flash is an ultra-fast variant of FLUX.2 designed for instant image generation with minimal latency.
    image, generation, flux, ultra-fast, flash, text-to-image, txt2img

    Use cases:
    - Instant preview generation for user interfaces
    - Real-time collaborative design tools
    - Lightning-fast concept exploration
    - High-speed batch processing
    - Interactive gaming and entertainment applications
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="Size preset for the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="If set to true, the prompt will be expanded for better results."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    guidance_scale: float = Field(
        default=2.5, description="Guidance Scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/flash",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size"]

class IdeogramV3(FALNode):
    """
    Ideogram V3 is the latest generation with enhanced text rendering, superior image quality, and expanded creative controls.
    image, generation, ideogram, typography, text-rendering, text-to-image, txt2img

    Use cases:
    - Create professional graphics with embedded text
    - Design social media posts with perfect typography
    - Generate logos and brand identities
    - Produce marketing materials with text overlays
    - Create educational content with clear text
    """

    class RenderingSpeed(Enum):
        """
        The rendering speed to use.
        """
        TURBO = "TURBO"
        BALANCED = "BALANCED"
        QUALITY = "QUALITY"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The resolution of the generated image"
    )
    style: str = Field(
        default="", description="The style preset for the generated image"
    )
    style_preset: str = Field(
        default="", description="Style preset for generation. The chosen style preset will guide the generation."
    )
    expand_prompt: bool = Field(
        default=True, description="Automatically enhance the prompt for better results"
    )
    rendering_speed: RenderingSpeed = Field(
        default=RenderingSpeed.BALANCED, description="The rendering speed to use."
    )
    style_codes: str = Field(
        default="", description="A list of 8 character hexadecimal codes representing the style of the image. Cannot be used in conjunction with style_reference_images or style"
    )
    color_palette: str = Field(
        default="", description="A color palette for generation, must EITHER be specified via one of the presets (name) or explicitly via hexadecimal representations of the color with optional weights (members)"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seed: str = Field(
        default="", description="Seed for the random number generator"
    )
    images: list[ImageRef] = Field(
        default=ImageRef(), description="A set of images to use as style references (maximum total size 10MB across all style references). The images should be in JPEG, PNG or WebP format"
    )
    negative_prompt: str = Field(
        default="", description="Description of what to exclude from an image. Descriptions in the prompt take precedence to descriptions in the negative prompt."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        images_data_urls = []
        for image in self.images or []:
            image_base64 = await context.image_to_base64(image)
            images_data_urls.append(f"data:image/png;base64,{image_base64}")
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "style": self.style,
            "style_preset": self.style_preset,
            "expand_prompt": self.expand_prompt,
            "rendering_speed": self.rendering_speed.value,
            "style_codes": self.style_codes,
            "color_palette": self.color_palette,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "image_urls": images_data_urls,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v3",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "aspect_ratio", "style"]

class OmniGenV1(FALNode):
    """
    OmniGen V1 is a versatile unified model for multi-modal image generation and editing with text, supporting complex compositional tasks.
    image, generation, multi-modal, editing, unified, text-to-image, txt2img

    Use cases:
    - Generate images with multiple input modalities
    - Edit existing images with text instructions
    - Create complex compositional scenes
    - Combine text and image inputs for generation
    - Perform advanced image manipulations
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate or edit an image"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    img_guidance_scale: float = Field(
        default=1.6, description="The Image Guidance scale is a measure of how close you want the model to stick to your input image when looking for a related image to show you."
    )
    input_images: list[str] = Field(
        default=[], description="URL of images to use while generating the image, Use <img><|image_1|></img> for the first image and so on."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=3, description="How strictly to follow the prompt and inputs"
    )
    num_inference_steps: int = Field(
        default=50, description="Number of denoising steps for generation quality"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "img_guidance_scale": self.img_guidance_scale,
            "input_image_urls": self.input_images,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/omnigen-v1",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "guidance_scale", "num_inference_steps"]

class Sana(FALNode):
    """
    Sana is an efficient high-resolution image generation model that balances quality and speed for practical applications.
    image, generation, efficient, high-resolution, text-to-image, txt2img

    Use cases:
    - Generate high-resolution images efficiently
    - Create detailed artwork with good performance
    - Produce quality visuals with limited compute
    - Generate images for web and mobile applications
    - Balanced quality-speed image production
    """

    class StyleName(Enum):
        """
        The style to generate the image in.
        """
        NO_STYLE = "(No style)"
        CINEMATIC = "Cinematic"
        PHOTOGRAPHIC = "Photographic"
        ANIME = "Anime"
        MANGA = "Manga"
        DIGITAL_ART = "Digital Art"
        PIXEL_ART = "Pixel art"
        FANTASY_ART = "Fantasy art"
        NEONPUNK = "Neonpunk"
        MODEL_3D = "3D Model"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: ImageSizePreset = Field(
        default=ImageSizePreset.LANDSCAPE_4_3, description="Size preset for the generated image"
    )
    style_name: StyleName = Field(
        default=StyleName.NO_STYLE, description="The style to generate the image in."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=5, description="How strictly to follow the prompt"
    )
    num_inference_steps: int = Field(
        default=18, description="Number of denoising steps"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="Elements to avoid in the generated image"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "style_name": self.style_name.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sana",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "image_size", "guidance_scale"]

class HunyuanImageV3InstructTextToImage(FALNode):
    """
    Hunyuan Image v3 Instruct generates high-quality images from text with advanced instruction understanding.
    image, generation, hunyuan, v3, instruct, text-to-image

    Use cases:
    - Generate images with detailed instructions
    - Create artwork with precise text control
    - Produce high-quality visual content
    - Generate images with advanced understanding
    - Create professional visuals from text
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The text prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="auto", description="The desired size of the generated image. If auto, image size will be determined by the model."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible results. If None, a random seed is used."
    )
    guidance_scale: float = Field(
        default=3.5, description="Controls how much the model adheres to the prompt. Higher values mean stricter adherence."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-image/v3/instruct/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class QwenImageMaxTextToImage(FALNode):
    """
    Qwen Image Max generates premium quality images from text with superior detail and accuracy.
    image, generation, qwen, max, premium, text-to-image

    Use cases:
    - Generate premium quality images
    - Create detailed artwork from text
    - Produce high-fidelity visual content
    - Generate professional-grade images
    - Create superior quality visuals
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="Text prompt describing the desired image. Supports Chinese and English. Max 800 characters."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Enable LLM prompt optimization for better results."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility (0-2147483647)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable content moderation for input and output."
    )
    negative_prompt: str = Field(
        default="", description="Content to avoid in the generated image. Max 500 characters."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-max/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class QwenImage2512(FALNode):
    """
    Qwen Image 2512 generates high-resolution images from text with excellent quality and detail.
    image, generation, qwen, 2512, high-resolution, text-to-image

    Use cases:
    - Generate high-resolution images
    - Create detailed visual content
    - Produce quality artwork from text
    - Generate images with fine details
    - Create high-quality visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=4, description="The guidance scale to use for the image generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate an image from."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-2512",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class QwenImage2512Lora(FALNode):
    """
    Qwen Image 2512 with LoRA support enables custom-trained models for specialized image generation.
    image, generation, qwen, 2512, lora, custom

    Use cases:
    - Generate images with custom models
    - Create specialized visual content
    - Produce domain-specific artwork
    - Generate images with fine-tuned models
    - Create customized visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use up to 3 LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=4, description="The guidance scale to use for the image generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate an image from."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image-2512/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ZImageBase(FALNode):
    """
    Z-Image Base generates quality images from text with efficient processing and good results.
    image, generation, z-image, base, efficient, text-to-image

    Use cases:
    - Generate images efficiently
    - Create quality artwork from text
    - Produce visual content quickly
    - Generate images with good performance
    - Create efficient visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=4, description="The guidance scale to use for the image generation."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use for the image generation."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/base",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ZImageBaseLora(FALNode):
    """
    Z-Image Base with LoRA enables efficient custom-trained models for specialized generation tasks.
    image, generation, z-image, base, lora, custom

    Use cases:
    - Generate images with custom efficient models
    - Create specialized content quickly
    - Produce domain-specific visuals
    - Generate with fine-tuned base model
    - Create efficient custom visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    loras: list[LoRAInput] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
    )
    guidance_scale: float = Field(
        default=4, description="The guidance scale to use for the image generation."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use for the image generation."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/base/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ZImageTurbo(FALNode):
    """
    Z-Image Turbo generates images from text with maximum speed for rapid iteration and prototyping.
    image, generation, z-image, turbo, fast, text-to-image

    Use cases:
    - Generate images at maximum speed
    - Create rapid prototypes from text
    - Produce quick visual iterations
    - Generate images for fast workflows
    - Create instant visual content
    """

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. Note: this will increase the price by 0.0025 credits per request."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/turbo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ZImageTurboLora(FALNode):
    """
    Z-Image Turbo with LoRA combines maximum speed with custom models for fast specialized generation.
    image, generation, z-image, turbo, lora, fast

    Use cases:
    - Generate custom images at turbo speed
    - Create specialized content rapidly
    - Produce quick domain-specific visuals
    - Generate with fast fine-tuned models
    - Create instant custom visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    loras: list[LoRAInput] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. Note: this will increase the price by 0.0025 credits per request."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/z-image/turbo/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2Klein4B(FALNode):
    """
    FLUX-2 Klein 4B generates images with the efficient 4-billion parameter model for balanced quality and speed.
    image, generation, flux-2, klein, 4b, text-to-image

    Use cases:
    - Generate images with 4B model
    - Create balanced quality-speed content
    - Produce efficient visual artwork
    - Generate images with good performance
    - Create optimized visuals
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the image to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI. Output is not stored when this is True."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=4, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/4b",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2Klein4BBase(FALNode):
    """
    FLUX-2 Klein 4B Base provides foundation model generation with 4-billion parameters.
    image, generation, flux-2, klein, 4b, base

    Use cases:
    - Generate with base 4B model
    - Create foundation quality content
    - Produce standard visual artwork
    - Generate images with base model
    - Create baseline visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use for image generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the image to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for image generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI. Output is not stored when this is True."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for classifier-free guidance. Describes what to avoid in the image."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/4b/base",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2Klein4BBaseLora(FALNode):
    """
    FLUX-2 Klein 4B Base with LoRA enables custom-trained 4B models for specialized generation.
    image, generation, flux-2, klein, 4b, base, lora

    Use cases:
    - Generate with custom 4B base model
    - Create specialized foundation content
    - Produce domain-specific visuals
    - Generate with fine-tuned 4B model
    - Create customized baseline visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use for image generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the image to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for image generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI. Output is not stored when this is True."
    )
    loras: list[LoRAInput] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for classifier-free guidance. Describes what to avoid in the image."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/4b/base/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2Klein9B(FALNode):
    """
    FLUX-2 Klein 9B generates high-quality images with the powerful 9-billion parameter model.
    image, generation, flux-2, klein, 9b, text-to-image

    Use cases:
    - Generate high-quality images with 9B model
    - Create superior visual content
    - Produce detailed artwork
    - Generate images with powerful model
    - Create premium quality visuals
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the image to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI. Output is not stored when this is True."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=4, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/9b",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2Klein9BBase(FALNode):
    """
    FLUX-2 Klein 9B Base provides foundation generation with the full 9-billion parameter model.
    image, generation, flux-2, klein, 9b, base

    Use cases:
    - Generate with base 9B model
    - Create high-quality foundation content
    - Produce superior baseline artwork
    - Generate images with powerful base
    - Create premium baseline visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use for image generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the image to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for image generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI. Output is not stored when this is True."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for classifier-free guidance. Describes what to avoid in the image."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/9b/base",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2Klein9BBaseLora(FALNode):
    """
    FLUX-2 Klein 9B Base with LoRA combines powerful generation with custom-trained models.
    image, generation, flux-2, klein, 9b, base, lora

    Use cases:
    - Generate with custom 9B base model
    - Create specialized high-quality content
    - Produce custom superior visuals
    - Generate with fine-tuned 9B model
    - Create advanced customized visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use for image generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the image to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use for image generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI. Output is not stored when this is True."
    )
    loras: list[LoRAInput] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for classifier-free guidance. Describes what to avoid in the image."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2/klein/9b/base/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2Max(FALNode):
    """
    FLUX-2 Max generates maximum quality images with the most advanced FLUX-2 model for premium results.
    image, generation, flux-2, max, premium, text-to-image

    Use cases:
    - Generate maximum quality images
    - Create premium visual content
    - Produce professional-grade artwork
    - Generate images with best model
    - Create superior quality visuals
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-max",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class GlmImage(FALNode):
    """
    GLM Image generates images from text with advanced AI understanding and quality output.
    image, generation, glm, ai, text-to-image

    Use cases:
    - Generate images with GLM AI
    - Create intelligent visual content
    - Produce AI-powered artwork
    - Generate images with understanding
    - Create smart visuals from text
    """

    class OutputFormat(Enum):
        """
        Output image format.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="Text prompt for image generation."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="Output image size."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="Output image format."
    )
    sync_mode: bool = Field(
        default=False, description="If True, the image will be returned as a base64 data URI instead of a URL."
    )
    guidance_scale: float = Field(
        default=1.5, description="Classifier-free guidance scale. Higher values make the model follow the prompt more closely."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. The same seed with the same prompt will produce the same image."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="If True, the prompt will be enhanced using an LLM for more detailed and higher quality results."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of diffusion denoising steps. More steps generally produce higher quality images."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable NSFW safety checking on the generated images."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/glm-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class GptImage15(FALNode):
    """
    GPT Image 1.5 generates images from text with GPT-powered language understanding and visual creation.
    image, generation, gpt, language-ai, text-to-image

    Use cases:
    - Generate images with GPT understanding
    - Create language-aware visual content
    - Produce intelligent artwork
    - Generate images with natural language
    - Create GPT-powered visuals
    """

    class ImageSize(Enum):
        """
        Aspect ratio for the generated image
        """
        VALUE_1024X1024 = "1024x1024"
        VALUE_1536X1024 = "1536x1024"
        VALUE_1024X1536 = "1024x1536"

    class Background(Enum):
        """
        Background for the generated image
        """
        AUTO = "auto"
        TRANSPARENT = "transparent"
        OPAQUE = "opaque"

    class Quality(Enum):
        """
        Quality for the generated image
        """
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        Output format for the images
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt for image generation"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: ImageSize = Field(
        default=ImageSize.VALUE_1024X1024, description="Aspect ratio for the generated image"
    )
    background: Background = Field(
        default=Background.AUTO, description="Background for the generated image"
    )
    quality: Quality = Field(
        default=Quality.HIGH, description="Quality for the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output format for the images"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "background": self.background.value,
            "quality": self.quality.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/gpt-image-1.5",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV26TextToImage(FALNode):
    """
    Wan v2.6 generates high-quality images from text with advanced capabilities and consistent results.
    image, generation, wan, v2.6, quality, text-to-image

    Use cases:
    - Generate quality images with Wan v2.6
    - Create consistent visual content
    - Produce reliable artwork from text
    - Generate images with advanced model
    - Create high-quality visuals
    """

    prompt: str = Field(
        default="", description="Text prompt describing the desired image. Supports Chinese and English. Max 2000 characters."
    )
    image_size: str = Field(
        default="", description="Output image size. If not set: matches input image size (up to 1280*1280). Use presets like 'square_hd', 'landscape_16_9', or specify exact dimensions."
    )
    max_images: int = Field(
        default=1, description="Maximum number of images to generate (1-5). Actual count may be less depending on model inference."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Optional reference image (0 or 1). When provided, can be used for style guidance. Resolution: 384-5000px each dimension. Max size: 10MB. Formats: JPEG, JPG, PNG (no alpha), BMP, WEBP."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable content moderation for input and output."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility (0-2147483647)."
    )
    negative_prompt: str = Field(
        default="", description="Content to avoid in the generated image. Max 500 characters."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "max_images": self.max_images,
            "image_url": f"data:image/png;base64,{image_base64}",
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="wan/v2.6/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LongcatImage(FALNode):
    """
    Longcat Image generates creative and unique images from text with distinctive AI characteristics.
    image, generation, longcat, creative, text-to-image

    Use cases:
    - Generate creative images
    - Create unique visual content
    - Produce distinctive artwork
    - Generate images with character
    - Create artistic visuals
    """

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    guidance_scale: float = Field(
        default=4.5, description="The guidance scale to use for the image generation."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/longcat-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BytedanceSeedreamV45TextToImage(FALNode):
    """
    ByteDance SeeDream v4.5 generates advanced images from text with cutting-edge AI technology.
    image, generation, bytedance, seedream, v4.5, text-to-image

    Use cases:
    - Generate images with SeeDream v4.5
    - Create cutting-edge visual content
    - Produce advanced AI artwork
    - Generate images with latest tech
    - Create modern AI visuals
    """

    prompt: str = Field(
        default="", description="The text prompt used to generate the image"
    )
    num_images: int = Field(
        default=1, description="Number of separate model generations to be run with the prompt."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. Width and height must be between 1920 and 4096, or total number of pixels must be between 2560*1440 and 4096*4096."
    )
    max_images: int = Field(
        default=1, description="If set to a number greater than one, enables multi-image generation. The model will potentially return up to `max_images` images every generation, and in total, `num_images` generations will be carried out. In total, the number of images generated will be between `num_images` and `max_images*num_images`."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="Random seed to control the stochasticity of image generation."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "max_images": self.max_images,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedream/v4.5/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ViduQ2TextToImage(FALNode):
    """
    Vidu Q2 generates quality images from text with optimized performance and consistent results.
    image, generation, vidu, q2, optimized, text-to-image

    Use cases:
    - Generate optimized quality images
    - Create consistent visual content
    - Produce balanced artwork
    - Generate images efficiently
    - Create reliable visuals
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the output video
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"


    prompt: str = Field(
        default="", description="Text prompt for video generation, max 1500 characters"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="The aspect ratio of the output video"
    )
    seed: int = Field(
        default=-1, description="Random seed for generation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/vidu/q2/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ImagineartImagineart15ProPreviewTextToImage(FALNode):
    """
    ImagineArt 1.5 Pro Preview
    generation, text-to-image, txt2img, ai-art, professional

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        Image aspect ratio: 1:1, 3:1, 1:3, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3
        """
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"
        RATIO_3_1 = "3:1"
        RATIO_1_3 = "1:3"
        RATIO_3_2 = "3:2"
        RATIO_2_3 = "2:3"


    prompt: str = Field(
        default="", description="Text prompt describing the desired image"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="Image aspect ratio: 1:1, 3:1, 1:3, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3"
    )
    seed: int = Field(
        default=-1, description="Seed for the image generation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="imagineart/imagineart-1.5-pro-preview/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BriaFiboLiteGenerate(FALNode):
    """
    Fibo Lite
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        Aspect ratio. Options: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9
        """
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"
        RATIO_3_2 = "3:2"
        RATIO_3_4 = "3:4"
        RATIO_4_3 = "4:3"
        RATIO_4_5 = "4:5"
        RATIO_5_4 = "5:4"
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"


    prompt: str = Field(
        default="", description="Prompt for image generation."
    )
    steps_num: int = Field(
        default=8, description="Number of inference steps for Fibo Lite."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="Aspect ratio. Options: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9"
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Reference image (file or URL)."
    )
    sync_mode: bool = Field(
        default=False, description="If true, returns the image directly in the response (increases latency)."
    )
    seed: int = Field(
        default=5555, description="Random seed for reproducibility."
    )
    structured_prompt: str = Field(
        default="", description="The structured prompt to generate an image from."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "steps_num": self.steps_num,
            "aspect_ratio": self.aspect_ratio.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "structured_prompt": self.structured_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo-lite/generate",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class OvisImage(FALNode):
    """
    Ovis Image
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        The acceleration level to use.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The acceleration level to use."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=5, description="The guidance scale to use for the image generation."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate an image from."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ovis-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2LoraGallerySepiaVintage(FALNode):
    """
    Flux 2 Lora Gallery
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. 'regular' balances speed and quality.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the output image
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate a sepia vintage photography style image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the sepia vintage photography effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/sepia-vintage",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2LoraGallerySatelliteViewStyle(FALNode):
    """
    Flux 2 Lora Gallery
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. 'regular' balances speed and quality.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the output image
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate a satellite/aerial view style image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the satellite view style effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/satellite-view-style",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2LoraGalleryRealism(FALNode):
    """
    Flux 2 Lora Gallery
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. 'regular' balances speed and quality.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the output image
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate a realistic image with natural lighting and authentic details."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the realism effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/realism",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2LoraGalleryHdrStyle(FALNode):
    """
    Flux 2 Lora Gallery
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. 'regular' balances speed and quality.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the output image
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate an HDR style image. The trigger word 'Hyp3rRe4list1c' will be automatically prepended."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the HDR style effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/hdr-style",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2LoraGalleryDigitalComicArt(FALNode):
    """
    Flux 2 Lora Gallery
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. 'regular' balances speed and quality.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the output image
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate a digital comic art style image. Use 'd1g1t4l' trigger word for best results."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the digital comic art effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/digital-comic-art",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2LoraGalleryBallpointPenSketch(FALNode):
    """
    Flux 2 Lora Gallery
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. 'regular' balances speed and quality.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the output image
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to generate a ballpoint pen sketch style image. Use 'b4llp01nt' trigger word for best results."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    lora_scale: float = Field(
        default=1, description="The strength of the ballpoint pen sketch effect."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and won't be saved in history."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale. Controls how closely the model follows the prompt."
    )
    seed: str = Field(
        default="", description="Random seed for reproducibility. Same seed with same prompt will produce same result."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker for the generated image."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "lora_scale": self.lora_scale,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-lora-gallery/ballpoint-pen-sketch",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux2Flex(FALNode):
    """
    Flux 2 Flex
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to expand the prompt using the model's own knowledge."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation."
    )
    guidance_scale: float = Field(
        default=3.5, description="The guidance scale to use for the generation."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-2-flex",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Gemini3ProImagePreview(FALNode):
    """
    Gemini 3 Pro Image Preview
    generation, text-to-image, txt2img, ai-art, professional

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Resolution(Enum):
        """
        The resolution of the image to generate.
        """
        VALUE_1K = "1K"
        VALUE_2K = "2K"
        VALUE_4K = "4K"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for content moderation. 1 is the most strict (blocks most content), 6 is the least strict.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"
        VALUE_6 = "6"


    prompt: str = Field(
        default="", description="The text prompt to generate an image from."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1K, description="The resolution of the image to generate."
    )
    enable_web_search: bool = Field(
        default=False, description="Enable web search for the image generation task. This will allow the model to use the latest information from the web to generate the image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    aspect_ratio: str = Field(
        default="1:1", description="The aspect ratio of the generated image. Use \"auto\" to let the model decide based on the prompt."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_4, description="The safety tolerance level for content moderation. 1 is the most strict (blocks most content), 6 is the least strict."
    )
    seed: str = Field(
        default="", description="The seed for the random number generator."
    )
    limit_generations: bool = Field(
        default=False, description="Experimental parameter to limit the number of generations from each round of prompting to 1. Set to `True` to to disregard any instructions in the prompt regarding the number of images to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "enable_web_search": self.enable_web_search,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "seed": self.seed,
            "limit_generations": self.limit_generations,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/gemini-3-pro-image-preview",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class NanoBananaPro(FALNode):
    """
    Nano Banana Pro
    generation, text-to-image, txt2img, ai-art, professional

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Resolution(Enum):
        """
        The resolution of the image to generate.
        """
        VALUE_1K = "1K"
        VALUE_2K = "2K"
        VALUE_4K = "4K"

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image. Use "auto" to let the model decide based on the prompt.
        """
        AUTO = "auto"
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_3_2 = "3:2"
        RATIO_4_3 = "4:3"
        RATIO_5_4 = "5:4"
        RATIO_1_1 = "1:1"
        RATIO_4_5 = "4:5"
        RATIO_3_4 = "3:4"
        RATIO_2_3 = "2:3"
        RATIO_9_16 = "9:16"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for content moderation. 1 is the most strict (blocks most content), 6 is the least strict.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"
        VALUE_6 = "6"


    prompt: str = Field(
        default="", description="The text prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    enable_web_search: bool = Field(
        default=False, description="Enable web search for the image generation task. This will allow the model to use the latest information from the web to generate the image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1K, description="The resolution of the image to generate."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image. Use \"auto\" to let the model decide based on the prompt."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_4, description="The safety tolerance level for content moderation. 1 is the most strict (blocks most content), 6 is the least strict."
    )
    seed: int = Field(
        default=-1, description="The seed for the random number generator."
    )
    limit_generations: bool = Field(
        default=False, description="Experimental parameter to limit the number of generations from each round of prompting to 1. Set to `True` to to disregard any instructions in the prompt regarding the number of images to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "enable_web_search": self.enable_web_search,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "seed": self.seed,
            "limit_generations": self.limit_generations,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/nano-banana-pro",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ImagineartImagineart15PreviewTextToImage(FALNode):
    """
    Imagineart 1.5 Preview
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        Image aspect ratio: 1:1, 3:1, 1:3, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3
        """
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"
        RATIO_3_1 = "3:1"
        RATIO_1_3 = "1:3"
        RATIO_3_2 = "3:2"
        RATIO_2_3 = "2:3"


    prompt: str = Field(
        default="", description="Text prompt describing the desired image"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="Image aspect ratio: 1:1, 3:1, 1:3, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3"
    )
    seed: int = Field(
        default=-1, description="Seed for the image generation"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="imagineart/imagineart-1.5-preview/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Emu35ImageTextToImage(FALNode):
    """
    Emu 3.5 Image
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Resolution(Enum):
        """
        The resolution of the output image.
        """
        VALUE_480P = "480p"
        VALUE_720P = "720p"

    class AspectRatio(Enum):
        """
        The aspect ratio of the output image.
        """
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_3_2 = "3:2"
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"
        RATIO_9_21 = "9:21"

    class OutputFormat(Enum):
        """
        The format of the output image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt to create the image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_720P, description="The resolution of the output image."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the output image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the output image."
    )
    sync_mode: bool = Field(
        default=False, description="Whether to return the image in sync mode."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Whether to enable the safety checker."
    )
    seed: int = Field(
        default=-1, description="The seed for the inference."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "resolution": self.resolution.value,
            "aspect_ratio": self.aspect_ratio.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/emu-3.5-image/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BriaFiboGenerate(FALNode):
    """
    Fibo
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        Aspect ratio. Options: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9
        """
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"
        RATIO_3_2 = "3:2"
        RATIO_3_4 = "3:4"
        RATIO_4_3 = "4:3"
        RATIO_4_5 = "4:5"
        RATIO_5_4 = "5:4"
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"


    prompt: str = Field(
        default="", description="Prompt for image generation."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="Aspect ratio. Options: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9"
    )
    steps_num: int = Field(
        default=50, description="Number of inference steps."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Reference image (file or URL)."
    )
    sync_mode: bool = Field(
        default=False, description="If true, returns the image directly in the response (increases latency)."
    )
    guidance_scale: int = Field(
        default=5, description="Guidance scale for text."
    )
    seed: int = Field(
        default=5555, description="Random seed for reproducibility."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for image generation."
    )
    structured_prompt: str = Field(
        default="", description="The structured prompt to generate an image from."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "steps_num": self.steps_num,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "structured_prompt": self.structured_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="bria/fibo/generate",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Piflow(FALNode):
    """
    Piflow
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image. You can choose between some presets or custom height and width that **must be multiples of 8**."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation. If set to None, a random seed will be used."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/piflow",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class GptImage1Mini(FALNode):
    """
    GPT Image 1 Mini
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Background(Enum):
        """
        Background for the generated image
        """
        AUTO = "auto"
        TRANSPARENT = "transparent"
        OPAQUE = "opaque"

    class ImageSize(Enum):
        """
        Aspect ratio for the generated image
        """
        AUTO = "auto"
        VALUE_1024X1024 = "1024x1024"
        VALUE_1536X1024 = "1536x1024"
        VALUE_1024X1536 = "1024x1536"

    class Quality(Enum):
        """
        Quality for the generated image
        """
        AUTO = "auto"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        Output format for the images
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    background: Background = Field(
        default=Background.AUTO, description="Background for the generated image"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: ImageSize = Field(
        default=ImageSize.AUTO, description="Aspect ratio for the generated image"
    )
    prompt: str = Field(
        default="", description="The prompt for image generation"
    )
    quality: Quality = Field(
        default=Quality.AUTO, description="Quality for the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output format for the images"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "background": self.background.value,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "prompt": self.prompt,
            "quality": self.quality.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/gpt-image-1-mini",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class ReveTextToImage(FALNode):
    """
    Reve
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The desired aspect ratio of the generated image.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_3_2 = "3:2"
        RATIO_2_3 = "2:3"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"
        RATIO_1_1 = "1:1"

    class OutputFormat(Enum):
        """
        Output format for the generated image.
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The text description of the desired image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_3_2, description="The desired aspect ratio of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output format for the generated image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "output_format": self.output_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/reve/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class HunyuanImageV3TextToImage(FALNode):
    """
    Hunyuan Image
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The text prompt for image-to-image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The desired size of the generated image."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=7.5, description="Controls how much the model adheres to the prompt. Higher values mean stricter adherence."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible results. If None, a random seed is used."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to guide the image generation away from certain concepts."
    )
    num_inference_steps: int = Field(
        default=28, description="Number of denoising steps."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-image/v3/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Wan25PreviewTextToImage(FALNode):
    """
    Wan 2.5 Text to Image
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt for image generation. Supports Chinese and English, max 2000 characters."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate. Values from 1 to 4."
    )
    image_size: str = Field(
        default="square", description="The size of the generated image. Can use preset names like 'square', 'landscape_16_9', etc., or specific dimensions. Total pixels must be between 768×768 and 1440×1440, with aspect ratio between [1:4, 4:1]."
    )
    enable_prompt_expansion: bool = Field(
        default=True, description="Whether to enable prompt rewriting using LLM. Improves results for short prompts but increases processing time."
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

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan-25-preview/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxSrpo(FALNode):
    """
    FLUX.1 SRPO [dev]
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        The speed of the generation. The higher the speed, the faster the generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=4.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/srpo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux1Srpo(FALNode):
    """
    FLUX.1 SRPO [dev]
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        The speed of the generation. The higher the speed, the faster the generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=4.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-1/srpo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class HunyuanImageV21TextToImage(FALNode):
    """
    Hunyuan Image
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The text prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The desired size of the generated image."
    )
    use_reprompt: bool = Field(
        default=True, description="Enable prompt enhancement for potentially better results."
    )
    use_refiner: bool = Field(
        default=False, description="Enable the refiner model for improved image quality."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=3.5, description="Controls how much the model adheres to the prompt. Higher values mean stricter adherence."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible results. If None, a random seed is used."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to guide the image generation away from certain concepts."
    )
    num_inference_steps: int = Field(
        default=28, description="Number of denoising steps."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "use_reprompt": self.use_reprompt,
            "use_refiner": self.use_refiner,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/hunyuan-image/v2.1/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BytedanceSeedreamV4TextToImage(FALNode):
    """
    Bytedance Seedream v4
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class EnhancePromptMode(Enum):
        """
        The mode to use for enhancing prompt enhancement. Standard mode provides higher quality results but takes longer to generate. Fast mode provides average quality results but takes less time to generate.
        """
        STANDARD = "standard"
        FAST = "fast"


    prompt: str = Field(
        default="", description="The text prompt used to generate the image"
    )
    num_images: int = Field(
        default=1, description="Number of separate model generations to be run with the prompt."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. Total pixels must be between 960x960 and 4096x4096."
    )
    max_images: int = Field(
        default=1, description="If set to a number greater than one, enables multi-image generation. The model will potentially return up to `max_images` images every generation, and in total, `num_images` generations will be carried out. In total, the number of images generated will be between `num_images` and `max_images*num_images`."
    )
    enhance_prompt_mode: EnhancePromptMode = Field(
        default=EnhancePromptMode.STANDARD, description="The mode to use for enhancing prompt enhancement. Standard mode provides higher quality results but takes longer to generate. Fast mode provides average quality results but takes less time to generate."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="Random seed to control the stochasticity of image generation."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "max_images": self.max_images,
            "enhance_prompt_mode": self.enhance_prompt_mode.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedream/v4/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Gemini25FlashImage(FALNode):
    """
    Gemini 2.5 Flash Image
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image.
        """
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_3_2 = "3:2"
        RATIO_4_3 = "4:3"
        RATIO_5_4 = "5:4"
        RATIO_1_1 = "1:1"
        RATIO_4_5 = "4:5"
        RATIO_3_4 = "3:4"
        RATIO_2_3 = "2:3"
        RATIO_9_16 = "9:16"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The text prompt to generate an image from."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    limit_generations: bool = Field(
        default=False, description="Experimental parameter to limit the number of generations from each round of prompting to 1. Set to `True` to to disregard any instructions in the prompt regarding the number of images to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "limit_generations": self.limit_generations,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/gemini-25-flash-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class NanoBanana(FALNode):
    """
    Nano Banana
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image.
        """
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_3_2 = "3:2"
        RATIO_4_3 = "4:3"
        RATIO_5_4 = "5:4"
        RATIO_1_1 = "1:1"
        RATIO_4_5 = "4:5"
        RATIO_3_4 = "3:4"
        RATIO_2_3 = "2:3"
        RATIO_9_16 = "9:16"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The text prompt to generate an image from."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    limit_generations: bool = Field(
        default=False, description="Experimental parameter to limit the number of generations from each round of prompting to 1. Set to `True` to to disregard any instructions in the prompt regarding the number of images to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_images": self.num_images,
            "sync_mode": self.sync_mode,
            "output_format": self.output_format.value,
            "limit_generations": self.limit_generations,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/nano-banana",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BytedanceDreaminaV31TextToImage(FALNode):
    """
    Bytedance
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The text prompt used to generate the image"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="The size of the generated image. Width and height must be between 512 and 2048."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seed: int = Field(
        default=-1, description="Random seed to control the stochasticity of image generation."
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to use an LLM to enhance the prompt"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/dreamina/v3.1/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV22A14BTextToImageLora(FALNode):
    """
    Wan v2.2 A14B Text-to-Image A14B with LoRAs
    generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'.
        """
        NONE = "none"
        REGULAR = "regular"

    class ImageFormat(Enum):
        """
        The format of the output image.
        """
        PNG = "png"
        JPEG = "jpeg"


    shift: float = Field(
        default=2, description="Shift value for the image. Must be between 1.0 and 10.0."
    )
    prompt: str = Field(
        default="", description="The text prompt to guide image generation."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
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
    guidance_scale: float = Field(
        default=3.5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    image_format: ImageFormat = Field(
        default=ImageFormat.JPEG, description="The format of the output image."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    guidance_scale_2: float = Field(
        default=4, description="Guidance scale for the second stage of the model. This is used to control the adherence to the prompt in the second stage of the model."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    num_inference_steps: int = Field(
        default=27, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "shift": self.shift,
            "prompt": self.prompt,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "reverse_video": self.reverse_video,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "image_format": self.image_format.value,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "guidance_scale_2": self.guidance_scale_2,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-a14b/text-to-image/lora",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV225BTextToImage(FALNode):
    """
    Wan
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class ImageFormat(Enum):
        """
        The format of the output image.
        """
        PNG = "png"
        JPEG = "jpeg"


    prompt: str = Field(
        default="", description="The text prompt to guide image generation."
    )
    image_format: ImageFormat = Field(
        default=ImageFormat.JPEG, description="The format of the output image."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    shift: float = Field(
        default=2, description="Shift value for the image. Must be between 1.0 and 10.0."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_inference_steps: int = Field(
        default=40, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "image_format": self.image_format.value,
            "image_size": self.image_size,
            "shift": self.shift,
            "seed": self.seed,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-5b/text-to-image",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class WanV22A14BTextToImage(FALNode):
    """
    Wan
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'.
        """
        NONE = "none"
        REGULAR = "regular"


    prompt: str = Field(
        default="", description="The text prompt to guide image generation."
    )
    shift: float = Field(
        default=2, description="Shift value for the image. Must be between 1.0 and 10.0."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="Acceleration level to use. The more acceleration, the faster the generation, but with lower quality. The recommended value is 'regular'."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    enable_output_safety_checker: bool = Field(
        default=False, description="If set to true, output video will be checked for safety after generation."
    )
    guidance_scale_2: float = Field(
        default=4, description="Guidance scale for the second stage of the model. This is used to control the adherence to the prompt in the second stage of the model."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier-free guidance scale. Higher values give better adherence to the prompt but may decrease quality."
    )
    num_inference_steps: int = Field(
        default=27, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, input data will be checked for safety before processing."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for video generation."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion. This will use a large language model to expand the prompt with additional details while maintaining the original meaning."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "shift": self.shift,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "seed": self.seed,
            "enable_output_safety_checker": self.enable_output_safety_checker,
            "guidance_scale_2": self.guidance_scale_2,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "enable_prompt_expansion": self.enable_prompt_expansion,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/wan/v2.2-a14b/text-to-image",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class QwenImage(FALNode):
    """
    Qwen Image
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. Options: 'none', 'regular', 'high'. Higher acceleration increases speed. 'regular' balances speed and quality. 'high' is recommended for images without text.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate the image with"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Acceleration level for image generation. Options: 'none', 'regular', 'high'. Higher acceleration increases speed. 'regular' balances speed and quality. 'high' is recommended for images without text."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to perform."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use up to 3 LoRAs and they will be merged together to generate the final image."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    use_turbo: bool = Field(
        default=False, description="Enable turbo mode for faster generation with high quality. When enabled, uses optimized settings (10 steps, CFG=1.2)."
    )
    negative_prompt: str = Field(
        default=" ", description="The negative prompt for the generation"
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "num_inference_steps": self.num_inference_steps,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "use_turbo": self.use_turbo,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/qwen-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxKreaLoraStream(FALNode):
    """
    Flux Krea Lora
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate. This is always set to 1 for streaming output."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-krea-lora/stream",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxKreaLora(FALNode):
    """
    FLUX.1 Krea [dev] with LoRAs
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate. This is always set to 1 for streaming output."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-krea-lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxKrea(FALNode):
    """
    FLUX.1 Krea [dev]
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        The speed of the generation. The higher the speed, the faster the generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=4.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux/krea",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux1Krea(FALNode):
    """
    FLUX.1 Krea [dev]
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        The speed of the generation. The higher the speed, the faster the generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=4.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-1/krea",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class SkyRaccoon(FALNode):
    """
    Sky Raccoon
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The text prompt to guide video generation."
    )
    image_size: str = Field(
        default="", description="The size of the generated image."
    )
    turbo_mode: bool = Field(
        default=False, description="If true, the video will be generated faster with no noticeable degradation in the visual quality."
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="Whether to enable prompt expansion."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducibility. If None, a random seed is chosen."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards", description="Negative prompt for video generation."
    )
    num_inference_steps: int = Field(
        default=30, description="Number of inference steps for sampling. Higher values give better quality but take longer."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "turbo_mode": self.turbo_mode,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sky-raccoon",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxKontextLoraTextToImage(FALNode):
    """
    Flux Kontext Lora
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        The speed of the generation. The higher the speed, the faster the generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate the image with"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=2.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to perform."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-kontext-lora/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class OmnigenV2(FALNode):
    """
    Omnigen V2
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Scheduler(Enum):
        """
        The scheduler to use for the diffusion process.
        """
        EULER = "euler"
        DPMSOLVER = "dpmsolver"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate or edit an image. Use specific language like 'Add the bird from image 1 to the desk in image 2' for better results."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULER, description="The scheduler to use for the diffusion process."
    )
    cfg_range_end: float = Field(
        default=1, description="CFG range end value."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar", description="Negative prompt to guide what should not be in the image."
    )
    text_guidance_scale: float = Field(
        default=5, description="The Text Guidance scale controls how closely the model follows the text prompt. Higher values make the model stick more closely to the prompt."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_guidance_scale: float = Field(
        default=2, description="The Image Guidance scale controls how closely the model follows the input images. For image editing: 1.3-2.0, for in-context generation: 2.0-3.0"
    )
    input_images: list[str] = Field(
        default=[], description="URLs of input images to use for image editing or multi-image generation. Support up to 3 images."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    cfg_range_start: float = Field(
        default=0, description="CFG range start value."
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "scheduler": self.scheduler.value,
            "cfg_range_end": self.cfg_range_end,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "text_guidance_scale": self.text_guidance_scale,
            "num_images": self.num_images,
            "image_guidance_scale": self.image_guidance_scale,
            "input_image_urls": self.input_images,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "cfg_range_start": self.cfg_range_start,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/omnigen-v2",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BytedanceSeedreamV3TextToImage(FALNode):
    """
    Bytedance
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The text prompt used to generate the image"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: str = Field(
        default="", description="Use for finer control over the output image size. Will be used over aspect_ratio, if both are provided. Width and height must be between 512 and 2048."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=2.5, description="Controls how closely the output image aligns with the input prompt. Higher values mean stronger prompt correlation."
    )
    seed: int = Field(
        default=-1, description="Random seed to control the stochasticity of image generation."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bytedance/seedream/v3/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux1Schnell(FALNode):
    """
    Fastest inference in the world for the 12 billion parameter FLUX.1 [schnell] text-to-image model. 
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        The speed of the generation. The higher the speed, the faster the generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=4, description="The number of inference steps to perform."
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-1/schnell",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Flux1Dev(FALNode):
    """
    FLUX.1 [dev] is a 12 billion parameter flow transformer that generates high-quality images from text. It is suitable for personal and commercial use. 
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        The speed of the generation. The higher the speed, the faster the generation.
        """
        NONE = "none"
        REGULAR = "regular"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.REGULAR, description="The speed of the generation. The higher the speed, the faster the generation."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    seed: str = Field(
        default="", description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-1/dev",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxProKontextMaxTextToImage(FALNode):
    """
    FLUX.1 Kontext [max] text-to-image is a new premium model brings maximum performance across all aspects – greatly improved prompt adherence.
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image.
        """
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_3_2 = "3:2"
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"
        RATIO_9_21 = "9:21"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"
        VALUE_6 = "6"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/kontext/max/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxProKontextTextToImage(FALNode):
    """
    The FLUX.1 Kontext [pro] text-to-image delivers state-of-the-art image generation results with unprecedented prompt following, photorealistic rendering, and flawless typography.
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image.
        """
        RATIO_21_9 = "21:9"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_3_2 = "3:2"
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"
        RATIO_9_21 = "9:21"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"
        VALUE_6 = "6"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enhance_prompt": self.enhance_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/kontext/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Bagel(FALNode):
    """
    Bagel is a 7B parameter from Bytedance-Seed multimodal model that can generate both text and images.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The seed to use for the generation."
    )
    use_thought: bool = Field(
        default=False, description="Whether to use thought tokens for generation. If set to true, the model will \"think\" to potentially improve generation quality. Increases generation time and increases the cost by 20%."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "use_thought": self.use_thought,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bagel",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Imagen4PreviewUltra(FALNode):
    """
    Google’s highest quality image generation model
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image.
        """
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"

    class Resolution(Enum):
        """
        The resolution of the generated image.
        """
        VALUE_1K = "1K"
        VALUE_2K = "2K"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The text prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1K, description="The resolution of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "sync_mode": self.sync_mode,
            "output_format": self.output_format.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/imagen4/preview/ultra",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Dreamo(FALNode):
    """
    DreamO is an image customization framework designed to support a wide range of tasks while facilitating seamless integration of multiple conditions.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class SecondReferenceTask(Enum):
        """
        Task for second reference image (ip/id/style).
        """
        IP = "ip"
        ID = "id"
        STYLE = "style"

    class FirstReferenceTask(Enum):
        """
        Task for first reference image (ip/id/style).
        """
        IP = "ip"
        ID = "id"
        STYLE = "style"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    first_image: ImageRef = Field(
        default=ImageRef(), description="URL of first reference image to use for generation."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    second_image: ImageRef = Field(
        default=ImageRef(), description="URL of second reference image to use for generation."
    )
    second_reference_task: SecondReferenceTask = Field(
        default=SecondReferenceTask.IP, description="Task for second reference image (ip/id/style)."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    first_reference_task: FirstReferenceTask = Field(
        default=FirstReferenceTask.IP, description="Task for first reference image (ip/id/style)."
    )
    negative_prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    ref_resolution: int = Field(
        default=512, description="Resolution for reference images."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    true_cfg: float = Field(
        default=1, description="The weight of the CFG loss."
    )
    num_inference_steps: int = Field(
        default=12, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        first_image_base64 = await context.image_to_base64(self.first_image)
        second_image_base64 = await context.image_to_base64(self.second_image)
        arguments = {
            "prompt": self.prompt,
            "first_image_url": f"data:image/png;base64,{first_image_base64}",
            "image_size": self.image_size,
            "second_image_url": f"data:image/png;base64,{second_image_base64}",
            "second_reference_task": self.second_reference_task.value,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "first_reference_task": self.first_reference_task.value,
            "negative_prompt": self.negative_prompt,
            "ref_resolution": self.ref_resolution,
            "sync_mode": self.sync_mode,
            "true_cfg": self.true_cfg,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/dreamo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxLoraStream(FALNode):
    """
    Super fast endpoint for the FLUX.1 [dev] model with LoRA support, enabling rapid and high-quality image generation using pre-trained LoRA adaptations for personalization, specific styles, brand identities, and product-specific outputs.
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. 'regular' balances speed and quality.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate. This is always set to 1 for streaming output."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "acceleration": self.acceleration.value,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-lora/stream",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class MinimaxImage01(FALNode):
    """
    Generate high quality images from text prompts using MiniMax Image-01. Longer text prompts will result in better quality images.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        Aspect ratio of the generated image
        """
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"
        RATIO_4_3 = "4:3"
        RATIO_3_2 = "3:2"
        RATIO_2_3 = "2:3"
        RATIO_3_4 = "3:4"
        RATIO_9_16 = "9:16"
        RATIO_21_9 = "21:9"


    prompt: str = Field(
        default="", description="Text prompt for image generation (max 1500 characters)"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate (1-9)"
    )
    prompt_optimizer: bool = Field(
        default=False, description="Whether to enable automatic prompt optimization"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="Aspect ratio of the generated image"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "prompt_optimizer": self.prompt_optimizer,
            "aspect_ratio": self.aspect_ratio.value,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/minimax/image-01",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PonyV7(FALNode):
    """
    Pony V7 is a finetuned text to image for superior aesthetics and prompt following.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class NoiseSource(Enum):
        """
        The source of the noise to use for generating images.
        If set to 'gpu', the noise will be generated on the GPU.
        If set to 'cpu', the noise will be generated on the CPU.
        """
        GPU = "gpu"
        CPU = "cpu"


    prompt: str = Field(
        default="", description="The prompt to generate images from"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate"
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    noise_source: NoiseSource = Field(
        default=NoiseSource.GPU, description="The source of the noise to use for generating images. If set to 'gpu', the noise will be generated on the GPU. If set to 'cpu', the noise will be generated on the CPU."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier free guidance scale"
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to take"
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The seed to use for generating images"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "noise_source": self.noise_source.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pony-v7",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FLiteStandard(FALNode):
    """
    F Lite is a 10B parameter diffusion model created by Fal and Freepik, trained exclusively on copyright-safe and SFW content.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative Prompt for generation."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/f-lite/standard",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FLiteTexture(FALNode):
    """
    F Lite is a 10B parameter diffusion model created by Fal and Freepik, trained exclusively on copyright-safe and SFW content. This is a high texture density variant of the model.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="Negative Prompt for generation."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/f-lite/texture",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class GptImage1TextToImage(FALNode):
    """
    OpenAI's latest image generation and editing model: gpt-1-image.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class ImageSize(Enum):
        """
        Aspect ratio for the generated image
        """
        AUTO = "auto"
        VALUE_1024X1024 = "1024x1024"
        VALUE_1536X1024 = "1536x1024"
        VALUE_1024X1536 = "1024x1536"

    class Background(Enum):
        """
        Background for the generated image
        """
        AUTO = "auto"
        TRANSPARENT = "transparent"
        OPAQUE = "opaque"

    class Quality(Enum):
        """
        Quality for the generated image
        """
        AUTO = "auto"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class OutputFormat(Enum):
        """
        Output format for the images
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="The prompt for image generation"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    image_size: ImageSize = Field(
        default=ImageSize.AUTO, description="Aspect ratio for the generated image"
    )
    background: Background = Field(
        default=Background.AUTO, description="Background for the generated image"
    )
    quality: Quality = Field(
        default=Quality.AUTO, description="Quality for the generated image"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="Output format for the images"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "background": self.background.value,
            "quality": self.quality.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/gpt-image-1/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class SanaV1516b(FALNode):
    """
    Sana v1.5 1.6B is a lightweight text-to-image model that delivers 4K image generation with impressive efficiency.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class StyleName(Enum):
        """
        The style to generate the image in.
        """
        NO_STYLE = "(No style)"
        CINEMATIC = "Cinematic"
        PHOTOGRAPHIC = "Photographic"
        ANIME = "Anime"
        MANGA = "Manga"
        DIGITAL_ART = "Digital Art"
        PIXEL_ART = "Pixel art"
        FANTASY_ART = "Fantasy art"
        NEONPUNK = "Neonpunk"
        MODEL_3D = "3D Model"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image."
    )
    style_name: StyleName = Field(
        default=StyleName.NO_STYLE, description="The style to generate the image in."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=18, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "style_name": self.style_name.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sana/v1.5/1.6b",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class SanaV1548b(FALNode):
    """
    Sana v1.5 4.8B is a powerful text-to-image model that generates ultra-high quality 4K images with remarkable detail.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class StyleName(Enum):
        """
        The style to generate the image in.
        """
        NO_STYLE = "(No style)"
        CINEMATIC = "Cinematic"
        PHOTOGRAPHIC = "Photographic"
        ANIME = "Anime"
        MANGA = "Manga"
        DIGITAL_ART = "Digital Art"
        PIXEL_ART = "Pixel art"
        FANTASY_ART = "Fantasy art"
        NEONPUNK = "Neonpunk"
        MODEL_3D = "3D Model"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image."
    )
    style_name: StyleName = Field(
        default=StyleName.NO_STYLE, description="The style to generate the image in."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=18, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "style_name": self.style_name.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sana/v1.5/4.8b",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class SanaSprint(FALNode):
    """
    Sana Sprint is a text-to-image model capable of generating 4K images with exceptional speed.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class StyleName(Enum):
        """
        The style to generate the image in.
        """
        NO_STYLE = "(No style)"
        CINEMATIC = "Cinematic"
        PHOTOGRAPHIC = "Photographic"
        ANIME = "Anime"
        MANGA = "Manga"
        DIGITAL_ART = "Digital Art"
        PIXEL_ART = "Pixel art"
        FANTASY_ART = "Fantasy art"
        NEONPUNK = "Neonpunk"
        MODEL_3D = "3D Model"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image."
    )
    style_name: StyleName = Field(
        default=StyleName.NO_STYLE, description="The style to generate the image in."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=2, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "style_name": self.style_name.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sana/sprint",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class RundiffusionFalJuggernautFluxLora(FALNode):
    """
    Juggernaut Base Flux LoRA by RunDiffusion is a drop-in replacement for Flux [Dev] that delivers sharper details, richer colors, and enhanced realism to all your LoRAs and LyCORIS with full compatibility.
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="rundiffusion-fal/juggernaut-flux-lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class RundiffusionFalJuggernautFluxBase(FALNode):
    """
    Juggernaut Base Flux by RunDiffusion is a drop-in replacement for Flux [Dev] that delivers sharper details, richer colors, and enhanced realism, while instantly boosting LoRAs and LyCORIS with full compatibility.
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="rundiffusion-fal/juggernaut-flux/base",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class RundiffusionFalJuggernautFluxLightning(FALNode):
    """
    Juggernaut Lightning Flux by RunDiffusion provides blazing-fast, high-quality images rendered at five times the speed of Flux. Perfect for mood boards and mass ideation, this model excels in both realism and prompt adherence.
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=4, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="rundiffusion-fal/juggernaut-flux/lightning",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class RundiffusionFalJuggernautFluxPro(FALNode):
    """
    Juggernaut Pro Flux by RunDiffusion is the flagship Juggernaut model rivaling some of the most advanced image models available, often surpassing them in realism. It combines Juggernaut Base with RunDiffusion Photo and features enhancements like reduced background blurriness.
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="rundiffusion-fal/juggernaut-flux/pro",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class RundiffusionFalRundiffusionPhotoFlux(FALNode):
    """
    RunDiffusion Photo Flux provides insane realism. With this enhancer, textures and skin details burst to life, turning your favorite prompts into vivid, lifelike creations. Recommended to keep it at 0.65 to 0.80 weight. Supports resolutions up to 1536x1536.
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    photo_lora_scale: float = Field(
        default=0.75, description="LoRA Scale of the photo lora model"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "photo_lora_scale": self.photo_lora_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="rundiffusion-fal/rundiffusion-photo-flux",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Cogview4(FALNode):
    """
    Generate high quality images from text prompts using CogView4. Longer text prompts will result in better quality images.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to perform."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/cogview4",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class IdeogramV2a(FALNode):
    """
    Generate high-quality images, posters, and logos with Ideogram V2A. Features exceptional typography handling and realistic outputs optimized for commercial and creative use.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image
        """
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

    class Style(Enum):
        """
        The style of the generated image
        """
        AUTO = "auto"
        GENERAL = "general"
        REALISTIC = "realistic"
        DESIGN = "design"
        RENDER_3D = "render_3D"
        ANIME = "anime"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    style: Style = Field(
        default=Style.AUTO, description="The style of the generated image"
    )
    seed: str = Field(
        default="", description="Seed for the random number generator"
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to expand the prompt with MagicPrompt functionality."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "style": self.style.value,
            "seed": self.seed,
            "expand_prompt": self.expand_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2a",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class IdeogramV2aTurbo(FALNode):
    """
    Accelerated image generation with Ideogram V2A Turbo. Create high-quality visuals, posters, and logos with enhanced speed while maintaining Ideogram's signature quality.
    generation, text-to-image, txt2img, ai-art, fast

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image
        """
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

    class Style(Enum):
        """
        The style of the generated image
        """
        AUTO = "auto"
        GENERAL = "general"
        REALISTIC = "realistic"
        DESIGN = "design"
        RENDER_3D = "render_3D"
        ANIME = "anime"


    prompt: str = Field(
        default=""
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    style: Style = Field(
        default=Style.AUTO, description="The style of the generated image"
    )
    seed: str = Field(
        default="", description="Seed for the random number generator"
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to expand the prompt with MagicPrompt functionality."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "style": self.style.value,
            "seed": self.seed,
            "expand_prompt": self.expand_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/ideogram/v2a/turbo",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxControlLoraCanny(FALNode):
    """
    FLUX Control LoRA Canny is a high-performance endpoint that uses a control image to transfer structure to the generated image, using a Canny edge map.
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    control_lora_strength: float = Field(
        default=1, description="The strength of the control lora."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    control_lora_image: ImageRef = Field(
        default=ImageRef(), description="The image to use for control lora. This is used to control the style of the generated image."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_lora_image_base64 = await context.image_to_base64(self.control_lora_image)
        arguments = {
            "control_lora_strength": self.control_lora_strength,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "prompt": self.prompt,
            "output_format": self.output_format.value,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "control_lora_image_url": f"data:image/png;base64,{control_lora_image_base64}",
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-control-lora-canny",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxControlLoraDepth(FALNode):
    """
    FLUX Control LoRA Depth is a high-performance endpoint that uses a control image to transfer structure to the generated image, using a depth map.
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    control_lora_strength: float = Field(
        default=1, description="The strength of the control lora."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    preprocess_depth: bool = Field(
        default=True, description="If set to true, the input image will be preprocessed to extract depth information. This is useful for generating depth maps from images."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    control_lora_image: ImageRef = Field(
        default=ImageRef(), description="The image to use for control lora. This is used to control the style of the generated image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_lora_image_base64 = await context.image_to_base64(self.control_lora_image)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "control_lora_strength": self.control_lora_strength,
            "output_format": self.output_format.value,
            "preprocess_depth": self.preprocess_depth,
            "sync_mode": self.sync_mode,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "guidance_scale": self.guidance_scale,
            "control_lora_image_url": f"data:image/png;base64,{control_lora_image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-control-lora-depth",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Imagen3Fast(FALNode):
    """
    Imagen3 Fast is a high-quality text-to-image model that generates realistic images from text prompts.
    generation, text-to-image, txt2img, ai-art, fast

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image
        """
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_3_4 = "3:4"
        RATIO_4_3 = "4:3"


    prompt: str = Field(
        default="", description="The text prompt describing what you want to see"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate (1-4)"
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation"
    )
    negative_prompt: str = Field(
        default="", description="A description of what to discourage in the generated images"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_images": self.num_images,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/imagen3/fast",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Imagen3(FALNode):
    """
    Imagen3 is a high-quality text-to-image model that generates realistic images from text prompts.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the generated image
        """
        RATIO_1_1 = "1:1"
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_3_4 = "3:4"
        RATIO_4_3 = "4:3"


    prompt: str = Field(
        default="", description="The text prompt describing what you want to see"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the generated image"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate (1-4)"
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation"
    )
    negative_prompt: str = Field(
        default="", description="A description of what to discourage in the generated images"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "num_images": self.num_images,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/imagen3",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LuminaImageV2(FALNode):
    """
    Lumina-Image-2.0 is a 2 billion parameter flow-based diffusion transforer which features improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    cfg_trunc_ratio: float = Field(
        default=1, description="The ratio of the timestep interval to apply normalization-based guidance scale."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    system_prompt: str = Field(
        default="You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts.", description="The system prompt to use."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of inference steps to perform."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    cfg_normalization: bool = Field(
        default=True, description="Whether to apply normalization-based guidance scale."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "cfg_trunc_ratio": self.cfg_trunc_ratio,
            "seed": self.seed,
            "output_format": self.output_format.value,
            "system_prompt": self.system_prompt,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "cfg_normalization": self.cfg_normalization,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/lumina-image/v2",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Janus(FALNode):
    """
    DeepSeek Janus-Pro is a novel text-to-image model that unifies multimodal understanding and generation through an autoregressive framework
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate in parallel."
    )
    image_size: str = Field(
        default="square", description="The size of the generated image."
    )
    cfg_weight: float = Field(
        default=5, description="Classifier Free Guidance scale - how closely to follow the prompt."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    temperature: float = Field(
        default=1, description="Controls randomness in the generation. Higher values make output more random."
    )
    seed: int = Field(
        default=-1, description="Random seed for reproducible generation."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "cfg_weight": self.cfg_weight,
            "enable_safety_checker": self.enable_safety_checker,
            "temperature": self.temperature,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/janus",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxProV11UltraFinetuned(FALNode):
    """
    FLUX1.1 [pro] ultra fine-tuned is the newest version of FLUX1.1 [pro] with a fine-tuned LoRA, maintaining professional-grade image quality while delivering up to 2K resolution with improved photo realism.
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class SafetyTolerance(Enum):
        """
        The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_3 = "3"
        VALUE_4 = "4"
        VALUE_5 = "5"
        VALUE_6 = "6"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    finetune_id: str = Field(
        default="", description="References your specific model"
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    image_prompt_strength: float = Field(
        default=0.1, description="The strength of the image prompt, between 0 and 1."
    )
    raw: bool = Field(
        default=False, description="Generate less processed, more natural-looking images."
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )
    aspect_ratio: str = Field(
        default="16:9", description="The aspect ratio of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The image URL to generate an image from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    finetune_strength: float = Field(
        default=0.0, description="Controls finetune influence. Increase this value if your target concept isn't showing up strongly enough. The optimal setting depends on your finetune and prompt"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "finetune_id": self.finetune_id,
            "safety_tolerance": self.safety_tolerance.value,
            "enable_safety_checker": self.enable_safety_checker,
            "image_prompt_strength": self.image_prompt_strength,
            "raw": self.raw,
            "enhance_prompt": self.enhance_prompt,
            "aspect_ratio": self.aspect_ratio,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "finetune_strength": self.finetune_strength,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-pro/v1.1-ultra-finetuned",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Switti(FALNode):
    """
    Switti is a scale-wise transformer for fast text-to-image generation that outperforms existing T2I AR models and competes with state-of-the-art T2I diffusion models while being faster than distilled diffusion models.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    sampling_top_k: int = Field(
        default=400, description="The number of top-k tokens to sample from."
    )
    turn_off_cfg_start_si: int = Field(
        default=8, description="Disable CFG starting scale"
    )
    guidance_scale: float = Field(
        default=6, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    smooth_start_si: int = Field(
        default=2, description="Smoothing starting scale"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    last_scale_temp: float = Field(
        default=0.1, description="Temperature after disabling CFG"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    more_diverse: bool = Field(
        default=False, description="More diverse sampling"
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    more_smooth: bool = Field(
        default=True, description="Smoothing with Gumbel softmax sampling"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    sampling_top_p: float = Field(
        default=0.95, description="The top-p probability to sample from."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "sampling_top_k": self.sampling_top_k,
            "turn_off_cfg_start_si": self.turn_off_cfg_start_si,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "smooth_start_si": self.smooth_start_si,
            "negative_prompt": self.negative_prompt,
            "last_scale_temp": self.last_scale_temp,
            "output_format": self.output_format.value,
            "more_diverse": self.more_diverse,
            "sync_mode": self.sync_mode,
            "more_smooth": self.more_smooth,
            "seed": self.seed,
            "sampling_top_p": self.sampling_top_p,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/switti",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Switti512(FALNode):
    """
    Switti is a scale-wise transformer for fast text-to-image generation that outperforms existing T2I AR models and competes with state-of-the-art T2I diffusion models while being faster than distilled diffusion models.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    sampling_top_k: int = Field(
        default=400, description="The number of top-k tokens to sample from."
    )
    turn_off_cfg_start_si: int = Field(
        default=8, description="Disable CFG starting scale"
    )
    guidance_scale: float = Field(
        default=6, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    smooth_start_si: int = Field(
        default=2, description="Smoothing starting scale"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    last_scale_temp: float = Field(
        default=0.1, description="Temperature after disabling CFG"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    more_diverse: bool = Field(
        default=False, description="More diverse sampling"
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    more_smooth: bool = Field(
        default=True, description="Smoothing with Gumbel softmax sampling"
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    sampling_top_p: float = Field(
        default=0.95, description="The top-p probability to sample from."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "sampling_top_k": self.sampling_top_k,
            "turn_off_cfg_start_si": self.turn_off_cfg_start_si,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "smooth_start_si": self.smooth_start_si,
            "negative_prompt": self.negative_prompt,
            "last_scale_temp": self.last_scale_temp,
            "output_format": self.output_format.value,
            "more_diverse": self.more_diverse,
            "sync_mode": self.sync_mode,
            "more_smooth": self.more_smooth,
            "seed": self.seed,
            "sampling_top_p": self.sampling_top_p,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/switti/512",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BriaTextToImageBase(FALNode):
    """
    Bria's Text-to-Image model, trained exclusively on licensed data for safe and risk-free commercial use. Available also as source code and weights. For access to weights: https://bria.ai/contact-us
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the image. When a guidance method is being used, the aspect ratio is defined by the guidance image and this parameter is ignored.
        """
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"
        RATIO_3_2 = "3:2"
        RATIO_3_4 = "3:4"
        RATIO_4_3 = "4:3"
        RATIO_4_5 = "4:5"
        RATIO_5_4 = "5:4"
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"

    class Medium(Enum):
        """
        Which medium should be included in your generated images. This parameter is optional.
        """
        PHOTOGRAPHY = "photography"
        ART = "art"


    prompt: str = Field(
        default="", description="The prompt you would like to use to generate images."
    )
    num_images: int = Field(
        default=4, description="How many images you would like to generate. When using any Guidance Method, Value is set to 1."
    )
    prompt_enhancement: bool = Field(
        default=False, description="When set to true, enhances the provided prompt by generating additional, more descriptive variations, resulting in more diverse and creative output images."
    )
    guidance: list[GuidanceInput] = Field(
        default=[], description="Guidance images to use for the generation. Up to 4 guidance methods can be combined during a single inference."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the image. When a guidance method is being used, the aspect ratio is defined by the guidance image and this parameter is ignored."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    medium: Medium | None = Field(
        default=None, description="Which medium should be included in your generated images. This parameter is optional."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt you would like to use to generate images."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of iterations the model goes through to refine the generated image. This parameter is optional."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "prompt_enhancement": self.prompt_enhancement,
            "guidance": [item.model_dump(exclude={"type"}) for item in self.guidance],
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "medium": self.medium.value if self.medium else None,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/text-to-image/base",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BriaTextToImageFast(FALNode):
    """
    Bria's Text-to-Image model with perfect harmony of latency and quality. Trained exclusively on licensed data for safe and risk-free commercial use. Available also as source code and weights. For access to weights: https://bria.ai/contact-us
    generation, text-to-image, txt2img, ai-art, fast

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the image. When a guidance method is being used, the aspect ratio is defined by the guidance image and this parameter is ignored.
        """
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"
        RATIO_3_2 = "3:2"
        RATIO_3_4 = "3:4"
        RATIO_4_3 = "4:3"
        RATIO_4_5 = "4:5"
        RATIO_5_4 = "5:4"
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"

    class Medium(Enum):
        """
        Which medium should be included in your generated images. This parameter is optional.
        """
        PHOTOGRAPHY = "photography"
        ART = "art"


    prompt: str = Field(
        default="", description="The prompt you would like to use to generate images."
    )
    num_images: int = Field(
        default=4, description="How many images you would like to generate. When using any Guidance Method, Value is set to 1."
    )
    prompt_enhancement: bool = Field(
        default=False, description="When set to true, enhances the provided prompt by generating additional, more descriptive variations, resulting in more diverse and creative output images."
    )
    guidance: list[GuidanceInput] = Field(
        default=[], description="Guidance images to use for the generation. Up to 4 guidance methods can be combined during a single inference."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the image. When a guidance method is being used, the aspect ratio is defined by the guidance image and this parameter is ignored."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    medium: Medium | None = Field(
        default=None, description="Which medium should be included in your generated images. This parameter is optional."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt you would like to use to generate images."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of iterations the model goes through to refine the generated image. This parameter is optional."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "prompt_enhancement": self.prompt_enhancement,
            "guidance": [item.model_dump(exclude={"type"}) for item in self.guidance],
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "medium": self.medium.value if self.medium else None,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/text-to-image/fast",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class BriaTextToImageHd(FALNode):
    """
    Bria's Text-to-Image model for HD images. Trained exclusively on licensed data for safe and risk-free commercial use. Available also as source code and weights. For access to weights: https://bria.ai/contact-us
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class AspectRatio(Enum):
        """
        The aspect ratio of the image. When a guidance method is being used, the aspect ratio is defined by the guidance image and this parameter is ignored.
        """
        RATIO_1_1 = "1:1"
        RATIO_2_3 = "2:3"
        RATIO_3_2 = "3:2"
        RATIO_3_4 = "3:4"
        RATIO_4_3 = "4:3"
        RATIO_4_5 = "4:5"
        RATIO_5_4 = "5:4"
        RATIO_9_16 = "9:16"
        RATIO_16_9 = "16:9"

    class Medium(Enum):
        """
        Which medium should be included in your generated images. This parameter is optional.
        """
        PHOTOGRAPHY = "photography"
        ART = "art"


    prompt: str = Field(
        default="", description="The prompt you would like to use to generate images."
    )
    num_images: int = Field(
        default=4, description="How many images you would like to generate. When using any Guidance Method, Value is set to 1."
    )
    prompt_enhancement: bool = Field(
        default=False, description="When set to true, enhances the provided prompt by generating additional, more descriptive variations, resulting in more diverse and creative output images."
    )
    guidance: list[GuidanceInput] = Field(
        default=[], description="Guidance images to use for the generation. Up to 4 guidance methods can be combined during a single inference."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_1_1, description="The aspect ratio of the image. When a guidance method is being used, the aspect ratio is defined by the guidance image and this parameter is ignored."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    medium: Medium | None = Field(
        default=None, description="Which medium should be included in your generated images. This parameter is optional."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt you would like to use to generate images."
    )
    num_inference_steps: int = Field(
        default=30, description="The number of iterations the model goes through to refine the generated image. This parameter is optional."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "prompt_enhancement": self.prompt_enhancement,
            "guidance": [item.model_dump(exclude={"type"}) for item in self.guidance],
            "aspect_ratio": self.aspect_ratio.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "medium": self.medium.value if self.medium else None,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/bria/text-to-image/hd",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Recraft20b(FALNode):
    """
    Recraft 20b is a new and affordable text-to-image model.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Style(Enum):
        """
        The style of the generated images. Vector images cost 2X as much.
        """
        ANY = "any"
        REALISTIC_IMAGE = "realistic_image"
        DIGITAL_ILLUSTRATION = "digital_illustration"
        VECTOR_ILLUSTRATION = "vector_illustration"
        REALISTIC_IMAGE_B_AND_W = "realistic_image/b_and_w"
        REALISTIC_IMAGE_ENTERPRISE = "realistic_image/enterprise"
        REALISTIC_IMAGE_HARD_FLASH = "realistic_image/hard_flash"
        REALISTIC_IMAGE_HDR = "realistic_image/hdr"
        REALISTIC_IMAGE_MOTION_BLUR = "realistic_image/motion_blur"
        REALISTIC_IMAGE_NATURAL_LIGHT = "realistic_image/natural_light"
        REALISTIC_IMAGE_STUDIO_PORTRAIT = "realistic_image/studio_portrait"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER = "digital_illustration/2d_art_poster"
        DIGITAL_ILLUSTRATION_2D_ART_POSTER_2 = "digital_illustration/2d_art_poster_2"
        DIGITAL_ILLUSTRATION_3D = "digital_illustration/3d"
        DIGITAL_ILLUSTRATION_80S = "digital_illustration/80s"
        DIGITAL_ILLUSTRATION_ENGRAVING_COLOR = "digital_illustration/engraving_color"
        DIGITAL_ILLUSTRATION_GLOW = "digital_illustration/glow"
        DIGITAL_ILLUSTRATION_GRAIN = "digital_illustration/grain"
        DIGITAL_ILLUSTRATION_HAND_DRAWN = "digital_illustration/hand_drawn"
        DIGITAL_ILLUSTRATION_HAND_DRAWN_OUTLINE = "digital_illustration/hand_drawn_outline"
        DIGITAL_ILLUSTRATION_HANDMADE_3D = "digital_illustration/handmade_3d"
        DIGITAL_ILLUSTRATION_INFANTILE_SKETCH = "digital_illustration/infantile_sketch"
        DIGITAL_ILLUSTRATION_KAWAII = "digital_illustration/kawaii"
        DIGITAL_ILLUSTRATION_PIXEL_ART = "digital_illustration/pixel_art"
        DIGITAL_ILLUSTRATION_PSYCHEDELIC = "digital_illustration/psychedelic"
        DIGITAL_ILLUSTRATION_SEAMLESS = "digital_illustration/seamless"
        DIGITAL_ILLUSTRATION_VOXEL = "digital_illustration/voxel"
        DIGITAL_ILLUSTRATION_WATERCOLOR = "digital_illustration/watercolor"
        VECTOR_ILLUSTRATION_CARTOON = "vector_illustration/cartoon"
        VECTOR_ILLUSTRATION_DOODLE_LINE_ART = "vector_illustration/doodle_line_art"
        VECTOR_ILLUSTRATION_ENGRAVING = "vector_illustration/engraving"
        VECTOR_ILLUSTRATION_FLAT_2 = "vector_illustration/flat_2"
        VECTOR_ILLUSTRATION_KAWAII = "vector_illustration/kawaii"
        VECTOR_ILLUSTRATION_LINE_ART = "vector_illustration/line_art"
        VECTOR_ILLUSTRATION_LINE_CIRCUIT = "vector_illustration/line_circuit"
        VECTOR_ILLUSTRATION_LINOCUT = "vector_illustration/linocut"
        VECTOR_ILLUSTRATION_SEAMLESS = "vector_illustration/seamless"


    prompt: str = Field(
        default=""
    )
    image_size: str = Field(
        default="square_hd"
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    colors: list[RGBColor] = Field(
        default=[], description="An array of preferable colors"
    )
    style: Style = Field(
        default=Style.REALISTIC_IMAGE, description="The style of the generated images. Vector images cost 2X as much."
    )
    style_id: str = Field(
        default="", description="The ID of the custom style reference (optional)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "enable_safety_checker": self.enable_safety_checker,
            "colors": [item.model_dump(exclude={"type"}) for item in self.colors],
            "style": self.style.value,
            "style_id": self.style_id,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/recraft-20b",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LumaPhotonFlash(FALNode):
    """
    Generate images from your prompts using Luma Photon Flash. Photon Flash is the most creative, personalizable, and intelligent visual models for creatives, bringing a step-function change in the cost of high-quality image generation.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
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
            application="fal-ai/luma-photon/flash",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class AuraFlow(FALNode):
    """
    AuraFlow v0.3 is an open-source flow-based text-to-image generation model that achieves state-of-the-art results on GenEval. The model is currently in beta.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt to generate images from"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate"
    )
    expand_prompt: bool = Field(
        default=True, description="Whether to perform prompt expansion (recommended)"
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=3.5, description="Classifier free guidance scale"
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to take"
    )
    seed: int = Field(
        default=-1, description="The seed to use for generating images"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "expand_prompt": self.expand_prompt,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/aura-flow",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class StableDiffusionV35Medium(FALNode):
    """
    Stable Diffusion 3.5 Medium is a Multimodal Diffusion Transformer (MMDiT) text-to-image model that features improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="landscape_4_3", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=4.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=40, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-diffusion-v35-medium",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxLoraInpainting(FALNode):
    """
    Super fast endpoint for the FLUX.1 [dev] inpainting model with LoRA support, enabling rapid and high-quality image inpaingting using pre-trained LoRA adaptations for personalization, specific styles, brand identities, and product-specific outputs.
    flux, generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Acceleration(Enum):
        """
        Acceleration level for image generation. 'regular' balances speed and quality.
        """
        NONE = "none"
        REGULAR = "regular"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    acceleration: Acceleration = Field(
        default=Acceleration.NONE, description="Acceleration level for image generation. 'regular' balances speed and quality."
    )
    image_size: str = Field(
        default="", description="The size of the generated image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate. This is always set to 1 for streaming output."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of image to use for inpainting. or img2img"
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    strength: float = Field(
        default=0.85, description="The strength to use for inpainting/image-to-image. Only used if the image_url is provided. 1.0 is completely remakes the image while 0.0 preserves the original."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    mask_url: str = Field(
        default="", description="The mask to area to Inpaint in."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "acceleration": self.acceleration.value,
            "image_size": self.image_size,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "mask_url": self.mask_url,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-lora/inpainting",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class StableDiffusionV3Medium(FALNode):
    """
    Stable Diffusion 3 Medium (Text to Image) is a Multimodal Diffusion Transformer (MMDiT) model that improves image quality, typography, prompt understanding, and efficiency.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt_expansion: bool = Field(
        default=False, description="If set to true, prompt will be upsampled with more details."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to generate an image from."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt_expansion": self.prompt_expansion,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "prompt": self.prompt,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-diffusion-v3-medium",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FooocusUpscaleOrVary(FALNode):
    """
    Default parameters with automated optimizations and quality improvements.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Performance(Enum):
        """
        You can choose Speed or Quality
        """
        SPEED = "Speed"
        QUALITY = "Quality"
        EXTREME_SPEED = "Extreme Speed"
        LIGHTNING = "Lightning"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"

    class RefinerModel(Enum):
        """
        Refiner (SDXL or SD 1.5)
        """
        NONE = "None"
        REALISTICVISIONV60B1_V51VAE_SAFETENSORS = "realisticVisionV60B1_v51VAE.safetensors"

    class UovMethod(Enum):
        """
        The method to use for upscaling or varying.
        """
        DISABLED = "Disabled"
        VARY_SUBTLE = "Vary (Subtle)"
        VARY_STRONG = "Vary (Strong)"
        UPSCALE_1_5X = "Upscale (1.5x)"
        UPSCALE_2X = "Upscale (2x)"
        UPSCALE_FAST_2X = "Upscale (Fast 2x)"


    styles: list[str] = Field(
        default=[], description="The style to use."
    )
    uov_image: ImageRef = Field(
        default=ImageRef(), description="The image to upscale or vary."
    )
    performance: Performance = Field(
        default=Performance.EXTREME_SPEED, description="You can choose Speed or Quality"
    )
    mixing_image_prompt_and_vary_upscale: bool = Field(
        default=False, description="Mixing Image Prompt and Vary/Upscale"
    )
    image_prompt_3: str = Field(
        default=""
    )
    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use up to 5 LoRAs and they will be merged together to generate the final image."
    )
    image_prompt_4: str = Field(
        default=""
    )
    image_prompt_1: str = Field(
        default=""
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to false, the safety checker will be disabled."
    )
    sharpness: float = Field(
        default=2, description="The sharpness of the generated image. Use it to control how sharp the generated image should be. Higher value means image and texture are sharper."
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    aspect_ratio: str = Field(
        default="1024x1024", description="The size of the generated image. You can choose between some presets or custom height and width that **must be multiples of 8**."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate in one request"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    refiner_model: RefinerModel = Field(
        default=RefinerModel.NONE, description="Refiner (SDXL or SD 1.5)"
    )
    image_prompt_2: str = Field(
        default=""
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    uov_method: UovMethod = Field(
        default=UovMethod.VARY_STRONG, description="The method to use for upscaling or varying."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    refiner_switch: float = Field(
        default=0.8, description="Use 0.4 for SD1.5 realistic models; 0.667 for SD1.5 anime models 0.8 for XL-refiners; or any value for switching two SDXL models."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        uov_image_base64 = await context.image_to_base64(self.uov_image)
        arguments = {
            "styles": self.styles,
            "uov_image_url": f"data:image/png;base64,{uov_image_base64}",
            "performance": self.performance.value,
            "mixing_image_prompt_and_vary_upscale": self.mixing_image_prompt_and_vary_upscale,
            "image_prompt_3": self.image_prompt_3,
            "prompt": self.prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "image_prompt_4": self.image_prompt_4,
            "image_prompt_1": self.image_prompt_1,
            "enable_safety_checker": self.enable_safety_checker,
            "sharpness": self.sharpness,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "aspect_ratio": self.aspect_ratio,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "refiner_model": self.refiner_model.value,
            "image_prompt_2": self.image_prompt_2,
            "sync_mode": self.sync_mode,
            "uov_method": self.uov_method.value,
            "seed": self.seed,
            "refiner_switch": self.refiner_switch,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fooocus/upscale-or-vary",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PixartSigma(FALNode):
    """
    Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Style(Enum):
        """
        The style to apply to the image.
        """
        NO_STYLE = "(No style)"
        CINEMATIC = "Cinematic"
        PHOTOGRAPHIC = "Photographic"
        ANIME = "Anime"
        MANGA = "Manga"
        DIGITAL_ART = "Digital Art"
        PIXEL_ART = "Pixel art"
        FANTASY_ART = "Fantasy art"
        NEONPUNK = "Neonpunk"
        MODEL_3D = "3D Model"

    class Scheduler(Enum):
        """
        The scheduler to use for the model.
        """
        DPM_SOLVER = "DPM-SOLVER"
        SA_SOLVER = "SA-SOLVER"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    style: Style = Field(
        default=Style.NO_STYLE, description="The style to apply to the image."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.DPM_SOLVER, description="The scheduler to use for the model."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=4.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=35, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "style": self.style.value,
            "scheduler": self.scheduler.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/pixart-sigma",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FluxSubject(FALNode):
    """
    Super fast endpoint for the FLUX.1 [schnell] model with subject input capabilities, enabling rapid and high-quality image generation for personalization, specific styles, brand identities, and product-specific outputs.
    flux, generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to generate an image from."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="URL of image of the subject"
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    guidance_scale: float = Field(
        default=3.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of the model will output the same image every time."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_base64}",
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/flux-subject",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class SdxlControlnetUnion(FALNode):
    """
    An efficent SDXL multi-controlnet text-to-image model.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    depth_preprocess: bool = Field(
        default=True, description="Whether to preprocess the depth image."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. Leave it none to automatically infer from the control image."
    )
    normal_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the control image."
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    teed_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the control image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The list of LoRA weights to use."
    )
    guidance_scale: float = Field(
        default=7.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    canny_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the control image."
    )
    segmentation_preprocess: bool = Field(
        default=True, description="Whether to preprocess the segmentation image."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    request_id: str = Field(
        default="", description="An id bound to a request, can be used with response to identify the request itself."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    segmentation_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the control image."
    )
    openpose_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the control image."
    )
    canny_preprocess: bool = Field(
        default=True, description="Whether to preprocess the canny image."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    depth_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the control image."
    )
    normal_preprocess: bool = Field(
        default=True, description="Whether to preprocess the normal image."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    teed_preprocess: bool = Field(
        default=True, description="Whether to preprocess the teed image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    controlnet_conditioning_scale: float = Field(
        default=0.5, description="The scale of the controlnet conditioning."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    openpose_preprocess: bool = Field(
        default=True, description="Whether to preprocess the openpose image."
    )
    num_inference_steps: int = Field(
        default=35, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        normal_image_base64 = await context.image_to_base64(self.normal_image)
        teed_image_base64 = await context.image_to_base64(self.teed_image)
        canny_image_base64 = await context.image_to_base64(self.canny_image)
        segmentation_image_base64 = await context.image_to_base64(self.segmentation_image)
        openpose_image_base64 = await context.image_to_base64(self.openpose_image)
        depth_image_base64 = await context.image_to_base64(self.depth_image)
        arguments = {
            "prompt": self.prompt,
            "depth_preprocess": self.depth_preprocess,
            "image_size": self.image_size,
            "normal_image_url": f"data:image/png;base64,{normal_image_base64}",
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "teed_image_url": f"data:image/png;base64,{teed_image_base64}",
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_scale": self.guidance_scale,
            "canny_image_url": f"data:image/png;base64,{canny_image_base64}",
            "segmentation_preprocess": self.segmentation_preprocess,
            "format": self.format.value,
            "sync_mode": self.sync_mode,
            "request_id": self.request_id,
            "seed": self.seed,
            "segmentation_image_url": f"data:image/png;base64,{segmentation_image_base64}",
            "openpose_image_url": f"data:image/png;base64,{openpose_image_base64}",
            "canny_preprocess": self.canny_preprocess,
            "expand_prompt": self.expand_prompt,
            "depth_image_url": f"data:image/png;base64,{depth_image_base64}",
            "normal_preprocess": self.normal_preprocess,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "teed_preprocess": self.teed_preprocess,
            "num_images": self.num_images,
            "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
            "safety_checker_version": self.safety_checker_version.value,
            "openpose_preprocess": self.openpose_preprocess,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/sdxl-controlnet-union",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Kolors(FALNode):
    """
    Photorealistic Text-to-Image
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class Scheduler(Enum):
        """
        The scheduler to use for the model.
        """
        EULERDISCRETESCHEDULER = "EulerDiscreteScheduler"
        EULERANCESTRALDISCRETESCHEDULER = "EulerAncestralDiscreteScheduler"
        DPMSOLVERMULTISTEPSCHEDULER = "DPMSolverMultistepScheduler"
        DPMSOLVERMULTISTEPSCHEDULER_SDE_KARRAS = "DPMSolverMultistepScheduler_SDE_karras"
        UNIPCMULTISTEPSCHEDULER = "UniPCMultistepScheduler"
        DEISMULTISTEPSCHEDULER = "DEISMultistepScheduler"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULERDISCRETESCHEDULER, description="The scheduler to use for the model."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    num_inference_steps: int = Field(
        default=50, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="Seed"
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "scheduler": self.scheduler.value,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kolors",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class StableCascade(FALNode):
    """
    Stable Cascade: Image generation on a smaller & cheaper latent space.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    second_stage_guidance_scale: float = Field(
        default=0, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the image will be returned as base64 encoded string."
    )
    first_stage_steps: int = Field(
        default=20, description="Number of steps to run the first stage for."
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Cascade will output the same image every time."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to false, the safety checker will be disabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    second_stage_steps: int = Field(
        default=10, description="Number of steps to run the second stage for."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "second_stage_guidance_scale": self.second_stage_guidance_scale,
            "sync_mode": self.sync_mode,
            "first_stage_steps": self.first_stage_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "second_stage_steps": self.second_stage_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-cascade",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FastSdxl(FALNode):
    """
    Run SDXL at the speed of light
    generation, text-to-image, txt2img, ai-art, fast

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The list of LoRA weights to use."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    guidance_scale: float = Field(
        default=7.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    request_id: str = Field(
        default="", description="An id bound to a request, can be used with response to identify the request itself."
    )
    num_inference_steps: int = Field(
        default=25, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "format": self.format.value,
            "num_images": self.num_images,
            "sync_mode": self.sync_mode,
            "safety_checker_version": self.safety_checker_version.value,
            "request_id": self.request_id,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-sdxl",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class StableCascadeSoteDiffusion(FALNode):
    """
    Anime finetune of Würstchen V3.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="", description="The size of the generated image."
    )
    second_stage_guidance_scale: float = Field(
        default=2, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the image will be returned as base64 encoded string."
    )
    first_stage_steps: int = Field(
        default=25, description="Number of steps to run the first stage for."
    )
    guidance_scale: float = Field(
        default=8, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Cascade will output the same image every time."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to false, the safety checker will be disabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    second_stage_steps: int = Field(
        default=10, description="Number of steps to run the second stage for."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "second_stage_guidance_scale": self.second_stage_guidance_scale,
            "sync_mode": self.sync_mode,
            "first_stage_steps": self.first_stage_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "second_stage_steps": self.second_stage_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-cascade/sote-diffusion",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LightningModels(FALNode):
    """
    Collection of SDXL Lightning models.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Scheduler(Enum):
        """
        Scheduler / sampler to use for the image denoising process.
        """
        DPM_PLUS_PLUS_2M = "DPM++ 2M"
        DPM_PLUS_PLUS_2M_KARRAS = "DPM++ 2M Karras"
        DPM_PLUS_PLUS_2M_SDE = "DPM++ 2M SDE"
        DPM_PLUS_PLUS_2M_SDE_KARRAS = "DPM++ 2M SDE Karras"
        DPM_PLUS_PLUS_SDE = "DPM++ SDE"
        DPM_PLUS_PLUS_SDE_KARRAS = "DPM++ SDE Karras"
        KDPM_2A = "KDPM 2A"
        EULER = "Euler"
        EULER_TRAILING_TIMESTEPS = "Euler (trailing timesteps)"
        EULER_A = "Euler A"
        LCM = "LCM"
        EDMDPMSOLVERMULTISTEPSCHEDULER = "EDMDPMSolverMultistepScheduler"
        TCDSCHEDULER = "TCDScheduler"

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default=""
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The list of LoRA weights to use."
    )
    scheduler: Scheduler | None = Field(
        default=None, description="Scheduler / sampler to use for the image denoising process."
    )
    guidance_scale: float = Field(
        default=2, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)", description="The negative prompt to use. Use it to address details that you don't want in the image."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    model_name: str = Field(
        default="", description="The Lightning model to use."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    num_inference_steps: int = Field(
        default=5, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "scheduler": self.scheduler.value if self.scheduler else None,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "format": self.format.value,
            "num_images": self.num_images,
            "model_name": self.model_name,
            "sync_mode": self.sync_mode,
            "safety_checker_version": self.safety_checker_version.value,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/lightning-models",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class PlaygroundV25(FALNode):
    """
    State-of-the-art open-source model in aesthetic quality
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    guidance_rescale: float = Field(
        default=0, description="The rescale factor for the CFG."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    guidance_scale: float = Field(
        default=3, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    request_id: str = Field(
        default="", description="An id bound to a request, can be used with response to identify the request itself."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    num_inference_steps: int = Field(
        default=25, description="The number of inference steps to perform."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "expand_prompt": self.expand_prompt,
            "guidance_rescale": self.guidance_rescale,
            "enable_safety_checker": self.enable_safety_checker,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "format": self.format.value,
            "num_images": self.num_images,
            "safety_checker_version": self.safety_checker_version.value,
            "request_id": self.request_id,
            "seed": self.seed,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/playground-v25",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class RealisticVision(FALNode):
    """
    Generate realistic images.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default=""
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The list of LoRA weights to use."
    )
    guidance_rescale: float = Field(
        default=0, description="The rescale factor for the CFG."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)", description="The negative prompt to use. Use it to address details that you don't want in the image."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    model_name: str = Field(
        default="", description="The Realistic Vision model to use."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    request_id: str = Field(
        default="", description="An id bound to a request, can be used with response to identify the request itself."
    )
    num_inference_steps: int = Field(
        default=35, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_rescale": self.guidance_rescale,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "format": self.format.value,
            "num_images": self.num_images,
            "model_name": self.model_name,
            "sync_mode": self.sync_mode,
            "safety_checker_version": self.safety_checker_version.value,
            "request_id": self.request_id,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/realistic-vision",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Dreamshaper(FALNode):
    """
    Dreamshaper model.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class ModelName(Enum):
        """
        The Dreamshaper model to use.
        """
        LYKON_DREAMSHAPER_XL_1_0 = "Lykon/dreamshaper-xl-1-0"
        LYKON_DREAMSHAPER_XL_V2_TURBO = "Lykon/dreamshaper-xl-v2-turbo"
        LYKON_DREAMSHAPER_8 = "Lykon/dreamshaper-8"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default=""
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The list of LoRA weights to use."
    )
    guidance_scale: float = Field(
        default=5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)", description="The negative prompt to use. Use it to address details that you don't want in the image."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    model_name: ModelName | None = Field(
        default=None, description="The Dreamshaper model to use."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    num_inference_steps: int = Field(
        default=35, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "format": self.format.value,
            "num_images": self.num_images,
            "model_name": self.model_name.value if self.model_name else None,
            "sync_mode": self.sync_mode,
            "safety_checker_version": self.safety_checker_version.value,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/dreamshaper",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class StableDiffusionV15(FALNode):
    """
    Stable Diffusion v1.5
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default="square", description="The size of the generated image."
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The list of LoRA weights to use."
    )
    guidance_scale: float = Field(
        default=7.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    request_id: str = Field(
        default="", description="An id bound to a request, can be used with response to identify the request itself."
    )
    num_inference_steps: int = Field(
        default=25, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "format": self.format.value,
            "num_images": self.num_images,
            "sync_mode": self.sync_mode,
            "safety_checker_version": self.safety_checker_version.value,
            "request_id": self.request_id,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/stable-diffusion-v15",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class LayerDiffusion(FALNode):
    """
    SDXL with an alpha channel.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    guidance_scale: float = Field(
        default=8, description="The guidance scale for the model."
    )
    num_inference_steps: int = Field(
        default=20, description="The number of inference steps for the model."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    negative_prompt: str = Field(
        default="text, watermark", description="The prompt to use for generating the negative image. Be as descriptive as possible for best results."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to false, the safety checker will be disabled."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        arguments = {
            "prompt": self.prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "enable_safety_checker": self.enable_safety_checker,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/layer-diffusion",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FastLightningSdxl(FALNode):
    """
    Run SDXL at the speed of light
    generation, text-to-image, txt2img, ai-art, fast

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"

    class NumInferenceSteps(Enum):
        """
        The number of inference steps to perform.
        """
        VALUE_1 = "1"
        VALUE_2 = "2"
        VALUE_4 = "4"
        VALUE_8 = "8"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    guidance_rescale: float = Field(
        default=0, description="The rescale factor for the CFG."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    num_inference_steps: NumInferenceSteps = Field(
        default=4, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    request_id: str = Field(
        default="", description="An id bound to a request, can be used with response to identify the request itself."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "format": self.format.value,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "expand_prompt": self.expand_prompt,
            "sync_mode": self.sync_mode,
            "guidance_rescale": self.guidance_rescale,
            "safety_checker_version": self.safety_checker_version.value,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps.value,
            "seed": self.seed,
            "request_id": self.request_id,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-lightning-sdxl",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FastFooocusSdxlImageToImage(FALNode):
    """
    Fooocus extreme speed mode as a standalone app.
    generation, text-to-image, txt2img, ai-art, fast

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    enable_refiner: bool = Field(
        default=True, description="If set to true, a smaller model will try to refine the output after it was processed."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. Leave it none to automatically infer from the prompt image."
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    expand_prompt: bool = Field(
        default=True, description="If set to true, the prompt will be expanded with additional prompts."
    )
    guidance_rescale: float = Field(
        default=0, description="The rescale factor for the CFG."
    )
    guidance_scale: float = Field(
        default=2, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use.Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The URL of the image to use as a starting point for the generation."
    )
    strength: float = Field(
        default=0.95, description="determines how much the generated image resembles the initial image"
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "enable_refiner": self.enable_refiner,
            "image_size": self.image_size,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "expand_prompt": self.expand_prompt,
            "guidance_rescale": self.guidance_rescale,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "format": self.format.value,
            "num_images": self.num_images,
            "image_url": f"data:image/png;base64,{image_base64}",
            "strength": self.strength,
            "safety_checker_version": self.safety_checker_version.value,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-fooocus-sdxl/image-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FastSdxlControlnetCanny(FALNode):
    """
    Generate Images with ControlNet.
    generation, text-to-image, txt2img, ai-art, fast

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. Leave it none to automatically infer from the control image."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The list of LoRA weights to use."
    )
    guidance_scale: float = Field(
        default=7.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    controlnet_conditioning_scale: float = Field(
        default=0.5, description="The scale of the controlnet conditioning."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    control_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the control image."
    )
    num_inference_steps: int = Field(
        default=25, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    enable_deep_cache: bool = Field(
        default=False, description="If set to true, DeepCache will be enabled. TBD"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        control_image_base64 = await context.image_to_base64(self.control_image)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "expand_prompt": self.expand_prompt,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
            "sync_mode": self.sync_mode,
            "control_image_url": f"data:image/png;base64,{control_image_base64}",
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "enable_deep_cache": self.enable_deep_cache,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-sdxl-controlnet-canny",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FastLcmDiffusion(FALNode):
    """
    Run SDXL at the speed of light
    generation, text-to-image, txt2img, ai-art, fast

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class ModelName(Enum):
        """
        The name of the model to use.
        """
        STABILITYAI_STABLE_DIFFUSION_XL_BASE_1_0 = "stabilityai/stable-diffusion-xl-base-1.0"
        RUNWAYML_STABLE_DIFFUSION_V1_5 = "runwayml/stable-diffusion-v1-5"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    expand_prompt: bool = Field(
        default=False, description="If set to true, the prompt will be expanded with additional prompts."
    )
    guidance_rescale: float = Field(
        default=0, description="The rescale factor for the CFG."
    )
    guidance_scale: float = Field(
        default=1.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    model_name: ModelName = Field(
        default=ModelName.STABILITYAI_STABLE_DIFFUSION_XL_BASE_1_0, description="The name of the model to use."
    )
    sync_mode: bool = Field(
        default=True, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    request_id: str = Field(
        default="", description="An id bound to a request, can be used with response to identify the request itself."
    )
    num_inference_steps: int = Field(
        default=6, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "expand_prompt": self.expand_prompt,
            "guidance_rescale": self.guidance_rescale,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "format": self.format.value,
            "num_images": self.num_images,
            "model_name": self.model_name.value,
            "sync_mode": self.sync_mode,
            "safety_checker_version": self.safety_checker_version.value,
            "request_id": self.request_id,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-lcm-diffusion",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FastFooocusSdxl(FALNode):
    """
    Fooocus extreme speed mode as a standalone app.
    generation, text-to-image, txt2img, ai-art, fast

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Format(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"

    class SafetyCheckerVersion(Enum):
        """
        The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model.
        """
        V1 = "v1"
        V2 = "v2"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    enable_refiner: bool = Field(
        default=True, description="If set to true, a smaller model will try to refine the output after it was processed."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image."
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The list of embeddings to use."
    )
    expand_prompt: bool = Field(
        default=True, description="If set to true, the prompt will be expanded with additional prompts."
    )
    guidance_rescale: float = Field(
        default=0, description="The rescale factor for the CFG."
    )
    guidance_scale: float = Field(
        default=2, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    format: Format = Field(
        default=Format.JPEG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate."
    )
    safety_checker_version: SafetyCheckerVersion = Field(
        default=SafetyCheckerVersion.V1, description="The version of the safety checker to use. v1 is the default CompVis safety checker. v2 uses a custom ViT model."
    )
    num_inference_steps: int = Field(
        default=8, description="The number of inference steps to perform."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "enable_refiner": self.enable_refiner,
            "image_size": self.image_size,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "expand_prompt": self.expand_prompt,
            "guidance_rescale": self.guidance_rescale,
            "guidance_scale": self.guidance_scale,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "format": self.format.value,
            "num_images": self.num_images,
            "safety_checker_version": self.safety_checker_version.value,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fast-fooocus-sdxl",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class IllusionDiffusion(FALNode):
    """
    Create illusions conditioned on image.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Scheduler(Enum):
        """
        Scheduler / sampler to use for the image denoising process.
        """
        DPM_PLUS_PLUS_KARRAS_SDE = "DPM++ Karras SDE"
        EULER = "Euler"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image. You can choose between some presets or custom height and width that **must be multiples of 8**."
    )
    controlnet_conditioning_scale: float = Field(
        default=1, description="The scale of the ControlNet."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Input image url."
    )
    scheduler: Scheduler = Field(
        default=Scheduler.EULER, description="Scheduler / sampler to use for the image denoising process."
    )
    control_guidance_start: float = Field(
        default=0
    )
    guidance_scale: float = Field(
        default=7.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="Seed of the generated Image. It will be the same value of the one passed in the input or the randomly generated that was used in case none was passed."
    )
    control_guidance_end: float = Field(
        default=1
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    num_inference_steps: int = Field(
        default=40, description="Increasing the amount of steps tells Stable Diffusion that it should take more steps to generate your final result which can increase the amount of detail in your image."
    )

    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "controlnet_conditioning_scale": self.controlnet_conditioning_scale,
            "image_url": f"data:image/png;base64,{image_base64}",
            "scheduler": self.scheduler.value,
            "control_guidance_start": self.control_guidance_start,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "control_guidance_end": self.control_guidance_end,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/illusion-diffusion",
            arguments=arguments,
        )
        return res

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FooocusImagePrompt(FALNode):
    """
    Default parameters with automated optimizations and quality improvements.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Performance(Enum):
        """
        You can choose Speed or Quality
        """
        SPEED = "Speed"
        QUALITY = "Quality"
        EXTREME_SPEED = "Extreme Speed"
        LIGHTNING = "Lightning"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"

    class RefinerModel(Enum):
        """
        Refiner (SDXL or SD 1.5)
        """
        NONE = "None"
        REALISTICVISIONV60B1_V51VAE_SAFETENSORS = "realisticVisionV60B1_v51VAE.safetensors"

    class InpaintMode(Enum):
        """
        The mode to use for inpainting.
        """
        INPAINT_OR_OUTPAINT_DEFAULT = "Inpaint or Outpaint (default)"
        IMPROVE_DETAIL_FACE_HAND_EYES_ETC = "Improve Detail (face, hand, eyes, etc.)"
        MODIFY_CONTENT_ADD_OBJECTS_CHANGE_BACKGROUND_ETC = "Modify Content (add objects, change background, etc.)"

    class UovMethod(Enum):
        """
        The method to use for upscaling or varying.
        """
        DISABLED = "Disabled"
        VARY_SUBTLE = "Vary (Subtle)"
        VARY_STRONG = "Vary (Strong)"
        UPSCALE_1_5X = "Upscale (1.5x)"
        UPSCALE_2X = "Upscale (2x)"
        UPSCALE_FAST_2X = "Upscale (Fast 2x)"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    uov_image: ImageRef = Field(
        default=ImageRef(), description="The image to upscale or vary."
    )
    performance: Performance = Field(
        default=Performance.EXTREME_SPEED, description="You can choose Speed or Quality"
    )
    image_prompt_3: str = Field(
        default=""
    )
    styles: list[str] = Field(
        default=[], description="The style to use."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use up to 5 LoRAs and they will be merged together to generate the final image."
    )
    image_prompt_4: str = Field(
        default=""
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    sharpness: float = Field(
        default=2, description="The sharpness of the generated image. Use it to control how sharp the generated image should be. Higher value means image and texture are sharper."
    )
    mixing_image_prompt_and_inpaint: bool = Field(
        default=False, description="Mixing Image Prompt and Inpaint"
    )
    outpaint_selections: list[str] = Field(
        default=[], description="The directions to outpaint."
    )
    inpaint_image: ImageRef = Field(
        default=ImageRef(), description="The image to use as a reference for inpainting."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    refiner_model: RefinerModel = Field(
        default=RefinerModel.NONE, description="Refiner (SDXL or SD 1.5)"
    )
    image_prompt_2: str = Field(
        default=""
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    inpaint_mode: InpaintMode = Field(
        default=InpaintMode.INPAINT_OR_OUTPAINT_DEFAULT, description="The mode to use for inpainting."
    )
    uov_method: UovMethod = Field(
        default=UovMethod.DISABLED, description="The method to use for upscaling or varying."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    refiner_switch: float = Field(
        default=0.8, description="Use 0.4 for SD1.5 realistic models; 0.667 for SD1.5 anime models 0.8 for XL-refiners; or any value for switching two SDXL models."
    )
    mixing_image_prompt_and_vary_upscale: bool = Field(
        default=False, description="Mixing Image Prompt and Vary/Upscale"
    )
    mask_image: ImageRef = Field(
        default=ImageRef(), description="The image to use as a mask for the generated image."
    )
    image_prompt_1: str = Field(
        default=""
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to false, the safety checker will be disabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate in one request"
    )
    aspect_ratio: str = Field(
        default="1024x1024", description="The size of the generated image. You can choose between some presets or custom height and width that **must be multiples of 8**."
    )
    inpaint_additional_prompt: str = Field(
        default="", description="Describe what you want to inpaint."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        uov_image_base64 = await context.image_to_base64(self.uov_image)
        inpaint_image_base64 = await context.image_to_base64(self.inpaint_image)
        mask_image_base64 = await context.image_to_base64(self.mask_image)
        arguments = {
            "prompt": self.prompt,
            "uov_image_url": f"data:image/png;base64,{uov_image_base64}",
            "performance": self.performance.value,
            "image_prompt_3": self.image_prompt_3,
            "styles": self.styles,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "image_prompt_4": self.image_prompt_4,
            "guidance_scale": self.guidance_scale,
            "sharpness": self.sharpness,
            "mixing_image_prompt_and_inpaint": self.mixing_image_prompt_and_inpaint,
            "outpaint_selections": self.outpaint_selections,
            "inpaint_image_url": f"data:image/png;base64,{inpaint_image_base64}",
            "output_format": self.output_format.value,
            "refiner_model": self.refiner_model.value,
            "image_prompt_2": self.image_prompt_2,
            "sync_mode": self.sync_mode,
            "inpaint_mode": self.inpaint_mode.value,
            "uov_method": self.uov_method.value,
            "seed": self.seed,
            "refiner_switch": self.refiner_switch,
            "mixing_image_prompt_and_vary_upscale": self.mixing_image_prompt_and_vary_upscale,
            "mask_image_url": f"data:image/png;base64,{mask_image_base64}",
            "image_prompt_1": self.image_prompt_1,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio,
            "inpaint_additional_prompt": self.inpaint_additional_prompt,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fooocus/image-prompt",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class FooocusInpaint(FALNode):
    """
    Default parameters with automated optimizations and quality improvements.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Performance(Enum):
        """
        You can choose Speed or Quality
        """
        SPEED = "Speed"
        QUALITY = "Quality"
        EXTREME_SPEED = "Extreme Speed"
        LIGHTNING = "Lightning"

    class RefinerModel(Enum):
        """
        Refiner (SDXL or SD 1.5)
        """
        NONE = "None"
        REALISTICVISIONV60B1_V51VAE_SAFETENSORS = "realisticVisionV60B1_v51VAE.safetensors"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"

    class InpaintMode(Enum):
        """
        The mode to use for inpainting.
        """
        INPAINT_OR_OUTPAINT_DEFAULT = "Inpaint or Outpaint (default)"
        IMPROVE_DETAIL_FACE_HAND_EYES_ETC = "Improve Detail (face, hand, eyes, etc.)"
        MODIFY_CONTENT_ADD_OBJECTS_CHANGE_BACKGROUND_ETC = "Modify Content (add objects, change background, etc.)"

    class InpaintEngine(Enum):
        """
        Version of Fooocus inpaint model
        """
        NONE = "None"
        V1 = "v1"
        V2_5 = "v2.5"
        V2_6 = "v2.6"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    performance: Performance = Field(
        default=Performance.EXTREME_SPEED, description="You can choose Speed or Quality"
    )
    styles: list[str] = Field(
        default=[], description="The style to use."
    )
    image_prompt_3: str = Field(
        default=""
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use up to 5 LoRAs and they will be merged together to generate the final image."
    )
    image_prompt_4: str = Field(
        default=""
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    sharpness: float = Field(
        default=2, description="The sharpness of the generated image. Use it to control how sharp the generated image should be. Higher value means image and texture are sharper."
    )
    mixing_image_prompt_and_inpaint: bool = Field(
        default=False, description="Mixing Image Prompt and Inpaint"
    )
    outpaint_selections: list[str] = Field(
        default=[], description="The directions to outpaint."
    )
    inpaint_image: ImageRef = Field(
        default=ImageRef(), description="The image to use as a reference for inpainting."
    )
    refiner_model: RefinerModel = Field(
        default=RefinerModel.NONE, description="Refiner (SDXL or SD 1.5)"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_prompt_2: str = Field(
        default=""
    )
    inpaint_respective_field: float = Field(
        default=0.618, description="The area to inpaint. Value 0 is same as \"Only Masked\" in A1111. Value 1 is same as \"Whole Image\" in A1111. Only used in inpaint, not used in outpaint. (Outpaint always use 1.0)"
    )
    inpaint_mode: InpaintMode = Field(
        default=InpaintMode.INPAINT_OR_OUTPAINT_DEFAULT, description="The mode to use for inpainting."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    refiner_switch: float = Field(
        default=0.8, description="Use 0.4 for SD1.5 realistic models; 0.667 for SD1.5 anime models 0.8 for XL-refiners; or any value for switching two SDXL models."
    )
    inpaint_disable_initial_latent: bool = Field(
        default=False, description="If set to true, the initial preprocessing will be disabled."
    )
    mask_image: ImageRef = Field(
        default=ImageRef(), description="The image to use as a mask for the generated image."
    )
    invert_mask: bool = Field(
        default=False, description="If set to true, the mask will be inverted."
    )
    image_prompt_1: str = Field(
        default=""
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to false, the safety checker will be disabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate in one request"
    )
    aspect_ratio: str = Field(
        default="1024x1024", description="The size of the generated image. You can choose between some presets or custom height and width that **must be multiples of 8**."
    )
    inpaint_additional_prompt: str = Field(
        default="", description="Describe what you want to inpaint."
    )
    inpaint_strength: float = Field(
        default=1, description="Same as the denoising strength in A1111 inpaint. Only used in inpaint, not used in outpaint. (Outpaint always use 1.0)"
    )
    override_inpaint_options: bool = Field(
        default=False, description="If set to true, the advanced inpaint options ('inpaint_disable_initial_latent', 'inpaint_engine', 'inpaint_strength', 'inpaint_respective_field', 'inpaint_erode_or_dilate') will be overridden. Otherwise, the default values will be used."
    )
    inpaint_engine: InpaintEngine = Field(
        default=InpaintEngine.V2_6, description="Version of Fooocus inpaint model"
    )
    inpaint_erode_or_dilate: float = Field(
        default=0, description="Positive value will make white area in the mask larger, negative value will make white area smaller. (default is 0, always process before any mask invert)"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        inpaint_image_base64 = await context.image_to_base64(self.inpaint_image)
        mask_image_base64 = await context.image_to_base64(self.mask_image)
        arguments = {
            "prompt": self.prompt,
            "performance": self.performance.value,
            "styles": self.styles,
            "image_prompt_3": self.image_prompt_3,
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "image_prompt_4": self.image_prompt_4,
            "guidance_scale": self.guidance_scale,
            "sharpness": self.sharpness,
            "mixing_image_prompt_and_inpaint": self.mixing_image_prompt_and_inpaint,
            "outpaint_selections": self.outpaint_selections,
            "inpaint_image_url": f"data:image/png;base64,{inpaint_image_base64}",
            "refiner_model": self.refiner_model.value,
            "output_format": self.output_format.value,
            "image_prompt_2": self.image_prompt_2,
            "inpaint_respective_field": self.inpaint_respective_field,
            "inpaint_mode": self.inpaint_mode.value,
            "sync_mode": self.sync_mode,
            "seed": self.seed,
            "refiner_switch": self.refiner_switch,
            "inpaint_disable_initial_latent": self.inpaint_disable_initial_latent,
            "mask_image_url": f"data:image/png;base64,{mask_image_base64}",
            "invert_mask": self.invert_mask,
            "image_prompt_1": self.image_prompt_1,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio,
            "inpaint_additional_prompt": self.inpaint_additional_prompt,
            "inpaint_strength": self.inpaint_strength,
            "override_inpaint_options": self.override_inpaint_options,
            "inpaint_engine": self.inpaint_engine.value,
            "inpaint_erode_or_dilate": self.inpaint_erode_or_dilate,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fooocus/inpaint",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Lcm(FALNode):
    """
    Produce high-quality images with minimal inference steps.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Model(Enum):
        """
        The model to use for generating the image.
        """
        SDXL = "sdxl"
        SDV1_5 = "sdv1-5"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    controlnet_inpaint: bool = Field(
        default=False, description="If set to true, the inpainting pipeline will use controlnet inpainting. Only effective for inpainting pipelines."
    )
    image_size: str = Field(
        default="", description="The size of the generated image. You can choose between some presets or custom height and width that **must be multiples of 8**. If not provided: - For text-to-image generations, the default size is 512x512. - For image-to-image generations, the default size is the same as the input image. - For inpainting generations, the default size is the same as the input image."
    )
    enable_safety_checks: bool = Field(
        default=True, description="If set to true, the resulting image will be checked whether it includes any potentially unsafe content. If it does, it will be replaced with a black image."
    )
    model: Model = Field(
        default=Model.SDV1_5, description="The model to use for generating the image."
    )
    lora: ImageRef = Field(
        default=ImageRef(), description="The url of the lora server to use for image generation."
    )
    guidance_scale: float = Field(
        default=1, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use.Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    inpaint_mask_only: bool = Field(
        default=False, description="If set to true, the inpainting pipeline will only inpaint the provided mask area. Only effective for inpainting pipelines."
    )
    num_images: int = Field(
        default=1, description="The number of images to generate. The function will return a list of images with the same prompt and negative prompt but different seeds."
    )
    lora_scale: float = Field(
        default=1, description="The scale of the lora server to use for image generation."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="The base image to use for guiding the image generation on image-to-image generations. If the either width or height of the image is larger than 1024 pixels, the image will be resized to 1024 pixels while keeping the aspect ratio."
    )
    strength: float = Field(
        default=0.8, description="The strength of the image that is passed as `image_url`. The strength determines how much the generated image will be similar to the image passed as `image_url`. The higher the strength the more model gets \"creative\" and generates an image that's different from the initial image. A strength of 1.0 means that the initial image is more or less ignored and the model will try to generate an image that's as close as possible to the prompt."
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    request_id: str = Field(
        default="", description="An id bound to a request, can be used with response to identify the request itself."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    mask: ImageRef = Field(
        default=ImageRef(), description="The mask to use for guiding the image generation on image inpainting. The model will focus on the mask area and try to fill it with the most relevant content. The mask must be a black and white image where the white area is the area that needs to be filled and the black area is the area that should be ignored. The mask must have the same dimensions as the image passed as `image_url`."
    )
    num_inference_steps: int = Field(
        default=4, description="The number of inference steps to use for generating the image. The more steps the better the image will be but it will also take longer to generate."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        lora_base64 = await context.image_to_base64(self.lora)
        image_base64 = await context.image_to_base64(self.image)
        mask_base64 = await context.image_to_base64(self.mask)
        arguments = {
            "prompt": self.prompt,
            "controlnet_inpaint": self.controlnet_inpaint,
            "image_size": self.image_size,
            "enable_safety_checks": self.enable_safety_checks,
            "model": self.model.value,
            "lora_url": f"data:image/png;base64,{lora_base64}",
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "inpaint_mask_only": self.inpaint_mask_only,
            "num_images": self.num_images,
            "lora_scale": self.lora_scale,
            "image_url": f"data:image/png;base64,{image_base64}",
            "strength": self.strength,
            "sync_mode": self.sync_mode,
            "request_id": self.request_id,
            "seed": self.seed,
            "mask_url": f"data:image/png;base64,{mask_base64}",
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/lcm",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class DiffusionEdge(FALNode):
    """
    Diffusion based high quality edge detection
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    image: ImageRef = Field(
        default=ImageRef(), description="The text prompt you would like to convert to speech."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_base64 = await context.image_to_base64(self.image)
        arguments = {
            "image_url": f"data:image/png;base64,{image_base64}",
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/diffusion-edge",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class KlingImageO3TextToImage(FALNode):
    """
    Kling Image O3 generates high-quality images from text prompts with refined detail.
    image, generation, kling, o3, text-to-image, txt2img

    Use cases:
    - Generate images from detailed text prompts
    - Create high-fidelity concept art
    - Produce marketing visuals from descriptions
    - Generate creative illustrations from ideas
    - Create polished images for presentations
    """

    class AspectRatio(Enum):
        """
        Aspect ratio of generated images.
        """
        RATIO_16_9 = "16:9"
        RATIO_9_16 = "9:16"
        RATIO_1_1 = "1:1"
        RATIO_4_3 = "4:3"
        RATIO_3_4 = "3:4"
        RATIO_3_2 = "3:2"
        RATIO_2_3 = "2:3"
        RATIO_21_9 = "21:9"

    class Resolution(Enum):
        """
        Image generation resolution. 1K: standard, 2K: high-res, 4K: ultra high-res.
        """
        VALUE_1K = "1K"
        VALUE_2K = "2K"
        VALUE_4K = "4K"

    class ResultType(Enum):
        """
        Result type. 'single' for one image, 'series' for a series of related images.
        """
        SINGLE = "single"
        SERIES = "series"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"
        WEBP = "webp"


    prompt: str = Field(
        default="", description="Text prompt for image generation. Max 2500 characters."
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.RATIO_16_9, description="Aspect ratio of generated images."
    )
    resolution: Resolution = Field(
        default=Resolution.VALUE_1K, description="Image generation resolution. 1K: standard, 2K: high-res, 4K: ultra high-res."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate (1-9). Only used when result_type is 'single'."
    )
    series_amount: int = Field(
        default=0, description="Number of images in series (2-9). Only used when result_type is 'series'."
    )
    result_type: ResultType = Field(
        default=ResultType.SINGLE, description="Result type. 'single' for one image, 'series' for a series of related images."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.PNG, description="The format of the generated image."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI."
    )
    elements: list[ElementInput] = Field(
        default=[], description="Optional: Elements (characters/objects) for face control. Reference in prompt as @Element1, @Element2, etc."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "aspect_ratio": self.aspect_ratio.value,
            "resolution": self.resolution.value,
            "num_images": self.num_images,
            "series_amount": self.series_amount,
            "result_type": self.result_type.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "elements": [item.model_dump(exclude={"type"}) for item in self.elements],
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/kling-image/o3/text-to-image",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt", "resolution", "aspect_ratio"]

class Fooocus(FALNode):
    """
    Default parameters with automated optimizations and quality improvements.
    generation, text-to-image, txt2img, ai-art

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Performance(Enum):
        """
        You can choose Speed or Quality
        """
        SPEED = "Speed"
        QUALITY = "Quality"
        EXTREME_SPEED = "Extreme Speed"
        LIGHTNING = "Lightning"

    class ControlType(Enum):
        """
        The type of image control
        """
        IMAGEPROMPT = "ImagePrompt"
        PYRACANNY = "PyraCanny"
        CPDS = "CPDS"
        FACESWAP = "FaceSwap"

    class OutputFormat(Enum):
        """
        The format of the generated image.
        """
        PNG = "png"
        JPEG = "jpeg"
        WEBP = "webp"

    class RefinerModel(Enum):
        """
        Refiner (SDXL or SD 1.5)
        """
        NONE = "None"
        REALISTICVISIONV60B1_V51VAE_SAFETENSORS = "realisticVisionV60B1_v51VAE.safetensors"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    performance: Performance = Field(
        default=Performance.EXTREME_SPEED, description="You can choose Speed or Quality"
    )
    styles: list[str] = Field(
        default=[], description="The style to use."
    )
    control_type: ControlType = Field(
        default=ControlType.PYRACANNY, description="The type of image control"
    )
    mask_image: ImageRef = Field(
        default=ImageRef(), description="The image to use as a mask for the generated image."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use up to 5 LoRAs and they will be merged together to generate the final image."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to false, the safety checker will be disabled."
    )
    sharpness: float = Field(
        default=2, description="The sharpness of the generated image. Use it to control how sharp the generated image should be. Higher value means image and texture are sharper."
    )
    guidance_scale: float = Field(
        default=4, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use. Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    inpaint_image: ImageRef = Field(
        default=ImageRef(), description="The image to use as a reference for inpainting."
    )
    mixing_image_prompt_and_inpaint: bool = Field(
        default=False
    )
    aspect_ratio: str = Field(
        default="1024x1024", description="The size of the generated image. You can choose between some presets or custom height and width that **must be multiples of 8**."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate in one request"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    refiner_model: RefinerModel = Field(
        default=RefinerModel.NONE, description="Refiner (SDXL or SD 1.5)"
    )
    sync_mode: bool = Field(
        default=False, description="If set to true, the function will wait for the image to be generated and uploaded before returning the response. This will increase the latency of the function but it allows you to get the image directly in the response without going through the CDN."
    )
    control_image: ImageRef = Field(
        default=ImageRef(), description="The image to use as a reference for the generated image."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    refiner_switch: float = Field(
        default=0.8, description="Use 0.4 for SD1.5 realistic models; 0.667 for SD1.5 anime models 0.8 for XL-refiners; or any value for switching two SDXL models."
    )
    control_image_weight: float = Field(
        default=1, description="The strength of the control image. Use it to control how much the generated image should look like the control image."
    )
    control_image_stop_at: float = Field(
        default=1, description="The stop at value of the control image. Use it to control how much the generated image should look like the control image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        mask_image_base64 = await context.image_to_base64(self.mask_image)
        inpaint_image_base64 = await context.image_to_base64(self.inpaint_image)
        control_image_base64 = await context.image_to_base64(self.control_image)
        arguments = {
            "prompt": self.prompt,
            "performance": self.performance.value,
            "styles": self.styles,
            "control_type": self.control_type.value,
            "mask_image_url": f"data:image/png;base64,{mask_image_base64}",
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "enable_safety_checker": self.enable_safety_checker,
            "sharpness": self.sharpness,
            "guidance_scale": self.guidance_scale,
            "negative_prompt": self.negative_prompt,
            "inpaint_image_url": f"data:image/png;base64,{inpaint_image_base64}",
            "mixing_image_prompt_and_inpaint": self.mixing_image_prompt_and_inpaint,
            "aspect_ratio": self.aspect_ratio,
            "num_images": self.num_images,
            "output_format": self.output_format.value,
            "refiner_model": self.refiner_model.value,
            "sync_mode": self.sync_mode,
            "control_image_url": f"data:image/png;base64,{control_image_base64}",
            "seed": self.seed,
            "refiner_switch": self.refiner_switch,
            "control_image_weight": self.control_image_weight,
            "control_image_stop_at": self.control_image_stop_at,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/fooocus",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]

class Lora(FALNode):
    """
    Run Any Stable Diffusion model with customizable LoRA weights.
    generation, text-to-image, txt2img, ai-art, lora

    Use cases:
    - AI-powered art generation
    - Marketing and advertising visuals
    - Concept art and ideation
    - Social media content creation
    - Rapid prototyping and mockups
    """

    class Scheduler(Enum):
        """
        Scheduler / sampler to use for the image denoising process.
        """
        DPM_PLUS_PLUS_2M = "DPM++ 2M"
        DPM_PLUS_PLUS_2M_KARRAS = "DPM++ 2M Karras"
        DPM_PLUS_PLUS_2M_SDE = "DPM++ 2M SDE"
        DPM_PLUS_PLUS_2M_SDE_KARRAS = "DPM++ 2M SDE Karras"
        EULER = "Euler"
        EULER_A = "Euler A"
        EULER_TRAILING_TIMESTEPS = "Euler (trailing timesteps)"
        LCM = "LCM"
        LCM_TRAILING_TIMESTEPS = "LCM (trailing timesteps)"
        DDIM = "DDIM"
        TCD = "TCD"

    class PredictionType(Enum):
        """
        The type of prediction to use for the image generation.
        The `epsilon` is the default.
        """
        V_PREDICTION = "v_prediction"
        EPSILON = "epsilon"

    class ImageFormat(Enum):
        """
        The format of the generated image.
        """
        JPEG = "jpeg"
        PNG = "png"


    prompt: str = Field(
        default="", description="The prompt to use for generating the image. Be as descriptive as possible for best results."
    )
    image_size: str = Field(
        default="square_hd", description="The size of the generated image. You can choose between some presets or custom height and width that **must be multiples of 8**."
    )
    tile_height: int = Field(
        default=4096, description="The size of the tiles to be used for the image generation."
    )
    embeddings: list[Embedding] = Field(
        default=[], description="The embeddings to use for the image generation. Only a single embedding is supported at the moment. The embeddings will be used to map the tokens in the prompt to the embedding weights."
    )
    ic_light_model: ImageRef = Field(
        default=ImageRef(), description="The URL of the IC Light model to use for the image generation."
    )
    image_encoder_weight_name: str = Field(
        default="pytorch_model.bin", description="The weight name of the image encoder model to use for the image generation."
    )
    ip_adapter: list[IPAdapter] = Field(
        default=[], description="The IP adapter to use for the image generation."
    )
    loras: list[LoraWeight] = Field(
        default=[], description="The LoRAs to use for the image generation. You can use any number of LoRAs and they will be merged together to generate the final image."
    )
    scheduler: Scheduler | None = Field(
        default=None, description="Scheduler / sampler to use for the image denoising process."
    )
    sigmas: str = Field(
        default="", description="Optionally override the sigmas to use for the denoising process. Only works with schedulers which support the `sigmas` argument in their `set_sigmas` method. Defaults to not overriding, in which case the scheduler automatically sets the sigmas based on the `num_inference_steps` parameter. If set to a custom sigma schedule, the `num_inference_steps` parameter will be ignored. Cannot be set if `timesteps` is set."
    )
    guidance_scale: float = Field(
        default=7.5, description="The CFG (Classifier Free Guidance) scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    tile_stride_width: int = Field(
        default=2048, description="The stride of the tiles to be used for the image generation."
    )
    debug_per_pass_latents: bool = Field(
        default=False, description="If set to true, the latents will be saved for debugging per pass."
    )
    timesteps: str = Field(
        default="", description="Optionally override the timesteps to use for the denoising process. Only works with schedulers which support the `timesteps` argument in their `set_timesteps` method. Defaults to not overriding, in which case the scheduler automatically sets the timesteps based on the `num_inference_steps` parameter. If set to a custom timestep schedule, the `num_inference_steps` parameter will be ignored. Cannot be set if `sigmas` is set."
    )
    image_encoder_subfolder: str = Field(
        default="", description="The subfolder of the image encoder model to use for the image generation."
    )
    prompt_weighting: bool = Field(
        default=False, description="If set to true, the prompt weighting syntax will be used. Additionally, this will lift the 77 token limit by averaging embeddings."
    )
    variant: str = Field(
        default="", description="The variant of the model to use for huggingface models, e.g. 'fp16'."
    )
    model_name: str = Field(
        default="", description="URL or HuggingFace ID of the base model to generate the image."
    )
    controlnet_guess_mode: bool = Field(
        default=False, description="If set to true, the controlnet will be applied to only the conditional predictions."
    )
    seed: int = Field(
        default=-1, description="The same seed and the same prompt given to the same version of Stable Diffusion will output the same image every time."
    )
    ic_light_model_background_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the IC Light model background image to use for the image generation. Make sure to use a background compatible with the model."
    )
    rescale_betas_snr_zero: bool = Field(
        default=False, description="Whether to set the rescale_betas_snr_zero option or not for the sampler"
    )
    tile_width: int = Field(
        default=4096, description="The size of the tiles to be used for the image generation."
    )
    prediction_type: PredictionType = Field(
        default=PredictionType.EPSILON, description="The type of prediction to use for the image generation. The `epsilon` is the default."
    )
    eta: float = Field(
        default=0, description="The eta value to be used for the image generation."
    )
    image_encoder_path: str = Field(
        default="", description="The path to the image encoder model to use for the image generation."
    )
    enable_safety_checker: bool = Field(
        default=False, description="If set to true, the safety checker will be enabled."
    )
    negative_prompt: str = Field(
        default="", description="The negative prompt to use.Use it to address details that you don't want in the image. This could be colors, objects, scenery and even the small details (e.g. moustache, blurry, low resolution)."
    )
    image_format: ImageFormat = Field(
        default=ImageFormat.PNG, description="The format of the generated image."
    )
    num_images: int = Field(
        default=1, description="Number of images to generate in one request. Note that the higher the batch size, the longer it will take to generate the images."
    )
    debug_latents: bool = Field(
        default=False, description="If set to true, the latents will be saved for debugging."
    )
    ic_light_image: ImageRef = Field(
        default=ImageRef(), description="The URL of the IC Light model image to use for the image generation."
    )
    unet_name: str = Field(
        default="", description="URL or HuggingFace ID of the custom U-Net model to use for the image generation."
    )
    clip_skip: int = Field(
        default=0, description="Skips part of the image generation process, leading to slightly different results. This means the image renders faster, too."
    )
    tile_stride_height: int = Field(
        default=2048, description="The stride of the tiles to be used for the image generation."
    )
    controlnets: list[ControlNet] = Field(
        default=[], description="The control nets to use for the image generation. You can use any number of control nets and they will be applied to the image at the specified timesteps."
    )
    num_inference_steps: int = Field(
        default=30, description="Increasing the amount of steps tells Stable Diffusion that it should take more steps to generate your final result which can increase the amount of detail in your image."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        ic_light_model_base64 = await context.image_to_base64(self.ic_light_model)
        ic_light_model_background_image_base64 = await context.image_to_base64(self.ic_light_model_background_image)
        ic_light_image_base64 = await context.image_to_base64(self.ic_light_image)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "tile_height": self.tile_height,
            "embeddings": [item.model_dump(exclude={"type"}) for item in self.embeddings],
            "ic_light_model_url": f"data:image/png;base64,{ic_light_model_base64}",
            "image_encoder_weight_name": self.image_encoder_weight_name,
            "ip_adapter": [item.model_dump(exclude={"type"}) for item in self.ip_adapter],
            "loras": [item.model_dump(exclude={"type"}) for item in self.loras],
            "scheduler": self.scheduler.value if self.scheduler else None,
            "sigmas": self.sigmas,
            "guidance_scale": self.guidance_scale,
            "tile_stride_width": self.tile_stride_width,
            "debug_per_pass_latents": self.debug_per_pass_latents,
            "timesteps": self.timesteps,
            "image_encoder_subfolder": self.image_encoder_subfolder,
            "prompt_weighting": self.prompt_weighting,
            "variant": self.variant,
            "model_name": self.model_name,
            "controlnet_guess_mode": self.controlnet_guess_mode,
            "seed": self.seed,
            "ic_light_model_background_image_url": f"data:image/png;base64,{ic_light_model_background_image_base64}",
            "rescale_betas_snr_zero": self.rescale_betas_snr_zero,
            "tile_width": self.tile_width,
            "prediction_type": self.prediction_type.value,
            "eta": self.eta,
            "image_encoder_path": self.image_encoder_path,
            "enable_safety_checker": self.enable_safety_checker,
            "negative_prompt": self.negative_prompt,
            "image_format": self.image_format.value,
            "num_images": self.num_images,
            "debug_latents": self.debug_latents,
            "ic_light_image_url": f"data:image/png;base64,{ic_light_image_base64}",
            "unet_name": self.unet_name,
            "clip_skip": self.clip_skip,
            "tile_stride_height": self.tile_stride_height,
            "controlnets": [item.model_dump(exclude={"type"}) for item in self.controlnets],
            "num_inference_steps": self.num_inference_steps,
        }

        # Remove None values
        arguments = {k: v for k, v in arguments.items() if v is not None}

        res = await self.submit_request(
            context=context,
            application="fal-ai/lora",
            arguments=arguments,
        )
        assert "images" in res
        assert len(res["images"]) > 0
        return ImageRef(uri=res["images"][0]["url"])

    @classmethod
    def get_basic_fields(cls):
        return ["prompt"]
