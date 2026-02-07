from enum import Enum
from pydantic import Field
from nodetool.metadata.types import ImageRef
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


class RenderingSpeed(Enum):
    """
    The rendering speed to use.
    """
    TURBO = "TURBO"
    BALANCED = "BALANCED"
    QUALITY = "QUALITY"


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

    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="Number of images to generate"
    )
    aspect_ratio: str = Field(
        default="16:9", description="Aspect ratio for the generated image"
    )
    enhance_prompt: bool = Field(
        default=False, description="Whether to enhance the prompt for better results."
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JPEG, description="The format of the generated image."
    )
    image_url: ImageRef = Field(
        default=ImageRef(), description="The image URL to generate an image from."
    )
    sync_mode: bool = Field(
        default=False, description="If `True`, the media will be returned as a data URI and the output data won't be available in the request history."
    )
    safety_tolerance: SafetyTolerance = Field(
        default=SafetyTolerance.VALUE_2, description="The safety tolerance level for the generated image. 1 being the most strict and 5 being the most permissive."
    )
    image_prompt_strength: float = Field(
        default=0.1, description="Strength of image prompt influence (0-1)"
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )
    raw: bool = Field(
        default=False, description="Generate less processed, more natural results"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "aspect_ratio": self.aspect_ratio,
            "enhance_prompt": self.enhance_prompt,
            "output_format": self.output_format.value,
            "image_url": f"data:image/png;base64,{image_url_base64}",
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "image_prompt_strength": self.image_prompt_strength,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
            "raw": self.raw,
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

    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    num_images: int = Field(
        default=1, description="The number of images to generate. This is always set to 1 for streaming output."
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
    loras: list[str] = Field(
        default=[], description="List of LoRA models to apply with their weights"
    )
    guidance_scale: float = Field(
        default=3.5, description="How strictly to follow the prompt"
    )
    num_inference_steps: int = Field(
        default=28, description="Number of denoising steps"
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enable_safety_checker: bool = Field(
        default=True, description="Enable safety checker to filter unsafe content"
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "loras": self.loras,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "enable_safety_checker": self.enable_safety_checker,
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

    prompt: str = Field(
        default="", description="The prompt to generate an image from"
    )
    image_size: str = Field(
        default="square_hd", description="Size preset for the generated image"
    )
    style: Style = Field(
        default=Style.REALISTIC_IMAGE, description="Visual style preset for the generated image"
    )
    colors: list[str] = Field(
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
            "colors": self.colors,
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
    loras: list[str] = Field(
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
            "loras": self.loras,
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
    num_inference_steps: int = Field(
        default=28, description="The number of inference steps to perform."
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
            "image_size": self.image_size.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "safety_tolerance": self.safety_tolerance.value,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
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
    guidance_scale: float = Field(
        default=2.5, description="Guidance Scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="If set to true, the prompt will be expanded for better results."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "enable_safety_checker": self.enable_safety_checker,
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
    guidance_scale: float = Field(
        default=2.5, description="Guidance Scale is a measure of how close you want the model to stick to your prompt when looking for a related image to show you."
    )
    seed: int = Field(
        default=-1, description="Seed for reproducible results. Use -1 for random"
    )
    enable_prompt_expansion: bool = Field(
        default=False, description="If set to true, the prompt will be expanded for better results."
    )
    enable_safety_checker: bool = Field(
        default=True, description="If set to true, the safety checker will be enabled."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size.value,
            "output_format": self.output_format.value,
            "sync_mode": self.sync_mode,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "enable_prompt_expansion": self.enable_prompt_expansion,
            "enable_safety_checker": self.enable_safety_checker,
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
    image_urls: ImageRef = Field(
        default=ImageRef(), description="A set of images to use as style references (maximum total size 10MB across all style references). The images should be in JPEG, PNG or WebP format"
    )
    negative_prompt: str = Field(
        default="", description="Description of what to exclude from an image. Descriptions in the prompt take precedence to descriptions in the negative prompt."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        image_urls_base64 = await context.image_to_base64(self.image_urls)
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
            "image_urls": f"data:image/png;base64,{image_urls_base64}",
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
    input_image_urls: list[str] = Field(
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
            "input_image_urls": self.input_image_urls,
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
    loras: list[str] = Field(
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
            "loras": self.loras,
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
    loras: list[str] = Field(
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
            "loras": self.loras,
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
    loras: list[str] = Field(
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
            "loras": self.loras,
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
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for classifier-free guidance. Describes what to avoid in the image."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
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
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
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
    loras: list[str] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
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
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for classifier-free guidance. Describes what to avoid in the image."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "loras": self.loras,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
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
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for classifier-free guidance. Describes what to avoid in the image."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
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
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
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
    loras: list[str] = Field(
        default=[], description="List of LoRA weights to apply (maximum 3)."
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
    seed: int = Field(
        default=-1, description="The seed to use for the generation. If not provided, a random seed will be used."
    )
    negative_prompt: str = Field(
        default="", description="Negative prompt for classifier-free guidance. Describes what to avoid in the image."
    )
    guidance_scale: float = Field(
        default=5, description="Guidance scale for classifier-free guidance."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        arguments = {
            "prompt": self.prompt,
            "num_images": self.num_images,
            "image_size": self.image_size,
            "acceleration": self.acceleration.value,
            "output_format": self.output_format.value,
            "loras": self.loras,
            "sync_mode": self.sync_mode,
            "enable_safety_checker": self.enable_safety_checker,
            "num_inference_steps": self.num_inference_steps,
            "seed": self.seed,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
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
    image_url: ImageRef = Field(
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
        image_url_base64 = await context.image_to_base64(self.image_url)
        arguments = {
            "prompt": self.prompt,
            "image_size": self.image_size,
            "max_images": self.max_images,
            "image_url": f"data:image/png;base64,{image_url_base64}",
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